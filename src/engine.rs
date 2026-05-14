use crate::config::Config;
use crate::file::{self, FileData, FileType, ProcessedFile};
use crate::provider::{ModelId, PromptPart, create_provider};
use anyhow::{Result, anyhow};
use std::path::PathBuf;

/// Rough chars-per-token ratio for typical UTF-8 prose / code.
const CHARS_PER_TOKEN: usize = 4;
/// Minimal overhead for separators, role tags, JSON structure, etc.
const OVERHEAD_TOKENS: usize = 512;

/// Returns how many **characters** of file content fit into the context window
/// after reserving space for the fixed instruction, the previous iteration's
/// result, the model's output, and structural overhead.
///
/// Recompute this before every batch so that a growing `previous_result`
/// automatically reduces the next batch size.
fn compute_file_budget(
    api_limit: usize,
    max_output_tokens: usize,
    instruction: &str,
    previous_result: Option<&str>,
) -> usize {
    let instruction_tokens = instruction.len() / CHARS_PER_TOKEN + 1;
    let previous_tokens = previous_result.map_or(0, |s| s.len() / CHARS_PER_TOKEN + 1);
    let overhead_tokens = (max_output_tokens / 16).min(OVERHEAD_TOKENS);

    let reserved = instruction_tokens
        .saturating_add(previous_tokens)
        .saturating_add(max_output_tokens)
        .saturating_add(overhead_tokens);

    api_limit
        .saturating_sub(reserved)
        .saturating_mul(CHARS_PER_TOKEN)
}

/// Estimates the token cost of an image from its raw bytes by reading the
/// image dimensions. Uses the approximation `(width * height) / 750`, aligned
/// with typical multimodal tile-based encoding costs.
///
/// Returns `None` if dimensions cannot be determined (unrecognised format,
/// truncated file, etc.).
fn try_estimate_image_tokens(data: &[u8]) -> Option<usize> {
    use std::io::Cursor;
    let (width, height) = image::ImageReader::new(Cursor::new(data))
        .with_guessed_format()
        .ok()?
        .into_dimensions()
        .ok()?;
    Some((width as usize) * (height as usize) / 750)
}

/// Returns the estimated token cost, falling back to 1000 for unknown images.
fn estimate_image_tokens(data: &[u8]) -> usize {
    try_estimate_image_tokens(data).unwrap_or(1000)
}

/// Estimates the "character equivalent" size of a file for budgeting purposes.
/// For text files: byte length (≈ char count for UTF-8).
/// For images: estimated tokens * CHARS_PER_TOKEN.
/// For audio/video: duration-based token estimate * CHARS_PER_TOKEN.
fn estimate_file_cost(processed: &ProcessedFile) -> usize {
    match &processed.data {
        FileData::Text(c) => c.len(),
        FileData::Image(bytes) => estimate_image_tokens(bytes) * CHARS_PER_TOKEN,
        FileData::Audio(_, duration) => {
            // Estimate 50 tokens per second for audio
            ((*duration * 50.0) as usize) * CHARS_PER_TOKEN
        }
        FileData::Video(_, duration) => {
            // Estimate 300 tokens per second for video
            ((*duration * 300.0) as usize) * CHARS_PER_TOKEN
        }
    }
}

fn build_prompt(
    instruction: &str,
    files: &[ProcessedFile],
    previous_result: Option<&str>,
) -> Vec<PromptPart> {
    let mut text = String::new();
    let mut parts: Vec<PromptPart> = Vec::new();

    // 1. System Instruction
    text.push_str("<system_instruction>\n");
    text.push_str(instruction);
    text.push_str("\n</system_instruction>\n\n");

    // 2. Iterative Task Definition
    text.push_str("<task>\n");
    text.push_str("You are in an iterative process. Your task is to update the 'previous_result' using the information found in 'new_files'.\n");
    text.push_str("- Do NOT change the requested output format defined in \"system_instruction\".\n");
    text.push_str("- Do NOT add conversational filler text (e.g. \"Here is the summary\").\n");
    text.push_str("- Merge intelligently without losing previous critical data.\n");
    text.push_str("</task>\n\n");

    // 3. Previous Result
    text.push_str("<previous_result>\n");
    if let Some(prev) = previous_result {
        text.push_str(prev);
    } else {
        text.push_str("None yet, this is the first batch.");
    }
    text.push_str("\n</previous_result>\n\n");

    // 4. New Files (text and images interleaved)
    text.push_str("<new_files>\n");
    for file in files {
        match &file.data {
            FileData::Text(content) => {
                let FileType::Text { encoding } = &file.metadata.file_type else {
                    continue;
                };
                text.push_str(&format!(
                    "<file path=\"{}\" encoding=\"{}\">\n",
                    file.metadata.file_name, encoding
                ));
                text.push_str(content);
                text.push_str("\n</file>\n");
            }
            FileData::Image(bytes) => {
                let FileType::Image { mime_type } = &file.metadata.file_type else {
                    continue;
                };
                text.push_str(&format!(
                    "<file path=\"{}\" type=\"image\">\n</file>\n",
                    file.metadata.file_name
                ));
                parts.push(PromptPart::Text(std::mem::take(&mut text)));
                parts.push(PromptPart::Image {
                    mime_type: mime_type.clone(),
                    data: bytes.clone(),
                });
            }
            FileData::Audio(bytes, duration) => {
                let FileType::Audio { mime_type } = &file.metadata.file_type else {
                    continue;
                };
                text.push_str(&format!(
                    "<file path=\"{}\" type=\"audio\" duration=\"{:.2}s\">\n</file>\n",
                    file.metadata.file_name, duration
                ));
                parts.push(PromptPart::Text(std::mem::take(&mut text)));
                parts.push(PromptPart::Audio {
                    mime_type: mime_type.clone(),
                    data: bytes.clone(),
                });
            }
            FileData::Video(bytes, duration) => {
                let FileType::Video { mime_type } = &file.metadata.file_type else {
                    continue;
                };
                text.push_str(&format!(
                    "<file path=\"{}\" type=\"video\" duration=\"{:.2}s\">\n</file>\n",
                    file.metadata.file_name, duration
                ));
                parts.push(PromptPart::Text(std::mem::take(&mut text)));
                parts.push(PromptPart::Video {
                    mime_type: mime_type.clone(),
                    data: bytes.clone(),
                });
            }
        }
    }
    text.push_str("</new_files>\n\n");

    // 5. Final Reminder (Sandwich)
    text.push_str("<reminder>\n");
    text.push_str("Merge the provided new files into the previous result. Strictly adhere to the original instruction provided in \"system_instruction\" at the very beginning of this prompt.\n");
    text.push_str("</reminder>\n");

    // Flush any remaining text
    if !text.is_empty() {
        parts.push(PromptPart::Text(text));
    }

    parts
}

/// Core execution loop for summarization. Processes files in batches
/// based on the model's context limit, passing the previous batch's
/// result into the next prompt to produce a rolling summary.
pub async fn run_summarize_loop(
    files: Vec<PathBuf>,
    config: Config,
    model_str: &str,
    debug: bool,
    instruction: &str,
) -> Result<()> {
    if files.is_empty() {
        return Err(anyhow!("No files provided."));
    }

    let model_id = ModelId::parse(model_str)?;
    let provider = create_provider(&model_id.provider, &config)?;

    if debug {
        eprintln!("--- Debug Info ---");
        eprintln!("Provider: {}", model_id.provider);
        eprintln!("Model:    {}", model_id.model);
    }

    let api_limit = provider.get_context_limit(&model_id.model).await?;

    if debug {
        eprintln!("Context Window: {} tokens", api_limit);
        eprintln!("Max output token: {} tokens", config.max_output_tokens);
        eprintln!("------------------");
    }

    // Budget for the *first* batch: previous_result is None yet, but we
    // conservatively assume the output may grow up to max_output_tokens chars,
    // so later batches (with a real previous_result) will automatically shrink.
    let initial_file_budget =
        compute_file_budget(api_limit, config.max_output_tokens, instruction, None);

    // 1. Convert list of files into list of batches using an iterator.
    //    We use the initial budget here; the per-batch budget is re-evaluated
    //    below once we know the actual previous_result size.
    let batches: Vec<Vec<PathBuf>> = files
        .into_iter()
        .try_fold(vec![(0_usize, Vec::new())], |mut acc, path| -> Result<Vec<(usize, Vec<PathBuf>)>> {
            // Estimate token usage via file size (bytes ≈ chars for ASCII/UTF-8).
            // For images we use a rough token estimate based on file size as a proxy
            // before we read the file, then refine during actual processing.
            let size = std::fs::metadata(&path)
                .map(|m| m.len() as usize)
                .unwrap_or(0);

            let (current_size, batch) = acc.last_mut().ok_or_else(|| anyhow!("Batch accumulator is unexpectedly empty"))?;

            if !batch.is_empty() && (*current_size + size > initial_file_budget) {
                acc.push((size, vec![path]));
            } else {
                *current_size += size;
                batch.push(path);
            }
            Ok(acc)
        })?
        .into_iter()
        .map(|(_, batch)| batch)
        .filter(|batch| !batch.is_empty())
        .collect();

    let mut previous_result: Option<String> = None;
    let total_files: usize = batches.iter().map(|b| b.len()).sum();
    let mut processed_count = 0;
    let mut images_support_checked = false;

    // 2. Process each batch
    for (batch_idx, batch_paths) in batches.iter().enumerate() {
        let mut current_batch: Vec<ProcessedFile> = Vec::new();
        let mut batch_cost = 0;

        for file_path in batch_paths {
            processed_count += 1;

            if !file_path.exists() {
                eprintln!("Warning: File not found: {}", file_path.display());
                continue;
            }

            match file::read_file(file_path).await {
                Ok(processed) => {
                    // Check multimodal support once per batch/run
                    if matches!(processed.data, FileData::Image(_)) && !images_support_checked {
                        images_support_checked = true;
                        if !provider.supports_images(&model_id.model).await? {
                            return Err(anyhow!(
                                "The model '{}' does not support image analysis.",
                                model_id.model
                            ));
                        }
                    }
                    if matches!(processed.data, FileData::Audio(_, _))
                        && !provider.supports_audio(&model_id.model).await? {
                            return Err(anyhow!(
                                "The model '{}' does not support audio analysis.",
                                model_id.model
                            ));
                    }
                    if matches!(processed.data, FileData::Video(_, _))
                        && !provider.supports_video(&model_id.model).await? {
                            return Err(anyhow!(
                                "The model '{}' does not support video analysis.",
                                model_id.model
                            ));
                    }

                    let file_cost = estimate_file_cost(&processed);
                    batch_cost += file_cost;

                    if debug {
                        eprintln!(
                            "[{}/{}] Adding to batch: {} (~{} tokens)",
                            processed_count,
                            total_files,
                            file_path.display(),
                            file_cost / CHARS_PER_TOKEN,
                        );
                    }
                    current_batch.push(processed);
                }
                Err(e) => {
                    eprintln!("Warning: {} - Skipping: {}", e, file_path.display());
                }
            }
        }

        if !current_batch.is_empty() {
            // Show percentage of completion
            eprint!("{}% ", (batch_idx * 100) / batches.len());

            if debug {
                let file_budget = compute_file_budget(
                    api_limit,
                    config.max_output_tokens,
                    instruction,
                    previous_result.as_deref(),
                );
                eprintln!(
                    "[Batch {}/{}] file budget: {} chars (~{} tokens), batch content: ~{} tokens",
                    batch_idx + 1,
                    batches.len(),
                    file_budget,
                    file_budget / CHARS_PER_TOKEN,
                    batch_cost / CHARS_PER_TOKEN,
                );
            }

            let prompt = build_prompt(instruction, &current_batch, previous_result.as_deref());
            let new_result = provider.complete(&prompt, &model_id.model).await?;
            previous_result = Some(new_result);
        }
    }

    if let Some(final_result) = previous_result {
        println!("{}", final_result);
    }

    Ok(())
}
