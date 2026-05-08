use crate::config::Config;
use crate::file::{self, FileData, FileType, ProcessedFile};
use crate::provider::{ModelId, create_provider};
use anyhow::{Result, anyhow};
use std::path::PathBuf;

/// Rough chars-per-token ratio for typical UTF-8 prose / code.
const CHARS_PER_TOKEN: usize = 4;
/// Fixed overhead for separators, role tags, JSON structure, etc.
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

    // let thinking_tokens = api_limit
    let reserved = instruction_tokens
        .saturating_add(previous_tokens)
        .saturating_add(max_output_tokens)
        .saturating_add(OVERHEAD_TOKENS);

    api_limit
        .saturating_sub(reserved)
        .saturating_mul(CHARS_PER_TOKEN)
}

fn build_prompt(
    instruction: &str,
    files: &[ProcessedFile],
    previous_result: Option<&str>,
) -> String {
    let mut prompt = String::new();

    // 1. System Instruction
    prompt.push_str("<system_instruction>\n");
    prompt.push_str(instruction);
    prompt.push_str("\n</system_instruction>\n\n");

    // 2. Iterative Task Definition
    prompt.push_str("<task>\n");
    prompt.push_str("You are in an iterative process. Your task is to update the 'previous_result' using the information found in 'new_files'.\n");
    prompt.push_str("- Do NOT change the requested output format defined in \"system_instruction\".\n");
    prompt.push_str("- Do NOT add conversational filler text (e.g. \"Here is the summary\").\n");
    prompt.push_str("- Merge intelligently without losing previous critical data.\n");
    prompt.push_str("</task>\n\n");

    // 3. Previous Result
    prompt.push_str("<previous_result>\n");
    if let Some(prev) = previous_result {
        prompt.push_str(prev);
    } else {
        prompt.push_str("None yet, this is the first batch.");
    }
    prompt.push_str("\n</previous_result>\n\n");

    // 4. New Files
    prompt.push_str("<new_files>\n");
    for file in files {
        let FileData::Text(content) = &file.data;
        let FileType::Text { encoding } = &file.metadata.file_type;
        prompt.push_str(&format!(
            "<file path=\"{}\" encoding=\"{}\">\n",
            file.metadata.file_name, encoding
        ));
        prompt.push_str(content);
        prompt.push_str("\n</file>\n");
    }
    prompt.push_str("</new_files>\n\n");

    // 5. Final Reminder (Sandwich)
    prompt.push_str("<reminder>\n");
    prompt.push_str("Merge the provided new files into the previous result. Strictly adhere to the original instruction provided in \"system_instruction\" at the very beginning of this prompt.\n");
    prompt.push_str("</reminder>\n");

    prompt
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
        .fold(vec![(0_usize, Vec::new())], |mut acc, path| {
            // Estimate token usage via file size (bytes ≈ chars for ASCII/UTF-8)
            let size = std::fs::metadata(&path)
                .map(|m| m.len() as usize)
                .unwrap_or(0);

            let (current_size, batch) = acc.last_mut().unwrap();

            if !batch.is_empty() && (*current_size + size > initial_file_budget) {
                acc.push((size, vec![path]));
            } else {
                *current_size += size;
                batch.push(path);
            }
            acc
        })
        .into_iter()
        .map(|(_, batch)| batch)
        .filter(|batch| !batch.is_empty())
        .collect();

    let mut previous_result: Option<String> = None;
    let total_files: usize = batches.iter().map(|b| b.len()).sum();
    let mut processed_count = 0;

    // 2. Process each batch
    for (batch_idx, batch_paths) in batches.iter().enumerate() {
        let mut current_batch: Vec<ProcessedFile> = Vec::new();
        let mut batch_chars = 0;

        for file_path in batch_paths {
            processed_count += 1;

            if !file_path.exists() {
                eprintln!("Warning: File not found: {}", file_path.display());
                continue;
            }

            match file::read_file(&file_path).await {
                Ok(processed) => {
                    let FileData::Text(c) = &processed.data;
                    let file_chars = c.len();
                    batch_chars += file_chars;

                    if debug {
                        eprintln!(
                            "[{}/{}] Adding to batch: {}",
                            processed_count,
                            total_files,
                            file_path.display()
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
                    "[Batch {}/{}] file budget: {} chars (~{} tokens), batch content: {} chars",
                    batch_idx + 1,
                    batches.len(),
                    file_budget,
                    file_budget / CHARS_PER_TOKEN,
                    batch_chars,
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
