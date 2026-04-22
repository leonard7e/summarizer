use crate::config::Config;
use crate::file::{self, FileData, ProcessedFile};
use crate::provider::{ModelId, create_provider};
use anyhow::{Result, anyhow};
use std::ops::Div;
use std::path::PathBuf;

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
        eprintln!(
            "Abfrage des Kontext-Limits für Modell '{}'...",
            model_id.model
        );
        eprintln!("Context Window: {} tokens", api_limit);
        eprintln!("------------------");
    }

    let effective_limit = api_limit * 2 / 3; //.saturating_sub(4000);
    let max_chars = effective_limit * 4;

    // 1. Convert list of files into list of batches using an iterator
    let batches: Vec<Vec<PathBuf>> = files
        .into_iter()
        .fold(vec![(0_usize, Vec::new())], |mut acc, path| {
            // Estimate token usage via file size (bytes roughly equal chars for ASCII/UTF-8)
            let size = std::fs::metadata(&path)
                .map(|m| m.len() as usize)
                .unwrap_or(0);

            let last_idx = acc.len() - 1;
            let (current_size, batch) = &mut acc[last_idx];

            if !batch.is_empty() && (*current_size + size > max_chars) {
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
                    let file_chars = match &processed.data {
                        FileData::Text(c) => c.len(),
                    };
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
            eprint!("{}% ", (batch_idx * 100).div(batches.len()));

            let new_result = provider
                .complete(
                    instruction,
                    &current_batch,
                    previous_result.as_deref(),
                    &model_id.model,
                )
                .await?;
            previous_result = Some(new_result);
        }
    }

    if let Some(final_result) = previous_result {
        println!("{}", final_result);
    }

    Ok(())
}
