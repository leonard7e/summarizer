use crate::config::Config;
use crate::file::{self, FileData, ProcessedFile};
use crate::provider::{create_provider, ModelId};
use anyhow::{anyhow, Result};
use std::path::PathBuf;

pub async fn run_summarize_loop(files: Vec<PathBuf>, config: Config, model_str: &str, debug: bool) -> Result<()> {
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

    eprintln!("Abfrage des Kontext-Limits für Modell '{}'...", model_id.model);
    let api_limit = provider.get_context_limit(&model_id.model).await?;
    
    if debug {
        eprintln!("Context Window: {} tokens", api_limit);
        eprintln!("------------------");
    }

    let effective_limit = api_limit.saturating_sub(4000);
    let max_chars = effective_limit * 4;

    let instruction = config
        .instruction
        .clone()
        .unwrap_or_else(|| "Fasse die bisherigen Inhalte und den neuen Text logisch zusammen.".to_string());

    let mut previous_result: Option<String> = None;
    let mut current_batch: Vec<ProcessedFile> = Vec::new();
    let mut current_batch_chars = 0;

    for (i, file_path) in files.iter().enumerate() {
        if !file_path.exists() {
            eprintln!("Warning: File not found: {}", file_path.display());
            continue;
        }

        let processed = match file::read_file(file_path).await {
            Ok(p) => p,
            Err(e) => {
                eprintln!("Warning: {} - Skipping: {}", e, file_path.display());
                continue;
            }
        };

        let file_chars = match &processed.data {
            FileData::Text(c) => c.len(),
        };

        // If batch is not empty and adding this file exceeds limit, process batch
        if !current_batch.is_empty() && (current_batch_chars + file_chars > max_chars) {
            eprintln!("Batch-Limit erreicht. Sende Request für {} Dateien...", current_batch.len());
            let new_result = provider
                .complete(
                    &instruction,
                    &current_batch,
                    previous_result.as_deref(),
                    &model_id.model,
                )
                .await?;
            previous_result = Some(new_result);
            current_batch.clear();
            current_batch_chars = 0;
        }

        if debug {
            eprintln!("[{}/{}] Adding to batch: {}", i + 1, files.len(), file_path.display());
        }
        current_batch_chars += file_chars;
        current_batch.push(processed);

        // If a single file is already over the limit, we send it anyway (per plan)
        if current_batch_chars > max_chars {
             eprintln!("Einzeldatei überschreitet Kontext-Limit. Sende sofort...");
             let new_result = provider
                .complete(
                    &instruction,
                    &current_batch,
                    previous_result.as_deref(),
                    &model_id.model,
                )
                .await?;
            previous_result = Some(new_result);
            current_batch.clear();
            current_batch_chars = 0;
        }
    }

    // Process remaining batch
    if !current_batch.is_empty() {
        eprintln!("Sende finalen Batch mit {} Dateien...", current_batch.len());
        let new_result = provider
            .complete(
                &instruction,
                &current_batch,
                previous_result.as_deref(),
                &model_id.model,
            )
            .await?;
        previous_result = Some(new_result);
    }

    if let Some(final_result) = previous_result {
        println!("{}", final_result);
    }

    Ok(())
}
