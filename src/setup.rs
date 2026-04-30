use crate::config::{Config, GeminiConfig, OllamaConfig, OpenRouterConfig};
use crate::provider;
use anyhow::{anyhow, Result};
use dialoguer::{Confirm, Input, Select};

pub async fn get_all_models(config: &Config) -> Vec<String> {
    let providers: &[(&str, bool)] = &[
        ("google", config.providers.gemini.is_some()),
        ("openrouter", config.providers.openrouter.is_some()),
        ("ollama", true),
    ];

    let mut all_models = Vec::new();
    for &(name, enabled) in providers {
        if !enabled { continue; }
        if let Ok(p) = provider::create_provider(name, config) {
            if let Ok(models) = p.list_models().await {
                all_models.extend(models.into_iter().map(|m| format!("{}/{}", name, m)));
            }
        }
    }

    all_models.sort();
    all_models
}

pub async fn select_default_model(config: &mut Config) -> Result<()> {
    eprintln!("Fetching available models...");
    
    let all_models = get_all_models(config).await;

    if all_models.is_empty() {
        return Err(anyhow!("No models found. Please configure at least one provider correctly."));
    }

    let selection = Select::new()
        .with_prompt("Select the default model")
        .items(&all_models)
        .default(0)
        .interact()?;

    config.default_model = Some(all_models[selection].clone());
    config.save()?;

    println!("Default model set to '{}' and saved.", config.default_model.as_ref().unwrap());
    Ok(())
}

pub async fn run_initialization() -> Result<()> {
    let config_path = Config::path()?;
    if config_path.exists() {
        let proceed = Confirm::new()
            .with_prompt(format!(
                "Configuration file '{}' already exists. Overwrite?",
                config_path.display()
            ))
            .default(false)
            .interact()?;
        
        if !proceed {
            println!("Initialization aborted.");
            return Ok(());
        }
    }

    println!("--- Summarizer Initialization ---");

    let openrouter_key: String = Input::new()
        .with_prompt("OpenRouter API Key (leave empty to skip)")
        .allow_empty(true)
        .interact_text()?;

    let gemini_key: String = Input::new()
        .with_prompt("Gemini API Key (leave empty to skip)")
        .allow_empty(true)
        .interact_text()?;

    let ollama_host: String = Input::new()
        .with_prompt("Ollama Host")
        .default("http://localhost:11434".to_string())
        .interact_text()?;

    let mut config = Config::default();

    if !openrouter_key.is_empty() {
        config.providers.openrouter = Some(OpenRouterConfig { api_key: openrouter_key });
    }

    if !gemini_key.is_empty() {
        config.providers.gemini = Some(GeminiConfig { api_key: gemini_key });
    }

    config.providers.ollama = Some(OllamaConfig { host: ollama_host, num_ctx: 4096 });

    // Save initial keys so select_default_model can use them
    config.save()?;

    // Now select default model
    select_default_model(&mut config).await?;

    println!("Initialization completed successfully.");
    Ok(())
}
