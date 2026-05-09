use crate::config::{Config, GeminiConfig, OllamaConfig, OpenAiCompatibleConfig, OpenRouterConfig};
use crate::provider;
use anyhow::{anyhow, Result};
use dialoguer::{Confirm, Input, Select};

/// Aggregates a list of all available models from enabled providers.
pub async fn get_all_models(config: &Config) -> Vec<String> {
    let static_providers = vec![
        ("google", config.providers.gemini.is_some()),
        ("openrouter", config.providers.openrouter.is_some()),
        ("ollama", config.providers.ollama.is_some()),
    ];

    let custom_providers = config
        .providers
        .openai_compatible
        .iter()
        .map(|c| (c.name.as_str(), true));

    let active_provider_names = static_providers
        .into_iter()
        .chain(custom_providers)
        .filter_map(|(name, enabled)| enabled.then_some(name));

    let mut all_models = Vec::new();
    for name in active_provider_names {
        let Ok(p) = provider::create_provider(name, config) else {
            continue;
        };
        let Ok(models) = p.list_models().await else {
            continue;
        };
        all_models.extend(models.into_iter().map(|m| format!("{}/{}", name, m)));
    }

    all_models.sort();
    all_models
}

/// Prompts the user interactively to select a default model and saves it to config.
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

/// Interactive wizard to set up the configuration file and provider API keys.
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

    loop {
        let add_openai: bool = Confirm::new()
            .with_prompt("Add an OpenAI-compatible provider (e.g. Mistral, Groq, local)?")
            .default(false)
            .interact()?;
        
        if !add_openai {
            break;
        }

        let name: String = Input::new()
            .with_prompt("Provider Name (e.g. 'mistral', 'groq')")
            .interact_text()?;
            
        let api_key: String = Input::new()
            .with_prompt("API Key")
            .interact_text()?;
            
        let base_url: String = Input::new()
            .with_prompt("Base URL")
            .interact_text()?;

        config.providers.openai_compatible.push(OpenAiCompatibleConfig {
            name,
            api_key,
            base_url,
        });
    }

    // Save initial keys so select_default_model can use them
    config.save()?;

    // Now select default model
    select_default_model(&mut config).await?;

    println!("Initialization completed successfully.");
    Ok(())
}
