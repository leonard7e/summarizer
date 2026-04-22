use crate::config::{Config, GeminiConfig, OllamaConfig, OpenRouterConfig};
use crate::provider;
use anyhow::{anyhow, Result};
use dialoguer::{Confirm, Input, Select};

pub async fn get_all_models(config: &Config) -> Vec<String> {
    let mut all_models = Vec::new();

    // Gemini
    if config.providers.gemini.is_some() {
        if let Ok(p) = provider::create_provider("google", config) {
            if let Ok(models) = p.list_models().await {
                for m in models {
                    all_models.push(format!("google/{}", m));
                }
            }
        }
    }

    // OpenRouter
    if config.providers.openrouter.is_some() {
        if let Ok(p) = provider::create_provider("openrouter", config) {
            if let Ok(models) = p.list_models().await {
                for m in models {
                    all_models.push(format!("openrouter/{}", m));
                }
            }
        }
    }

    // Ollama
    if let Ok(p) = provider::create_provider("ollama", config) {
        if let Ok(models) = p.list_models().await {
            for m in models {
                all_models.push(format!("ollama/{}", m));
            }
        }
    }

    all_models.sort();
    all_models
}

pub async fn select_default_model(config: &mut Config) -> Result<()> {
    eprintln!("Abfrage der verfügbaren Modelle...");
    
    let all_models = get_all_models(config).await;

    if all_models.is_empty() {
        return Err(anyhow!("Keine Modelle gefunden. Bitte konfiguriere mindestens einen Provider korrekt."));
    }

    let selection = Select::new()
        .with_prompt("Wähle das Standard-Modell aus")
        .items(&all_models)
        .default(0)
        .interact()?;

    let selected_model = all_models[selection].clone();
    config.default_model = Some(selected_model.clone());
    config.save()?;

    println!("Standard-Modell auf '{}' gesetzt und gespeichert.", selected_model);
    Ok(())
}

pub async fn run_initialization() -> Result<()> {
    let config_path = Config::path()?;
    if config_path.exists() {
        let proceed = Confirm::new()
            .with_prompt(format!(
                "Konfigurationsdatei '{}' existiert bereits. Überschreiben?",
                config_path.display()
            ))
            .default(false)
            .interact()?;
        
        if !proceed {
            println!("Initialisierung abgebrochen.");
            return Ok(());
        }
    }

    println!("--- Initialisierung von summarizer ---");

    let openrouter_key: String = Input::new()
        .with_prompt("OpenRouter API Key (leer lassen zum Überspringen)")
        .allow_empty(true)
        .interact_text()?;

    let gemini_key: String = Input::new()
        .with_prompt("Gemini API Key (leer lassen zum Überspringen)")
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

    println!("Initialisierung erfolgreich abgeschlossen.");
    Ok(())
}
