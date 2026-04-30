use crate::config::Config;
use anyhow::{Result, anyhow};
use async_trait::async_trait;

pub mod gemini;
pub mod ollama;
pub mod openai_compatible;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ModelId {
    pub provider: String,
    pub model: String,
}

impl ModelId {
    pub fn parse(s: &str) -> Result<Self> {
        let (provider, model) = s
            .split_once('/')
            .ok_or_else(|| anyhow!("Invalid model format. Expected 'provider/model_id', got '{}'", s))?;
        
        Ok(Self {
            provider: provider.to_lowercase(),
            model: model.to_string(),
        })
    }
}

pub const DEFAULT_CONTEXT_LIMIT: usize = 8192;

#[async_trait]
pub trait LlmProvider {
    async fn complete(&self, prompt: &str, model: &str) -> Result<String>;

    async fn list_models(&self) -> Result<Vec<String>>;

    async fn get_context_limit(&self, model: &str) -> Result<usize>;
}

pub fn create_provider(provider_name: &str, config: &Config) -> Result<Box<dyn LlmProvider>> {
    match provider_name {
        "google" | "gemini" => {
            let api_key = config
                .providers
                .gemini
                .as_ref()
                .ok_or_else(|| anyhow!("Gemini provider not configured"))?
                .api_key
                .clone();
            Ok(Box::new(gemini::GeminiProvider::new(api_key)))
        }
        "openrouter" => {
            let api_key = config
                .providers
                .openrouter
                .as_ref()
                .ok_or_else(|| anyhow!("OpenRouter provider not configured"))?
                .api_key
                .clone();
            Ok(Box::new(openai_compatible::OpenAiCompatibleProvider::new(
                api_key,
                "https://openrouter.ai/api/v1".to_string(),
            )))
        }
        "openai-compatible" => {
            let conf = config
                .providers
                .openai_compatible
                .as_ref()
                .ok_or_else(|| anyhow!("OpenAI Compatible provider not configured"))?;
            Ok(Box::new(openai_compatible::OpenAiCompatibleProvider::new(
                conf.api_key.clone(),
                conf.base_url.clone(),
            )))
        }
        "ollama" => {
            let config_ollama = config.providers.ollama.clone().unwrap_or_default();
            Ok(Box::new(ollama::OllamaProvider::new(config_ollama.host, config_ollama.num_ctx)))
        }
        _ => Err(anyhow!("Unknown provider: {}", provider_name)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_model_id_parse() {
        let id = ModelId::parse("google/gemini-1.5-flash").unwrap();
        assert_eq!(id.provider, "google");
        assert_eq!(id.model, "gemini-1.5-flash");

        let err = ModelId::parse("invalid-model").unwrap_err();
        assert!(err.to_string().contains("Invalid model format"));
    }
}
