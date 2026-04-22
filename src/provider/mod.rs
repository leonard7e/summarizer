use crate::config::Config;
use crate::file::ProcessedFile;
use anyhow::{anyhow, Result};
use async_trait::async_trait;

pub mod gemini;
pub mod ollama;
pub mod openrouter;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ModelId {
    pub provider: String,
    pub model: String,
}

impl ModelId {
    pub fn parse(s: &str) -> Result<Self> {
        let parts: Vec<&str> = s.splitn(2, '/').collect();
        if parts.len() != 2 {
            return Err(anyhow!("Invalid model format. Expected 'provider/model_id', got '{}'", s));
        }
        Ok(Self {
            provider: parts[0].to_lowercase(),
            model: parts[1].to_string(),
        })
    }
}

#[async_trait]
pub trait LlmProvider {
    async fn complete(
        &self,
        prompt: &str,
        files: &[ProcessedFile],
        previous_result: Option<&str>,
        model: &str,
    ) -> Result<String>;

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
            Ok(Box::new(openrouter::OpenRouterProvider::new(api_key)))
        }
        "ollama" => {
            let config_ollama = config.providers.ollama.as_ref();
            let host = config_ollama
                .map(|o| o.host.clone())
                .unwrap_or_else(|| "http://localhost:11434".to_string());
            let num_ctx = config_ollama.map(|o| o.num_ctx).unwrap_or(4096);
            Ok(Box::new(ollama::OllamaProvider::new(host, num_ctx)))
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
