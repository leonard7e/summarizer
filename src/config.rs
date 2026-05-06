use serde::{Deserialize, Serialize};
use std::path::PathBuf;

/// The main configuration structure for the summarizer application.
#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct Config {
    pub default_model: Option<String>,
    /// How many tokens to reserve for the model's output response.
    /// The remaining context window is available for input (instruction +
    /// previous result + file contents). Defaults to 2048.
    #[serde(default = "default_max_output_tokens")]
    pub max_output_tokens: usize,
    #[serde(default)]
    pub providers: ProvidersConfig,
}

fn default_max_output_tokens() -> usize {
    2048
}

impl Default for Config {
    fn default() -> Self {
        Self {
            default_model: None,
            max_output_tokens: default_max_output_tokens(),
            providers: ProvidersConfig::default(),
        }
    }
}

#[derive(Debug, Serialize, Deserialize, Default, Clone)]
pub struct ProvidersConfig {
    pub gemini: Option<GeminiConfig>,
    pub openrouter: Option<OpenRouterConfig>,
    pub ollama: Option<OllamaConfig>,
    #[serde(rename = "openai-compatible")]
    pub openai_compatible: Option<OpenAiCompatibleConfig>,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct GeminiConfig {
    pub api_key: String,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct OpenRouterConfig {
    pub api_key: String,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct OpenAiCompatibleConfig {
    pub api_key: String,
    pub base_url: String,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct OllamaConfig {
    #[serde(default = "default_ollama_host")]
    pub host: String,
    #[serde(default = "default_ollama_num_ctx")]
    pub num_ctx: usize,
}

fn default_ollama_host() -> String {
    "http://localhost:11434".to_string()
}

fn default_ollama_num_ctx() -> usize {
    4096
}

impl Default for OllamaConfig {
    fn default() -> Self {
        Self {
            host: default_ollama_host(),
            num_ctx: default_ollama_num_ctx(),
        }
    }
}

impl Config {
    /// Loads the configuration from the user's config directory, creating a default one if it doesn't exist.
    pub fn load() -> anyhow::Result<Self> {
        let config_path = Self::path()?;
        if !config_path.exists() {
            let default_config = Config::default();
            default_config.save()?;
            return Ok(default_config);
        }

        let content = std::fs::read_to_string(config_path)?;
        let config: Config = serde_yaml::from_str(&content)?;
        Ok(config)
    }

    pub fn save(&self) -> anyhow::Result<()> {
        let path = Self::path()?;
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let yaml = serde_yaml::to_string(self)?;
        std::fs::write(&path, yaml)?;

        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o600))?;
        }

        Ok(())
    }

    pub fn path() -> anyhow::Result<PathBuf> {
        let mut path = dirs::config_dir().ok_or_else(|| anyhow::anyhow!("Could not find config directory"))?;
        path.push("summarizer");
        path.push("config.yaml");
        Ok(path)
    }
}
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_defaults() {
        let config = Config::default();
        assert!(config.default_model.is_none());
        assert_eq!(config.max_output_tokens, 2048);
    }

    #[test]
    fn test_ollama_default_host() {
        let config = OllamaConfig { host: default_ollama_host(), num_ctx: default_ollama_num_ctx() };
        assert_eq!(config.host, "http://localhost:11434");
    }
}
