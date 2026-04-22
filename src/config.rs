use serde::{Deserialize, Serialize};
use std::path::PathBuf;

#[derive(Debug, Serialize, Deserialize, Default, Clone)]
pub struct Config {
    pub default_model: Option<String>,
    #[serde(default)]
    pub providers: ProvidersConfig,
}

#[derive(Debug, Serialize, Deserialize, Default, Clone)]
pub struct ProvidersConfig {
    pub gemini: Option<GeminiConfig>,
    pub openrouter: Option<OpenRouterConfig>,
    pub ollama: Option<OllamaConfig>,
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

impl Config {
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
    }

    #[test]
    fn test_ollama_default_host() {
        let config = OllamaConfig { host: default_ollama_host(), num_ctx: default_ollama_num_ctx() };
        assert_eq!(config.host, "http://localhost:11434");
    }
}
