use super::LlmProvider;
use anyhow::{anyhow, Result};
use async_trait::async_trait;
use reqwest::Client;
use serde::{Deserialize, Serialize};

pub struct OllamaProvider {
    host: String,
    num_ctx: usize,
    client: Client,
}

impl OllamaProvider {
    pub fn new(host: String, num_ctx: usize) -> Self {
        Self {
            host: host.trim_end_matches('/').to_string(),
            num_ctx,
            client: Client::new(),
        }
    }
}

#[derive(Serialize)]
struct OllamaOptions {
    num_ctx: usize,
}

#[derive(Serialize)]
struct OllamaRequest<'a> {
    model: &'a str,
    prompt: String,
    stream: bool,
    options: OllamaOptions,
}

#[derive(Deserialize)]
struct OllamaResponse {
    response: String,
}

#[async_trait]
impl LlmProvider for OllamaProvider {
    async fn complete(&self, prompt: &str, model: &str) -> Result<String> {
        let req_body = OllamaRequest {
            model,
            prompt: prompt.to_string(),
            stream: false,
            options: OllamaOptions {
                num_ctx: self.num_ctx,
            },
        };

        let url = format!("{}/api/generate", self.host);

        let res = self.client.post(&url).json(&req_body).send().await?;

        let status = res.status();
        if !status.is_success() {
            let error_text = res.text().await.unwrap_or_default();
            return Err(anyhow!("Ollama API Error ({}): {}", status, error_text));
        }

        let resp: OllamaResponse = res.json().await?;
        Ok(resp.response)
    }

    async fn list_models(&self) -> Result<Vec<String>> {
        #[derive(Deserialize)]
        struct ModelsResponse {
            models: Vec<ModelData>,
        }
        #[derive(Deserialize)]
        struct ModelData {
            name: String,
        }

        let url = format!("{}/api/tags", self.host);
        let res: ModelsResponse = self.client.get(&url).send().await?.json().await?;

        Ok(res.models.into_iter().map(|m| m.name).collect())
    }

    async fn get_context_limit(&self, _model: &str) -> Result<usize> {
        Ok(self.num_ctx)
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    use mockito::Server;

    #[tokio::test]
    async fn test_ollama_complete() {
        let mut server = Server::new_async().await;
        let url = server.url();
        let provider = OllamaProvider::new(url, 4096);

        let mock = server
            .mock("POST", "/api/generate")
            .with_status(200)
            .with_header("content-type", "application/json")
            .with_body(r#"{"response": "Ollama Zusammenfassung"}"#)
            .create_async()
            .await;

        let result = provider.complete("Prompt", "test-model").await.unwrap();
        assert_eq!(result, "Ollama Zusammenfassung");
        mock.assert_async().await;
    }

    #[tokio::test]
    async fn test_ollama_api_error() {
        let mut server = Server::new_async().await;
        let url = server.url();
        let provider = OllamaProvider::new(url, 4096);

        let mock = server
            .mock("POST", "/api/generate")
            .with_status(500)
            .with_body("Internal Server Error")
            .create_async()
            .await;

        let result = provider.complete("Prompt", "test-model").await;
        assert!(result.is_err());
        let err_msg = result.unwrap_err().to_string();
        assert!(err_msg.contains("Ollama API Error"));
        assert!(err_msg.contains("500"));
        mock.assert_async().await;
    }
}
