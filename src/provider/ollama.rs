use super::{LlmProvider, PromptPart};
use anyhow::{Result, anyhow, ensure};
use async_trait::async_trait;
use base64::{Engine as _, engine::general_purpose::STANDARD};
use reqwest::Client;
use serde::{Deserialize, Serialize};

pub struct OllamaProvider {
    base_url: String,
    num_ctx: usize,
    client: Client,
}

impl OllamaProvider {
    pub fn new(base_url: String, num_ctx: usize) -> Self {
        Self {
            base_url: base_url.trim_end_matches('/').to_string(),
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
    #[serde(skip_serializing_if = "Option::is_none")]
    system: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    images: Option<Vec<String>>,
    stream: bool,
    options: OllamaOptions,
}

#[derive(Serialize)]
struct OllamaShowRequest<'a> {
    model: &'a str,
}

#[derive(Deserialize)]
struct OllamaShowResponse {
    #[serde(default)]
    model_info: std::collections::HashMap<String, serde_json::Value>,
    capabilities: Option<Vec<String>>,
}

#[derive(Deserialize)]
struct OllamaResponse {
    response: String,
}

#[async_trait]
impl LlmProvider for OllamaProvider {
    async fn complete(
        &self,
        system_instruction: &str,
        prompt_parts: &[PromptPart],
        model: &str,
    ) -> Result<String> {
        let (prompt_text, images): (String, Vec<String>) =
            prompt_parts
                .iter()
                .fold((String::new(), Vec::new()), |(mut text, mut imgs), part| {
                    match part {
                        PromptPart::Text(t) => text.push_str(t),
                        PromptPart::Image { data, .. } => imgs.push(STANDARD.encode(data)),
                        _ => {} // Ignore audio/video as Ollama doesn't support them natively yet
                    }
                    (text, imgs)
                });

        let images_opt = (!images.is_empty()).then_some(images);

        let sys_opt = (!system_instruction.is_empty()).then(|| system_instruction.to_string());

        let req_body = OllamaRequest {
            model,
            prompt: prompt_text,
            system: sys_opt,
            images: images_opt,
            stream: false,
            options: OllamaOptions {
                num_ctx: self.num_ctx,
            },
        };

        let url = format!("{}/api/generate", self.base_url);

        let res = self.client.post(&url).json(&req_body).send().await?;

        let status = res.status();
        ensure!(
            status.is_success(),
            anyhow!(
                "Ollama API Error ({}): {}",
                status,
                res.text().await.unwrap_or_default()
            )
        );

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

        let url = format!("{}/api/tags", self.base_url);
        let res: ModelsResponse = self.client.get(&url).send().await?.json().await?;

        Ok(res.models.into_iter().map(|m| m.name).collect())
    }

    async fn get_context_limit(&self, model: &str) -> Result<usize> {
        let url = format!("{}/api/show", self.base_url);
        let req_body = OllamaShowRequest { model };

        // Silently fall back to self.num_ctx if the request fails or returns
        // unexpected data — we don't want a missing /api/show to break the run.
        let model_max_ctx: Option<usize> = async {
            let info = self
                .client
                .post(&url)
                .json(&req_body)
                .send()
                .await
                .ok()?
                .error_for_status()
                .ok()?
                .json::<OllamaShowResponse>()
                .await
                .ok()?;

            info.model_info
                .into_iter()
                .filter(|(k, _)| k.ends_with(".context_length"))
                .filter_map(|(_, v)| v.as_u64().map(|n| n as usize))
                .next()
        }
        .await;

        match model_max_ctx {
            Some(limit) if self.num_ctx > limit => {
                eprintln!(
                    "Warning: Configured num_ctx ({}) exceeds model's context window ({}). \
                     Using model's context window instead.",
                    self.num_ctx, limit
                );
                Ok(limit)
            }
            _ => Ok(self.num_ctx),
        }
    }

    async fn supports_images(&self, model: &str) -> Result<bool> {
        let url = format!("{}/api/show", self.base_url);
        let req_body = OllamaShowRequest { model };

        let supported = self
            .client
            .post(&url)
            .json(&req_body)
            .send()
            .await?
            .error_for_status()?
            .json::<OllamaShowResponse>()
            .await?
            .capabilities
            .is_some_and(|caps| caps.iter().any(|c| c == "vision"));

        Ok(supported)
    }

    async fn supports_audio(&self, _model: &str) -> Result<bool> {
        Ok(false)
    }

    async fn supports_video(&self, _model: &str) -> Result<bool> {
        Ok(false)
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

        let result = provider
            .complete("", &[PromptPart::Text("Prompt".to_string())], "test-model")
            .await
            .unwrap();
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

        let result = provider
            .complete("", &[PromptPart::Text("Prompt".to_string())], "test-model")
            .await;
        assert!(result.is_err());
        let err_msg = result.unwrap_err().to_string();
        assert!(err_msg.contains("Ollama API Error"));
        assert!(err_msg.contains("500"));
        mock.assert_async().await;
    }
}
