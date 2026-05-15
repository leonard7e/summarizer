use super::{LlmProvider, PromptPart};
use anyhow::{Result, anyhow};
use async_trait::async_trait;
use base64::{Engine as _, engine::general_purpose::STANDARD};
use reqwest::Client;
use serde::{Deserialize, Serialize};

pub struct GeminiProvider {
    api_key: String,
    client: Client,
    base_url: String,
}

impl GeminiProvider {
    pub fn new(api_key: String) -> Self {
        Self {
            api_key,
            client: Client::new(),
            base_url: "https://generativelanguage.googleapis.com".to_string(),
        }
    }

    #[cfg(test)]
    pub fn with_base_url(api_key: String, base_url: String) -> Self {
        Self {
            api_key,
            client: Client::new(),
            base_url: base_url.trim_end_matches('/').to_string(),
        }
    }
}

#[derive(Serialize)]
struct GeminiRequest {
    contents: Vec<GeminiContent>,
}

#[derive(Serialize, Deserialize)]
struct GeminiContent {
    parts: Vec<GeminiPart>,
}

#[derive(Serialize, Deserialize)]
struct GeminiPart {
    #[serde(skip_serializing_if = "Option::is_none")]
    text: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    inline_data: Option<GeminiInlineData>,
}

#[derive(Serialize, Deserialize)]
struct GeminiInlineData {
    mime_type: String,
    data: String,
}

#[derive(Deserialize)]
struct GeminiResponse {
    candidates: Option<Vec<GeminiCandidate>>,
    error: Option<GeminiError>,
}

#[derive(Deserialize)]
struct GeminiCandidate {
    content: GeminiContent,
}

#[derive(Deserialize)]
struct GeminiError {
    message: String,
}

#[async_trait]
impl LlmProvider for GeminiProvider {
    async fn complete(&self, prompt_parts: &[PromptPart], model: &str) -> Result<String> {
        let url = format!(
            "{}/v1beta/models/{}:generateContent?key={}",
            self.base_url, model, self.api_key
        );

        let parts: Vec<GeminiPart> = prompt_parts
            .iter()
            .map(|p| match p {
                PromptPart::Text(t) => GeminiPart {
                    text: Some(t.clone()),
                    inline_data: None,
                },
                PromptPart::Image { mime_type, data } => GeminiPart {
                    text: None,
                    inline_data: Some(GeminiInlineData {
                        mime_type: mime_type.clone(),
                        data: STANDARD.encode(data),
                    }),
                },
                PromptPart::Audio { mime_type, data } => GeminiPart {
                    text: None,
                    inline_data: Some(GeminiInlineData {
                        mime_type: mime_type.clone(),
                        data: STANDARD.encode(data),
                    }),
                },
                PromptPart::Video { mime_type, data } => GeminiPart {
                    text: None,
                    inline_data: Some(GeminiInlineData {
                        mime_type: mime_type.clone(),
                        data: STANDARD.encode(data),
                    }),
                },
            })
            .collect();

        let req_body = GeminiRequest {
            contents: vec![GeminiContent { parts }],
        };

        let res = self.client.post(&url).json(&req_body).send().await?;

        let status = res.status();
        if !status.is_success() {
            let error_text = res.text().await.unwrap_or_default();
            return Err(anyhow!(
                "Gemini API HTTP Error ({}): {}",
                status,
                error_text
            ));
        }

        let resp: GeminiResponse = res.json().await?;

        if let Some(err) = resp.error {
            return Err(anyhow!("Gemini API Error ({}): {}", status, err.message));
        }

        let text = resp
            .candidates
            .and_then(|c| c.into_iter().next())
            .and_then(|c| c.content.parts.into_iter().next())
            .and_then(|p| p.text)
            .ok_or_else(|| anyhow!("Invalid or empty Gemini response"))?;

        Ok(text)
    }

    async fn list_models(&self) -> Result<Vec<String>> {
        #[derive(Deserialize)]
        struct ModelsResponse {
            models: Vec<ModelInfo>,
        }
        #[derive(Deserialize)]
        struct ModelInfo {
            name: String,
        }

        let url = format!("{}/v1beta/models?key={}", self.base_url, self.api_key);

        let res: ModelsResponse = self.client.get(&url).send().await?.json().await?;

        Ok(res
            .models
            .into_iter()
            .map(|m| m.name.replace("models/", ""))
            .collect())
    }

    async fn get_context_limit(&self, model: &str) -> Result<usize> {
        #[derive(Deserialize)]
        struct ModelInfo {
            #[serde(rename = "inputTokenLimit")]
            input_token_limit: Option<usize>,
        }

        let url = format!(
            "{}/v1beta/models/{}?key={}",
            self.base_url, model, self.api_key
        );

        let res = self.client.get(&url).send().await?;
        if !res.status().is_success() {
            return Ok(32768); // Fallback
        }

        let info: ModelInfo = res.json().await?;
        Ok(info
            .input_token_limit
            .unwrap_or(crate::provider::DEFAULT_CONTEXT_LIMIT))
    }

    async fn supports_images(&self, _model: &str) -> Result<bool> {
        Ok(true)
    }
    async fn supports_audio(&self, _model: &str) -> Result<bool> {
        Ok(true)
    }
    async fn supports_video(&self, _model: &str) -> Result<bool> {
        Ok(true)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use mockito::Server;

    #[tokio::test]
    async fn test_gemini_complete() {
        let mut server = Server::new_async().await;
        let url = server.url();
        let provider = GeminiProvider::with_base_url("test_key".to_string(), url);

        let mock = server
            .mock(
                "POST",
                "/v1beta/models/test-model:generateContent?key=test_key",
            )
            .with_status(200)
            .with_header("content-type", "application/json")
            .with_body(
                r#"{
                "candidates": [
                    {
                        "content": {
                            "parts": [
                                { "text": "Zusammenfassung" }
                            ]
                        }
                    }
                ]
            }"#,
            )
            .create_async()
            .await;

        let result = provider
            .complete(&[PromptPart::Text("Prompt".to_string())], "test-model")
            .await
            .unwrap();
        assert_eq!(result, "Zusammenfassung");
        mock.assert_async().await;
    }

    #[tokio::test]
    async fn test_gemini_api_error() {
        let mut server = Server::new_async().await;
        let url = server.url();
        let provider = GeminiProvider::with_base_url("test_key".to_string(), url);

        let mock = server
            .mock(
                "POST",
                "/v1beta/models/test-model:generateContent?key=test_key",
            )
            .with_status(400)
            .with_header("content-type", "application/json")
            .with_body(
                r#"{
                "error": {
                    "message": "Invalid request"
                }
            }"#,
            )
            .create_async()
            .await;

        let result = provider
            .complete(&[PromptPart::Text("Prompt".to_string())], "test-model")
            .await;
        assert!(result.is_err());
        let err_msg = result.unwrap_err().to_string();
        assert!(err_msg.contains("Gemini API HTTP Error"));
        assert!(err_msg.contains("Invalid request"));
        mock.assert_async().await;
    }
}
