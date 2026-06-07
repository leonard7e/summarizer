use super::{LlmProvider, PromptPart};
use anyhow::{Result, anyhow, ensure};
use async_trait::async_trait;
use base64::{Engine as _, engine::general_purpose::STANDARD};
use reqwest::{Client, header};
use serde::{Deserialize, Serialize};

pub struct OpenAiCompatibleProvider {
    client: Client,
    base_url: String,
}

impl OpenAiCompatibleProvider {
    pub fn new(api_key: String, base_url: String) -> Result<Self> {
        let mut headers = header::HeaderMap::new();
        // Some local providers don't strictly require an API key but we send it anyway if present
        let auth_val = if api_key.is_empty() {
            "Bearer dummy_key".to_string()
        } else {
            format!("Bearer {}", api_key)
        };
        let mut auth_value = header::HeaderValue::from_str(&auth_val)
            .map_err(|e| anyhow!("Invalid authorization header value: {}", e))?;
        auth_value.set_sensitive(true);
        headers.insert(header::AUTHORIZATION, auth_value);

        Ok(Self {
            client: Client::builder()
                .default_headers(headers)
                .build()
                .map_err(|e| anyhow!("Failed to build HTTP client: {}", e))?,
            base_url: base_url.trim_end_matches('/').to_string(),
        })
    }
}

#[derive(Serialize)]
struct ChatCompletionRequest<'a> {
    model: &'a str,
    messages: Vec<Message<'a>>,
}

#[derive(Serialize)]
struct Message<'a> {
    role: &'a str,
    content: Vec<OpenAiContentPart>,
}

#[derive(Serialize)]
#[serde(tag = "type")]
enum OpenAiContentPart {
    #[serde(rename = "text")]
    Text { text: String },
    #[serde(rename = "image_url")]
    ImageUrl { image_url: OpenAiImageUrl },
    #[serde(rename = "input_audio")]
    InputAudio { input_audio: OpenAiInputAudio },
    #[serde(rename = "video_url")]
    VideoUrl { video_url: OpenAiVideoUrl },
}

#[derive(Serialize)]
struct OpenAiInputAudio {
    data: String,
    format: String,
}

#[derive(Serialize)]
struct OpenAiVideoUrl {
    url: String,
}

#[derive(Serialize)]
struct OpenAiImageUrl {
    url: String,
}

#[derive(Deserialize)]
struct ChatCompletionResponse {
    choices: Option<Vec<Choice>>,
    error: Option<ApiError>,
}

#[derive(Deserialize)]
struct Choice {
    message: MessageResponse,
}

#[derive(Deserialize)]
struct MessageResponse {
    content: String,
}

#[derive(Deserialize)]
struct ApiError {
    message: String,
}

#[async_trait]
impl LlmProvider for OpenAiCompatibleProvider {
    async fn complete(
        &self,
        system_instruction: &str,
        prompt_parts: &[PromptPart],
        model: &str,
    ) -> Result<String> {
        let content_parts: Vec<OpenAiContentPart> = prompt_parts
            .iter()
            .map(|p| match p {
                PromptPart::Text(t) => OpenAiContentPart::Text { text: t.clone() },
                PromptPart::Image { mime_type, data } => OpenAiContentPart::ImageUrl {
                    image_url: OpenAiImageUrl {
                        url: format!("data:{};base64,{}", mime_type, STANDARD.encode(data)),
                    },
                },
                PromptPart::Audio { mime_type, data } => OpenAiContentPart::InputAudio {
                    input_audio: OpenAiInputAudio {
                        data: STANDARD.encode(data),
                        format: mime_type.split('/').nth(1).unwrap_or("mp3").to_string(),
                    },
                },
                PromptPart::Video { mime_type, data } => OpenAiContentPart::VideoUrl {
                    video_url: OpenAiVideoUrl {
                        url: format!("data:{};base64,{}", mime_type, STANDARD.encode(data)),
                    },
                },
            })
            .collect();

        let mut messages = Vec::new();
        if !system_instruction.is_empty() {
            messages.push(Message {
                role: "system",
                content: vec![OpenAiContentPart::Text {
                    text: system_instruction.to_string(),
                }],
            });
        }
        messages.push(Message {
            role: "user",
            content: content_parts,
        });

        let req_body = ChatCompletionRequest { model, messages };

        let url = format!("{}/chat/completions", self.base_url);

        let res = self.client.post(&url).json(&req_body).send().await?;

        let status = res.status();
        ensure!(
            status.is_success(),
            anyhow!(
                "OpenAI API HTTP Error ({}): {}",
                status,
                res.text().await.unwrap_or_default()
            )
        );

        let resp: ChatCompletionResponse = res.json().await?;

        ensure!(
            resp.error.is_none(),
            anyhow!(
                "OpenAI API Error ({}): {}",
                status,
                resp.error.unwrap().message
            )
        );

        let content = resp
            .choices
            .and_then(|c| c.into_iter().next())
            .map(|choice| choice.message.content)
            .ok_or_else(|| anyhow!("No choices in API response"))?;

        Ok(content)
    }

    async fn list_models(&self) -> Result<Vec<String>> {
        #[derive(Deserialize)]
        struct ModelsResponse {
            data: Vec<ModelData>,
        }
        #[derive(Deserialize)]
        struct ModelData {
            id: String,
        }

        let url = format!("{}/models", self.base_url);

        let res: ModelsResponse = self.client.get(&url).send().await?.json().await?;

        Ok(res.data.into_iter().map(|m| m.id).collect())
    }

    async fn get_context_limit(&self, model: &str) -> Result<usize> {
        #[derive(Deserialize)]
        struct ModelsResponse {
            data: Vec<ModelData>,
        }
        #[derive(Deserialize)]
        struct ModelData {
            id: String,
            context_length: Option<usize>,
            max_position_embeddings: Option<usize>,
        }

        let url = format!("{}/models", self.base_url);

        let res = self.client.get(&url).send().await?;
        if !res.status().is_success() {
            // Safe fallback if /models endpoint doesn't exist or fails
            return Ok(8192);
        }

        let limit = if let Ok(resp) = res.json::<ModelsResponse>().await {
            resp.data
                .into_iter()
                .find(|m| m.id == model)
                .and_then(|info| info.context_length.or(info.max_position_embeddings))
        } else {
            None
        };

        Ok(limit.unwrap_or(crate::provider::DEFAULT_CONTEXT_LIMIT))
    }

    async fn supports_images(&self, model: &str) -> Result<bool> {
        #[derive(Deserialize)]
        struct ModelsResponse {
            data: Vec<ModelData>,
        }
        #[derive(Deserialize)]
        struct ModelData {
            id: String,
            architecture: Option<Architecture>,
        }
        #[derive(Deserialize)]
        struct Architecture {
            input_modalities: Option<Vec<String>>,
        }

        let url = format!("{}/models", self.base_url);

        let image_support: Option<bool> = async {
            let resp: ModelsResponse = self
                .client
                .get(&url)
                .send()
                .await
                .ok()?
                .error_for_status()
                .ok()?
                .json()
                .await
                .ok()?;

            let modalities = resp
                .data
                .into_iter()
                .find(|m| m.id == model)?
                .architecture?
                .input_modalities?;

            Some(modalities.iter().any(|m| m == "image"))
        }
        .await;

        Ok(image_support.unwrap_or(true))
    }

    async fn supports_audio(&self, _model: &str) -> Result<bool> {
        // Similar optimistic fallback for audio
        Ok(true)
    }

    async fn supports_video(&self, _model: &str) -> Result<bool> {
        // Similar optimistic fallback for video
        Ok(true)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use mockito::Server;

    #[tokio::test]
    async fn test_openai_compatible_complete() {
        let mut server = Server::new_async().await;
        let url = server.url();
        let provider = OpenAiCompatibleProvider::new("test_key".to_string(), url.clone()).unwrap();

        let mock = server
            .mock("POST", "/chat/completions")
            .with_status(200)
            .with_header("content-type", "application/json")
            .with_body(
                r#"{
                "choices": [
                    {
                        "message": {
                            "content": "Compatible Zusammenfassung"
                        }
                    }
                ]
            }"#,
            )
            .create_async()
            .await;

        let result = provider
            .complete("", &[PromptPart::Text("Prompt".to_string())], "test-model")
            .await
            .unwrap();
        assert_eq!(result, "Compatible Zusammenfassung");
        mock.assert_async().await;
    }

    #[tokio::test]
    async fn test_openai_compatible_api_error() {
        let mut server = Server::new_async().await;
        let url = server.url();
        let provider = OpenAiCompatibleProvider::new("test_key".to_string(), url.clone()).unwrap();

        let mock = server
            .mock("POST", "/chat/completions")
            .with_status(401)
            .with_header("content-type", "application/json")
            .with_body(
                r#"{
                "error": {
                    "message": "Invalid API Key"
                }
            }"#,
            )
            .create_async()
            .await;

        let result = provider
            .complete("", &[PromptPart::Text("Prompt".to_string())], "test-model")
            .await;
        assert!(result.is_err());
        let err_msg = result.unwrap_err().to_string();
        assert!(err_msg.contains("OpenAI API HTTP Error"));
        assert!(err_msg.contains("401"));
        mock.assert_async().await;
    }

    #[tokio::test]
    async fn test_openai_compatible_get_context_limit() {
        let mut server = Server::new_async().await;
        let url = server.url();
        let provider = OpenAiCompatibleProvider::new("test_key".to_string(), url.clone()).unwrap();

        let mock = server
            .mock("GET", "/models")
            .with_status(200)
            .with_header("content-type", "application/json")
            .with_body(
                r#"{
                "data": [
                    {
                        "id": "test-model",
                        "context_length": 16384
                    }
                ]
            }"#,
            )
            .create_async()
            .await;

        let limit = provider.get_context_limit("test-model").await.unwrap();
        assert_eq!(limit, 16384);
        mock.assert_async().await;
    }

    #[tokio::test]
    async fn test_openai_compatible_get_context_limit_fallback() {
        let mut server = Server::new_async().await;
        let url = server.url();
        let provider = OpenAiCompatibleProvider::new("test_key".to_string(), url.clone()).unwrap();

        let mock = server
            .mock("GET", "/models")
            .with_status(200)
            .with_header("content-type", "application/json")
            .with_body(
                r#"{
                "data": [
                    {
                        "id": "test-model"
                    }
                ]
            }"#,
            )
            .create_async()
            .await;

        let limit = provider.get_context_limit("test-model").await.unwrap();
        assert_eq!(limit, 8192);
        mock.assert_async().await;
    }
}
