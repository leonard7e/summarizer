use super::LlmProvider;
use anyhow::{anyhow, Result};
use async_trait::async_trait;
use reqwest::{header, Client};
use serde::{Deserialize, Serialize};

pub struct OpenRouterProvider {
    client: Client,
    base_url: String,
}

impl OpenRouterProvider {
    pub fn new(api_key: String) -> Self {
        let mut headers = header::HeaderMap::new();
        let mut auth_value = header::HeaderValue::from_str(&format!("Bearer {}", api_key)).unwrap();
        auth_value.set_sensitive(true);
        headers.insert(header::AUTHORIZATION, auth_value);

        Self {
            client: Client::builder().default_headers(headers).build().unwrap(),
            base_url: "https://openrouter.ai".to_string(),
        }
    }

    #[cfg(test)]
    pub fn with_base_url(api_key: String, base_url: String) -> Self {
        let mut headers = header::HeaderMap::new();
        let mut auth_value = header::HeaderValue::from_str(&format!("Bearer {}", api_key)).unwrap();
        auth_value.set_sensitive(true);
        headers.insert(header::AUTHORIZATION, auth_value);

        Self {
            client: Client::builder().default_headers(headers).build().unwrap(),
            base_url,
        }
    }
}

#[derive(Serialize)]
struct OpenRouterRequest<'a> {
    model: &'a str,
    messages: Vec<Message<'a>>,
}

#[derive(Serialize)]
struct Message<'a> {
    role: &'a str,
    content: &'a str,
}

#[derive(Deserialize)]
struct OpenRouterResponse {
    choices: Option<Vec<Choice>>,
    error: Option<OpenRouterError>,
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
struct OpenRouterError {
    message: String,
}

#[async_trait]
impl LlmProvider for OpenRouterProvider {
    async fn complete(&self, prompt: &str, model: &str) -> Result<String> {
        let messages = vec![Message {
            role: "user",
            content: prompt,
        }];

        let req_body = OpenRouterRequest { model, messages };

        let url = format!("{}/api/v1/chat/completions", self.base_url.trim_end_matches('/'));

        let res = self
            .client
            .post(&url)
            .json(&req_body)
            .send()
            .await?;

        let status = res.status();
        let resp: OpenRouterResponse = res.json().await?;

        if let Some(err) = resp.error {
            return Err(anyhow!("OpenRouter API Error ({}): {}", status, err.message));
        }

        let choices = resp
            .choices
            .ok_or_else(|| anyhow!("No choices in OpenRouter response"))?;
        let first = choices
            .into_iter()
            .next()
            .ok_or_else(|| anyhow!("Empty choices list"))?;

        Ok(first.message.content)
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

        let url = format!("{}/api/v1/models", self.base_url.trim_end_matches('/'));

        let res: ModelsResponse = self
            .client
            .get(&url)
            .send()
            .await?
            .json()
            .await?;

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
            context_length: usize,
        }

        let url = format!("{}/api/v1/models", self.base_url.trim_end_matches('/'));
        let res: ModelsResponse = self.client.get(&url).send().await?.json().await?;

        let model_info = res.data.into_iter().find(|m| m.id == model);
        Ok(model_info.map(|m| m.context_length).unwrap_or(4096))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use mockito::Server;

    #[tokio::test]
    async fn test_openrouter_complete() {
        let mut server = Server::new_async().await;
        let url = server.url();
        let provider = OpenRouterProvider::with_base_url("test_key".to_string(), url);

        let mock = server
            .mock("POST", "/api/v1/chat/completions")
            .with_status(200)
            .with_header("content-type", "application/json")
            .with_body(r#"{
                "choices": [
                    {
                        "message": {
                            "content": "OpenRouter Zusammenfassung"
                        }
                    }
                ]
            }"#)
            .create_async()
            .await;

        let result = provider.complete("Prompt", "test-model").await.unwrap();
        assert_eq!(result, "OpenRouter Zusammenfassung");
        mock.assert_async().await;
    }

    #[tokio::test]
    async fn test_openrouter_api_error() {
        let mut server = Server::new_async().await;
        let url = server.url();
        let provider = OpenRouterProvider::with_base_url("test_key".to_string(), url);

        let mock = server
            .mock("POST", "/api/v1/chat/completions")
            .with_status(401)
            .with_header("content-type", "application/json")
            .with_body(r#"{
                "error": {
                    "message": "Invalid API Key"
                }
            }"#)
            .create_async()
            .await;

        let result = provider.complete("Prompt", "test-model").await;
        assert!(result.is_err());
        let err_msg = result.unwrap_err().to_string();
        assert!(err_msg.contains("OpenRouter API Error"));
        assert!(err_msg.contains("401"));
        mock.assert_async().await;
    }
}
