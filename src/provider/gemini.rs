use super::LlmProvider;
use crate::file::{FileData, FileType, ProcessedFile};
use anyhow::{anyhow, Result};
use async_trait::async_trait;
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
            base_url,
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
    text: String,
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
    async fn complete(
        &self,
        prompt: &str,
        files: &[ProcessedFile],
        previous_result: Option<&str>,
        model: &str,
    ) -> Result<String> {
        let mut full_prompt = prompt.to_string();
        if let Some(prev) = previous_result {
            full_prompt.push_str("\n\n--- Bisheriges Ergebnis ---\n");
            full_prompt.push_str(prev);
        }

        for file in files {
            match &file.data {
                FileData::Text(content) => {
                    let encoding = match &file.metadata.file_type {
                        FileType::Text { encoding } => encoding,
                    };
                    full_prompt.push_str(&format!(
                        "\n\n--- Datei: {} (Encoding: {}) ---\n",
                        file.metadata.file_name, encoding
                    ));
                    full_prompt.push_str(content);
                }
            }
        }

        let url = format!(
            "{}/v1beta/models/{}:generateContent?key={}",
            self.base_url, model, self.api_key
        );

        let req_body = GeminiRequest {
            contents: vec![GeminiContent {
                parts: vec![GeminiPart { text: full_prompt }],
            }],
        };

        let res = self
            .client
            .post(&url)
            .json(&req_body)
            .send()
            .await?;

        let status = res.status();
        let resp: GeminiResponse = res.json().await?;

        if let Some(err) = resp.error {
            return Err(anyhow!("Gemini API Error ({}): {}", status, err.message));
        }

        let candidates = resp
            .candidates
            .ok_or_else(|| anyhow!("No candidates in Gemini response"))?;
        let first = candidates
            .into_iter()
            .next()
            .ok_or_else(|| anyhow!("Empty candidates list"))?;
        let text = first
            .content
            .parts
            .into_iter()
            .next()
            .ok_or_else(|| anyhow!("Empty parts list"))?
            .text;

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

        let url = format!(
            "{}/v1beta/models?key={}",
            self.base_url, self.api_key
        );

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
        Ok(info.input_token_limit.unwrap_or(32768))
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
            .mock("POST", "/v1beta/models/test-model:generateContent?key=test_key")
            .with_status(200)
            .with_header("content-type", "application/json")
            .with_body(r#"{
                "candidates": [
                    {
                        "content": {
                            "parts": [
                                { "text": "Zusammenfassung" }
                            ]
                        }
                    }
                ]
            }"#)
            .create_async()
            .await;

        let result = provider.complete("Prompt", &[], None, "test-model").await.unwrap();
        assert_eq!(result, "Zusammenfassung");
        mock.assert_async().await;
    }

    #[tokio::test]
    async fn test_gemini_api_error() {
        let mut server = Server::new_async().await;
        let url = server.url();
        let provider = GeminiProvider::with_base_url("test_key".to_string(), url);

        let mock = server
            .mock("POST", "/v1beta/models/test-model:generateContent?key=test_key")
            .with_status(400)
            .with_header("content-type", "application/json")
            .with_body(r#"{
                "error": {
                    "message": "Invalid request"
                }
            }"#)
            .create_async()
            .await;

        let result = provider.complete("Prompt", &[], None, "test-model").await;
        assert!(result.is_err());
        let err_msg = result.unwrap_err().to_string();
        assert!(err_msg.contains("Gemini API Error"));
        assert!(err_msg.contains("Invalid request"));
        mock.assert_async().await;
    }
}
