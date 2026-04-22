use anyhow::{anyhow, Result};
use std::path::Path;
use tokio::fs;

pub enum FileType {
    Text { encoding: String },
}

pub enum FileData {
    Text(String),
}

pub struct FileMetadata {
    pub file_name: String,
    pub file_type: FileType,
}

pub struct ProcessedFile {
    pub metadata: FileMetadata,
    pub data: FileData,
}

pub async fn read_file(path: &Path) -> Result<ProcessedFile> {
    let file_name = path
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or("unknown")
        .to_string();

    match fs::read_to_string(path).await {
        Ok(content) => Ok(ProcessedFile {
            metadata: FileMetadata {
                file_name,
                file_type: FileType::Text {
                    encoding: "utf-8".into(),
                },
            },
            data: FileData::Text(content),
        }),
        Err(e) if e.kind() == std::io::ErrorKind::InvalidData => {
            Err(anyhow!("File type not supported yet."))
        }
        Err(e) => Err(anyhow!("Failed to read file: {}", e)),
    }
}
