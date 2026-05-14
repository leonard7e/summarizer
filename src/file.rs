use anyhow::{anyhow, Result};
use std::path::Path;
use tokio::fs;

pub enum FileType {
    Text { encoding: String },
    Image { mime_type: String },
}

pub enum FileData {
    Text(String),
    Image(Vec<u8>),
}

pub struct FileMetadata {
    pub file_name: String,
    pub file_type: FileType,
}

/// Represents a file that has been successfully read and parsed into memory.
pub struct ProcessedFile {
    pub metadata: FileMetadata,
    pub data: FileData,
}

/// Reads a file from disk and wraps it in a ProcessedFile struct. 
/// Currently only supports UTF-8 text.
pub async fn read_file(path: &Path) -> Result<ProcessedFile> {
    let file_name = path
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or("unknown")
        .to_string();

    let ext = path
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or("")
        .to_lowercase();

    // Returns Some(mime_type) for known image extensions, None for text.
    let image_mime = match ext.as_str() {
        "png"        => Some("image/png"),
        "jpg" | "jpeg" => Some("image/jpeg"),
        "webp"       => Some("image/webp"),
        "gif"        => Some("image/gif"),
        _            => None,
    };

    if let Some(mime_type) = image_mime {
        let content = fs::read(path)
            .await
            .map_err(|e| anyhow!("Failed to read image file: {}", e))?;
        Ok(ProcessedFile {
            metadata: FileMetadata {
                file_name,
                file_type: FileType::Image { mime_type: mime_type.to_string() },
            },
            data: FileData::Image(content),
        })
    } else {
        let content = fs::read_to_string(path).await.map_err(|e| {
            if e.kind() == std::io::ErrorKind::InvalidData {
                anyhow!("File type not supported yet.")
            } else {
                anyhow!("Failed to read text file: {}", e)
            }
        })?;
        Ok(ProcessedFile {
            metadata: FileMetadata {
                file_name,
                file_type: FileType::Text { encoding: "utf-8".into() },
            },
            data: FileData::Text(content),
        })
    }
}
