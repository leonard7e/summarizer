use anyhow::{anyhow, Result};
use std::path::Path;
use tokio::fs;

pub enum FileType {
    Text { encoding: String },
    Image { mime_type: String },
    Audio { mime_type: String },
    Video { mime_type: String },
}

pub enum FileData {
    Text(String),
    Image(Vec<u8>),
    Audio(Vec<u8>, f64),
    Video(Vec<u8>, f64),
}

pub struct FileMetadata {
    pub file_name: String,
    pub file_type: FileType,
}

pub struct ProcessedFile {
    pub metadata: FileMetadata,
    pub data: FileData,
}

enum MediaCategory {
    Image,
    Audio,
    Video,
}

impl MediaCategory {
    fn from_ext(ext: &str) -> Option<(Self, &'static str)> {
        match ext {
            "png" => Some((Self::Image, "image/png")),
            "jpg" | "jpeg" => Some((Self::Image, "image/jpeg")),
            "webp" => Some((Self::Image, "image/webp")),
            "gif" => Some((Self::Image, "image/gif")),
            "mp3" => Some((Self::Audio, "audio/mpeg")),
            "wav" => Some((Self::Audio, "audio/wav")),
            "flac" => Some((Self::Audio, "audio/flac")),
            "ogg" => Some((Self::Audio, "audio/ogg")),
            "m4a" | "aac" => Some((Self::Audio, "audio/mp4")),
            "mp4" => Some((Self::Video, "video/mp4")),
            "mov" => Some((Self::Video, "video/quicktime")),
            "webm" => Some((Self::Video, "video/webm")),
            "avi" => Some((Self::Video, "video/x-msvideo")),
            "mpeg" | "mpg" => Some((Self::Video, "video/mpeg")),
            _ => None,
        }
    }

    fn process(self, mime: &str, name: String, data: Vec<u8>) -> ProcessedFile {
        let (file_type, data) = match self {
            Self::Image => (
                FileType::Image { mime_type: mime.into() },
                FileData::Image(data)
            ),
            Self::Audio => {
                let dur = get_media_duration(&data);
                (FileType::Audio { mime_type: mime.into() }, FileData::Audio(data, dur))
            }
            Self::Video => {
                let dur = get_media_duration(&data);
                (FileType::Video { mime_type: mime.into() }, FileData::Video(data, dur))
            }
        };
        ProcessedFile {
            metadata: FileMetadata { file_name: name, file_type },
            data,
        }
    }
}

fn get_media_duration(data: &[u8]) -> f64 {
    use std::io::Cursor;
    use symphonia::core::formats::FormatOptions;
    use symphonia::core::io::{MediaSourceStream, ReadOnlySource};
    use symphonia::core::probe::Hint;

    let source = Box::new(ReadOnlySource::new(Cursor::new(data.to_vec())));
    let mss = MediaSourceStream::new(source, Default::default());

    symphonia::default::get_probe()
        .format(&Hint::new(), mss, &FormatOptions::default(), &Default::default())
        .ok()
        .map(|probed| {
            probed.format
                .tracks()
                .iter()
                .filter_map(|t| {
                    let p = &t.codec_params;
                    p.n_frames.zip(p.time_base)
                        .map(|(n, tb)| tb.calc_time(n))
                        .map(|ts| ts.seconds as f64 + ts.frac)
                })
                .fold(0.0, f64::max)
        })
        .unwrap_or(0.0)
}

pub async fn read_file(path: &Path) -> Result<ProcessedFile> {
    let file_name = path.file_name()
        .and_then(|n| n.to_str())
        .unwrap_or("unknown")
        .to_string();

    let ext = path.extension()
        .and_then(|e| e.to_str())
        .unwrap_or("")
        .to_lowercase();

    if let Some((cat, mime)) = MediaCategory::from_ext(&ext) {
        let data = fs::read(path).await.map_err(|e| anyhow!("Failed to read file: {}", e))?;
        Ok(cat.process(mime, file_name, data))
    } else {
        let text = fs::read_to_string(path).await.map_err(|e| {
            match e.kind() {
                std::io::ErrorKind::InvalidData => anyhow!("File type not supported yet: {}", file_name),
                _ => anyhow!("Failed to read text file: {}", e),
            }
        })?;
        Ok(ProcessedFile {
            metadata: FileMetadata {
                file_name,
                file_type: FileType::Text { encoding: "utf-8".into() },
            },
            data: FileData::Text(text),
        })
    }
}
