use anyhow::{Result, anyhow};
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
                FileType::Image {
                    mime_type: mime.into(),
                },
                FileData::Image(data),
            ),
            Self::Audio => {
                let dur = get_media_duration(&data);
                (
                    FileType::Audio {
                        mime_type: mime.into(),
                    },
                    FileData::Audio(data, dur),
                )
            }
            Self::Video => {
                let dur = get_media_duration(&data);
                (
                    FileType::Video {
                        mime_type: mime.into(),
                    },
                    FileData::Video(data, dur),
                )
            }
        };
        ProcessedFile {
            metadata: FileMetadata {
                file_name: name,
                file_type,
            },
            data,
        }
    }
}

pub fn compress_image(data: &[u8], max_px: u32, quality: u8) -> Result<Vec<u8>> {
    use image::ImageReader;
    use std::io::Cursor;

    let img = ImageReader::new(Cursor::new(data))
        .with_guessed_format()?
        .decode()?;

    let (width, height) = (img.width(), img.height());
    let resized = if width > max_px || height > max_px {
        let nwidth = if width > height {
            max_px
        } else {
            (width as f64 * (max_px as f64 / height as f64)) as u32
        };
        let nheight = if height > width {
            max_px
        } else {
            (height as f64 * (max_px as f64 / width as f64)) as u32
        };
        img.resize(nwidth, nheight, image::imageops::FilterType::Lanczos3)
    } else {
        img
    };

    let mut out = Vec::new();
    let mut cursor = Cursor::new(&mut out);
    let jpeg_encoder = image::codecs::jpeg::JpegEncoder::new_with_quality(&mut cursor, quality);
    resized.write_with_encoder(jpeg_encoder)?;
    Ok(out)
}

pub async fn compress_audio_ffmpeg(
    data: &[u8],
    ext: &str,
    ffmpeg: &str,
    bitrate: Option<&str>,
    mono: bool,
    sample_rate: Option<u32>,
) -> Result<Vec<u8>> {
    use std::io::Write;
    use tokio::process::Command;

    let mut input_temp = tempfile::Builder::new()
        .prefix("sum_audio_in_")
        .suffix(&format!(".{}", ext))
        .tempfile()?;
    input_temp.write_all(data)?;

    let output_temp = tempfile::Builder::new()
        .prefix("sum_audio_out_")
        .suffix(".mp3")
        .tempfile()?;

    let mut cmd = Command::new(ffmpeg);
    cmd.arg("-y").arg("-i").arg(input_temp.path());

    if mono {
        cmd.arg("-ac").arg("1");
    }
    if let Some(br) = bitrate {
        cmd.arg("-b:a").arg(br);
    }
    if let Some(sr) = sample_rate {
        cmd.arg("-ar").arg(sr.to_string());
    }

    cmd.arg(output_temp.path());

    let status = cmd.status().await?;
    if !status.success() {
        return Err(anyhow!("ffmpeg failed with status: {:?}", status.code()));
    }

    let out_data = fs::read(output_temp.path()).await?;
    Ok(out_data)
}

pub async fn compress_video_ffmpeg(
    data: &[u8],
    ext: &str,
    ffmpeg: &str,
    max_height: Option<u32>,
    video_bitrate: Option<&str>,
    audio_bitrate: Option<&str>,
) -> Result<Vec<u8>> {
    use std::io::Write;
    use tokio::process::Command;

    let mut input_temp = tempfile::Builder::new()
        .prefix("sum_video_in_")
        .suffix(&format!(".{}", ext))
        .tempfile()?;
    input_temp.write_all(data)?;

    let output_temp = tempfile::Builder::new()
        .prefix("sum_video_out_")
        .suffix(".mp4")
        .tempfile()?;

    let mut cmd = Command::new(ffmpeg);
    cmd.arg("-y").arg("-i").arg(input_temp.path());

    if let Some(h) = max_height {
        // scale=-2:height ensures even width is computed, maintaining ratio
        cmd.arg("-vf").arg(format!("scale=-2:{}", h));
    }
    if let Some(vbr) = video_bitrate {
        cmd.arg("-b:v").arg(vbr);
    }
    if let Some(abr) = audio_bitrate {
        cmd.arg("-b:a").arg(abr);
    }

    cmd.arg(output_temp.path());

    let status = cmd.status().await?;
    if !status.success() {
        return Err(anyhow!("ffmpeg failed with status: {:?}", status.code()));
    }

    let out_data = fs::read(output_temp.path()).await?;
    Ok(out_data)
}

pub fn get_media_duration(data: &[u8]) -> f64 {
    use std::io::Cursor;
    use symphonia::core::formats::FormatOptions;
    use symphonia::core::io::{MediaSourceStream, ReadOnlySource};
    use symphonia::core::probe::Hint;

    let source = Box::new(ReadOnlySource::new(Cursor::new(data.to_vec())));
    let mss = MediaSourceStream::new(source, Default::default());

    symphonia::default::get_probe()
        .format(
            &Hint::new(),
            mss,
            &FormatOptions::default(),
            &Default::default(),
        )
        .ok()
        .map(|probed| {
            probed
                .format
                .tracks()
                .iter()
                .filter_map(|t| {
                    let p = &t.codec_params;
                    p.n_frames
                        .zip(p.time_base)
                        .map(|(n, tb)| tb.calc_time(n))
                        .map(|ts| ts.seconds as f64 + ts.frac)
                })
                .fold(0.0, f64::max)
        })
        .unwrap_or(0.0)
}

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

    if let Some((cat, mime)) = MediaCategory::from_ext(&ext) {
        let data = fs::read(path)
            .await
            .map_err(|e| anyhow!("Failed to read file: {}", e))?;
        Ok(cat.process(mime, file_name, data))
    } else {
        let text = fs::read_to_string(path).await.map_err(|e| match e.kind() {
            std::io::ErrorKind::InvalidData => {
                anyhow!("File type not supported yet: {}", file_name)
            }
            _ => anyhow!("Failed to read text file: {}", e),
        })?;
        Ok(ProcessedFile {
            metadata: FileMetadata {
                file_name,
                file_type: FileType::Text {
                    encoding: "utf-8".into(),
                },
            },
            data: FileData::Text(text),
        })
    }
}
