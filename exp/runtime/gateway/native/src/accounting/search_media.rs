//! Canonical media translation for conversations rebuilt after input inspection.

use crate::errors::PublicError;
use serde_json::{json, Value};

pub(super) fn optional(target: &mut Value, key: &str, value: &Value) {
    if !value.is_null() {
        target[key] = value.clone();
    }
}

fn reference(part: &Value, providers: &[&str]) -> Result<Option<Value>, PublicError> {
    if part["handle"].is_null() {
        return Ok(None);
    }
    if !providers.contains(&part["handle"]["provider"].as_str().unwrap_or("")) {
        return Err(PublicError::internal());
    }
    Ok(Some(part["handle"]["reference"].clone()))
}

fn data_url(part: &Value) -> Value {
    if part["data"].is_null() {
        part["url"].clone()
    } else {
        json!(format!(
            "data:{};base64,{}",
            part["media_type"].as_str().unwrap_or(""),
            part["data"].as_str().unwrap_or("")
        ))
    }
}

pub(super) fn part(part: &Value, dialect: &str, ordinal: usize) -> Result<Value, PublicError> {
    let kind = part["kind"].as_str().ok_or_else(PublicError::internal)?;
    if kind == "text" {
        return Ok(match dialect {
            "chat" | "anthropic" => json!({"type": "text", "text": part["text"]}),
            "responses" => json!({"type": "input_text", "text": part["text"]}),
            _ => json!({"text": part["text"]}),
        });
    }
    match dialect {
        "chat" => chat(part, kind),
        "responses" => responses(part, kind),
        "anthropic" => anthropic(part, kind),
        "gemini" => gemini(part, kind),
        "bedrock" => bedrock(part, kind, ordinal),
        _ => Err(PublicError::internal()),
    }
}

fn chat(p: &Value, kind: &str) -> Result<Value, PublicError> {
    Ok(match kind {
        "image" | "video" => {
            if !p["handle"].is_null() {
                return Err(PublicError::internal());
            }
            let mut value = json!({"url": data_url(p)});
            if kind == "image" {
                optional(&mut value, "detail", &p["detail"]);
            }
            let key = format!("{kind}_url");
            json!({"type": key, key: value})
        }
        "audio" => {
            json!({"type": "input_audio", "input_audio": {"data": p["data"], "format": audio_format(p)?}})
        }
        "document" => {
            let value = if let Some(id) = reference(p, &["openai"])? {
                json!({"file_id": id})
            } else if p["data"].is_null() {
                return Err(PublicError::internal());
            } else {
                json!({"filename": p["name"].as_str().filter(|s| !s.is_empty()).unwrap_or("document.pdf"), "file_data": data_url(p)})
            };
            json!({"type": "file", "file": value})
        }
        _ => return Err(PublicError::internal()),
    })
}

fn responses(p: &Value, kind: &str) -> Result<Value, PublicError> {
    let mut value = match kind {
        "image" => {
            let mut value = json!({"type": "input_image"});
            if let Some(id) = reference(p, &["openai"])? {
                value["file_id"] = id;
            } else {
                value["image_url"] = data_url(p);
            }
            optional(&mut value, "detail", &p["detail"]);
            value
        }
        "document" => json!({"type": "input_file"}),
        _ => return Err(PublicError::internal()),
    };
    if kind == "document" {
        if let Some(id) = reference(p, &["openai"])? {
            value["file_id"] = id;
        } else if p["data"].is_null() {
            value["file_url"] = p["url"].clone();
        } else {
            value["filename"] = json!(p["name"]
                .as_str()
                .filter(|s| !s.is_empty())
                .unwrap_or("document.pdf"));
            value["file_data"] = data_url(p);
        }
    }
    Ok(value)
}

fn anthropic(p: &Value, kind: &str) -> Result<Value, PublicError> {
    if !matches!(kind, "image" | "document") {
        return Err(PublicError::internal());
    }
    let source = if let Some(id) = reference(p, &["anthropic"])? {
        json!({"type": "file", "file_id": id})
    } else if p["data"].is_null() {
        json!({"type": "url", "url": p["url"]})
    } else {
        json!({"type": "base64", "media_type": p["media_type"], "data": p["data"]})
    };
    let mut value = json!({"type": kind, "source": source});
    if kind == "document" {
        optional(&mut value, "title", &p["name"]);
    }
    optional(&mut value, "cache_control", &p["cache_control"]);
    Ok(value)
}

fn gemini(p: &Value, kind: &str) -> Result<Value, PublicError> {
    if let Some(id) = reference(p, &["gemini", "vertex"])? {
        let mut file = json!({"file_uri": id});
        optional(&mut file, "mime_type", &p["media_type"]);
        return Ok(json!({"file_data": file}));
    }
    if p["data"].is_null() {
        if kind != "video" {
            return Err(PublicError::internal());
        }
        let mut file = json!({"file_uri": p["url"]});
        optional(&mut file, "mime_type", &p["media_type"]);
        return Ok(json!({"file_data": file}));
    }
    Ok(json!({"inline_data": {"mime_type": p["media_type"], "data": p["data"]}}))
}

fn audio_format(p: &Value) -> Result<&str, PublicError> {
    match p["media_type"].as_str() {
        Some("audio/wav" | "audio/x-wav") => Ok("wav"),
        Some("audio/mpeg" | "audio/mp3") => Ok("mp3"),
        _ => Err(PublicError::internal()),
    }
}

fn bedrock(p: &Value, kind: &str, ordinal: usize) -> Result<Value, PublicError> {
    let source = if let Some(id) = reference(p, &["bedrock"])? {
        let mut location = json!({"uri": id});
        optional(&mut location, "bucketOwner", &p["handle"]["bucket_owner"]);
        json!({"s3Location": location})
    } else if p["data"].is_null() {
        return Err(PublicError::internal());
    } else {
        json!({"bytes": p["data"]})
    };
    let format = match (kind, p["media_type"].as_str()) {
        ("document", _) => "pdf",
        ("image", Some("image/jpeg")) => "jpeg",
        ("image", Some("image/png")) => "png",
        ("image", Some("image/gif")) => "gif",
        ("image", Some("image/webp")) => "webp",
        ("video", Some("video/mp4")) => "mp4",
        ("video", Some("video/mpeg")) => "mpeg",
        ("video", Some("video/quicktime")) => "mov",
        ("video", Some("video/webm")) => "webm",
        ("video", Some("video/x-msvideo")) => "avi",
        ("video", Some("video/x-flv")) => "flv",
        ("video", Some("video/x-ms-wmv")) => "wmv",
        ("video", Some("video/3gpp")) => "three_gp",
        _ => return Err(PublicError::internal()),
    };
    let mut value = json!({"format": format, "source": source});
    if kind == "document" {
        static DISALLOWED: std::sync::LazyLock<regex::Regex> =
            std::sync::LazyLock::new(|| regex::Regex::new(r"[^A-Za-z0-9 \-()\[\]]+").unwrap());
        static SPACES: std::sync::LazyLock<regex::Regex> =
            std::sync::LazyLock::new(|| regex::Regex::new(" {2,}").unwrap());
        let cleaned = DISALLOWED.replace_all(p["name"].as_str().unwrap_or(""), "-");
        let cleaned = SPACES.replace_all(&cleaned, " ");
        let cleaned: String = cleaned.trim_matches([' ', '-']).chars().take(200).collect();
        value["name"] = json!(if cleaned.is_empty() {
            format!("document-{ordinal}")
        } else {
            cleaned
        });
    }
    Ok(json!({kind: value}))
}
