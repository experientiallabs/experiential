//! Media capture: binary parts, embedding vectors and provider jobs beside a record.
//!
//! A media surface (embeddings, images, speech, transcription, video) answers
//! with content that does not belong in a JSON record: base64 images, audio
//! bytes, embedding vectors. Each such part leaves the record as a
//! [`MediaPart`]: its role, kind, MIME type, size and SHA-256, plus the path of
//! a spill file holding its bytes when the collector spills. The bytes are
//! written by the spill thread, never through Python. The record keeps a
//! reference-free JSON projection of the body (vectors and base64 removed), so
//! a destination can store the request and response without parsing binary
//! content, and decide per part what to keep.
//!
//! A part's spill file lives as long as the record holding it and is removed
//! once the destination acknowledges the record, exactly like a split record's
//! parts. Without a spill directory the part is still described (size and
//! hash) but carries no path.

use std::sync::Arc;
use std::time::Duration;

use axum::body::Body;
use axum::response::Response;
use base64::Engine;
use serde::ser::SerializeStruct;
use serde::Serialize;
use serde_json::{json, Value};
use sha2::{Digest, Sha256};

use super::collector::Collector;
use super::record::Response as CapturedResponse;
use super::spill::{SpillFile, SpillStore};

/// How long a destination waits for a part's spill file before describing it without a path.
const PART_WAIT: Duration = Duration::from_secs(120);

/// Whether a part was sent by the caller or produced for it.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum MediaRole {
    Input,
    Output,
}

/// What a part holds.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum MediaKind {
    Image,
    Audio,
    Video,
    Vector,
}

/// A record half a host's settlement chose not to keep.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum OmittedPart {
    Request,
    Response,
}

/// One binary part: its description and, when spilled, the file holding it.
#[derive(Clone)]
pub(crate) struct MediaPart {
    pub role: MediaRole,
    pub kind: MediaKind,
    pub mime: String,
    pub bytes: u64,
    /// Lowercase hexadecimal SHA-256 of the exact bytes.
    pub sha256: String,
    pub file: Option<SpillFile>,
}

impl std::fmt::Debug for MediaPart {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("MediaPart")
            .field("role", &self.role)
            .field("kind", &self.kind)
            .field("mime", &self.mime)
            .field("bytes", &self.bytes)
            .field("spilled", &self.file.is_some())
            .finish()
    }
}

impl MediaPart {
    /// Describe `data` and, when `store` is present, queue it to a spill file.
    /// Hashing runs on the caller's task; the write runs on the spill thread.
    pub(crate) fn new(
        role: MediaRole,
        kind: MediaKind,
        mime: &str,
        data: bytes::Bytes,
        store: Option<&SpillStore>,
    ) -> Self {
        let sha256 = hex(&Sha256::digest(&data));
        let length = data.len() as u64;
        let file = store.and_then(|store| {
            let mut stream = store.open_stream()?;
            stream.push(data).then(|| stream.finish())
        });
        Self {
            role,
            kind,
            mime: bounded_mime(mime),
            bytes: length,
            sha256,
            file,
        }
    }

    fn heap_bytes(&self) -> usize {
        std::mem::size_of::<Self>()
            + self.mime.capacity()
            + self.sha256.capacity()
            + self.file.as_ref().map_or(0, SpillFile::heap_bytes)
    }
}

impl Serialize for MediaPart {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        // Only a complete, untrimmed file is named; a lost or cut one is
        // described without a path so no destination stores partial media.
        let path = self.file.as_ref().and_then(|file| {
            let written = file.wait(PART_WAIT)?;
            (written == self.bytes && !file.truncated())
                .then(|| file.path().to_str().map(str::to_owned))
                .flatten()
        });
        let mut part = serializer.serialize_struct("MediaPart", 6)?;
        part.serialize_field("part", &self.role)?;
        part.serialize_field("kind", &self.kind)?;
        part.serialize_field("mime", &self.mime)?;
        part.serialize_field("bytes", &self.bytes)?;
        part.serialize_field("sha256", &self.sha256)?;
        part.serialize_field("spill_path", &path)?;
        part.end()
    }
}

/// Count and width of the vectors an embeddings response returned.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
pub(crate) struct VectorSummary {
    pub count: u64,
    pub dimensions: u64,
}

/// A provider job a record's response belongs to (a video render).
#[derive(Clone, Debug, PartialEq, Serialize)]
pub(crate) struct ProviderJob {
    pub id: String,
    /// Unix seconds by which the job settles; the capture entry lives until then.
    pub deadline: f64,
}

/// Everything a media surface adds to one record.
#[derive(Clone, Debug, Default)]
pub(crate) struct MediaCapture {
    pub parts: Vec<MediaPart>,
    pub vector_summary: Option<VectorSummary>,
    pub omitted_parts: Vec<OmittedPart>,
    pub provider_job: Option<ProviderJob>,
}

impl MediaCapture {
    /// Write only the keys that carry something; a fully kept text record is unchanged.
    pub(crate) fn serialize_fields<S: SerializeStruct>(
        &self,
        record: &mut S,
    ) -> Result<(), S::Error> {
        if !self.parts.is_empty() {
            record.serialize_field("media", &self.parts)?;
        }
        if let Some(summary) = &self.vector_summary {
            record.serialize_field("vector_summary", summary)?;
        }
        if !self.omitted_parts.is_empty() {
            record.serialize_field("omitted_parts", &self.omitted_parts)?;
        }
        if let Some(job) = &self.provider_job {
            record.serialize_field("provider_job", job)?;
        }
        Ok(())
    }

    pub(crate) fn heap_bytes(&self) -> usize {
        self.parts.iter().map(MediaPart::heap_bytes).sum::<usize>()
            + self
                .provider_job
                .as_ref()
                .map_or(0, |job| job.id.capacity() + 16)
    }

    /// The spill files the parts hold, for a destination to keep until it acknowledges.
    pub(crate) fn files(&self) -> Vec<SpillFile> {
        self.parts
            .iter()
            .filter_map(|part| part.file.clone())
            .collect()
    }

    /// Drop one side's parts; the files go when their last handle does.
    pub(crate) fn omit(&mut self, role: MediaRole) {
        self.parts.retain(|part| part.role != role);
        if role == MediaRole::Output {
            self.vector_summary = None;
        }
        let omitted = match role {
            MediaRole::Input => OmittedPart::Request,
            MediaRole::Output => OmittedPart::Response,
        };
        if !self.omitted_parts.contains(&omitted) {
            self.omitted_parts.push(omitted);
        }
    }
}

/// The surfaces whose buffered answers [`capture_buffered`] shapes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum MediaSurface {
    Embeddings,
    Images,
    Speech,
    Transcription,
}

/// A buffered answer split into its record projection and its binary parts.
pub(crate) struct Shaped {
    pub body: Value,
    pub parts: Vec<MediaPart>,
    pub vector_summary: Option<VectorSummary>,
}

/// Split one complete answer into a JSON projection and media parts.
///
/// Embeddings move every `data[].embedding` into one `vector` part (a JSON
/// array in data order) and record a [`VectorSummary`]. Images move every
/// `data[].b64_json` into one decoded `image` part each, in data order. A
/// non-JSON answer (speech audio) becomes one output part, projected as its
/// content type and size; a non-JSON transcript (text, SRT, VTT) is kept as
/// text. An error body is kept as JSON with no parts.
pub(crate) fn shape(
    surface: MediaSurface,
    status: u16,
    content_type: &str,
    body: &[u8],
    store: Option<&SpillStore>,
) -> Option<Shaped> {
    let json_body = content_type.starts_with("application/json") || content_type.is_empty();
    if json_body || !(200..300).contains(&status) {
        let mut value = serde_json::from_slice::<Value>(body).ok()?;
        let mut parts = Vec::new();
        let mut vector_summary = None;
        if (200..300).contains(&status) {
            match surface {
                MediaSurface::Embeddings => {
                    vector_summary = extract_vectors(&mut value, store, &mut parts);
                }
                MediaSurface::Images => extract_images(&mut value, store, &mut parts),
                MediaSurface::Speech | MediaSurface::Transcription => {}
            }
        }
        return Some(Shaped {
            body: value,
            parts,
            vector_summary,
        });
    }
    let mime = content_type.split(';').next().unwrap_or("").trim();
    match surface {
        MediaSurface::Transcription => Some(Shaped {
            body: json!({
                "content_type": mime,
                "text": String::from_utf8_lossy(body),
            }),
            parts: Vec::new(),
            vector_summary: None,
        }),
        MediaSurface::Speech | MediaSurface::Images | MediaSurface::Embeddings => {
            let kind = if mime.starts_with("image/") {
                MediaKind::Image
            } else {
                MediaKind::Audio
            };
            let part = MediaPart::new(
                MediaRole::Output,
                kind,
                mime,
                bytes::Bytes::copy_from_slice(body),
                store,
            );
            Some(Shaped {
                body: json!({"content_type": mime, "bytes": body.len()}),
                parts: vec![part],
                vector_summary: None,
            })
        }
    }
}

fn extract_vectors(
    value: &mut Value,
    store: Option<&SpillStore>,
    parts: &mut Vec<MediaPart>,
) -> Option<VectorSummary> {
    let data = value.get_mut("data")?.as_array_mut()?;
    let mut vectors = Vec::with_capacity(data.len());
    let mut dimensions = 0u64;
    for item in data.iter_mut() {
        let Some(embedding) = item
            .as_object_mut()
            .and_then(|item| item.remove("embedding"))
        else {
            continue;
        };
        let width = match &embedding {
            Value::Array(values) => values.len() as u64,
            // base64 little-endian float32, as OpenAI encodes it.
            Value::String(encoded) => base64::engine::general_purpose::STANDARD
                .decode(encoded)
                .map_or(0, |raw| raw.len() as u64 / 4),
            _ => 0,
        };
        dimensions = dimensions.max(width);
        vectors.push(embedding);
    }
    if vectors.is_empty() {
        return None;
    }
    let count = vectors.len() as u64;
    let encoded = serde_json::to_vec(&Value::Array(vectors)).ok()?;
    parts.push(MediaPart::new(
        MediaRole::Output,
        MediaKind::Vector,
        "application/json",
        bytes::Bytes::from(encoded),
        store,
    ));
    Some(VectorSummary { count, dimensions })
}

fn extract_images(value: &mut Value, store: Option<&SpillStore>, parts: &mut Vec<MediaPart>) {
    let format = value
        .get("output_format")
        .and_then(Value::as_str)
        .map(str::to_owned);
    let Some(data) = value.get_mut("data").and_then(Value::as_array_mut) else {
        return;
    };
    for item in data.iter_mut() {
        let Some(object) = item.as_object_mut() else {
            continue;
        };
        let Some(Value::String(encoded)) = object.remove("b64_json") else {
            continue;
        };
        let Ok(raw) = base64::engine::general_purpose::STANDARD.decode(encoded.as_bytes()) else {
            // Undecodable text stays where it was rather than vanish.
            object.insert("b64_json".into(), Value::String(encoded));
            continue;
        };
        let mime = image_mime(format.as_deref(), &raw);
        parts.push(MediaPart::new(
            MediaRole::Output,
            MediaKind::Image,
            mime,
            bytes::Bytes::from(raw),
            store,
        ));
    }
}

/// The image's MIME type: the response's stated format, else its magic bytes.
pub(crate) fn image_mime(format: Option<&str>, raw: &[u8]) -> &'static str {
    match format {
        Some("png") => return "image/png",
        Some("jpeg" | "jpg") => return "image/jpeg",
        Some("webp") => return "image/webp",
        _ => {}
    }
    if raw.starts_with(b"\xff\xd8\xff") {
        "image/jpeg"
    } else if raw.len() >= 12 && &raw[..4] == b"RIFF" && &raw[8..12] == b"WEBP" {
        "image/webp"
    } else if raw.starts_with(b"GIF8") {
        "image/gif"
    } else {
        "image/png"
    }
}

fn bounded_mime(mime: &str) -> String {
    let mime = mime.split(';').next().unwrap_or("").trim();
    if mime.is_empty() || mime.len() > 128 || !mime.bytes().all(|byte| byte.is_ascii_graphic()) {
        "application/octet-stream".to_owned()
    } else {
        mime.to_ascii_lowercase()
    }
}

fn hex(digest: &[u8]) -> String {
    use std::fmt::Write;
    digest
        .iter()
        .fold(String::with_capacity(64), |mut out, byte| {
            let _ = write!(out, "{byte:02x}");
            out
        })
}

/// Capture one fully buffered media answer and return it unchanged.
///
/// The surfaces that use this answer only after settlement, from a body
/// already in memory, so reading it back costs no stream. The record gets the
/// shaped projection and the media parts; the caller receives the original
/// bytes and headers.
pub(crate) async fn capture_buffered(
    collector: Option<Arc<Collector>>,
    request_id: &str,
    surface: MediaSurface,
    response: Response,
) -> Response {
    let Some(collector) = collector else {
        return response;
    };
    if collector.attach(request_id).is_none() {
        return response;
    }
    let (parts, body) = response.into_parts();
    let Ok(bytes) = axum::body::to_bytes(body, usize::MAX).await else {
        collector.finish_unexposed(request_id);
        return Response::from_parts(parts, Body::empty());
    };
    let content_type = parts
        .headers
        .get("content-type")
        .and_then(|value| value.to_str().ok())
        .unwrap_or_default()
        .to_ascii_lowercase();
    let deployment_id = parts
        .headers
        .get("x-gateway-deployment")
        .and_then(|value| value.to_str().ok())
        .map(str::to_owned);
    let status = parts.status.as_u16();
    let shaped = shape(
        surface,
        status,
        &content_type,
        &bytes,
        collector.spill().map(Arc::as_ref),
    );
    match shaped {
        Some(shaped) => {
            let mut body = shaped.body;
            let source_json = super::response::lossless_projection(&mut body);
            collector.finish_media(
                request_id,
                CapturedResponse::Json {
                    status,
                    body,
                    source_json,
                },
                shaped.parts,
                shaped.vector_summary,
                deployment_id,
            );
        }
        None => collector.finish_unexposed(request_id),
    }
    Response::from_parts(parts, Body::from(bytes))
}

/// Describe caller-sent media (an edit's images, a transcription's audio).
/// `inputs` runs only for a registered capture, so uncaptured traffic pays nothing.
pub(crate) fn attach_inputs(
    collector: Option<&Arc<Collector>>,
    request_id: &str,
    inputs: impl FnOnce() -> Vec<(MediaKind, String, bytes::Bytes)>,
) {
    let Some(collector) = collector else {
        return;
    };
    if !collector.registered(request_id) {
        return;
    }
    let store = collector.spill();
    let parts = inputs()
        .into_iter()
        .map(|(kind, mime, data)| {
            MediaPart::new(MediaRole::Input, kind, &mime, data, store.map(Arc::as_ref))
        })
        .collect();
    collector.attach_media(request_id, parts);
}

#[cfg(test)]
#[path = "media_test.rs"]
mod tests;
