//! Split records: a small JSON skeleton plus disk parts, for captures larger than memory.
//!
//! A record whose response spilled to disk, whose caller body is large, or
//! whose encoding exceeds the record bound is handed to the destination as a
//! skeleton in which each large value is replaced by `{"<marker>": index}`.
//! Part `index` holds the exact JSON text of that value (`json`), or for a
//! spilled stream one compact JSON frame per line (`frames`). The marker key
//! carries a per-record nonce, so caller content cannot forge a reference.
//! Nothing is trimmed: the destination splices every part back in place.

use std::fmt;
use std::fs::File;
use std::io::{BufRead, BufReader, Read, Write};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use serde::de::{DeserializeSeed, Deserializer, MapAccess, SeqAccess, Visitor};
use serde_json::{Map, Number, Value};
use sha2::{Digest, Sha256};

use super::record::{Record, Response};
use super::spill::{SpillFile, SpillStore};

/// The destination waits at most this long for the relay writer to finish a file.
const SPILL_WAIT: Duration = Duration::from_secs(120);
/// One SSE event larger than this is not a frame anyone can replay; it is dropped
/// and the stream is marked truncated rather than buffered without bound.
const MAXIMUM_EVENT_BYTES: usize = 64 * 1024 * 1024;

/// The smallest string worth replacing with a reference to its own part.
const MINIMUM_PART_BYTES: usize = 4096;

static NONCE: AtomicU64 = AtomicU64::new(0);

/// A provider response retained on disk rather than in memory.
#[derive(Clone)]
pub(crate) struct SpilledResponse {
    pub status: u16,
    pub sse: bool,
    pub truncated: bool,
    pub client_disconnected: bool,
    pub file: SpillFile,
}

impl fmt::Debug for SpilledResponse {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("SpilledResponse")
            .field("status", &self.status)
            .field("sse", &self.sse)
            .field("truncated", &self.truncated)
            .field("client_disconnected", &self.client_disconnected)
            .finish_non_exhaustive()
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum PartKind {
    /// The exact JSON text of one value.
    Json,
    /// A stream's frames, one compact JSON value per line.
    Frames,
}

impl PartKind {
    pub(crate) fn as_str(self) -> &'static str {
        match self {
            Self::Json => "json",
            Self::Frames => "frames",
        }
    }
}

pub(crate) struct Part {
    pub kind: PartKind,
    pub file: SpillFile,
    pub bytes: u64,
}

pub(crate) struct SplitRecord {
    pub skeleton: String,
    pub marker: String,
    pub parts: Vec<Part>,
}

struct Builder<'a> {
    store: &'a SpillStore,
    marker: String,
    parts: Vec<Part>,
    lost: bool,
}

impl Builder<'_> {
    fn reference(&self, index: usize) -> Value {
        let mut object = Map::new();
        object.insert(self.marker.clone(), Value::Number(Number::from(index)));
        Value::Object(object)
    }

    fn add(
        &mut self,
        kind: PartKind,
        write: impl FnOnce(&mut dyn Write) -> std::io::Result<()>,
    ) -> Option<Value> {
        let file = self.store.write_with(write)?;
        let bytes = file.wait(Duration::ZERO)?;
        self.parts.push(Part { kind, file, bytes });
        Some(self.reference(self.parts.len() - 1))
    }

    /// Replace every string of at least `threshold` bytes in `value` with a
    /// reference to its own part.
    fn extract_strings(&mut self, value: &mut Value, threshold: usize) {
        match value {
            Value::String(text) if text.len() >= threshold => {
                let text = std::mem::take(text);
                match self.add(PartKind::Json, |out| {
                    serde_json::to_writer(out, &text).map_err(std::io::Error::other)
                }) {
                    Some(reference) => *value = reference,
                    None => {
                        self.lost = true;
                        *value = Value::String(text);
                    }
                }
            }
            Value::Array(items) => items
                .iter_mut()
                .for_each(|item| self.extract_strings(item, threshold)),
            Value::Object(items) => items
                .iter_mut()
                .for_each(|(_, item)| self.extract_strings(item, threshold)),
            _ => {}
        }
    }
}

fn nonce(request_id: &str) -> String {
    let sequence = NONCE.fetch_add(1, Ordering::Relaxed);
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |duration| duration.as_nanos());
    let digest = Sha256::new()
        .chain_update(request_id.as_bytes())
        .chain_update(nanos.to_le_bytes())
        .chain_update(sequence.to_le_bytes())
        .chain_update(std::process::id().to_le_bytes())
        .finalize();
    digest[..12]
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect()
}

/// Build the skeleton and parts of one record, or `None` when its parts cannot be kept.
pub(crate) fn split(
    record: &Record,
    store: &SpillStore,
    maximum_skeleton_bytes: usize,
) -> Option<SplitRecord> {
    let mut builder = Builder {
        store,
        marker: format!(
            "$experiential_capture_part_{}",
            nonce(&record.request.request_id)
        ),
        parts: Vec::new(),
        lost: false,
    };
    // The messages are spliced verbatim; the destination never parses them.
    let mut context = context_without_messages(&record.request.context);
    let reference = if let Some(pending) = &record.messages_part {
        // Admission already wrote them in the background.
        let file = pending.wait(SPILL_WAIT)?;
        let bytes = file.wait(Duration::ZERO)?;
        builder.parts.push(Part {
            kind: PartKind::Json,
            file,
            bytes,
        });
        Some(builder.reference(builder.parts.len() - 1))
    } else if let Some(messages) = record
        .request
        .context
        .get("request")
        .and_then(|request| request.get("messages"))
    {
        Some(builder.add(PartKind::Json, |out| {
            serde_json::to_writer(out, messages).map_err(std::io::Error::other)
        })?)
    } else {
        None
    };
    if let (Some(reference), Some(request)) = (
        reference,
        context.get_mut("request").and_then(Value::as_object_mut),
    ) {
        request.insert("messages".into(), reference);
    }
    let spilled = match &record.response {
        Some(Response::Spilled(spilled)) => Some(spilled.clone()),
        _ => None,
    };
    let mut skeleton = record.skeleton_value(context)?;
    if let Some(spilled) = spilled {
        let response = spilled_response(&mut builder, &spilled);
        if response.is_none() {
            // Like an undecodable in-memory body: the request keeps no response.
            for key in ["provider_reasoning", "provider_tool_calls_json"] {
                skeleton[key] = Value::Null;
            }
        }
        skeleton["response"] = response.unwrap_or(Value::Null);
    }
    if let Some(body) = &record.transport_body {
        let (kind, value) = parse_with_parts(&mut builder, body.as_slice()).map_or_else(
            || {
                (
                    "text",
                    Value::String(String::from_utf8_lossy(body).into_owned()),
                )
            },
            |value| ("json", value),
        );
        if let Some(wire) = skeleton
            .get_mut("transport")
            .and_then(|transport| transport.get_mut("wire_request"))
            .and_then(Value::as_object_mut)
        {
            wire.insert("body_kind".into(), Value::String(kind.into()));
            wire.insert("body".into(), value);
        }
    }
    builder.extract_strings(&mut skeleton, store.part_threshold_bytes);
    if builder.lost {
        return None;
    }
    let skeleton = match super::budget::encode(&skeleton, maximum_skeleton_bytes) {
        Some(encoded) => encoded,
        None => {
            // Many medium strings: move every string longer than a reference.
            builder.extract_strings(&mut skeleton, MINIMUM_PART_BYTES);
            if builder.lost {
                return None;
            }
            super::budget::encode(&skeleton, maximum_skeleton_bytes)?
        }
    };
    Some(SplitRecord {
        skeleton,
        marker: builder.marker,
        parts: builder.parts,
    })
}

fn spilled_response(builder: &mut Builder<'_>, spilled: &SpilledResponse) -> Option<Value> {
    let bytes = spilled.file.wait(SPILL_WAIT);
    let truncated = spilled.truncated || bytes.is_none() || spilled.file.truncated();
    if spilled.sse {
        let mut frames_truncated = truncated;
        let reference = match bytes {
            Some(_) => builder.add(PartKind::Frames, |out| {
                frames_truncated |= write_frames(spilled.file.path(), out)?;
                Ok(())
            })?,
            // The relay lost bytes: keep the stream's facts with no frames.
            None => builder.add(PartKind::Frames, |_| Ok(()))?,
        };
        Some(serde_json::json!({
            "kind": "sse",
            "status": spilled.status,
            "frames": reference,
            "truncated": frames_truncated,
            "client_disconnected": spilled.client_disconnected,
            "source_json": null,
        }))
    } else {
        if truncated || spilled.client_disconnected {
            return None;
        }
        let file = File::open(spilled.file.path()).ok()?;
        let body = parse_with_parts(builder, BufReader::new(file))?;
        Some(serde_json::json!({
            "kind": "json",
            "status": spilled.status,
            "body": body,
            "source_json": null,
        }))
    }
}

/// Convert a spilled SSE body into one compact JSON frame per line.
/// Returns whether any event was dropped for exceeding the event bound.
fn write_frames(path: &std::path::Path, out: &mut dyn Write) -> std::io::Result<bool> {
    let mut reader = BufReader::new(File::open(path)?);
    let mut line = Vec::new();
    let mut data = Vec::new();
    let mut has_data = false;
    let mut dropped = false;
    loop {
        line.clear();
        let read = reader
            .by_ref()
            .take(MAXIMUM_EVENT_BYTES as u64 + 1)
            .read_until(b'\n', &mut line)?;
        if read == 0 || !line.ends_with(b"\n") {
            // An unterminated final line is not a complete event (as in memory).
            if read > MAXIMUM_EVENT_BYTES {
                dropped = true;
                // Skip the rest of the oversized line.
                let mut sink = Vec::new();
                reader.read_until(b'\n', &mut sink)?;
                data.clear();
                has_data = false;
                continue;
            }
            break;
        }
        let trimmed = line.strip_suffix(b"\n").unwrap_or(&line);
        let trimmed = trimmed.strip_suffix(b"\r").unwrap_or(trimmed);
        if trimmed.is_empty() {
            if has_data {
                let value = serde_json::from_slice::<Value>(&data)
                    .unwrap_or_else(|_| Value::String(String::from_utf8_lossy(&data).into_owned()));
                serde_json::to_writer(&mut *out, &value).map_err(std::io::Error::other)?;
                out.write_all(b"\n")?;
                data.clear();
                has_data = false;
            }
        } else if let Some(payload) = trimmed.strip_prefix(b"data:") {
            let payload = payload.strip_prefix(b" ").unwrap_or(payload);
            if data.len() + payload.len() + 1 > MAXIMUM_EVENT_BYTES {
                dropped = true;
                data.clear();
                has_data = false;
                continue;
            }
            if has_data {
                data.push(b'\n');
            }
            data.extend_from_slice(payload);
            has_data = true;
        }
    }
    Ok(dropped)
}

/// Parse JSON with every long string written to its own part as it is read,
/// so memory holds the skeleton plus at most one string.
fn parse_with_parts(builder: &mut Builder<'_>, input: impl Read) -> Option<Value> {
    let mut deserializer = serde_json::Deserializer::from_reader(input);
    let value = PartSeed { builder }.deserialize(&mut deserializer).ok()?;
    deserializer.end().ok()?;
    Some(value)
}

struct PartSeed<'b, 'a> {
    builder: &'b mut Builder<'a>,
}

impl<'de> DeserializeSeed<'de> for PartSeed<'_, '_> {
    type Value = Value;

    fn deserialize<D: Deserializer<'de>>(self, deserializer: D) -> Result<Value, D::Error> {
        deserializer.deserialize_any(self)
    }
}

impl<'de> Visitor<'de> for PartSeed<'_, '_> {
    type Value = Value;

    fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("a JSON value")
    }

    fn visit_bool<E>(self, value: bool) -> Result<Value, E> {
        Ok(Value::Bool(value))
    }

    fn visit_i64<E>(self, value: i64) -> Result<Value, E> {
        Ok(Value::Number(value.into()))
    }

    fn visit_u64<E>(self, value: u64) -> Result<Value, E> {
        Ok(Value::Number(value.into()))
    }

    fn visit_f64<E: serde::de::Error>(self, value: f64) -> Result<Value, E> {
        Number::from_f64(value)
            .map(Value::Number)
            .ok_or_else(|| E::custom("non-finite number"))
    }

    fn visit_str<E: serde::de::Error>(self, value: &str) -> Result<Value, E> {
        if value.len() < self.builder.store.part_threshold_bytes {
            return Ok(Value::String(value.to_owned()));
        }
        self.builder
            .add(PartKind::Json, |out| {
                serde_json::to_writer(out, value).map_err(std::io::Error::other)
            })
            .ok_or_else(|| E::custom("capture part could not be kept"))
    }

    fn visit_string<E: serde::de::Error>(self, value: String) -> Result<Value, E> {
        self.visit_str(&value)
    }

    fn visit_unit<E>(self) -> Result<Value, E> {
        Ok(Value::Null)
    }

    fn visit_none<E>(self) -> Result<Value, E> {
        Ok(Value::Null)
    }

    fn visit_seq<A: SeqAccess<'de>>(self, mut sequence: A) -> Result<Value, A::Error> {
        let mut items = Vec::new();
        while let Some(item) = sequence.next_element_seed(PartSeed {
            builder: &mut *self.builder,
        })? {
            items.push(item);
        }
        Ok(Value::Array(items))
    }

    fn visit_map<A: MapAccess<'de>>(self, mut map: A) -> Result<Value, A::Error> {
        let mut object = Map::new();
        while let Some(key) = map.next_key::<String>()? {
            let value = map.next_value_seed(PartSeed {
                builder: &mut *self.builder,
            })?;
            object.insert(key, value);
        }
        Ok(Value::Object(object))
    }
}

/// Copy of a request context without its message list (which may be large).
pub(crate) fn context_without_messages(context: &Value) -> Value {
    {
        let Some(context) = context.as_object() else {
            return context.clone();
        };
        let mut copy = Map::new();
        for (key, value) in context {
            if key == "request" {
                if let Some(request) = value.as_object() {
                    let mut request_copy = Map::new();
                    for (name, field) in request {
                        if name != "messages" {
                            request_copy.insert(name.clone(), field.clone());
                        }
                    }
                    copy.insert(key.clone(), Value::Object(request_copy));
                    continue;
                }
            }
            copy.insert(key.clone(), value.clone());
        }
        Value::Object(copy)
    }
}

#[cfg(test)]
#[path = "split_test.rs"]
pub(crate) mod tests;
