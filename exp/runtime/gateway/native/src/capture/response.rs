//! Native response tap. Forward bytes unchanged and acknowledge destination completion.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use axum::body::{Body, HttpBody};
use axum::response::Response;
use futures_util::StreamExt;
use serde_json::{value::RawValue, Value};
use tokio::sync::OwnedSemaphorePermit;

use super::collector::Collector;
use super::record::Response as CapturedResponse;

struct Tap {
    collector: Arc<Collector>,
    request_id: String,
    deployment_id: Option<String>,
    sse: bool,
    status: u16,
    bytes: Vec<u8>,
    charged: usize,
    truncated: bool,
    finished: bool,
    permit: Option<OwnedSemaphorePermit>,
    discarded: Arc<AtomicBool>,
}

struct BodyCharge {
    collector: Arc<Collector>,
    bytes: usize,
    _permit: Option<OwnedSemaphorePermit>,
}

impl Drop for BodyCharge {
    fn drop(&mut self) {
        self.collector.release_body(self.bytes);
    }
}

/// Retain original wire bytes until the single destination worker needs JSON.
/// The response permit covers queued, blocked and actively decoded bodies alike.
pub(super) struct WireResponse {
    pub(super) relay: Option<super::relay::Relay>,
    bytes: Vec<u8>,
    sse: bool,
    status: u16,
    truncated: bool,
    disconnected: bool,
    maximum_bytes: usize,
    _charge: BodyCharge,
}

impl WireResponse {
    pub(super) fn heap_bytes(&self) -> usize {
        std::mem::size_of::<Self>()
            + self.bytes.capacity()
            + self
                .relay
                .as_ref()
                .map_or(0, super::relay::Relay::heap_bytes)
    }

    pub(super) fn decode(self) -> Option<CapturedResponse> {
        if self.sse {
            let (mut frames, number_sources) = data_frames_with_sources(&self.bytes);
            let mut truncated = self.truncated;
            loop {
                let mut frame_value = Value::Array(frames);
                let mut source_json = lossless_projection(&mut frame_value);
                if !number_sources.is_empty() {
                    let encoded = source_json
                        .take()
                        .unwrap_or_else(|| serde_json::to_string(&frame_value).unwrap());
                    let mut raw: Vec<Box<RawValue>> = serde_json::from_str(&encoded).unwrap();
                    for (index, source) in &number_sources {
                        if let Some(frame) = raw.get_mut(*index) {
                            *frame = RawValue::from_string(source.clone()).unwrap();
                        }
                    }
                    source_json = Some(serde_json::to_string(&raw).unwrap());
                }
                let Value::Array(moved_frames) = frame_value else {
                    unreachable!()
                };
                let response = CapturedResponse::Sse {
                    status: self.status,
                    frames: moved_frames,
                    truncated,
                    client_disconnected: self.disconnected,
                    source_json,
                };
                if response.json_bytes() <= self.maximum_bytes {
                    break Some(response);
                }
                let CapturedResponse::Sse {
                    frames: mut reduced,
                    source_json,
                    ..
                } = response
                else {
                    unreachable!()
                };
                if let Some(source) = source_json {
                    reduced = serde_json::from_str(&source).unwrap_or(reduced);
                }
                if reduced.is_empty() {
                    break None;
                }
                reduced.truncate(reduced.len() / 2);
                frames = reduced;
                truncated = true;
            }
        } else if !self.truncated && !self.disconnected {
            serde_json::from_slice::<Value>(&self.bytes)
                .ok()
                .map(|mut body| {
                    let wide = contains_wide_number(&body);
                    let mut source_json = lossless_projection(&mut body);
                    if wide {
                        // Successful JSON parsing already proved valid UTF-8.
                        source_json = Some(String::from_utf8(self.bytes).unwrap());
                    }
                    CapturedResponse::Json {
                        status: self.status,
                        body,
                        source_json,
                    }
                })
                .filter(|response| response.json_bytes() <= self.maximum_bytes)
        } else {
            None
        }
    }
}

impl Tap {
    fn push(&mut self, chunk: &[u8]) {
        if self.truncated {
            return;
        }
        let Some(required) = self.bytes.len().checked_add(chunk.len()) else {
            self.truncated = true;
            return;
        };
        let maximum = self.collector.config.maximum_response_bytes;
        if required > maximum {
            self.truncated = true;
            return;
        }
        if required > self.bytes.capacity() {
            let target = required.saturating_add(4095).min(maximum);
            let extra = target - self.bytes.capacity();
            if !self.collector.reserve_body(extra) {
                self.truncated = true;
                return;
            }
            self.charged += extra;
            if self
                .bytes
                .try_reserve_exact(target - self.bytes.len())
                .is_err()
            {
                self.truncated = true;
                return;
            }
            let excess = self.bytes.capacity().saturating_sub(self.charged);
            if !self.collector.reserve_body(excess) {
                self.bytes = Vec::new();
                self.collector.release_body(self.charged);
                self.charged = 0;
                self.truncated = true;
                return;
            }
            self.charged += excess;
        }
        self.bytes.extend_from_slice(chunk);
    }

    fn finish(&mut self, disconnected: bool) -> bool {
        if self.finished {
            return true;
        }
        self.finished = true;
        // While streaming, reserve room for the largest permitted body. Once
        // complete, retain only allocated bytes so small queued responses do
        // not pin a maximum-size slot until the destination decodes them.
        if let Some(permit) = self.permit.as_mut() {
            let unused = permit.num_permits().saturating_sub(self.charged);
            drop(permit.split(unused));
        }
        let wire = WireResponse {
            relay: None,
            bytes: std::mem::take(&mut self.bytes),
            sse: self.sse,
            status: self.status,
            truncated: self.truncated,
            disconnected,
            maximum_bytes: self.collector.config.maximum_response_bytes,
            _charge: BodyCharge {
                collector: self.collector.clone(),
                bytes: std::mem::take(&mut self.charged),
                _permit: self.permit.take(),
            },
        };
        self.collector
            .finish_wire(&self.request_id, wire, self.deployment_id.take())
            || self.discarded.load(Ordering::Acquire)
    }
}

impl Drop for Tap {
    fn drop(&mut self) {
        self.finish(true);
        self.collector.release_body(self.charged);
    }
}

/// Attach only to content registered by authenticated admission, never to arbitrary traffic.
pub(crate) fn capture_response(
    collector: Option<Arc<Collector>>,
    request_id: &str,
    response: Response,
) -> Response {
    let Some(collector) = collector else {
        return response;
    };
    if !response.headers().contains_key("x-request-id") {
        collector.without_relay(request_id);
    }
    let Some(discarded) = collector.attach(request_id) else {
        return response;
    };
    let sse = response
        .headers()
        .get("content-type")
        .and_then(|value| value.to_str().ok())
        .is_some_and(|value| value.starts_with("text/event-stream"));
    let deployment_id = response
        .headers()
        .get("x-gateway-deployment")
        .and_then(|value| value.to_str().ok())
        .map(str::to_owned);
    // A relay can stop polling once Content-Length bytes have arrived, without
    // polling EOF on a streaming body wrapper (notably retained Responses JSON).
    let expected = response.body().size_hint().exact().or_else(|| {
        response
            .headers()
            .get("content-length")?
            .to_str()
            .ok()?
            .parse::<u64>()
            .ok()
    });
    let status = response.status().as_u16();
    let (parts, body) = response.into_parts();
    let tap = Tap {
        collector,
        request_id: request_id.to_owned(),
        deployment_id,
        sse,
        status,
        bytes: Vec::new(),
        charged: 0,
        truncated: false,
        finished: false,
        permit: None,
        discarded,
    };
    let stream = futures_util::stream::unfold(
        (body.into_data_stream(), tap, 0u64),
        move |(mut body, mut tap, mut sent)| async move {
            if tap.permit.is_none() && !tap.finished {
                tap.permit = tap.collector.body_permit().await;
            }
            match body.next().await {
                Some(mut item) => {
                    match &item {
                        Ok(chunk) => {
                            tap.push(chunk);
                            sent = sent.saturating_add(chunk.len() as u64);
                            if expected == Some(sent) && !tap.finish(false) {
                                item = Err(axum::Error::new(std::io::Error::other(
                                    "capture persistence failed",
                                )));
                            }
                        }
                        Err(_) => {
                            tap.finish(true);
                        }
                    }
                    Some((item, (body, tap, sent)))
                }
                None => {
                    if tap.finish(false) {
                        None
                    } else {
                        Some((
                            Err(axum::Error::new(std::io::Error::other(
                                "capture persistence failed",
                            ))),
                            (body, tap, sent),
                        ))
                    }
                }
            }
        },
    );
    Response::from_parts(parts, Body::from_stream(stream))
}

/// Only whole SSE events enter the record, including non-JSON data such as [DONE].
#[cfg(test)]
fn data_frames(bytes: &[u8]) -> Vec<Value> {
    data_frames_with_sources(bytes).0
}

fn data_frames_with_sources(bytes: &[u8]) -> (Vec<Value>, Vec<(usize, String)>) {
    let mut frames = Vec::new();
    let mut number_sources = Vec::new();
    let mut data: Vec<&[u8]> = Vec::new();
    for line in bytes.split_inclusive(|byte| *byte == b'\n') {
        if !line.ends_with(b"\n") {
            break;
        }
        let line = line.strip_suffix(b"\n").unwrap_or(line);
        let line = line.strip_suffix(b"\r").unwrap_or(line);
        if line.is_empty() {
            if !data.is_empty() {
                let payload = data.join(&b'\n');
                let value = serde_json::from_slice(&payload).unwrap_or_else(|_| {
                    Value::String(String::from_utf8_lossy(&payload).into_owned())
                });
                if contains_wide_number(&value) {
                    number_sources.push((frames.len(), String::from_utf8(payload).unwrap()));
                }
                frames.push(value);
                data.clear();
            }
        } else if let Some(payload) = line.strip_prefix(b"data:") {
            data.push(payload.strip_prefix(b" ").unwrap_or(payload));
        }
    }
    (frames, number_sources)
}

/// Finite-width JSON can round integers beyond its signed/unsigned 64-bit range.
/// A wide float may also trigger a sidecar; preserving its source is harmless.
pub(super) fn contains_wide_number(value: &Value) -> bool {
    match value {
        Value::Number(number) if !number.is_i64() && !number.is_u64() => {
            number.as_f64().is_some_and(|number| {
                number >= 18_446_744_073_709_551_616.0 || number <= -9_223_372_036_854_775_808.0
            })
        }
        Value::Array(values) => values.iter().any(contains_wide_number),
        Value::Object(values) => values.values().any(contains_wide_number),
        _ => false,
    }
}

/// Preserve exact JSON text whenever storage requires a normalized projection.
pub(super) fn lossless_projection(value: &mut Value) -> Option<String> {
    if !contains_nul(value) {
        return None;
    }
    let source = serde_json::to_string(value).ok()?;
    normalize(value);
    Some(source)
}

/// Inspect decoded text without allocating a complete JSON representation.
pub(super) fn contains_nul(value: &Value) -> bool {
    match value {
        Value::String(text) => text.contains('\0'),
        Value::Array(values) => values.iter().any(contains_nul),
        Value::Object(object) => object
            .iter()
            .any(|(key, value)| key.contains('\0') || contains_nul(value)),
        _ => false,
    }
}

/// Storage normalization never touches forwarded bytes. Preserve colliding object keys.
fn normalize(value: &mut Value) {
    match value {
        Value::String(text) if text.contains('\0') => {
            *text = text.replace('\0', "\u{fffd}");
        }
        Value::Array(values) => values.iter_mut().for_each(normalize),
        Value::Object(object) => {
            for value in object.values_mut() {
                normalize(value);
            }
            let keys: Vec<String> = object
                .keys()
                .filter(|key| key.contains('\0'))
                .cloned()
                .collect();
            for key in keys {
                if let Some(value) = object.remove(&key) {
                    let base = key.replace('\0', "\u{fffd}");
                    let mut destination = base.clone();
                    let mut suffix = 1;
                    while object.contains_key(&destination) {
                        destination = format!("{base}~{suffix}");
                        suffix += 1;
                    }
                    object.insert(destination, value);
                }
            }
        }
        _ => {}
    }
}

#[cfg(test)]
#[path = "response_test.rs"]
mod tests;
