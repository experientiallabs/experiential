//! Native response tap. Bytes are forwarded unchanged; storage never runs here.

use std::sync::Arc;

use axum::body::{Body, HttpBody};
use axum::response::Response;
use futures_util::StreamExt;
use serde_json::Value;

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
            // The allocator may grant more than requested; never leave that capacity uncharged.
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

    fn finish(&mut self, disconnected: bool) {
        if self.finished {
            return;
        }
        self.finished = true;
        let response = if self.sse {
            let mut frames = data_frames(&self.bytes);
            let mut truncated = self.truncated;
            loop {
                let response = CapturedResponse::Sse {
                    status: self.status,
                    frames,
                    truncated,
                    client_disconnected: disconnected,
                };
                let size = serde_json::to_vec(&response).map_or(usize::MAX, |value| value.len());
                if size <= self.collector.config.maximum_response_bytes {
                    break Some(response);
                }
                let CapturedResponse::Sse {
                    frames: mut reduced,
                    ..
                } = response
                else {
                    unreachable!()
                };
                if reduced.is_empty() {
                    break None;
                }
                reduced.truncate(reduced.len() / 2);
                frames = reduced;
                truncated = true;
            }
        } else if !self.truncated && !disconnected {
            serde_json::from_slice::<Value>(&self.bytes)
                .ok()
                .map(|mut body| {
                    normalize(&mut body);
                    CapturedResponse::Json {
                        status: self.status,
                        body,
                    }
                })
        } else {
            None
        };
        self.collector
            .finish(&self.request_id, response, self.deployment_id.take());
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
    if !collector.attach(request_id) {
        return response;
    }
    if !response.status().is_success() {
        collector.finish(request_id, None, None);
        return response;
    }
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
    };
    let stream = futures_util::stream::unfold(
        (body.into_data_stream(), tap, 0u64),
        move |(mut body, mut tap, mut sent)| async move {
            match body.next().await {
                Some(item) => {
                    match &item {
                        Ok(chunk) => {
                            tap.push(chunk);
                            sent = sent.saturating_add(chunk.len() as u64);
                            if expected == Some(sent) {
                                tap.finish(false);
                            }
                        }
                        Err(_) => tap.finish(true),
                    }
                    Some((item, (body, tap, sent)))
                }
                None => {
                    tap.finish(false);
                    None
                }
            }
        },
    );
    Response::from_parts(parts, Body::from_stream(stream))
}

/// Only whole SSE events enter the record, including non-JSON data such as [DONE].
fn data_frames(bytes: &[u8]) -> Vec<Value> {
    let mut frames = Vec::new();
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
                let mut value = serde_json::from_slice(&payload).unwrap_or_else(|_| {
                    Value::String(String::from_utf8_lossy(&payload).into_owned())
                });
                normalize(&mut value);
                frames.push(value);
                data.clear();
            }
        } else if let Some(payload) = line.strip_prefix(b"data:") {
            data.push(payload.strip_prefix(b" ").unwrap_or(payload));
        }
    }
    frames
}

/// Storage normalization never touches forwarded bytes. Preserve colliding object keys.
fn normalize(value: &mut Value) {
    match value {
        Value::String(text) => {
            if text.contains('\0') {
                *text = text.replace('\0', "\u{fffd}");
            }
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
