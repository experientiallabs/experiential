//! Caller-facing metadata supplied by an outer HTTP relay, decoded off-path.

use serde::Deserialize;
use serde_json::{json, Value};

use super::budget;

/// Marks a body dropped for memory until decode renders its kind.
const DROPPED: &str = "body_dropped_for_memory";

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct Metadata {
    pub wire_request: Option<Value>,
    pub headers: Vec<[String; 2]>,
    pub timing: Value,
    pub relay_completed: bool,
    pub client_disconnected: bool,
}

pub(super) struct Relay {
    pub metadata: Metadata,
    pub body: Vec<u8>,
}

impl Relay {
    /// Keep the facts but not the body: capture memory had no room for it.
    pub(super) fn drop_body(&mut self) {
        self.body = Vec::new();
        if let Some(wire) = self
            .metadata
            .wire_request
            .as_mut()
            .and_then(Value::as_object_mut)
        {
            wire.insert("truncated".into(), Value::Bool(true));
            wire.insert(DROPPED.into(), Value::Bool(true));
        }
    }

    pub(super) fn heap_bytes(&self) -> usize {
        std::mem::size_of::<Self>()
            + self.body.capacity()
            + self
                .metadata
                .wire_request
                .as_ref()
                .map_or(0, budget::heap_bytes)
            + budget::heap_bytes(&self.metadata.timing)
            + self
                .metadata
                .headers
                .iter()
                .map(|pair| {
                    std::mem::size_of::<[String; 2]>() + pair[0].capacity() + pair[1].capacity()
                })
                .sum::<usize>()
    }

    /// Decode for a split-capable destination: a body above `keep_raw_above`
    /// bytes is returned raw instead of parsed in memory, for the split builder.
    pub(super) fn decode_split(mut self, keep_raw_above: usize) -> (Value, Option<Vec<u8>>) {
        if self.body.len() <= keep_raw_above {
            return (self.decode(), None);
        }
        let body = std::mem::take(&mut self.body);
        let mut value = self.decode();
        if let Some(wire) = value.get_mut("wire_request").and_then(Value::as_object_mut) {
            // The split builder sets the real kind and body.
            wire.insert("body_kind".into(), Value::String("json".into()));
            wire.insert("body".into(), Value::Null);
        }
        (value, Some(body))
    }

    pub(super) fn decode(self) -> Value {
        let mut wire = self.metadata.wire_request;
        if let Some(wire) = wire.as_mut().and_then(Value::as_object_mut) {
            let (kind, body) = if self.body.is_empty() {
                ("empty", Value::Null)
            } else if let Ok(body) = serde_json::from_slice::<Value>(&self.body) {
                if super::response::contains_wide_number(&body) {
                    wire.insert(
                        "body_source_json".into(),
                        Value::String(String::from_utf8(self.body).unwrap()),
                    );
                }
                ("json", body)
            } else {
                (
                    "text",
                    Value::String(String::from_utf8_lossy(&self.body).into_owned()),
                )
            };
            let dropped = wire.remove(DROPPED).is_some();
            let kind = if dropped { "dropped" } else { kind };
            wire.insert("body_kind".into(), Value::String(kind.into()));
            wire.insert("body".into(), body);
        }
        // The hosted wire column already used this bound. Keep facts when
        // the captured body cannot fit, without changing the served input.
        if wire
            .as_ref()
            .is_some_and(|wire| budget::json_bytes(wire) > 3_670_016)
        {
            if let Some(wire) = wire.as_mut().and_then(Value::as_object_mut) {
                wire.insert("body".into(), Value::Null);
                wire.remove("body_source_json");
                wire.insert("body_kind".into(), Value::String("dropped".into()));
                wire.insert("truncated".into(), Value::Bool(true));
            }
        }
        let mut value = json!({
            "wire_request": wire,
            "headers": self.metadata.headers,
            "timing": self.metadata.timing,
            "relay_completed": self.metadata.relay_completed,
            "client_disconnected": self.metadata.client_disconnected,
        });
        super::response::lossless_projection(&mut value);
        value
    }
}
