//! Host-ledger billing extensions, applied before final publication and replay.

use serde::Deserialize;
use serde_json::{json, Value};

use crate::bridge::Bridge;
use crate::encode::compact_json;

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SettledBilling {
    paid_nano_usd: u64,
    byok_nano_usd: u64,
    is_byok: bool,
}

impl SettledBilling {
    /// Missing or failed annotation never substitutes an invented zero cost.
    pub async fn read(bridge: &Bridge, request_id: &str) -> Option<Self> {
        let answer = bridge
            .call(
                "settled_billing",
                compact_json(&json!({"request_id": request_id})),
            )
            .await
            .ok()?;
        serde_json::from_str::<Option<Self>>(&answer).ok().flatten()
    }

    fn dollars(value: u64) -> Value {
        // Parse an exact decimal spelling; never use floating arithmetic to
        // compute the charged amount before JSON rendering.
        serde_json::from_str(&format!(
            "{}.{:09}",
            value / 1_000_000_000,
            value % 1_000_000_000
        ))
        .expect("bounded nano-dollar decimal")
    }

    /// Enrich only terminal usage objects, preserving protocol-native fields.
    pub fn annotate(&self, value: &mut Value) {
        let holder = if value.get("response").is_some() {
            &mut value["response"]
        } else {
            value
        };
        let Some(usage) = holder.get_mut("usage").and_then(Value::as_object_mut) else {
            return;
        };
        usage.insert("cost".into(), Self::dollars(self.paid_nano_usd));
        usage.insert("is_byok".into(), Value::Bool(self.is_byok));
        if self.is_byok {
            usage.insert(
                "cost_details".into(),
                json!({"upstream_inference_cost": Self::dollars(self.byok_nano_usd)}),
            );
        }
    }

    /// A terminal frame may contain multiple SSE events. Event names and [DONE]
    /// stay untouched; only a data object's existing usage gains the extension.
    pub fn annotate_sse(&self, frame: &str) -> String {
        frame
            .split_inclusive('\n')
            .map(|line| {
                let Some(data) = line.strip_prefix("data: ") else {
                    return line.to_owned();
                };
                let Ok(mut payload) = serde_json::from_str::<Value>(data.trim_end()) else {
                    return line.to_owned();
                };
                if payload["type"] == "message_start" {
                    return line.to_owned();
                }
                self.annotate(&mut payload);
                format!(
                    "data: {}{}",
                    compact_json(&payload),
                    if line.ends_with('\n') { "\n" } else { "" }
                )
            })
            .collect()
    }
}

/// Annotate a completed non-streaming body after settlement.
pub async fn body(bridge: &Bridge, request_id: &str, value: &mut Value) {
    if let Some(billing) = SettledBilling::read(bridge, request_id).await {
        billing.annotate(value);
    }
}

/// Annotate a buffered SSE result before a keyed owner stores it.
pub async fn sse(bridge: &Bridge, request_id: &str, bytes: Vec<u8>) -> Vec<u8> {
    match (
        SettledBilling::read(bridge, request_id).await,
        std::str::from_utf8(&bytes),
    ) {
        (Some(billing), Ok(text)) => billing.annotate_sse(text).into_bytes(),
        _ => bytes,
    }
}

/// Annotate terminal frames only, leaving already-emitted content untouched.
pub async fn frames(bridge: &Bridge, request_id: &str, frames: Vec<String>) -> Vec<String> {
    match SettledBilling::read(bridge, request_id).await {
        Some(billing) => frames
            .iter()
            .map(|frame| billing.annotate_sse(frame))
            .collect(),
        None => frames,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn annotates_each_protocol_without_changing_token_shapes() {
        let billing = SettledBilling {
            paid_nano_usd: 1_234_567,
            byok_nano_usd: 0,
            is_byok: false,
        };
        for mut value in [
            json!({"id":"chatcmpl_1","usage":{"prompt_tokens":7}}),
            json!({"type":"response.completed","response":{"usage":{"input_tokens":7}}}),
            json!({"type":"message_delta","usage":{"output_tokens":7}}),
        ] {
            billing.annotate(&mut value);
            let holder = value.get("response").unwrap_or(&value);
            assert_eq!(holder["usage"]["cost"], json!(0.001234567));
            assert_eq!(holder["usage"]["is_byok"], false);
            assert!(holder["usage"].get("cost_details").is_none());
        }
    }

    #[test]
    fn sse_keeps_event_envelope_and_does_not_annotate_start_estimate() {
        let billing = SettledBilling {
            paid_nano_usd: 0,
            byok_nano_usd: 9,
            is_byok: true,
        };
        let frame = "event: message_delta\ndata: {\"type\":\"message_delta\",\"usage\":{\"output_tokens\":2}}\n\ndata: [DONE]\n\n";
        let result = billing.annotate_sse(frame);
        assert!(result.starts_with("event: message_delta\n"));
        assert!(result.contains("upstream_inference_cost"));
        assert!(result.ends_with("data: [DONE]\n\n"));
        let start =
            "data: {\"type\":\"message_start\",\"message\":{\"usage\":{\"input_tokens\":2}}}\n\n";
        assert_eq!(billing.annotate_sse(start), start);
    }
}
