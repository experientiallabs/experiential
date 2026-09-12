//! Host-ledger billing extensions, applied before final publication and replay.

use serde::Deserialize;
use serde_json::{json, Value};

use crate::bridge::Bridge;
use crate::encode::compact_json;

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SettledBilling {
    paid_nano_usd: u64,
    byok_nano_usd: u64,
    is_byok: bool,
}

impl SettledBilling {
    /// Missing or failed annotation never substitutes an invented zero cost.
    pub async fn read(
        bridge: &Bridge,
        request_id: &str,
        deadline: std::time::Instant,
    ) -> Option<Self> {
        // Preserve time to publish the paid success even if the ledger stalls.
        let budget = (deadline.saturating_duration_since(std::time::Instant::now()) / 2)
            .min(std::time::Duration::from_millis(100));
        if budget < std::time::Duration::from_millis(1) {
            return None;
        }
        let answer = bridge
            .call_optional(
                "settled_billing",
                compact_json(&json!({"request_id": request_id})),
                budget,
            )
            .await?;
        serde_json::from_str::<Option<Self>>(&answer).ok().flatten()
    }

    fn dollars(value: u64) -> Value {
        // Format ledger nanos before conversion to the wire's JSON number.
        // serde_json stores that number as f64; this is a display projection,
        // not an exact full-u64 accounting representation.
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
        let Some(usage) = holder.get_mut("usage") else {
            return;
        };
        if usage.is_null() {
            *usage = json!({});
        }
        let Some(usage) = usage.as_object_mut() else {
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
                let holder = payload.get("response").unwrap_or(&payload);
                if payload["type"] == "message_start" || holder.get("usage").is_none() {
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

/// Buffered encoders pass content through without another JSON parse or copy.
pub fn terminal_frame(
    billing: Option<&SettledBilling>,
    event: &crate::events::Event,
    frame: String,
) -> String {
    match billing.filter(|_| event.is_terminal()) {
        Some(billing) => billing.annotate_sse(&frame),
        None => frame,
    }
}

/// Annotate terminal frames only, leaving already-emitted content untouched.
pub async fn frames(
    bridge: &Bridge,
    request_id: &str,
    deadline: std::time::Instant,
    frames: Vec<String>,
) -> Vec<String> {
    match SettledBilling::read(bridge, request_id, deadline).await {
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
    fn dollar_numbers_are_display_values_not_full_range_nano_counters() {
        let nanos = 10_000_000_000_000_001;
        assert_eq!(
            format!("{}.{:09}", nanos / 1_000_000_000, nanos % 1_000_000_000),
            "10000000.000000001"
        );
        assert_eq!(compact_json(&SettledBilling::dollars(nanos)), "10000000.0");
    }

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
    fn buffered_content_passes_through_without_parsing_or_copying() {
        let billing = SettledBilling {
            paid_nano_usd: 1,
            byok_nano_usd: 0,
            is_byok: false,
        };
        let content = "data: { \"content\" : \"large content\" }\n\n".to_string();
        let allocation = content.as_ptr();
        let result = terminal_frame(
            Some(&billing),
            &crate::events::Event::TextDelta("x".into()),
            content,
        );
        assert_eq!(result.as_ptr(), allocation);
        let terminal = "event: response.completed\ndata: {\"type\":\"response.completed\",\"response\":{\"usage\":null}}\n\n".to_string();
        let result = terminal_frame(Some(&billing), &crate::events::Event::Completed, terminal);
        assert!(result.contains("\"cost\":1e-9"));
        assert!(!result.contains("input_tokens"));
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
