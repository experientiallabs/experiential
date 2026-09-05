//! Final public envelope and SSE framing for the Responses encoder.

use serde_json::{json, Value};

use super::{aggregate, OutputSlot, ResponsesSseEncoder};
use crate::encode::compact_json;
use crate::errors::Failure;
use crate::events::ProviderOutputItemStatus;

impl ResponsesSseEncoder {
    /// Build one SDK-readable Responses envelope for the current lifecycle state.
    pub(super) fn response(&self, status: &str, failure: Option<&Failure>) -> Value {
        let include_content = status != "in_progress";
        let fallback_status = match status {
            "in_progress" => ProviderOutputItemStatus::InProgress,
            "completed" => ProviderOutputItemStatus::Completed,
            _ => ProviderOutputItemStatus::Incomplete,
        };
        let output: Vec<Value> = self
            .output_order
            .iter()
            .map(|slot| match slot {
                OutputSlot::Message(key) => {
                    self.messages[key].item(include_content, fallback_status)
                }
                OutputSlot::Tool(index) => self.tools[index].item(fallback_status),
                // Hosted tool items re-serve the provider's verbatim JSON.
                OutputSlot::HostedTool(index) => self.hosted[index].item.clone(),
                OutputSlot::Reasoning(index) => self.reasoning[index].item(
                    include_content,
                    fallback_status,
                    self.envelope.include_encrypted_reasoning,
                ),
                OutputSlot::FireworksReasoning => self
                    .fireworks_reasoning
                    .as_ref()
                    .expect("Fireworks output slot has state")
                    .item(
                        include_content,
                        fallback_status,
                        self.envelope.include_encrypted_reasoning,
                    ),
            })
            .collect();
        let error = if status == "failed" {
            json!({
                "code": "server_error",
                "message": failure
                    .map(|failure| failure.safe_message.clone())
                    .unwrap_or_else(|| "Gateway stream failed.".to_string()),
            })
        } else {
            Value::Null
        };
        let mut response = json!({
            "id": self.response_id,
            "object": "response",
            "created_at": self.created_at,
            "completed_at": if status == "completed" { json!(self.created_at) } else { Value::Null },
            "status": status,
            "error": error,
            "incomplete_details": if status == "incomplete" {
                json!({"reason": "max_output_tokens"})
            } else {
                Value::Null
            },
            "instructions": Value::Null,
            "metadata": self.envelope.metadata,
            "model": self.model,
            "output": output,
            "parallel_tool_calls": self.envelope.parallel_tool_calls,
            "temperature": self.envelope.temperature,
            "top_p": self.envelope.top_p,
            "reasoning": self.envelope.reasoning,
            "tool_choice": self.envelope.tool_choice,
            "tools": self.envelope.tools,
            "max_output_tokens": self.envelope.max_output_tokens,
            "previous_response_id": self.envelope.previous_response_id,
            "usage": if include_content {
                aggregate::responses_usage(self.usage.as_ref())
            } else {
                Value::Null
            },
        });
        if !self.envelope.ignored_parameters.is_empty() {
            response
                .as_object_mut()
                .expect("response envelope is an object")
                .insert(
                    "x-experiential-ignored-parameters".to_string(),
                    json!(self.envelope.ignored_parameters),
                );
        }
        response
    }

    /// Assign one monotonic sequence number and frame a named SSE event.
    pub(super) fn event(&mut self, event_type: &str, fields: Value) -> String {
        let mut payload = serde_json::Map::new();
        payload.insert("type".to_string(), Value::String(event_type.to_string()));
        payload.insert("sequence_number".to_string(), json!(self.sequence));
        if let Value::Object(entries) = fields {
            for (key, value) in entries {
                payload.insert(key, value);
            }
        }
        self.sequence += 1;
        let encoded = compact_json(&Value::Object(payload));
        format!("event: {event_type}\ndata: {encoded}\n\n")
    }
}
