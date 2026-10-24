//! OpenAI Responses probability observations.

use serde_json::{Map, Value};
use std::io::{self, Write};

use super::openai::{openai_identity, openai_index};
use super::{malformed, optional_text, Normalizer};
use crate::errors::{Failure, FailureClass};
use crate::events::{Event, ProviderOutputItemKind};

const MAX_TOKEN_CHARS: usize = 256;
const MAX_BYTES: usize = 4096;
const MAX_RECORDS_BYTES: usize = 1_048_576;

/// Check the bounded JSON shape accepted for one provider probability phase.
pub(crate) fn records_are_bounded(records: &Value, optional_alternatives: bool) -> bool {
    let Some(records) = records.as_array() else {
        return false;
    };
    records_retained_bytes_slice(records).is_some()
        && records
            .iter()
            .all(|record| valid_record(record, optional_alternatives))
}

/// Count one value's JSON bytes without allocating a serialized copy or
/// applying the probability-record bound; callers charge the count to their
/// own retention budget.
pub(crate) fn json_retained_bytes(value: &Value) -> usize {
    let mut counter = ByteCounter { size: 0 };
    // Serializing a `Value` into a counter cannot fail.
    let _ = serde_json::to_writer(&mut counter, value);
    counter.size
}

/// Count JSON bytes without allocating a serialized copy.
pub(crate) fn records_retained_bytes(records: &Value) -> Option<usize> {
    let mut counter = ByteCounter { size: 0 };
    serde_json::to_writer(&mut counter, records).ok()?;
    (counter.size <= MAX_RECORDS_BYTES).then_some(counter.size)
}

fn records_retained_bytes_slice(records: &[Value]) -> Option<usize> {
    let mut counter = ByteCounter { size: 0 };
    serde_json::to_writer(&mut counter, records).ok()?;
    (counter.size <= MAX_RECORDS_BYTES).then_some(counter.size)
}

struct ByteCounter {
    size: usize,
}

impl Write for ByteCounter {
    fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
        self.size = self.size.saturating_add(bytes.len());
        Ok(bytes.len())
    }

    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

fn valid_record(record: &Value, optional_alternatives: bool) -> bool {
    let Some(record) = record.as_object() else {
        return false;
    };
    let token_ok = record.get("token").is_some_and(|token| {
        token
            .as_str()
            .is_some_and(|token| token.chars().count() <= MAX_TOKEN_CHARS)
    });
    let logprob_ok = record
        .get("logprob")
        .is_some_and(|logprob| logprob.as_f64().is_some_and(f64::is_finite));
    let bytes_ok = record.get("bytes").is_none_or(|bytes| {
        bytes.as_array().is_some_and(|bytes| {
            bytes.len() <= MAX_BYTES
                && bytes
                    .iter()
                    .all(|byte| byte.as_u64().is_some_and(|byte| byte <= u8::MAX as u64))
        })
    });
    let alternatives_ok = record.get("top_logprobs").is_none_or(|alternatives| {
        (optional_alternatives && alternatives.is_null())
            || alternatives.as_array().is_some_and(|alternatives| {
                alternatives.len() <= 20
                    && alternatives
                        .iter()
                        .all(|value| valid_alternative(value, optional_alternatives))
            })
    });
    token_ok && logprob_ok && bytes_ok && alternatives_ok
}

fn valid_alternative(value: &Value, optional: bool) -> bool {
    let Some(value) = value.as_object() else {
        return false;
    };
    value.get("token").map_or(optional, |token| {
        if optional && token.is_null() {
            return true;
        }
        token
            .as_str()
            .is_some_and(|token| token.chars().count() <= MAX_TOKEN_CHARS)
    }) && value.get("logprob").map_or(optional, |logprob| {
        (optional && logprob.is_null()) || logprob.as_f64().is_some_and(f64::is_finite)
    }) && value.get("bytes").is_none_or(|bytes| {
        bytes.as_array().is_some_and(|bytes| {
            bytes.len() <= MAX_BYTES
                && bytes
                    .iter()
                    .all(|byte| byte.as_u64().is_some_and(|byte| byte <= u8::MAX as u64))
        })
    }) && !value.contains_key("top_logprobs")
}

fn validate(records: &Value) -> Result<(), Failure> {
    (records.is_null() || records_are_bounded(records, false))
        .then_some(())
        .ok_or_else(|| {
            Failure::new(
                FailureClass::MalformedResponse,
                "Responses probability records exceeded the supported shape",
            )
        })
}

/// Map unusable provider logprob values onto the contract's negligible value
/// before the shape check, so a probability value never fails the answer
/// (see `crate::logprobs::logprob_value`); structure is still validated.
fn normalized(records: &Value, keep_null_alternatives: bool) -> Value {
    let mut records = records.clone();
    crate::logprobs::normalize_records_value(&mut records, keep_null_alternatives, "responses");
    records
}

/// Extract probability records from a Responses output text event.
pub(crate) fn payload_records(payload: &Map<String, Value>) -> Option<Value> {
    payload
        .get("logprobs")
        .or_else(|| {
            payload
                .get("part")
                .and_then(Value::as_object)?
                .get("logprobs")
        })
        .cloned()
}

/// Convert array positions without allowing distinct provider items to alias.
fn bounded_position(position: usize, label: &str) -> Result<u32, Failure> {
    u32::try_from(position).map_err(|_| malformed(&format!("{label} exceeds the supported range")))
}

/// Extract item completion probability observations, retaining each content
/// part identity and the provider's exact record values.
pub(crate) fn item_done_events(
    output_index: u32,
    item: &Map<String, Value>,
) -> Result<Vec<Event>, Failure> {
    let Some(item_id) = item.get("id").and_then(Value::as_str) else {
        return Ok(Vec::new());
    };
    let Some(content) = item.get("content").and_then(Value::as_array) else {
        return Ok(Vec::new());
    };
    let mut events = Vec::new();
    for (content_index, part) in content.iter().enumerate() {
        let Some(records) = part.as_object().and_then(|part| part.get("logprobs")) else {
            continue;
        };
        let records = normalized(records, false);
        validate(&records)?;
        events.push(Event::ProviderResponsesLogprobs {
            output_index,
            item_id: item_id.to_string(),
            content_index: bounded_position(content_index, "OpenAI content_index")?,
            phase: "item_done".to_string(),
            records,
        });
    }
    Ok(events)
}

/// Extract terminal probability observations from the final response object.
pub(crate) fn terminal_events(response: &Map<String, Value>) -> Result<Vec<Event>, Failure> {
    let mut events = Vec::new();
    let Some(output) = response.get("output").and_then(Value::as_array) else {
        return Ok(events);
    };
    for (output_position, item) in output.iter().enumerate() {
        let Some(item_object) = item.as_object() else {
            continue;
        };
        if item_object.get("type").and_then(Value::as_str) != Some("message") {
            continue;
        }
        let Some(item_id) = item_object.get("id").and_then(Value::as_str) else {
            continue;
        };
        let output_index = if item_object.contains_key("output_index") {
            openai_index(item_object, "output_index", "OpenAI output_index")?
        } else {
            bounded_position(output_position, "OpenAI output_index")?
        };
        let Some(content) = item_object.get("content").and_then(Value::as_array) else {
            continue;
        };
        for (content_index, part) in content.iter().enumerate() {
            let Some(records) = part.as_object().and_then(|part| part.get("logprobs")) else {
                continue;
            };
            let records = normalized(records, false);
            validate(&records)?;
            events.push(Event::ProviderResponsesLogprobs {
                output_index,
                item_id: item_id.to_string(),
                content_index: bounded_position(content_index, "OpenAI content_index")?,
                phase: "terminal".to_string(),
                records,
            });
        }
    }
    Ok(events)
}

impl Normalizer {
    pub(super) fn responses_probability_text_delta(
        &mut self,
        payload: &Map<String, Value>,
    ) -> Result<Vec<Event>, Failure> {
        let mut events = Vec::new();
        let output_index = openai_index(payload, "output_index", "OpenAI output_index")?;
        let item_id = openai_identity(payload, "item_id", "OpenAI message item ID")?;
        let delta = optional_text(payload, "delta", "OpenAI text delta")?;
        let records = self
            .responses_logprobs
            .then(|| payload_records(payload))
            .flatten();
        // Empty observations stay private and cannot manufacture a billed completion.
        let populated = records
            .as_ref()
            .and_then(Value::as_array)
            .is_some_and(|v| !v.is_empty());
        if (records.is_none() || !delta.is_empty() || populated)
            && self.bind_openai_output_item(
                output_index,
                ProviderOutputItemKind::Message,
                Some(item_id.clone()),
            )?
            && (records.is_none() || !delta.is_empty())
        {
            events.push(Event::ProviderOutputItemStarted {
                output_index,
                item_id: Some(item_id.clone()),
                kind: ProviderOutputItemKind::Message,
                status: None,
                phase: None,
            });
        }
        if let Some(records) = self
            .responses_logprobs
            .then(|| payload_records(payload))
            .flatten()
        {
            let records = normalized(&records, true);
            if !records_are_bounded(&records, true) {
                return Err(malformed(
                    "OpenAI Responses probability records are invalid",
                ));
            }
            if !records.is_null() {
                events.push(Event::ProviderResponsesLogprobs {
                    output_index,
                    item_id: item_id.clone(),
                    content_index: payload
                        .get("content_index")
                        .map(|_| openai_index(payload, "content_index", "OpenAI content_index"))
                        .transpose()?
                        .unwrap_or(0),
                    phase: "delta".to_string(),
                    records: records.clone(),
                });
            }
        }
        if !delta.is_empty() {
            events.push(Event::ProviderTextDelta {
                output_index,
                item_id,
                delta,
            });
        }
        Ok(events)
    }

    pub(super) fn responses_probability_part_done(
        &mut self,
        payload: &Map<String, Value>,
        event_type: &str,
    ) -> Result<Vec<Event>, Failure> {
        let mut events = Vec::new();
        if let Some(records) = self
            .responses_logprobs
            .then(|| payload_records(payload))
            .flatten()
        {
            let records = normalized(&records, event_type == "response.output_text.done");
            if !(records_are_bounded(&records, event_type == "response.output_text.done")
                || (records.is_null() && event_type == "response.content_part.done"))
            {
                return Err(malformed(
                    "OpenAI Responses probability records are invalid",
                ));
            }
            let output_index = openai_index(payload, "output_index", "OpenAI output_index")?;
            let item_id = openai_identity(payload, "item_id", "OpenAI message item ID")?;
            if records.as_array().is_some_and(|items| !items.is_empty()) {
                self.bind_openai_output_item(
                    output_index,
                    ProviderOutputItemKind::Message,
                    Some(item_id.clone()),
                )?;
            }
            let content_index = payload
                .get("content_index")
                .map(|_| openai_index(payload, "content_index", "OpenAI content_index"))
                .transpose()?
                .unwrap_or(0);
            events.push(Event::ProviderResponsesLogprobs {
                output_index,
                item_id,
                content_index,
                phase: if event_type.ends_with(".done") && event_type.contains("text") {
                    "text_done"
                } else {
                    "content_part_done"
                }
                .to_string(),
                records: records.clone(),
            });
        }
        Ok(events)
    }
}
