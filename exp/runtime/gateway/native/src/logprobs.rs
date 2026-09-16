//! Typed, bounded Chat Completions token probabilities.

use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::errors::{Failure, FailureClass};

pub const MAX_LOGPROB_RECORDS: usize = 4096;
pub const MAX_TOP_LOGPROBS: usize = 20;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LogprobCandidate {
    pub token: String,
    pub logprob: f64,
    pub bytes: Option<Vec<u8>>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TokenLogprob {
    pub token: String,
    pub logprob: f64,
    pub bytes: Option<Vec<u8>>,
    pub top_logprobs: Vec<LogprobCandidate>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ChoiceLogprobs {
    pub content: Option<Vec<TokenLogprob>>,
    pub refusal: Option<Vec<TokenLogprob>>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ChoiceLogprobsDelta {
    pub choice_index: u32,
    pub logprobs: Option<ChoiceLogprobs>,
}

impl ChoiceLogprobsDelta {
    pub fn retained_bytes(&self) -> usize {
        fn candidate_bytes(candidate: &LogprobCandidate) -> usize {
            64usize
                .saturating_add(candidate.token.len().saturating_mul(6))
                .saturating_add(
                    candidate
                        .bytes
                        .as_ref()
                        .map_or(0, |v| v.len().saturating_mul(4)),
                )
        }
        fn records_bytes(records: Option<&Vec<TokenLogprob>>) -> usize {
            records.map_or(0, |records| {
                records
                    .iter()
                    .map(|record| {
                        96usize
                            .saturating_add(record.token.len().saturating_mul(6))
                            .saturating_add(
                                record
                                    .bytes
                                    .as_ref()
                                    .map_or(0, |v| v.len().saturating_mul(4)),
                            )
                            .saturating_add(
                                record
                                    .top_logprobs
                                    .iter()
                                    .map(candidate_bytes)
                                    .sum::<usize>(),
                            )
                    })
                    .sum()
            })
        }
        64usize
            .saturating_add(records_bytes(
                self.logprobs.as_ref().and_then(|v| v.content.as_ref()),
            ))
            .saturating_add(records_bytes(
                self.logprobs.as_ref().and_then(|v| v.refusal.as_ref()),
            ))
    }
}

fn malformed(message: &str) -> Failure {
    Failure::new(FailureClass::MalformedResponse, message).with_retry(false, true)
}

fn bytes(value: Option<&Value>) -> Result<Option<Vec<u8>>, Failure> {
    match value {
        None | Some(Value::Null) => Ok(None),
        Some(Value::Array(values)) => values
            .iter()
            .map(|value| {
                let number = value
                    .as_u64()
                    .ok_or_else(|| malformed("logprob bytes must be integers"))?;
                u8::try_from(number).map_err(|_| malformed("logprob bytes must be in range 0..255"))
            })
            .collect::<Result<Vec<_>, _>>()
            .map(Some),
        Some(_) => Err(malformed("logprob bytes must be an array")),
    }
}

fn number(value: Option<&Value>) -> Result<f64, Failure> {
    let value = value
        .and_then(Value::as_f64)
        .ok_or_else(|| malformed("logprob must be a number"))?;
    if value.is_finite() {
        Ok(value)
    } else {
        Err(malformed("logprob must be finite"))
    }
}

fn candidate(value: &Value) -> Result<LogprobCandidate, Failure> {
    let object = value
        .as_object()
        .ok_or_else(|| malformed("logprob candidate must be an object"))?;
    Ok(LogprobCandidate {
        token: object
            .get("token")
            .and_then(Value::as_str)
            .ok_or_else(|| malformed("logprob token must be text"))?
            .to_string(),
        logprob: number(object.get("logprob"))?,
        bytes: bytes(object.get("bytes"))?,
    })
}

fn records(value: Option<&Value>) -> Result<Option<Vec<TokenLogprob>>, Failure> {
    let Some(value) = value else { return Ok(None) };
    if value.is_null() {
        return Ok(None);
    }
    let values = value
        .as_array()
        .ok_or_else(|| malformed("logprob channel must be an array"))?;
    if values.len() > MAX_LOGPROB_RECORDS {
        return Err(malformed("logprob channel is too large"));
    }
    values
        .iter()
        .map(|value| {
            let object = value
                .as_object()
                .ok_or_else(|| malformed("token logprob must be an object"))?;
            let alternatives = object
                .get("top_logprobs")
                .and_then(Value::as_array)
                .ok_or_else(|| malformed("token logprob top_logprobs must be an array"))?;
            if alternatives.len() > MAX_TOP_LOGPROBS {
                return Err(malformed("top_logprobs is too large"));
            }
            Ok(TokenLogprob {
                token: object
                    .get("token")
                    .and_then(Value::as_str)
                    .ok_or_else(|| malformed("logprob token must be text"))?
                    .to_string(),
                logprob: number(object.get("logprob"))?,
                bytes: bytes(object.get("bytes"))?,
                top_logprobs: alternatives
                    .iter()
                    .map(candidate)
                    .collect::<Result<_, _>>()?,
            })
        })
        .collect::<Result<Vec<_>, _>>()
        .map(Some)
}

/// Parse the OpenAI Chat choice-level `logprobs` object, preserving null and
/// empty channels while rejecting malformed populated records.
pub fn parse(
    value: Option<&Value>,
    choice_index: u32,
) -> Result<Option<ChoiceLogprobsDelta>, Failure> {
    let Some(value) = value else { return Ok(None) };
    if value.is_null() {
        return Ok(Some(ChoiceLogprobsDelta {
            choice_index,
            logprobs: None,
        }));
    }
    let object = value
        .as_object()
        .ok_or_else(|| malformed("choice logprobs must be an object"))?;
    let content = records(object.get("content"))?;
    let refusal = records(object.get("refusal"))?;
    if content.as_ref().map_or(0, Vec::len) + refusal.as_ref().map_or(0, Vec::len)
        > MAX_LOGPROB_RECORDS
    {
        return Err(malformed("choice logprobs has too many token records"));
    }
    Ok(Some(ChoiceLogprobsDelta {
        choice_index,
        logprobs: Some(ChoiceLogprobs { content, refusal }),
    }))
}

pub fn aggregate(events: &[crate::events::Event]) -> Option<ChoiceLogprobs> {
    let mut result = None;
    for event in events {
        let crate::events::Event::ChoiceLogprobsDelta(update) = event else {
            continue;
        };
        let Some(value) = &update.logprobs else {
            continue;
        };
        let current = result.get_or_insert(ChoiceLogprobs {
            content: None,
            refusal: None,
        });
        if let Some(records) = &value.content {
            current
                .content
                .get_or_insert_with(Vec::new)
                .extend(records.clone());
        }
        if let Some(records) = &value.refusal {
            current
                .refusal
                .get_or_insert_with(Vec::new)
                .extend(records.clone());
        }
    }
    result
}

/// Whether actual refusal text will be visible to the caller.
pub fn is_refusal_text(event: &crate::events::Event) -> bool {
    matches!(event, crate::events::Event::RefusalDelta(text) if !text.is_empty())
        || matches!(event, crate::events::Event::ProviderRefusalDelta { delta, .. } if !delta.is_empty())
}

/// Whether an event contains refusal evidence, including withheld probabilities.
pub fn is_refusal(event: &crate::events::Event) -> bool {
    is_refusal_text(event)
        || matches!(event, crate::events::Event::ChoiceLogprobsDelta(update)
        if update.logprobs.as_ref().and_then(|v| v.refusal.as_ref()).is_some_and(|v| !v.is_empty()))
}

/// Hold empty metadata without committing, and refusal-only output when retrying.
pub fn withhold_before_commit(event: &crate::events::Event, refusal_failover: bool) -> bool {
    if let crate::events::Event::ChoiceLogprobsDelta(update) = event {
        if update
            .logprobs
            .as_ref()
            .and_then(|v| v.content.as_ref())
            .is_some_and(|v| !v.is_empty())
        {
            return false;
        }
        return !is_refusal(event) || refusal_failover;
    }
    refusal_failover && is_refusal_text(event)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn preserves_zero_bytes_empty_arrays_and_partial_utf8() {
        let value = serde_json::json!({"content":[{"token":"é","logprob":-0.0,"bytes":[195,169,0],"top_logprobs":[]}]});
        let parsed = parse(Some(&value), 0).unwrap().unwrap();
        assert_eq!(
            parsed.logprobs.unwrap().content.unwrap()[0].bytes,
            Some(vec![195, 169, 0])
        );
    }
    #[test]
    fn distinguishes_null_and_empty_channels() {
        let value = serde_json::json!({"content":[],"refusal":null});
        let parsed = parse(Some(&value), 0).unwrap().unwrap().logprobs.unwrap();
        assert_eq!(parsed.content, Some(vec![]));
        assert_eq!(parsed.refusal, None);
    }
}
