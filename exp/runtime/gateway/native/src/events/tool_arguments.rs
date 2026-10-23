//! Preserve valid argument bytes while isolating malformed provider tails.

use serde_json::Value;

use super::ToolAccumulator;

/// Validate one raw tool-argument accumulation as a single JSON object.
///
/// The parse-failure reason carries serde's positional description (token
/// category and line/column, never input bytes), so an unparsable shape is
/// diagnosable from the boundary log without ever logging payload.
pub fn require_json_object_text(raw: &str) -> Result<(), String> {
    match serde_json::from_str::<Value>(raw) {
        Ok(Value::Object(_)) => Ok(()),
        Ok(_) => Err("streamed tool arguments must decode to an object".to_string()),
        Err(error) => Err(format!(
            "streamed tool arguments are not valid JSON: {error}"
        )),
    }
}

impl ToolAccumulator {
    /// Preserve valid JSON whitespace after the object too. Withhold only
    /// non-whitespace tails for repair, keeping emitted and completed bytes equal.
    /// Custom input remains opaque and passes through whole.
    pub fn push_arguments(&mut self, fragment: &str) -> Option<String> {
        if self.custom {
            self.raw_arguments.push_str(fragment);
            return Some(fragment.to_string());
        }
        match self.scan.feed(fragment) {
            None => {
                self.raw_arguments.push_str(fragment);
                Some(fragment.to_string())
            }
            Some(closed_at) => {
                let closed_at = if self.withheld_tail.is_empty() {
                    closed_at
                        + fragment[closed_at..]
                            .bytes()
                            .take_while(|byte| matches!(byte, b' ' | b'\t' | b'\r' | b'\n'))
                            .count()
                } else {
                    closed_at
                };
                let (value, tail) = fragment.split_at(closed_at);
                self.raw_arguments.push_str(value);
                self.withheld_tail.push_str(tail);
                (!value.is_empty()).then(|| value.to_string())
            }
        }
    }
}

/// Why a tail streamed after a complete argument object was dropped rather
/// than failing the call. Each shape reproduces exactly one parse, so no
/// argument content is ever invented or chosen between alternatives.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RedundantTail {
    /// Only whitespace (a pretty-printed buffer's trailing newline).
    Whitespace,
    /// One or more exact repetitions of the complete value (`{}{}`, a
    /// duplicated whole-object delta).
    DuplicateValue,
    /// Empty JSON literals after a zero-argument call (`{}""`: Azure
    /// Foundry's DeepSeek shim, live 2026-09-10, 222 attempts in one day).
    EmptyLiterals,
}

impl RedundantTail {
    pub(super) fn as_str(self) -> &'static str {
        match self {
            RedundantTail::Whitespace => "whitespace",
            RedundantTail::DuplicateValue => "duplicate_value",
            RedundantTail::EmptyLiterals => "empty_literals",
        }
    }
}

/// Classify the bytes a provider streamed AFTER its argument object closed.
///
/// Accepts only tails whose removal is unambiguous: whitespace; exact
/// repetitions of the whole value; and, after a zero-argument `{}` only,
/// empty literals (`""`, `{}`, `[]`) that carry no argument content. A tail
/// that is merely a suffix of the value (`{"a":1}}`) is NOT accepted: the same
/// bytes arise from a dropped inner delta (`{"a":1,"b":{` lost from
/// `{"a":1,"b":{"c":2}}`), so the parse would be a guess.
pub fn redundant_tail(value: &str, tail: &str) -> Option<RedundantTail> {
    let value = value.trim();
    let tail = tail.trim();
    if tail.is_empty() {
        return Some(RedundantTail::Whitespace);
    }
    let mut rest = tail;
    let mut duplicates = true;
    while !rest.is_empty() {
        match rest.strip_prefix(value) {
            Some(after) if !value.is_empty() => rest = after.trim_start(),
            _ => {
                duplicates = false;
                break;
            }
        }
    }
    if duplicates {
        return Some(RedundantTail::DuplicateValue);
    }
    if value != "{}" {
        return None;
    }
    let mut rest = tail;
    while !rest.is_empty() {
        rest = rest
            .strip_prefix("\"\"")
            .or_else(|| rest.strip_prefix("{}"))
            .or_else(|| rest.strip_prefix("[]"))?
            .trim_start();
    }
    Some(RedundantTail::EmptyLiterals)
}
