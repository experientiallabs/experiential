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
