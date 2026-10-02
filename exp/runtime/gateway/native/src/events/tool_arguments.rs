//! Preserve valid argument bytes while isolating malformed provider tails.

use super::ToolAccumulator;

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
