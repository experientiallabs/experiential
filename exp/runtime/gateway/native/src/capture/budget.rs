//! Allocation-free JSON sizing and bounded final encoding for capture destinations.

use std::io::{self, Write};

use serde::Serialize;
use serde_json::Value;

/// UTF-8 compact JSON length, without constructing escaped strings.
pub(super) fn string_bytes(text: &str) -> usize {
    // Count escape overhead separately from the existing UTF-8 bytes. These
    // reductions avoid a dependent saturating addition for every input byte.
    let escaped = text
        .bytes()
        .filter(|&byte| byte < 32 || byte == b'"' || byte == b'\\')
        .count();
    if escaped == 0 {
        return text.len().saturating_add(2);
    }
    let long = text
        .bytes()
        .filter(|&byte| byte < 32 && !matches!(byte, b'\n' | b'\r' | b'\t' | 8 | 12))
        .count();
    text.len()
        .saturating_add(2)
        .saturating_add(escaped)
        .saturating_add(long.saturating_mul(4))
}

pub(super) fn optional_string_bytes(text: Option<&str>) -> usize {
    text.map_or(4, string_bytes)
}

/// Count an already parsed JSON tree; only scalar numbers require formatting.
pub(super) fn json_bytes(value: &Value) -> usize {
    match value {
        Value::Null => 4,
        Value::Bool(value) => {
            if *value {
                4
            } else {
                5
            }
        }
        Value::Number(value) => value.to_string().len(),
        Value::String(value) => string_bytes(value),
        Value::Array(values) => values.iter().fold(
            2usize.saturating_add(values.len().saturating_sub(1)),
            |size, value| size.saturating_add(json_bytes(value)),
        ),
        Value::Object(values) => values.iter().fold(
            2usize.saturating_add(values.len().saturating_sub(1)),
            |size, (key, value)| {
                size.saturating_add(string_bytes(key))
                    .saturating_add(1)
                    .saturating_add(json_bytes(value))
            },
        ),
    }
}

/// Charge retained tree nodes as well as string and vector capacity.
/// Objects enter through serde, without caller-controlled spare map capacity.
pub(super) fn heap_bytes(value: &Value) -> usize {
    // serde-grown ordered maps can reserve three entry slots for the first key.
    // Charge four slots plus index/control storage per key, using actual node sizes.
    const MAP_ENTRY_BYTES: usize = 4
        * (std::mem::size_of::<Value>()
            + std::mem::size_of::<String>()
            + 2 * std::mem::size_of::<usize>());
    match value {
        Value::String(text) => text.capacity(),
        Value::Array(values) => values.iter().fold(
            values
                .capacity()
                .saturating_mul(std::mem::size_of::<Value>()),
            |size, value| size.saturating_add(heap_bytes(value)),
        ),
        Value::Object(values) => values.iter().fold(
            values.len().saturating_mul(MAP_ENTRY_BYTES),
            |size, (key, value)| {
                size.saturating_add(key.capacity())
                    .saturating_add(heap_bytes(value))
            },
        ),
        _ => 0,
    }
}

/// Stop the final serializer before an oversized payload can allocate without bound.
pub(super) fn encode(value: &impl Serialize, maximum: usize) -> Option<String> {
    struct Limited {
        bytes: Vec<u8>,
        maximum: usize,
    }
    impl Write for Limited {
        fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
            if bytes.len() > self.maximum.saturating_sub(self.bytes.len()) {
                return Err(io::Error::other("capture record exceeds storage budget"));
            }
            let required = self.bytes.len() + bytes.len();
            if required > self.bytes.capacity() {
                // Geometric growth must not retain spare capacity beyond the
                // destination's UTF-8 preparation reservation.
                let capacity = self
                    .bytes
                    .capacity()
                    .saturating_mul(2)
                    .max(required)
                    .min(self.maximum);
                self.bytes.reserve_exact(capacity - self.bytes.len());
            }
            self.bytes.extend_from_slice(bytes);
            Ok(bytes.len())
        }
        fn flush(&mut self) -> io::Result<()> {
            Ok(())
        }
    }
    let mut output = Limited {
        bytes: Vec::new(),
        maximum,
    };
    serde_json::to_writer(&mut output, value).ok()?;
    String::from_utf8(output.bytes).ok()
}

#[cfg(test)]
#[path = "budget_test.rs"]
mod tests;
