//! Compact JSON for model-readable tool results and reservation material.

use serde_json::Value;

use crate::errors::PublicError;

pub(super) fn dumps(value: &Value, ascii: bool) -> Result<String, PublicError> {
    encode(value, ascii, false)
}

pub(super) fn spaced(value: &Value) -> Result<String, PublicError> {
    encode(value, true, true)
}

fn encode(value: &Value, ascii: bool, spaced: bool) -> Result<String, PublicError> {
    let mut out = String::new();
    write(value, &mut out, spaced)?;
    if !ascii {
        return Ok(out);
    }
    let mut escaped = String::with_capacity(out.len());
    for c in out.chars() {
        if c.is_ascii() {
            escaped.push(c);
        } else {
            for unit in c.encode_utf16(&mut [0; 2]) {
                escaped.push_str(&format!("\\u{unit:04x}"));
            }
        }
    }
    Ok(escaped)
}

fn write(value: &Value, out: &mut String, spaced: bool) -> Result<(), PublicError> {
    match value {
        Value::Array(items) => {
            out.push('[');
            for (i, item) in items.iter().enumerate() {
                if i > 0 {
                    out.push(',');
                    if spaced {
                        out.push(' ');
                    }
                }
                write(item, out, spaced)?;
            }
            out.push(']');
        }
        Value::Object(items) => {
            out.push('{');
            for (i, (key, value)) in items.iter().enumerate() {
                if i > 0 {
                    out.push(',');
                    if spaced {
                        out.push(' ');
                    }
                }
                out.push_str(&serde_json::to_string(key).map_err(|_| PublicError::internal())?);
                out.push(':');
                if spaced {
                    out.push(' ');
                }
                write(value, out, spaced)?;
            }
            out.push('}');
        }
        Value::Number(number) => out.push_str(&super::canonical::number(number, false)?),
        _ => out.push_str(&serde_json::to_string(value).map_err(|_| PublicError::internal())?),
    }
    Ok(())
}
