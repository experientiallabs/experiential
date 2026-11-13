//! Canonical JSON shared with the request's existing continuation format.

use serde_json::{Number, Value};

use crate::errors::PublicError;

/// Sort object keys and normalize semantically integral tool arguments. JSON
/// integers keep arbitrary precision; floating values use Python's established
/// exponent spelling so a carrier remains readable by older serving workers.
pub(super) fn bytes(value: &Value, normalize_numbers: bool) -> Result<Vec<u8>, PublicError> {
    let mut out = String::new();
    write(value, normalize_numbers, &mut out)?;
    Ok(out.into_bytes())
}

fn write(value: &Value, normalize: bool, out: &mut String) -> Result<(), PublicError> {
    match value {
        Value::Array(items) => {
            out.push('[');
            for (i, value) in items.iter().enumerate() {
                if i > 0 {
                    out.push(',');
                }
                write(value, normalize, out)?;
            }
            out.push(']');
        }
        Value::Object(map) => {
            out.push('{');
            let mut keys: Vec<_> = map.keys().collect();
            keys.sort();
            for (i, key) in keys.into_iter().enumerate() {
                if i > 0 {
                    out.push(',');
                }
                out.push_str(&serde_json::to_string(key).map_err(|_| PublicError::internal())?);
                out.push(':');
                write(&map[key], normalize, out)?;
            }
            out.push('}');
        }
        Value::Number(n) => out.push_str(&number(n, normalize)?),
        _ => out.push_str(&serde_json::to_string(value).map_err(|_| PublicError::internal())?),
    }
    Ok(())
}

pub(super) fn number(n: &Number, normalize: bool) -> Result<String, PublicError> {
    let raw = n.to_string();
    if !raw.contains(['.', 'e', 'E']) {
        return Ok(raw);
    }
    let value = n
        .as_f64()
        .filter(|v| v.is_finite())
        .ok_or_else(PublicError::internal)?;
    floating(value, normalize)
}

fn floating(value: f64, normalize: bool) -> Result<String, PublicError> {
    if !value.is_finite() {
        return Err(PublicError::internal());
    }
    if normalize && value.abs() < 9_007_199_254_740_992.0 && value.fract() == 0.0 {
        return Ok((value as i64).to_string());
    }
    // Rust Debug and CPython repr both use the shortest round-tripping decimal.
    // CPython spells signed exponents with at least two digits.
    let shortest = format!("{value:?}");
    if let Some((mantissa, exponent)) = shortest.split_once('e') {
        let exponent: i32 = exponent.parse().map_err(|_| PublicError::internal())?;
        return Ok(format!("{mantissa}e{exponent:+03}"));
    }
    Ok(shortest)
}

/// Parse tool arguments with the engine's lossless JSON number contract.
/// Raw values preserve arbitrary-width integers exactly; only JSON
/// floating literals take the established semantic normalization path.
pub(super) fn arguments_bytes(raw: &str) -> Result<Vec<u8>, PublicError> {
    use serde_json::value::RawValue;
    let value: Box<RawValue> =
        crate::lossless_json::from_str(raw).map_err(|_| PublicError::internal())?;
    if !value.get().trim_start().starts_with('{') {
        return Err(PublicError::internal());
    }
    let mut out = String::new();
    raw_write(&value, &mut out)?;
    Ok(out.into_bytes())
}

fn raw_write(value: &serde_json::value::RawValue, out: &mut String) -> Result<(), PublicError> {
    use serde_json::value::RawValue;
    let raw = value.get().trim();
    match raw.as_bytes().first() {
        Some(b'{') => {
            let map: std::collections::BTreeMap<String, Box<RawValue>> =
                crate::lossless_json::from_str(raw).map_err(|_| PublicError::internal())?;
            out.push('{');
            for (i, (key, value)) in map.iter().enumerate() {
                if i > 0 {
                    out.push(',');
                }
                out.push_str(&serde_json::to_string(key).map_err(|_| PublicError::internal())?);
                out.push(':');
                raw_write(value, out)?;
            }
            out.push('}');
        }
        Some(b'[') => {
            let values: Vec<Box<RawValue>> =
                crate::lossless_json::from_str(raw).map_err(|_| PublicError::internal())?;
            out.push('[');
            for (i, value) in values.iter().enumerate() {
                if i > 0 {
                    out.push(',');
                }
                raw_write(value, out)?;
            }
            out.push(']');
        }
        Some(b'"') => {
            let text: String =
                crate::lossless_json::from_str(raw).map_err(|_| PublicError::internal())?;
            out.push_str(&serde_json::to_string(&text).map_err(|_| PublicError::internal())?);
        }
        Some(b'-' | b'0'..=b'9') if raw.contains(['.', 'e', 'E']) => {
            out.push_str(&floating(
                raw.parse().map_err(|_| PublicError::internal())?,
                true,
            )?);
        }
        Some(b'-' | b'0'..=b'9') => out.push_str(if raw == "-0" { "0" } else { raw }),
        Some(b'n' | b't' | b'f') => out.push_str(raw),
        _ => return Err(PublicError::internal()),
    }
    Ok(())
}
