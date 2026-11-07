//! Validate hosted replacement messages against the admitted canonical contract.

use std::sync::OnceLock;

use serde_json::{json, Value};

struct Schema {
    root: Value,
    validator: jsonschema::Validator,
}

fn schema() -> &'static Schema {
    static SCHEMA: OnceLock<Schema> = OnceLock::new();
    SCHEMA.get_or_init(|| {
        let root = crate::lossless_json::from_str(include_str!(
            "../generated/gateway_message_schema.json"
        ))
        .expect("generated canonical message schema");
        // The schema contains only bundled local references. This crate is built
        // without file or HTTP resolution, including on malformed references.
        let validator = jsonschema::options()
            .build(&root)
            .expect("canonical message validator");
        Schema { root, validator }
    })
}

pub(super) fn boolean(value: &Value) -> Result<bool, ()> {
    match value {
        Value::Bool(value) => Ok(*value),
        Value::Number(value) if value.as_f64() == Some(0.0) => Ok(false),
        Value::Number(value) if value.as_f64() == Some(1.0) => Ok(true),
        Value::String(value) => match value.to_ascii_lowercase().as_str() {
            "1" | "true" | "t" | "on" | "yes" | "y" => Ok(true),
            "0" | "false" | "f" | "off" | "no" | "n" => Ok(false),
            _ => Err(()),
        },
        _ => Err(()),
    }
}

fn integer(value: &Value) -> Result<Value, ()> {
    if value.as_i64().is_some() || value.as_u64().is_some() {
        return Ok(value.clone());
    }
    if let Value::Bool(value) = value {
        return Ok(json!(u8::from(*value)));
    }
    if let Some(text) = value.as_str() {
        static INTEGER: OnceLock<regex::Regex> = OnceLock::new();
        let text = text.trim();
        if !INTEGER
            .get_or_init(|| regex::Regex::new(r"\A[+-]?[0-9]+(?:_[0-9]+)*(?:\.0+)?\z").unwrap())
            .is_match(text)
        {
            return Err(());
        }
        let text = text.split('.').next().ok_or(())?.replace('_', "");
        if let Ok(n) = text.parse::<i64>() {
            return Ok(json!(n));
        }
        if let Ok(n) = text.parse::<u64>() {
            return Ok(json!(n));
        }
        return Err(());
    }
    let n = value.as_f64().ok_or(())?;
    if !n.is_finite() || n.fract() != 0.0 {
        return Err(());
    }
    if (0.0..u64::MAX as f64).contains(&n) {
        return Ok(json!(n as u64));
    }
    if (i64::MIN as f64..0.0).contains(&n) {
        return Ok(json!(n as i64));
    }
    Err(())
}

fn normalize(value: &Value, rule: &Value, root: &Value, depth: usize) -> Result<Value, ()> {
    if depth > 64 {
        return Err(());
    }
    if let Some(reference) = rule["$ref"].as_str() {
        if reference == "#/$defs/JsonValue" {
            return Ok(value.clone());
        }
        let path = reference.strip_prefix('#').ok_or(())?;
        return normalize(value, root.pointer(path).ok_or(())?, root, depth + 1);
    }
    if let Some(union) = rule["oneOf"].as_array() {
        let discriminator = rule["discriminator"]["propertyName"].as_str().ok_or(())?;
        let tag = value[discriminator].as_str().ok_or(())?;
        let reference = rule["discriminator"]["mapping"][tag].as_str().ok_or(())?;
        let branch = union.iter().find(|s| s["$ref"] == reference).ok_or(())?;
        return normalize(value, branch, root, depth + 1);
    }
    if let Some(union) = rule["anyOf"].as_array() {
        if value.is_null() && union.iter().any(|s| s["type"] == "null") {
            return Ok(Value::Null);
        }
        for branch in union.iter().filter(|s| s["type"] != "null") {
            if let Ok(result) = normalize(value, branch, root, depth + 1) {
                return Ok(result);
            }
        }
        return Err(());
    }
    match rule["type"].as_str() {
        Some("object") => {
            let original = value.as_object().ok_or(())?;
            let Some(properties) = rule["properties"].as_object() else {
                return Ok(value.clone());
            };
            let mut result = original.clone();
            for (key, rule) in properties {
                if let Some(value) = original.get(key) {
                    result.insert(key.clone(), normalize(value, rule, root, depth + 1)?);
                } else if let Some(default) = rule.get("default") {
                    result.insert(key.clone(), default.clone());
                }
            }
            Ok(Value::Object(result))
        }
        Some("array") => Ok(Value::Array(
            value
                .as_array()
                .ok_or(())?
                .iter()
                .map(|v| normalize(v, &rule["items"], root, depth + 1))
                .collect::<Result<_, _>>()?,
        )),
        Some("integer") => integer(value),
        Some("boolean") => Ok(json!(boolean(value)?)),
        Some("string") if !value.is_string() => Err(()),
        Some("null") if !value.is_null() => Err(()),
        _ => Ok(value.clone()),
    }
}

pub(super) fn message(value: &Value) -> Result<Value, ()> {
    let s = schema();
    let result = normalize(value, &s.root, &s.root, 0)?;
    if !s.validator.is_valid(&result) {
        return Err(());
    }
    super::input_messages::coherent(&result)?;
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hosted_messages_match_python_normalization_and_validation() {
        for (i, line) in include_str!("../../tests/generated/input_messages.jsonl")
            .lines()
            .enumerate()
        {
            let case: Value = crate::lossless_json::from_str(line).unwrap();
            assert_eq!(
                json!(message(&case["input"]).ok()),
                case["expected"],
                "case {i}: {}",
                case["input"]
            );
        }
    }
}
