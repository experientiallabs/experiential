//! Semantic JSON equality at Python-authored history and argument boundaries.

use serde_json::Value;

fn integer(value: &Value) -> Option<String> {
    match value {
        Value::Bool(v) => Some(if *v { "1" } else { "0" }.into()),
        Value::Number(v) => {
            let raw = v.to_string();
            if !raw.contains(['.', 'e', 'E']) {
                Some(if raw == "-0" { "0".into() } else { raw })
            } else {
                None
            }
        }
        _ => None,
    }
}
fn float_integer(value: &Value) -> Option<String> {
    let n = value.as_f64()?;
    (n.is_finite() && n.fract() == 0.0).then(|| {
        if n == 0.0 {
            "0".into()
        } else {
            format!("{n:.0}")
        }
    })
}
pub(super) fn equal(a: &Value, b: &Value) -> bool {
    match (a, b) {
        (Value::Bool(_) | Value::Number(_), Value::Bool(_) | Value::Number(_)) => {
            match (integer(a), integer(b)) {
                (Some(a), Some(b)) => a == b,
                (Some(a), None) => float_integer(b).as_ref() == Some(&a),
                (None, Some(b)) => float_integer(a).as_ref() == Some(&b),
                (None, None) => a.as_f64() == b.as_f64(),
            }
        }
        (Value::Array(a), Value::Array(b)) => {
            a.len() == b.len() && a.iter().zip(b).all(|(a, b)| equal(a, b))
        }
        (Value::Object(a), Value::Object(b)) => {
            a.len() == b.len() && a.iter().all(|(k, a)| b.get(k).is_some_and(|b| equal(a, b)))
        }
        _ => a == b,
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    use crate::lossless_json::parse;
    #[test]
    fn booleans_and_floats_match_integers_without_rounding_large_integers() {
        for (a, b, expected) in [
            ("true", "1", true),
            ("false", "-0.0", true),
            ("1", "1.0", true),
            ("9007199254740993", "9007199254740992.0", false),
            (
                "10000000000000000000000000000000000000000000000000001",
                "1e49",
                false,
            ),
            (r#"{"a":[true,1.0]}"#, r#"{"a":[1,1]}"#, true),
            (r#"{"a":1,"b":2}"#, r#"{"b":2,"a":1}"#, true),
        ] {
            assert_eq!(
                equal(&parse(a).unwrap(), &parse(b).unwrap()),
                expected,
                "{a} {b}"
            );
        }
    }
}
