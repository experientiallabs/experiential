//! Public attribution header validation before admission or replay ownership.

use std::collections::BTreeMap;
use std::fmt;

use axum::http::HeaderMap;
use serde::de::{self, MapAccess, Visitor};
use serde::{Deserialize, Deserializer};

use crate::errors::PublicError;

const HEADER: &str = "x-explabs-tags";
const MAX_HEADER_BYTES: usize = 8 * 1024;

#[derive(Debug, Default, PartialEq, Eq)]
struct RequestTags(BTreeMap<String, String>);

impl<'de> Deserialize<'de> for RequestTags {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        struct TagsVisitor;
        impl<'de> Visitor<'de> for TagsVisitor {
            type Value = RequestTags;

            fn expecting(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
                formatter.write_str("a flat map of at most 16 string tags")
            }

            fn visit_map<M: MapAccess<'de>>(self, mut map: M) -> Result<Self::Value, M::Error> {
                let mut tags = BTreeMap::new();
                while let Some((key, value)) = map.next_entry::<String, String>()? {
                    let valid_key = !key.is_empty()
                        && key.len() <= 64
                        && key.as_bytes()[0].is_ascii_alphabetic()
                        && key
                            .bytes()
                            .all(|c| c.is_ascii_alphanumeric() || b"_.-".contains(&c))
                        && !key.to_ascii_lowercase().starts_with("explabs.");
                    let valid_value = !value.is_empty()
                        && value.chars().count() <= 256
                        && !value.chars().any(char::is_control);
                    if !valid_key
                        || !valid_value
                        || tags.len() == 16
                        || tags.insert(key, value).is_some()
                    {
                        return Err(de::Error::custom("invalid or duplicate tag"));
                    }
                }
                Ok(RequestTags(tags))
            }
        }
        deserializer.deserialize_map(TagsVisitor)
    }
}

/// Parse exactly one UTF-8 JSON header, rejecting duplicate fields and keys.
/// Absence and an empty object both represent an untagged request.
pub(crate) fn request_tags(headers: &HeaderMap) -> Result<BTreeMap<String, String>, PublicError> {
    let mut values = headers.get_all(HEADER).iter();
    let Some(value) = values.next() else {
        return Ok(BTreeMap::new());
    };
    if values.next().is_some() || value.as_bytes().len() > MAX_HEADER_BYTES {
        return Err(invalid_tags());
    }
    serde_json::from_slice::<RequestTags>(value.as_bytes())
        .map(|tags| tags.0)
        .map_err(|_| invalid_tags())
}

fn invalid_tags() -> PublicError {
    let mut error = PublicError::new(
        400,
        "invalid_parameter",
        "Invalid X-Explabs-Tags. Send one JSON object (at most 8 KiB) with at most 16 unique ASCII identifier keys (1-64 characters, starting with a letter) and nonempty UTF-8 string values (at most 256 characters, no controls). The explabs. prefix is reserved.",
        "invalid_request_error",
    );
    error.param = Some("X-Explabs-Tags".to_string());
    error
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::http::HeaderValue;
    use serde_json::json;

    fn parse(raw: &[u8]) -> Result<BTreeMap<String, String>, PublicError> {
        let mut headers = HeaderMap::new();
        headers.insert(HEADER, HeaderValue::from_bytes(raw).unwrap());
        request_tags(&headers)
    }

    #[test]
    fn absent_empty_unicode_and_exact_keys() {
        assert!(request_tags(&HeaderMap::new()).unwrap().is_empty());
        assert!(parse(b"{}").unwrap().is_empty());
        let tags =
            parse(r#"{"Team":"a","team":"b","cost-center.prod":"café"}"#.as_bytes()).unwrap();
        assert_eq!(tags["Team"], "a");
        assert_eq!(tags["team"], "b");
        assert_eq!(tags["cost-center.prod"], "café");
    }

    #[test]
    fn rejects_invalid_shapes_duplicates_and_text() {
        for raw in [
            "",
            "null",
            "[]",
            "1",
            "{} {}",
            r#"{"a":{}}"#,
            r#"{"a":[]}"#,
            r#"{"a":null}"#,
            r#"{"a":true}"#,
            r#"{"a":1}"#,
            r#"{"a":""}"#,
            r#"{"":"v"}"#,
            r#"{"1a":"v"}"#,
            r#"{"é":"v"}"#,
            r#"{"a b":"v"}"#,
            r#"{"a":"x","a":"y"}"#,
            concat!(r#"{"a":"x","\"#, "u0061", r#"":"y"}"#),
            r#"{"a":"\ud800"}"#,
            r#"{"explabs.team":"x"}"#,
            r#"{"ExPlAbS.team":"x"}"#,
        ] {
            let error = parse(raw.as_bytes()).expect_err(raw);
            assert_eq!(error.status_code, 400);
            assert_eq!(error.param.as_deref(), Some("X-Explabs-Tags"));
        }
        assert!(parse(b"{\"a\":\"\xff\"}").is_err());
        for control in [0, 9, 10, 13, 127, 133, 159] {
            let raw = format!(r#"{{"a":"{}u{control:04x}"}}"#, char::from(92));
            assert!(parse(raw.as_bytes()).is_err());
        }
        let mut headers = HeaderMap::new();
        headers.append(HEADER, "{}".parse().unwrap());
        headers.append(HEADER, "{}".parse().unwrap());
        assert!(request_tags(&headers).is_err());
    }

    #[test]
    fn exact_entry_key_character_and_wire_limits() {
        let mut tags: BTreeMap<String, String> =
            (0..16).map(|i| (format!("k{i}"), "v".into())).collect();
        assert!(parse(serde_json::to_string(&tags).unwrap().as_bytes()).is_ok());
        tags.insert("extra".into(), "v".into());
        assert!(parse(serde_json::to_string(&tags).unwrap().as_bytes()).is_err());
        for (key_len, value_len, valid) in [(64, 256, true), (65, 256, false), (64, 257, false)] {
            let raw = json!({"k".repeat(key_len): "é".repeat(value_len)}).to_string();
            assert_eq!(parse(raw.as_bytes()).is_ok(), valid);
        }
        let exact = format!("{}{{}}", " ".repeat(MAX_HEADER_BYTES - 2));
        assert!(parse(exact.as_bytes()).is_ok());
        assert!(parse(format!(" {exact}").as_bytes()).is_err());
    }
}
