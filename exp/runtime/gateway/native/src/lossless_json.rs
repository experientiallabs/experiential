//! Exact JSON values, including large integers and literal Serde marker keys.
//!
//! Serde's arbitrary-precision number protocol uses a synthetic map. Reading a
//! real object through that protocol can mistake a caller's key for a number.
//! Reader-backed input distinguishes real owned/transient keys from Serde's
//! borrowed synthetic number key. Typed JSON fields use the same parser through
//! `deserialize`; typed, non-JSON protocol fields stay with Serde.

use std::any::{Any, TypeId};
use std::collections::{BTreeMap, HashMap};
use std::sync::Arc;

use serde::{de::DeserializeOwned, Deserializer};
use serde_json::{Error, Map, Number, Value};

fn error(message: &str) -> Error {
    serde::de::Error::custom(message)
}

pub(crate) fn number(raw: &str) -> Result<Number, Error> {
    if raw.contains(['.', 'e', 'E']) {
        Number::from_f64(
            raw.parse::<f64>()
                .map_err(|_| error("invalid JSON number"))?,
        )
        .ok_or_else(|| error("non-finite JSON number"))
    } else {
        raw.parse()
    }
}

pub(crate) fn parse(input: &str) -> Result<Value, Error> {
    parse_unique(input, false)
}

pub(crate) fn parse_unique(input: &str, unique: bool) -> Result<Value, Error> {
    use serde::de::DeserializeSeed;
    let mut d = serde_json::Deserializer::from_reader(input.as_bytes());
    let value = ValueSeed { unique }.deserialize(&mut d)?;
    d.end()?;
    Ok(value)
}

/// Reader-backed object keys are transient/owned; only Serde's synthetic
/// arbitrary-precision number field is borrowed. Preserve that distinction
/// through typed enum/flatten buffers as well as direct value parsing.
pub(crate) enum ObjectKey {
    Number,
    Key(String),
}
impl<'de> serde::Deserialize<'de> for ObjectKey {
    fn deserialize<D: Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        struct KeyVisitor;
        impl<'de> serde::de::Visitor<'de> for KeyVisitor {
            type Value = ObjectKey;
            fn expecting(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
                f.write_str("JSON object key")
            }
            fn visit_str<E: serde::de::Error>(self, key: &str) -> Result<ObjectKey, E> {
                Ok(ObjectKey::Key(key.into()))
            }
            fn visit_string<E: serde::de::Error>(self, key: String) -> Result<ObjectKey, E> {
                Ok(ObjectKey::Key(key))
            }
            fn visit_borrowed_str<E: serde::de::Error>(
                self,
                key: &'de str,
            ) -> Result<ObjectKey, E> {
                if key == "$serde_json::private::Number" {
                    Ok(ObjectKey::Number)
                } else {
                    self.visit_str(key)
                }
            }
        }
        d.deserialize_str(KeyVisitor)
    }
}
struct ValueSeed {
    unique: bool,
}
impl<'de> serde::de::DeserializeSeed<'de> for ValueSeed {
    type Value = Value;
    fn deserialize<D: Deserializer<'de>>(self, d: D) -> Result<Value, D::Error> {
        d.deserialize_any(self)
    }
}
impl<'de> serde::de::Visitor<'de> for ValueSeed {
    type Value = Value;
    fn expecting(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        f.write_str("JSON value")
    }
    fn visit_bool<E: serde::de::Error>(self, v: bool) -> Result<Value, E> {
        Ok(v.into())
    }
    fn visit_unit<E: serde::de::Error>(self) -> Result<Value, E> {
        Ok(Value::Null)
    }
    fn visit_none<E: serde::de::Error>(self) -> Result<Value, E> {
        Ok(Value::Null)
    }
    fn visit_i64<E: serde::de::Error>(self, v: i64) -> Result<Value, E> {
        Ok(v.into())
    }
    fn visit_u64<E: serde::de::Error>(self, v: u64) -> Result<Value, E> {
        Ok(v.into())
    }
    fn visit_i128<E: serde::de::Error>(self, v: i128) -> Result<Value, E> {
        Ok(Value::Number(v.into()))
    }
    fn visit_u128<E: serde::de::Error>(self, v: u128) -> Result<Value, E> {
        Ok(Value::Number(v.into()))
    }
    fn visit_f64<E: serde::de::Error>(self, v: f64) -> Result<Value, E> {
        Number::from_f64(v)
            .map(Value::Number)
            .ok_or_else(|| E::custom("non-finite JSON number"))
    }
    fn visit_str<E: serde::de::Error>(self, v: &str) -> Result<Value, E> {
        Ok(v.into())
    }
    fn visit_string<E: serde::de::Error>(self, v: String) -> Result<Value, E> {
        Ok(v.into())
    }
    fn visit_seq<A: serde::de::SeqAccess<'de>>(self, mut seq: A) -> Result<Value, A::Error> {
        let mut values = Vec::new();
        while let Some(v) = seq.next_element_seed(ValueSeed {
            unique: self.unique,
        })? {
            values.push(v);
        }
        Ok(Value::Array(values))
    }
    fn visit_map<A: serde::de::MapAccess<'de>>(self, mut map: A) -> Result<Value, A::Error> {
        let mut values = Map::new();
        while let Some(key) = map.next_key::<ObjectKey>()? {
            let ObjectKey::Key(key) = key else {
                if !values.is_empty() {
                    return Err(serde::de::Error::custom("misplaced number field"));
                }
                let raw: String = map.next_value()?;
                return number(&raw)
                    .map(Value::Number)
                    .map_err(serde::de::Error::custom);
            };
            let value = map.next_value_seed(ValueSeed {
                unique: self.unique,
            })?;
            if values.insert(key, value).is_some() && self.unique {
                return Err(serde::de::Error::custom("duplicate JSON key"));
            }
        }
        Ok(Value::Object(values))
    }
}

/// Rebuild containers without passing their literal object keys through Value's visitor.
pub(crate) trait Exact: Sized {
    fn exact(value: Value) -> Result<Self, Error>;
}
impl Exact for f64 {
    fn exact(value: Value) -> Result<Self, Error> {
        value
            .as_f64()
            .filter(|v| v.is_finite())
            .ok_or_else(|| error("expected finite number"))
    }
}
impl Exact for Value {
    fn exact(value: Value) -> Result<Self, Error> {
        Ok(value)
    }
}
impl<T: Exact> Exact for Vec<T> {
    fn exact(value: Value) -> Result<Self, Error> {
        let Value::Array(values) = value else {
            return Err(error("expected JSON array"));
        };
        values.into_iter().map(T::exact).collect()
    }
}
impl<T: Exact> Exact for Option<T> {
    fn exact(value: Value) -> Result<Self, Error> {
        if value.is_null() {
            Ok(None)
        } else {
            T::exact(value).map(Some)
        }
    }
}
impl<T: Exact> Exact for Arc<T> {
    fn exact(value: Value) -> Result<Self, Error> {
        T::exact(value).map(Arc::new)
    }
}
impl<T: Exact> Exact for Box<T> {
    fn exact(value: Value) -> Result<Self, Error> {
        T::exact(value).map(Box::new)
    }
}
impl Exact for Map<String, Value> {
    fn exact(value: Value) -> Result<Self, Error> {
        match value {
            Value::Object(map) => Ok(map),
            _ => Err(error("expected JSON object")),
        }
    }
}
impl<T: Exact> Exact for HashMap<String, T> {
    fn exact(value: Value) -> Result<Self, Error> {
        Map::exact(value)?
            .into_iter()
            .map(|(k, v)| T::exact(v).map(|v| (k, v)))
            .collect()
    }
}
impl<T: Exact> Exact for BTreeMap<String, T> {
    fn exact(value: Value) -> Result<Self, Error> {
        Map::exact(value)?
            .into_iter()
            .map(|(k, v)| T::exact(v).map(|v| (k, v)))
            .collect()
    }
}

pub(crate) fn deserialize<'de, D: Deserializer<'de>, T: Exact>(d: D) -> Result<T, D::Error> {
    use serde::de::DeserializeSeed;
    T::exact(ValueSeed { unique: false }.deserialize(d)?).map_err(serde::de::Error::custom)
}

fn untyped<T: Any>() -> bool {
    let id = TypeId::of::<T>();
    id == TypeId::of::<Value>()
        || id == TypeId::of::<Vec<Value>>()
        || id == TypeId::of::<Map<String, Value>>()
        || id == TypeId::of::<HashMap<String, Value>>()
        || id == TypeId::of::<BTreeMap<String, Value>>()
        || id == TypeId::of::<Option<Value>>()
}

pub(crate) fn from_value<T: DeserializeOwned + Any>(value: Value) -> Result<T, Error> {
    let id = TypeId::of::<T>();
    let boxed: Box<dyn Any> = if id == TypeId::of::<Value>() {
        Box::new(value)
    } else if id == TypeId::of::<Vec<Value>>() {
        Box::new(Vec::<Value>::exact(value)?)
    } else if id == TypeId::of::<Map<String, Value>>() {
        Box::new(Map::<String, Value>::exact(value)?)
    } else if id == TypeId::of::<HashMap<String, Value>>() {
        Box::new(HashMap::<String, Value>::exact(value)?)
    } else if id == TypeId::of::<BTreeMap<String, Value>>() {
        Box::new(BTreeMap::<String, Value>::exact(value)?)
    } else if id == TypeId::of::<Option<Value>>() {
        Box::new(Option::<Value>::exact(value)?)
    } else {
        return serde_json::from_reader(value.to_string().as_bytes());
    };
    boxed
        .downcast::<T>()
        .map(|v| *v)
        .map_err(|_| error("JSON type mismatch"))
}
pub(crate) fn from_str<T: DeserializeOwned + Any>(input: &str) -> Result<T, Error> {
    if untyped::<T>() {
        from_value(parse(input)?)
    } else {
        serde_json::from_reader(input.as_bytes())
    }
}
pub(crate) fn from_slice<T: DeserializeOwned + Any>(input: &[u8]) -> Result<T, Error> {
    from_str(std::str::from_utf8(input).map_err(|_| error("JSON is not UTF-8"))?)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde::Deserialize;

    #[test]
    fn exact_numbers_and_marker_keys_survive_typed_boundaries() {
        #[derive(Deserialize)]
        struct Payload {
            #[serde(deserialize_with = "deserialize")]
            data: Vec<Value>,
        }
        let raw = r#"{"data":[123456789012345678901234567890,{"$serde_json::private::Number":"1"},{"$serde_json::private::RawValue":"1"}]}"#;
        let payload: Payload = from_str(raw).unwrap();
        assert_eq!(
            payload.data[0].to_string(),
            "123456789012345678901234567890"
        );
        assert!(payload.data[1].is_object());
        assert!(payload.data[2].is_object());
        let typed: Payload = from_value(parse(raw).unwrap()).unwrap();
        assert_eq!(typed.data, payload.data);
        assert_eq!(from_str::<Value>(raw).unwrap().to_string(), raw);
    }
    #[test]
    fn strict_and_ordinary_parsing_keep_their_duplicate_key_contracts() {
        assert_eq!(parse(r#"{"a":1,"a":2}"#).unwrap()["a"], 2);
        assert!(parse_unique(r#"{"outer":{"a":1,"a":2}}"#, true).is_err());
        for raw in [
            "01",
            "1e",
            "[1,]",
            "{\"a\":}",
            "null true",
            "true\u{000b}",
            "\"\\uD800\"",
        ] {
            assert!(parse(raw).is_err(), "{raw}");
        }
        assert_eq!(parse("\"hé\\uD83D\\uDE00\"").unwrap(), "hé😀");
    }
    #[test]
    fn tagged_and_flattened_payloads_keep_numbers_and_literal_marker_objects() {
        #[derive(Deserialize)]
        #[serde(tag = "kind")]
        enum Tagged {
            #[serde(rename = "payload")]
            Payload {
                #[serde(deserialize_with = "deserialize")]
                data: Value,
                #[serde(deserialize_with = "deserialize")]
                deadline: f64,
            },
        }
        #[derive(Deserialize)]
        struct Flat {
            #[serde(flatten, deserialize_with = "deserialize")]
            rest: Map<String, Value>,
        }
        let raw = r#"{"kind":"payload","deadline":1.5,"data":{"n":12345678901234567890123456789012345678901234567890,"\u0024serde_json::private::Number":{"$serde_json::private::Number":"123"}}}"#;
        let expected = parse(raw).unwrap();
        for source in [
            from_str::<Tagged>(raw).unwrap(),
            from_value::<Tagged>(expected.clone()).unwrap(),
        ] {
            let Tagged::Payload { data, deadline } = source;
            assert_eq!(data, expected["data"]);
            assert_eq!(deadline, 1.5);
        }
        let flat: Flat = from_str(raw).unwrap();
        assert_eq!(flat.rest["data"], expected["data"]);
    }
}
