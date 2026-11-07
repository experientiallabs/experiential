//! Bounded Responses turn construction under the winning route's authority.

use std::collections::HashSet;

use base64::{engine::general_purpose::URL_SAFE_NO_PAD, Engine};
use serde::Deserialize;
use serde_json::{json, Value};

use crate::errors::PublicError;

#[derive(Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct Context {
    #[serde(deserialize_with = "crate::lossless_json::deserialize")]
    namespace: Value,
    episode_key: String,
    response_id: String,
    #[serde(deserialize_with = "crate::lossless_json::deserialize")]
    messages: Vec<Value>,
    #[serde(deserialize_with = "crate::lossless_json::deserialize")]
    replay: Vec<Value>,
    #[serde(deserialize_with = "crate::lossless_json::deserialize")]
    message_template: Value,
    #[serde(deserialize_with = "crate::lossless_json::deserialize")]
    route_bindings: Vec<Value>,
}

#[derive(Default)]
struct Segment {
    message: Option<Value>,
    calls: Vec<Value>,
    reasoning: Vec<Value>,
}

enum Item {
    Message(Value),
    Call(Value),
    Reasoning(Value),
    Native(Value),
}

fn text<'a>(v: &'a Value, key: &str) -> Result<&'a str, PublicError> {
    v[key]
        .as_str()
        .filter(|s| !s.is_empty())
        .ok_or_else(PublicError::internal)
}
fn array<'a>(v: &'a Value, key: &str) -> Result<&'a [Value], PublicError> {
    match v.get(key) {
        None => Ok(&[]),
        Some(v) => v
            .as_array()
            .map(Vec::as_slice)
            .ok_or_else(PublicError::internal),
    }
}
fn index(v: &Value, seen: &mut HashSet<u64>) -> Result<u64, PublicError> {
    let i = v["output_index"]
        .as_u64()
        .ok_or_else(PublicError::internal)?;
    if !seen.insert(i) {
        return Err(PublicError::internal());
    }
    Ok(i)
}
fn status(v: &Value) -> Result<Value, PublicError> {
    if !v.is_null() && !v.is_string() {
        return Err(PublicError::internal());
    }
    match v.as_str().unwrap_or("completed") {
        s @ ("in_progress" | "completed" | "incomplete") => Ok(json!(s)),
        _ => Err(PublicError::internal()),
    }
}
fn optional_text(v: &Value, key: &str) -> Result<Value, PublicError> {
    if v[key].is_null() {
        Ok(Value::Null)
    } else {
        text(v, key).map(|s| json!(s))
    }
}

impl Context {
    pub fn remember(
        &self,
        request: &str,
        depth: Option<usize>,
        data: &Value,
    ) -> Result<Option<Value>, PublicError> {
        if data["refusal"].as_bool().unwrap_or(false) {
            return Ok(None);
        }
        let mut messages = self.messages.clone();
        let mut replay = self.replay.clone();
        let mut seen = HashSet::new();
        let mut items = Vec::new();
        let mut plain_calls = Vec::new();
        let mut native_tail = Vec::new();
        let mut text_output = data["text"].as_str().unwrap_or("").to_string();
        let mut message_outputs = if data["message_outputs"].is_null() {
            vec![]
        } else {
            array(data, "message_outputs")?.to_vec()
        };
        if data["message_outputs"].is_null()
            && (!data["message_item_id"].is_null() || !data["message_output_index"].is_null())
        {
            message_outputs.push(json!({"item_id": data["message_item_id"], "output_index": data["message_output_index"], "text": text_output, "status": "completed"}));
            text_output.clear();
        }
        for m in &message_outputs {
            let i = index(m, &mut seen)?;
            text(m, "item_id")?;
            m["text"].as_str().ok_or_else(PublicError::internal)?;
            if !m["phase"].is_null()
                && !matches!(m["phase"].as_str(), Some("commentary" | "final_answer"))
            {
                return Err(PublicError::internal());
            }
            status(&m["status"])?;
            items.push((i, Item::Message(m.clone())));
        }
        for c in array(data, "tool_calls")? {
            let call_id = text(c, "call_id")?;
            let name = text(c, "name")?;
            let raw = c["arguments"].as_str().ok_or_else(PublicError::internal)?;
            if call_id.chars().count() > 65_536
                || name.chars().count() > 256
                || raw.chars().count() > 4_000_000
            {
                return Err(PublicError::internal());
            }
            let id = optional_text(c, "item_id")?;
            let i = if id.is_null() && c["output_index"].is_null() {
                None
            } else {
                Some(index(c, &mut seen)?)
            };
            let namespace = optional_text(c, "namespace")?;
            let signature = optional_text(c, "thought_signature")?;
            if !c["caller"].is_null() && !c["caller"].is_object() {
                return Err(PublicError::internal());
            }
            if c["custom"] == true {
                let mut native = json!({"type": "custom_tool_call", "call_id": call_id, "name": name, "input": raw});
                for key in ["namespace", "caller", "status"] {
                    if !c[key].is_null() {
                        native[key] = c[key].clone();
                    }
                }
                if !id.is_null() {
                    native["id"] = id;
                }
                if let Some(i) = i {
                    items.push((i, Item::Native(native)));
                } else {
                    native_tail.push(native);
                }
                continue;
            }
            let arguments: Value =
                crate::lossless_json::from_str(raw).map_err(|_| PublicError::internal())?;
            if !arguments.is_object() {
                return Err(PublicError::internal());
            }
            let call = json!({"call_id": call_id, "name": name, "arguments": arguments,
                "raw_arguments": raw, "provider_item_id": id, "provider_output_index": i,
                "provider_status": if i.is_some() { status(&c["status"])? } else { Value::Null },
                "provider_namespace": namespace, "provider_caller": c["caller"], "provider_thought_signature": signature});
            if let Some(i) = i {
                items.push((i, Item::Call(call)));
            } else {
                plain_calls.push(call);
            }
        }
        for e in array(data, "encrypted_reasoning")? {
            let i = index(e, &mut seen)?;
            items.push((i, Item::Reasoning(json!({"kind": "encrypted_reasoning", "id": text(e, "item_id")?,
                "encrypted_content": text(e, "encrypted_content")?, "output_index": i, "status": status(&e["status"])?}))));
        }
        for h in array(data, "hosted_items")? {
            let i = index(h, &mut seen)?;
            text(&h["item"], "type")?;
            items.push((i, Item::Native(h["item"].clone())));
        }
        let carrier = match data
            .get("reasoning_content_carrier")
            .filter(|v| !v.is_null())
        {
            None => None,
            Some(value) => Some(parse_carrier(
                value.as_str().ok_or_else(PublicError::internal)?,
            )?),
        };
        if !items.is_empty() && (carrier.is_some() || !text_output.is_empty()) {
            return Err(PublicError::internal());
        }
        items.sort_by_key(|item| item.0);
        let mut segment = Segment::default();
        for (_, item) in items {
            match item {
                Item::Message(m) => {
                    if segment.message.is_some() {
                        self.flush(&mut segment, &mut messages, &mut replay)?;
                    }
                    segment.message = Some(m);
                }
                Item::Call(c) => segment.calls.push(c),
                Item::Reasoning(e) => segment.reasoning.push(e),
                Item::Native(item) => {
                    self.flush(&mut segment, &mut messages, &mut replay)?;
                    messages.push(self.message_template.clone());
                    replay.push(
                        json!({"message_index": messages.len() - 1, "provider_native_item": item}),
                    );
                }
            }
        }
        self.flush(&mut segment, &mut messages, &mut replay)?;
        if !text_output.is_empty() || !plain_calls.is_empty() || carrier.is_some() {
            segment.message = Some(json!({"text": text_output}));
            segment.calls = plain_calls;
            segment.reasoning = carrier.into_iter().collect();
            self.flush(&mut segment, &mut messages, &mut replay)?;
        }
        for item in native_tail {
            messages.push(self.message_template.clone());
            replay.push(json!({"message_index": messages.len()-1, "provider_native_item": item}));
        }
        let encrypted = replay.iter().any(|m| {
            m["provider_reasoning"]
                .as_array()
                .is_some_and(|a| a.iter().any(|b| b["kind"] == "encrypted_reasoning"))
        });
        let binding = if encrypted {
            depth
                .and_then(|d| self.route_bindings.get(d))
                .filter(|b| b.is_object())
                .cloned()
                .ok_or_else(PublicError::internal)?
        } else {
            Value::Null
        };
        let state = json!({"episode_key": self.episode_key, "messages": messages, "route_binding": binding});
        let mut size = serde_json::to_vec(&state)
            .map_err(|_| PublicError::internal())?
            .len();
        replay.retain(|v| v.as_object().is_some_and(|o| o.len() > 1));
        if !replay.is_empty() {
            size += serde_json::to_vec(&replay)
                .map_err(|_| PublicError::internal())?
                .len();
        }
        Ok(Some(
            json!({"request_id": request, "namespace": self.namespace, "response_id": self.response_id,
            "state": state, "replay": replay, "size_bytes": size}),
        ))
    }

    fn flush(
        &self,
        s: &mut Segment,
        messages: &mut Vec<Value>,
        replay: &mut Vec<Value>,
    ) -> Result<(), PublicError> {
        if s.message.is_none() && s.calls.is_empty() && s.reasoning.is_empty() {
            return Ok(());
        }
        let mut message = self.message_template.clone();
        let mut hidden = json!({"message_index": messages.len()});
        if let Some(m) = s.message.take() {
            message["content"] = m["text"]
                .as_str()
                .filter(|s| !s.is_empty())
                .map_or(Value::Null, |s| json!(s));
            if !m["item_id"].is_null() {
                hidden["provider_item_id"] = m["item_id"].clone();
                hidden["provider_output_index"] = m["output_index"].clone();
                hidden["provider_status"] = status(&m["status"])?;
                hidden["provider_phase"] = m["phase"].clone();
            }
        }
        let public_calls: Vec<_> = s.calls.iter().map(|c| json!({"call_id": c["call_id"], "name": c["name"], "arguments": c["arguments"]})).collect();
        message["tool_calls"] = json!(public_calls);
        if !s.calls.is_empty() {
            let calls: Vec<_> = s
                .calls
                .drain(..)
                .map(|mut c| {
                    c.as_object_mut().unwrap().remove("name");
                    c.as_object_mut().unwrap().remove("arguments");
                    for k in [
                        "provider_namespace",
                        "provider_caller",
                        "provider_thought_signature",
                    ] {
                        if c[k].is_null() {
                            c.as_object_mut().unwrap().remove(k);
                        }
                    }
                    c
                })
                .collect();
            hidden["tool_calls"] = json!(calls);
        }
        if !s.reasoning.is_empty() {
            hidden["provider_reasoning"] = json!(std::mem::take(&mut s.reasoning));
        }
        messages.push(message);
        replay.push(hidden);
        Ok(())
    }
}

fn parse_carrier(carrier: &str) -> Result<Value, PublicError> {
    if carrier.len() > 4 * ((8 * 1024 * 1024 + 32 * 1024 + 30) / 3) + 512 || !carrier.is_ascii() {
        return Err(PublicError::internal());
    }
    let encoded = [
        "x-experiential-fireworks-reasoning-v2:",
        "x-experiential-hunyuan-reasoning-v1:",
    ]
    .into_iter()
    .find_map(|prefix| carrier.strip_prefix(prefix))
    .ok_or_else(PublicError::internal)?;
    let (deployment, envelope) = encoded.split_once(':').ok_or_else(PublicError::internal)?;
    let hint = String::from_utf8(
        URL_SAFE_NO_PAD
            .decode(deployment)
            .map_err(|_| PublicError::internal())?,
    )
    .map_err(|_| PublicError::internal())?;
    if hint.is_empty()
        || hint.len() > 256
        || URL_SAFE_NO_PAD
            .decode(envelope)
            .map_err(|_| PublicError::internal())?
            .len()
            > 8 * 1024 * 1024 + 32 * 1024 + 28
    {
        return Err(PublicError::internal());
    }
    Ok(json!({"kind": "sealed_reasoning_content", "carrier": carrier, "deployment_hint": hint}))
}
