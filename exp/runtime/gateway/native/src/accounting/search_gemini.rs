//! Gemini conversation encoding with complete, ordered parallel function results.

use super::{
    search_encode::{self as encode, Options},
    search_history::instruction,
    search_media as media,
    search_subject::{folded, items},
};
use crate::errors::PublicError;
use serde_json::{json, Value};
use std::collections::HashMap;

const BYPASS: &str = "skip_thought_signature_validator";
const MISSING: &str = "[no result: the caller's history carries no tool result for this call]";

#[derive(Default)]
struct Turn {
    calls: Vec<(String, String)>,
    responses: HashMap<String, Value>,
    others: Vec<Value>,
}

impl Turn {
    fn close(&mut self, contents: &mut Vec<Value>) {
        if self.calls.is_empty() {
            return;
        }
        if !self.responses.is_empty() || !self.others.is_empty() {
            let mut parts = vec![];
            for (id, name) in &self.calls {
                if let Some(response) = self.responses.get(id) {
                    parts.extend(items(&response["parts"]).to_vec());
                } else {
                    parts.push(json!({"functionResponse": {"name": name, "response": {"content": MISSING}}}));
                }
            }
            contents.push(json!({"role": "user", "parts": parts}));
            contents.append(&mut self.others);
        }
        self.calls.clear();
        self.responses.clear();
        self.others.clear();
    }
}

pub(super) fn rebuild(
    payload: &mut Value,
    request: &Value,
    messages: Vec<Value>,
    o: &Options,
) -> Result<(), PublicError> {
    let mut system = vec![];
    let mut out = vec![];
    let mut names = HashMap::new();
    let mut turn = Turn::default();
    for m in &messages {
        if instruction(m) {
            system.push(json!({"text": m["content"].as_str().ok_or_else(PublicError::internal)?}));
            continue;
        }
        if m["role"] == "assistant" {
            turn.close(&mut out);
        }
        let content = message(m, &mut names)?;
        if m["role"] == "assistant" {
            out.push(content);
            turn.calls = items(&m["tool_calls"])
                .iter()
                .map(|c| {
                    Ok((
                        c["call_id"]
                            .as_str()
                            .ok_or_else(PublicError::internal)?
                            .to_owned(),
                        c["name"]
                            .as_str()
                            .ok_or_else(PublicError::internal)?
                            .to_owned(),
                    ))
                })
                .collect::<Result<_, PublicError>>()?;
        } else if !turn.calls.is_empty() && matches!(m["role"].as_str(), Some("user" | "tool")) {
            let id = m["tool_call_id"].as_str().unwrap_or("");
            if m["role"] == "tool"
                && !turn.responses.contains_key(id)
                && turn.calls.iter().any(|(known, _)| known == id)
            {
                turn.responses.insert(id.to_owned(), content);
            } else {
                turn.others.push(content);
            }
        } else {
            out.push(content);
        }
    }
    turn.close(&mut out);
    if request["json_object_output"] == true {
        system.push(json!({"text": o.json_instruction}));
    }
    encode::remove(payload, "systemInstruction");
    if !system.is_empty() {
        payload["systemInstruction"] = json!({"parts":system});
    }
    payload["contents"] = json!(out);
    Ok(())
}

fn message(m: &Value, names: &mut HashMap<String, String>) -> Result<Value, PublicError> {
    if m["role"] == "tool" {
        let name = names
            .get(m["tool_call_id"].as_str().unwrap_or(""))
            .ok_or_else(PublicError::internal)?;
        return Ok(
            json!({"role":"user","parts":[{"functionResponse":{"name":name,"response":{"content":folded(m)}}}]}),
        );
    }
    let assistant = m["role"] == "assistant";
    if !assistant && m["role"] != "user" {
        return Err(PublicError::internal());
    }
    let mut parts = vec![];
    if !items(&m["content_parts"]).is_empty() {
        for (i, p) in items(&m["content_parts"]).iter().enumerate() {
            if assistant && !matches!(p["kind"].as_str(), Some("text" | "image")) {
                return Err(PublicError::internal());
            }
            let mut part = media::part(p, "gemini", i + 1)?;
            if assistant {
                part["thoughtSignature"] = json!(BYPASS);
            }
            parts.push(part);
        }
    } else if !m["content"].is_null() {
        parts.push(json!({"text": m["content"]}));
    } else if !assistant {
        return Err(PublicError::internal());
    }
    if assistant {
        let signed = items(&m["tool_calls"]).iter().any(|c| {
            c["provider_thought_signature"]
                .as_str()
                .is_some_and(|s| !s.is_empty())
        });
        for c in items(&m["tool_calls"]) {
            names.insert(
                c["call_id"]
                    .as_str()
                    .ok_or_else(PublicError::internal)?
                    .to_owned(),
                c["name"]
                    .as_str()
                    .ok_or_else(PublicError::internal)?
                    .to_owned(),
            );
            let mut part = json!({"functionCall":{"name":c["name"],"args":c["arguments"]}});
            if !c["provider_thought_signature"].is_null() {
                part["thoughtSignature"] = c["provider_thought_signature"].clone();
            } else if !signed {
                part["thoughtSignature"] = json!(BYPASS);
            }
            parts.push(part);
        }
    }
    if parts.is_empty() {
        return Err(PublicError::internal());
    }
    Ok(json!({"role": if assistant {"model"} else {"user"}, "parts":parts}))
}
