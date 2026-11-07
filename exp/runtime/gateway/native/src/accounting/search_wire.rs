//! Render inspected conversations with admission-frozen provider controls and tools.

use std::collections::HashMap;

use serde::Deserialize;
use serde_json::{json, Value};

use crate::errors::PublicError;

use super::search::Tool;

#[derive(Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct Plan {
    #[serde(deserialize_with = "crate::lossless_json::deserialize")]
    wire: Value,
    #[serde(deserialize_with = "crate::lossless_json::deserialize")]
    payload: Value,
    signed: bool,
    dialect: String,
    #[serde(deserialize_with = "crate::lossless_json::deserialize")]
    definitions: HashMap<String, Option<Vec<Value>>>,
    #[serde(deserialize_with = "crate::lossless_json::deserialize")]
    static_tools: Vec<Value>,
    encoder: super::search_encode::Options,
}

impl Plan {
    pub fn render_subject(&self, tools: &[Tool], request: &Value) -> Result<Value, PublicError> {
        let mut payload = self.payload.clone();
        super::search_encode::rebuild(&mut payload, &self.dialect, request, &self.encoder)?;
        self.tools(&mut payload, tools)?;
        if self.dialect == "bedrock_converse_stream"
            && super::search_subject::items(&request["messages"])
                .iter()
                .flat_map(|m| super::search_subject::items(&m["content_parts"]))
                .any(|p| p["kind"] != "text" && !p["data"].is_null())
            && super::json_text::dumps(&payload, false)?.len() > 25_000_000
        {
            return Err(PublicError::internal());
        }
        let mut wire = self.wire.clone();
        if self.signed {
            wire["upstream_body"] = json!(super::json_text::dumps(&payload, false)?);
        } else {
            wire["upstream_payload"] = payload;
        }
        Ok(wire)
    }

    fn tools(&self, payload: &mut Value, tools: &[Tool]) -> Result<(), PublicError> {
        let mut entries = Vec::new();
        for tool in tools {
            let definition = self
                .definitions
                .get(&tool.key)
                .and_then(Option::as_ref)
                .ok_or_else(PublicError::internal)?;
            entries.extend(definition.iter().cloned());
        }
        if self.dialect == "openai_responses" {
            let mut natives = std::collections::BTreeMap::new();
            let count = entries.len() + self.static_tools.len();
            for tool in &self.static_tools {
                let index = tool["index"]
                    .as_u64()
                    .filter(|n| *n < count as u64)
                    .ok_or_else(PublicError::internal)?;
                if natives
                    .insert(index as usize, tool["tool"].clone())
                    .is_some()
                {
                    return Err(PublicError::internal());
                }
            }
            let mut functions = entries.into_iter();
            entries = (0..count)
                .map(|i| {
                    natives
                        .remove(&i)
                        .or_else(|| functions.next())
                        .ok_or_else(PublicError::internal)
                })
                .collect::<Result<Vec<_>, _>>()?;
        } else {
            entries.extend(self.static_tools.iter().map(|v| v["tool"].clone()));
        }
        match self.dialect.as_str() {
            "gemini_generate_content" if !entries.is_empty() => {
                payload["tools"] = json!([{"functionDeclarations": entries}])
            }
            "bedrock_converse_stream" if !entries.is_empty() => {
                payload["toolConfig"]["tools"] = json!(entries)
            }
            "bedrock_converse_stream" => {
                payload
                    .as_object_mut()
                    .ok_or_else(PublicError::internal)?
                    .remove("toolConfig");
            }
            _ if !entries.is_empty() => payload["tools"] = json!(entries),
            _ => (),
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::super::search_round::Search;
    use super::*;

    #[test]
    fn rebuilt_dispatches_match_all_five_python_provider_encoders() {
        for (i, line) in include_str!("../../tests/generated/search_wires.jsonl")
            .lines()
            .enumerate()
        {
            let case: Value = crate::lossless_json::from_str(line).unwrap();
            let plan: Plan = crate::lossless_json::from_value(case["plan"].clone()).unwrap();
            let counter: super::super::search_counter::Counter =
                crate::lossless_json::from_value(case["counter"].clone()).unwrap();
            let mut subject: super::super::search_subject::Subject =
                crate::lossless_json::from_value(case["subject"].clone()).unwrap();
            let mut search = Search::new(
                crate::lossless_json::from_value(case["configuration"].clone()).unwrap(),
            )
            .unwrap();
            for (j, step) in case["steps"].as_array().unwrap().iter().enumerate() {
                search.perform(&step["calls"]).unwrap();
                subject
                    .append(&search.tools(), search.frames.last().unwrap())
                    .unwrap();
                if let Some(messages) = step["replacement"].as_array() {
                    let restored =
                        super::super::input_rewrite::restore(&subject, messages.clone()).unwrap();
                    subject.set_messages(restored);
                }
                assert_eq!(
                    subject.request, step["subject"],
                    "case {i} round {j} complete subject"
                );
                assert_eq!(
                    subject.visible, step["visible"],
                    "case {i} round {j} visible subject"
                );
                assert_eq!(
                    json!(subject.size().unwrap()),
                    step["subject_size"],
                    "case {i} round {j} size"
                );
                assert_eq!(
                    json!(subject.subjects().ok()),
                    step["text_subjects"],
                    "case {i} round {j} subjects"
                );
                assert_eq!(
                    json!(
                        super::super::input_window::window(&subject.request, &case["window"])
                            .unwrap()
                    ),
                    step["window_text"],
                    "case {i} round {j} window"
                );
                let (counted, reserved) = counter
                    .count_messages(
                        &search.tools(),
                        subject.request["messages"].as_array().unwrap(),
                    )
                    .unwrap();
                if case["rewrites"] != true {
                    assert_eq!(
                        counter.count(&search.tools(), &search.frames).unwrap(),
                        (counted, reserved),
                        "case {i} round {j} canonical recount"
                    );
                }
                assert_eq!(
                    json!(counted),
                    step["counted_input"],
                    "case {i} round {j} counted"
                );
                assert_eq!(
                    json!(reserved),
                    step["reserved_input"],
                    "case {i} round {j} reserved"
                );
                let mut rebuilt = plan
                    .render_subject(&search.tools(), &subject.request)
                    .unwrap();
                assert_eq!(
                    json!(subject.recovery_prefix().unwrap()),
                    step["recovery_prefix"],
                    "case {i} round {j} recovery identity"
                );
                let mut expected = step["expected"].clone();
                // Signed bodies may order object members differently; the
                // signer binds the exact rendered bytes at each dispatch.
                for wire in [&mut rebuilt, &mut expected] {
                    if let Some(body) = wire["upstream_body"].as_str() {
                        wire["upstream_payload"] = crate::lossless_json::from_str(body).unwrap();
                        wire["upstream_body"] = Value::Null;
                    }
                }
                assert_eq!(
                    rebuilt, expected,
                    "rebuilt case {i} round {j} {}",
                    plan.dialect
                );
            }
        }
    }
}
