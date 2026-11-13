//! Ordered tool-search rounds and the conversation frames they add.

use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

use crate::errors::PublicError;

use super::{json_text, search};

#[derive(Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct Configuration {
    pub mode: Mode,
    pub tool_name: String,
    pub declared_name: String,
    pub max_rounds: usize,
    pub default_limit: usize,
    pub loaded: Vec<search::Tool>,
    pub deferred: Vec<search::Tool>,
    pub gateway_tool: search::Tool,
}

#[derive(Clone, Copy, Deserialize)]
#[serde(rename_all = "lowercase")]
pub(super) enum Mode {
    Any,
    Regex,
    Bm25,
}

#[derive(Clone, Serialize)]
pub(super) struct Call {
    pub call_id: String,
    pub name: String,
    pub arguments: Value,
    pub raw_arguments: String,
}

#[derive(Clone, Serialize)]
pub(super) struct Frame {
    pub calls: Vec<Call>,
    pub results: Vec<String>,
}

#[derive(Clone)]
pub(super) struct Search {
    config: Configuration,
    pub frames: Vec<Frame>,
}

impl Search {
    pub fn new(config: Configuration) -> Result<Self, PublicError> {
        if !(1..=8).contains(&config.max_rounds)
            || !(1..=10).contains(&config.default_limit)
            || config.tool_name.is_empty()
            || config.tool_name != config.gateway_tool.name
            || config.declared_name.is_empty()
        {
            return Err(PublicError::internal());
        }
        Ok(Self {
            config,
            frames: Vec::new(),
        })
    }

    pub fn exhausted(&self) -> bool {
        self.frames.len() >= self.config.max_rounds
    }

    pub fn tools(&self) -> Vec<search::Tool> {
        let mut tools = self.config.loaded.clone();
        if !self.exhausted() && !self.config.deferred.is_empty() {
            tools.push(self.config.gateway_tool.clone());
        }
        tools
    }

    pub fn perform(&mut self, raw: &Value) -> Result<Vec<Value>, PublicError> {
        if self.exhausted() {
            return Err(PublicError::internal());
        }
        let mut calls = Vec::new();
        for value in raw.as_array().into_iter().flatten() {
            let Some(id) = value["call_id"].as_str().filter(|s| !s.is_empty()) else {
                continue;
            };
            if !value["name"].is_string() {
                continue;
            }
            let raw = value["arguments"]
                .as_str()
                .filter(|s| !s.is_empty())
                .unwrap_or("{}");
            let arguments = crate::lossless_json::from_str::<Value>(raw)
                .ok()
                .filter(Value::is_object)
                .ok_or_else(PublicError::internal)?;
            calls.push(Call {
                call_id: id.to_owned(),
                name: self.config.tool_name.clone(),
                arguments,
                raw_arguments: raw.to_owned(),
            });
        }
        if calls.is_empty()
            || calls.len() > crate::tool_search::MAXIMUM_WITHHELD_SEARCH_CALLS
            || calls
                .iter()
                .any(|c| c.call_id.chars().count() > 65_536 || c.name.chars().count() > 256)
            || calls
                .iter()
                .map(|c| &c.call_id)
                .collect::<std::collections::HashSet<_>>()
                .len()
                != calls.len()
            || calls.iter().map(|c| c.raw_arguments.len()).sum::<usize>()
                > crate::tool_search::MAXIMUM_WITHHELD_SEARCH_BYTES
        {
            return Err(PublicError::internal());
        }
        let mut results = Vec::new();
        let mut rounds = Vec::new();
        for call in &calls {
            let (indices, query, pattern, error) = self.find(call);
            let matched: Vec<_> = indices
                .iter()
                .map(|i| self.config.deferred[*i].clone())
                .collect();
            for tool in &matched {
                // Python list.remove keeps the surviving declaration order.
                if let Some(i) = self.config.deferred.iter().position(|t| t == tool) {
                    self.config.deferred.remove(i);
                    self.config.loaded.push(tool.clone());
                }
            }
            let content = if let Some(error) = error {
                json!({"error": error, "matched": []})
            } else {
                json!({"matched": matched.iter().map(|t| json!({"name": t.name,
                    "description": t.description.as_deref().unwrap_or("")})).collect::<Vec<_>>(),
                    "loaded": !matched.is_empty(), "note": if matched.is_empty() {
                        "No deferred tool matched; answer with the tools you have or search again."
                    } else { "The matched tools are now loaded and can be called directly." } })
            };
            results.push(json_text::dumps(&content, true)?);
            rounds.push(json!({"call_id": call.call_id, "query": query, "pattern": pattern,
                "declared_name": self.config.declared_name, "matched": matched.iter().map(|t| &t.name).collect::<Vec<_>>(),
                "matched_tools": matched.iter().map(|t| {
                    let mut value = json!({"type": "function", "name": t.name});
                    if let Some(description) = &t.description { value["description"] = json!(description); }
                    value["parameters"] = t.parameters.clone(); value
                }).collect::<Vec<_>>() }));
        }
        self.frames.push(Frame { calls, results });
        Ok(rounds)
    }

    fn find(
        &self,
        call: &Call,
    ) -> (
        Vec<usize>,
        Option<String>,
        Option<String>,
        Option<&'static str>,
    ) {
        let limit = argument_limit(call, self.config.default_limit);
        let query: Option<String> = call.arguments["query"]
            .as_str()
            .map(|s| s.chars().take(500).collect());
        let pattern = call.arguments["pattern"].as_str().map(str::to_owned);
        if let Some(p) = pattern.as_deref().filter(|s| !s.is_empty()) {
            if matches!(self.config.mode, Mode::Regex | Mode::Any) {
                return match search::pattern(p, &self.config.deferred, limit) {
                    Ok(indices) => (indices, None, pattern, None),
                    Err(error) => (Vec::new(), None, pattern, Some(error)),
                };
            }
        }
        if let Some(q) = query.as_deref().filter(|s| !s.is_empty()) {
            if matches!(self.config.mode, Mode::Bm25 | Mode::Any) {
                return (
                    search::bm25(q, &self.config.deferred, limit),
                    query,
                    None,
                    None,
                );
            }
        }
        (
            Vec::new(),
            query,
            pattern,
            Some("provide a natural-language `query` or a regular-expression `pattern`"),
        )
    }
}

fn argument_limit(call: &Call, default: usize) -> usize {
    // A JSON integer keeps integer semantics even beyond the engine's u64
    // number representation; scientific notation and booleans use the default.
    if let Ok(object) = crate::lossless_json::from_str::<
        std::collections::HashMap<String, Box<serde_json::value::RawValue>>,
    >(&call.raw_arguments)
    {
        if let Some(value) = object.get("limit") {
            let raw = value.get();
            let digits = raw.strip_prefix('-').unwrap_or(raw);
            if !digits.is_empty() && digits.bytes().all(|b| b.is_ascii_digit()) {
                if raw.starts_with('-') {
                    return 1;
                }
                return digits.parse::<usize>().unwrap_or(10).clamp(1, 10);
            }
        }
    }
    search::limit(&call.arguments["limit"], default)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn evolving_rounds_match_python_results_and_loaded_order() {
        for (i, line) in include_str!("../../tests/generated/search_rounds.jsonl")
            .lines()
            .enumerate()
        {
            let case: Value = crate::lossless_json::from_str(line).unwrap();
            let mut search = Search::new(
                crate::lossless_json::from_value(case["configuration"].clone()).unwrap(),
            )
            .unwrap();
            for (j, step) in case["steps"].as_array().unwrap().iter().enumerate() {
                if step["error"] == true {
                    assert!(
                        search.perform(&step["calls"]).is_err(),
                        "case {i} round {j} rejection"
                    );
                    break;
                }
                let rounds = search.perform(&step["calls"]).unwrap();
                assert_eq!(json!(rounds), step["rounds"], "case {i} round {j} facts");
                assert_eq!(
                    json!(search.frames.last().unwrap()),
                    step["frame"],
                    "case {i} round {j} conversation"
                );
                assert_eq!(
                    json!(search.tools()),
                    step["tools"],
                    "case {i} round {j} tools"
                );
                assert_eq!(search.exhausted(), step["exhausted"].as_bool().unwrap());
            }
            if case["steps"].as_array().unwrap().last().unwrap()["error"] != true {
                assert!(search.perform(&json!([])).is_err());
            }
        }
    }
}

#[cfg(test)]
mod bounds_tests {
    use super::*;

    #[test]
    fn oversized_search_never_executes_or_exposes_the_query() {
        let config: Configuration = crate::lossless_json::from_value(json!({
            "mode":"any", "tool_name":"tool_search", "declared_name":"tool_search",
            "max_rounds":2,"default_limit":1,"loaded":[],"deferred":[],
            "gateway_tool":{"name":"tool_search","description":null,"parameters":{}}
        }))
        .unwrap();
        let mut search = Search::new(config).unwrap();
        let calls = json!([{"call_id":"id","name":"tool_search",
            "arguments": json!({"query":"private-marker".repeat(350_000)}).to_string()}]);
        let error = search.perform(&calls).unwrap_err();
        assert!(!error.message.contains("private-marker"));
        assert!(search.frames.is_empty());
    }
}
