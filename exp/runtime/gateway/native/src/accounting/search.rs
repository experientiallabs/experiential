//! Request-local deferred-tool search with bounded documents and stable ordering.

use std::collections::{BTreeSet, HashMap};

use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::guardrails::syntax::with_flags;

const STOPWORDS: &str = "a an the and or of to in on for with is are be was were do does did what which who whom whose how when where why i me my we our you your it its this that these those there here can could should would will shall may might must please want need use using";
const PATTERN_ERROR: &str = "invalid pattern: RE2 syntax only (no backreferences or lookaround)";

#[derive(Clone, PartialEq, Deserialize, Serialize)]
pub(super) struct Tool {
    #[serde(default, skip_serializing)]
    pub key: String,
    pub name: String,
    pub description: Option<String>,
    #[serde(deserialize_with = "crate::lossless_json::deserialize")]
    pub parameters: Value,
    #[serde(flatten, deserialize_with = "crate::lossless_json::deserialize")]
    pub extra: serde_json::Map<String, Value>,
}

/// Split ASCII camel boundaries before Unicode lowercase, matching admission.
fn tokenize(text: &str) -> Vec<String> {
    let mut spaced = String::with_capacity(text.len());
    let mut previous = ' ';
    for c in text.chars() {
        let c = if matches!(c, '_' | '-' | '.') { ' ' } else { c };
        if (previous.is_ascii_lowercase() || previous.is_ascii_digit()) && c.is_ascii_uppercase() {
            spaced.push(' ');
        }
        spaced.push(c);
        previous = c;
    }
    spaced
        .to_lowercase()
        .split(|c: char| !c.is_ascii_lowercase() && !c.is_ascii_digit())
        .filter(|s| !s.is_empty())
        .map(str::to_owned)
        .collect()
}

fn document(tool: &Tool) -> String {
    let mut parts = vec![tool.name.as_str()];
    if let Some(description) = tool.description.as_deref().filter(|s| !s.is_empty()) {
        parts.push(description);
    }
    if let Some(properties) = tool.parameters.get("properties").and_then(Value::as_object) {
        parts.extend(properties.keys().map(String::as_str));
        for value in properties.values() {
            if let Some(description) = value.get("description").and_then(Value::as_str) {
                parts.push(description);
            }
        }
    }
    parts.join(" ").chars().take(4000).collect()
}

pub(super) fn limit(value: &Value, default: usize) -> usize {
    if let Some(n) = value.as_i64() {
        return n.clamp(1, 10) as usize;
    }
    if value.as_u64().is_some() {
        return 10;
    }
    default.clamp(1, 10)
}

pub(super) fn bm25(query: &str, tools: &[Tool], limit: usize) -> Vec<usize> {
    let stopwords: BTreeSet<_> = STOPWORDS.split_whitespace().collect();
    let terms: BTreeSet<_> = tokenize(&query.chars().take(500).collect::<String>())
        .into_iter()
        .filter(|term| !stopwords.contains(term.as_str()))
        .collect();
    if terms.is_empty() || tools.is_empty() {
        return Vec::new();
    }
    let mut lengths = Vec::with_capacity(tools.len());
    let frequencies: Vec<HashMap<String, usize>> = tools
        .iter()
        .map(|tool| {
            let words = tokenize(&document(tool));
            lengths.push(words.len());
            let mut frequencies = HashMap::new();
            for term in words {
                *frequencies.entry(term).or_default() += 1;
            }
            frequencies
        })
        .collect();
    let average = (lengths.iter().sum::<usize>() as f64 / tools.len() as f64).max(1.0);
    let counts: Vec<_> = terms
        .iter()
        .map(|term| {
            let count = frequencies.iter().filter(|f| f.contains_key(term)).count();
            (
                term,
                (1.0 + (tools.len() as f64 - count as f64 + 0.5) / (count as f64 + 0.5)).ln(),
            )
        })
        .collect();
    let mut scored: Vec<_> = frequencies
        .iter()
        .enumerate()
        .filter_map(|(index, frequencies)| {
            let mut score = 0.0;
            for (term, idf) in &counts {
                let frequency = *frequencies.get(*term).unwrap_or(&0) as f64;
                if frequency == 0.0 {
                    continue;
                }
                let normalized = frequency * (1.2 + 1.0);
                let denominator =
                    frequency + 1.2 * (1.0 - 0.75 + 0.75 * lengths[index] as f64 / average);
                score += idf * normalized / denominator;
            }
            (score > 0.0).then_some((score, index))
        })
        .collect();
    scored.sort_by(|a, b| b.0.total_cmp(&a.0).then(a.1.cmp(&b.1)));
    scored
        .into_iter()
        .take(limit)
        .map(|(_, index)| index)
        .collect()
}

pub(super) fn pattern(
    pattern: &str,
    tools: &[Tool],
    limit: usize,
) -> Result<Vec<usize>, &'static str> {
    if pattern.is_empty() || pattern.chars().count() > 200 {
        return Err("pattern must be 1-200 characters");
    }
    let translated = with_flags(pattern, true).map_err(|_| PATTERN_ERROR)?;
    let expression = regex::bytes::RegexBuilder::new(&translated)
        .case_insensitive(true)
        .size_limit(262_144)
        .build()
        .map_err(|_| PATTERN_ERROR)?;
    Ok(tools
        .iter()
        .enumerate()
        .filter_map(|(i, tool)| {
            let text: String = format!(
                "{}\n{}",
                tool.name,
                tool.description.as_deref().unwrap_or("")
            )
            .chars()
            .take(4000)
            .collect();
            expression.is_match(text.as_bytes()).then_some(i)
        })
        .take(limit)
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn search_results_match_python() {
        for (i, line) in include_str!("../../tests/generated/search.jsonl")
            .lines()
            .enumerate()
        {
            let case: Value = crate::lossless_json::from_str(line).unwrap();
            let tools: Vec<Tool> = crate::lossless_json::from_value(case["tools"].clone()).unwrap();
            let limit = limit(&case["limit"], case["default"].as_u64().unwrap() as usize);
            let actual = match case["pattern"].as_str() {
                Some(p) => pattern(p, &tools, limit)
                    .map(|indices| json!(indices))
                    .unwrap_or_else(|error| json!({"error": error})),
                None => json!(bm25(case["query"].as_str().unwrap(), &tools, limit)),
            };
            assert_eq!(actual, case["expected"], "case {i}");
        }
    }
}
