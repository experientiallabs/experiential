//! Google Decisions API answers, translated back to the OpenAI Decisions API
//! shape the caller sent on `/v1/decisions`.
//!
//! Admission (`google_decisions_contracts.py`) keyed the questions
//! positionally (`q0`, `q1`, ...), spelled each requested choice value as its
//! Google option key (a boolean as `true`/`false`, a string as itself) and
//! each score level as `label: description` (or the bare label). This module
//! inverts exactly those keys: Google's `answers` map comes back as the
//! OpenAI answer list in question order, with the typed requested values and
//! the admitted labels, and Google's input-only usage as OpenAI's `usage`.

use serde_json::{json, Map, Value};

use super::{
    choice_can_be_highest, distribution, malformed, probability, score_matches_distribution,
    score_mismatch, token_count, wire_failure, DecisionsAdmission,
};
use crate::errors::Failure;
use crate::events::Usage;

/// The positional Google question key admission gave the question at `index`.
fn question_key(index: usize) -> String {
    format!("q{index}")
}

/// The Google option key of one requested choice value, as admission spelled it.
fn option_key(value: &Value) -> Option<String> {
    match value {
        Value::String(text) => Some(text.clone()),
        Value::Bool(flag) => Some(flag.to_string()),
        _ => None,
    }
}

/// The criterion text admission gave one ordered score level.
fn level_criterion(level: &Value) -> Option<String> {
    let label = level.get("label")?.as_str()?;
    Some(match level.get("description").and_then(Value::as_str) {
        Some(description) if !description.is_empty() => format!("{label}: {description}"),
        _ => label.to_string(),
    })
}

/// Validate a Google Decisions API answer against the admitted questions and
/// return it in the OpenAI shape.
///
/// Every admitted question needs its positional key answered with the
/// question's Google type. Probabilities must be finite, within `[0, 1]`,
/// keyed by exactly the requested option keys (choice) or level indices
/// (score), and sum to one within the published-rounding envelope; a score's
/// legend must be the admitted criteria and the score the probability-weighted
/// level average. Usage is Google's `total_input_tokens` with zero output.
pub(super) fn public_google_decisions(
    payload: Value,
    admission: &DecisionsAdmission,
) -> Result<(Value, Usage), Failure> {
    let answers = payload
        .get("answers")
        .and_then(Value::as_object)
        .ok_or_else(|| malformed("decision response omitted its answers object"))?;
    if answers.len() != admission.openai_questions.len() {
        return Err(malformed(
            "decision response answer count does not match the request",
        ));
    }
    let mut public = Vec::with_capacity(answers.len());
    for (index, question) in admission.openai_questions.iter().enumerate() {
        let answer = answers
            .get(&question_key(index))
            .ok_or_else(|| malformed("decision response question keys do not match the request"))?;
        public.push(translate_answer(question, answer)?);
    }
    let usage = google_decision_usage(&payload)?;
    let input = usage.input_tokens.unwrap_or(0);
    Ok((
        json!({
            "model": admission.alias,
            "answers": public,
            "usage": {"input_tokens": input, "output_tokens": 0, "total_tokens": input},
        }),
        usage,
    ))
}

fn translate_answer(question: &Value, answer: &Value) -> Result<Value, Failure> {
    let kind = question
        .get("type")
        .and_then(Value::as_str)
        .ok_or_else(wire_failure)?;
    let expected = match kind {
        "predicate" => "noul",
        "choice" => "choice",
        "score" => "score",
        _ => return Err(wire_failure()),
    };
    if answer.get("type").and_then(Value::as_str) != Some(expected) {
        return Err(malformed(
            "decision answer type does not match the question",
        ));
    }
    let mut public = Map::new();
    public.insert("type".to_string(), json!(kind));
    if let Some(name) = question.get("name").filter(|name| !name.is_null()) {
        public.insert("name".to_string(), name.clone());
    }
    match kind {
        "predicate" => {
            probability(&answer["noul"])?;
            public.insert("probability".to_string(), answer["noul"].clone());
        }
        "choice" => translate_choice(question, answer, &mut public)?,
        _ => translate_score(question, answer, &mut public)?,
    }
    Ok(Value::Object(public))
}

fn translate_choice(
    question: &Value,
    answer: &Value,
    public: &mut Map<String, Value>,
) -> Result<(), Failure> {
    let values: Vec<&Value> = question
        .get("choices")
        .and_then(Value::as_array)
        .ok_or_else(wire_failure)?
        .iter()
        .map(|choice| choice.get("value"))
        .collect::<Option<_>>()
        .ok_or_else(wire_failure)?;
    let keys: Vec<String> = values
        .iter()
        .map(|value| option_key(value))
        .collect::<Option<_>>()
        .ok_or_else(wire_failure)?;
    let probabilities = distribution(&answer["probabilities"], keys.iter().map(String::as_str))?;
    let chosen = answer
        .get("choice")
        .and_then(Value::as_str)
        .ok_or_else(|| malformed("decision choice is not a requested value"))?;
    let selected = keys
        .iter()
        .position(|key| key == chosen)
        .ok_or_else(|| malformed("decision choice is not a requested value"))?;
    probability(&answer["confidence"])?;
    if !choice_can_be_highest(selected, &probabilities) {
        return Err(malformed(
            "decision choice is not a highest-probability value",
        ));
    }
    public.insert("choice".to_string(), values[selected].clone());
    public.insert(
        "probabilities".to_string(),
        Value::Array(
            values
                .iter()
                .zip(&keys)
                .map(|(value, key)| json!({"value": value, "probability": answer["probabilities"][key]}))
                .collect(),
        ),
    );
    public.insert("confidence".to_string(), answer["confidence"].clone());
    Ok(())
}

fn translate_score(
    question: &Value,
    answer: &Value,
    public: &mut Map<String, Value>,
) -> Result<(), Failure> {
    let levels = question
        .get("levels")
        .and_then(Value::as_array)
        .ok_or_else(wire_failure)?;
    let legend: Map<String, Value> = levels
        .iter()
        .enumerate()
        .map(|(index, level)| level_criterion(level).map(|text| (index.to_string(), json!(text))))
        .collect::<Option<_>>()
        .ok_or_else(wire_failure)?;
    if answer.get("legend").and_then(Value::as_object) != Some(&legend) {
        return Err(malformed(
            "decision score legend does not match the requested levels",
        ));
    }
    let probabilities = distribution(&answer["probabilities"], legend.keys().map(String::as_str))?;
    probability(&answer["confidence"])?;
    let score = answer
        .get("score")
        .and_then(Value::as_f64)
        .filter(|value| value.is_finite())
        .ok_or_else(|| malformed("decision score is not finite numeric data"))?;
    if score < 0.0
        || score > levels.len().saturating_sub(1) as f64
        || !score_matches_distribution(score, &probabilities)
    {
        return Err(score_mismatch(
            "decision score does not match its probability distribution",
            score,
            &probabilities,
        ));
    }
    public.insert("score".to_string(), answer["score"].clone());
    public.insert(
        "probabilities".to_string(),
        Value::Array(
            levels
                .iter()
                .enumerate()
                .map(|(index, level)| {
                    json!({"value": index, "label": level["label"],
                        "probability": answer["probabilities"][index.to_string()]})
                })
                .collect(),
        ),
    );
    public.insert("confidence".to_string(), answer["confidence"].clone());
    Ok(())
}

/// Google's billing evidence: `usage.total_input_tokens` (text, image and
/// audio tokens together) and `usage.total_output_tokens`, documented as
/// always zero. No cache legs exist on this wire.
pub(super) fn google_decision_usage(payload: &Value) -> Result<Usage, Failure> {
    let input_tokens = token_count(&payload["usage"]["total_input_tokens"])?;
    let output_tokens = token_count(&payload["usage"]["total_output_tokens"])?;
    if input_tokens == 0 && output_tokens == 0 {
        return Err(malformed(
            "decision response reported no billable token usage",
        ));
    }
    Ok(Usage {
        input_tokens: Some(input_tokens),
        output_tokens: Some(output_tokens),
        cached_input_tokens: None,
        cache_creation_input_tokens: None,
        cache_creation_1h_input_tokens: None,
        cache_creation_12h_input_tokens: None,
        reasoning_tokens: None,
        billed_units: None,
        cache_writes_read_back: false,
        image_output_tokens: None,
    })
}
