//! OpenAI Decisions API answers on the decisions surface.
//!
//! OpenAI returns `answers` in question order. A question's `name` is
//! optional on both sides, so answers are matched by position, and a name the
//! provider echoes must equal the admitted one. Choice values are typed
//! (`"true"` and `true` are different values), so they are compared as JSON
//! values, never as text. The host may decline one question with a
//! `refusal` answer, which passes through beside the answered questions.

use serde_json::{json, Value};

use super::{
    choice_can_be_highest, decision_usage, malformed, probability, score_matches_distribution,
    token_count, unit_total, wire_failure, DecisionsAdmission,
};
use crate::errors::Failure;
use crate::events::Usage;

/// Validate an OpenAI Decisions API answer against the admitted questions.
///
/// Every admitted question needs exactly one answer at its position, either
/// of the question's type or a `refusal`. Probabilities must be finite,
/// within `[0, 1]`, keyed by exactly the requested values (choice) or levels
/// in order (score), and sum to one within the published-rounding envelope.
/// A choice must name a requested value that can be highest-probability; a
/// score must equal the probability-weighted average of its level values. Answers pass through
/// verbatim in question order with the public alias as `model` and the
/// provider's usage block.
pub(super) fn public_openai_decisions(
    payload: Value,
    admission: &DecisionsAdmission,
) -> Result<(Value, Usage), Failure> {
    let answers = payload
        .get("answers")
        .and_then(Value::as_array)
        .ok_or_else(|| malformed("decision response omitted its answers array"))?;
    if answers.len() != admission.openai_questions.len() {
        return Err(malformed(
            "decision response answer count does not match the request",
        ));
    }
    for (question, answer) in admission.openai_questions.iter().zip(answers) {
        validate_answer(question, answer)?;
    }
    let usage = openai_decision_usage(&payload)?;
    Ok((
        json!({"model": admission.alias, "answers": answers, "usage": payload["usage"]}),
        usage,
    ))
}

fn validate_answer(question: &Value, answer: &Value) -> Result<(), Failure> {
    let kind = question
        .get("type")
        .and_then(Value::as_str)
        .ok_or_else(wire_failure)?;
    let echoed = answer.get("name").filter(|name| !name.is_null());
    if let (Some(asked), Some(echoed)) = (question.get("name"), echoed) {
        if asked != echoed {
            return Err(malformed(
                "decision response question names do not match the request",
            ));
        }
    }
    let answered = answer.get("type").and_then(Value::as_str);
    if answered == Some("refusal") {
        return Ok(());
    }
    if answered != Some(kind) {
        return Err(malformed(
            "decision answer type does not match the question",
        ));
    }
    match kind {
        "predicate" => {
            probability(&answer["probability"])?;
        }
        "choice" => validate_choice(question, answer)?,
        "score" => validate_score(question, answer)?,
        _ => return Err(wire_failure()),
    }
    Ok(())
}

fn validate_choice(question: &Value, answer: &Value) -> Result<(), Failure> {
    let values: Vec<&Value> = question
        .get("choices")
        .and_then(Value::as_array)
        .ok_or_else(wire_failure)?
        .iter()
        .map(|choice| choice.get("value").filter(|value| typed_choice(value)))
        .collect::<Option<_>>()
        .ok_or_else(wire_failure)?;
    let probabilities = typed_distribution(&answer["probabilities"], &values)?;
    let choice = answer
        .get("choice")
        .filter(|value| typed_choice(value))
        .ok_or_else(|| malformed("decision choice is not a requested value"))?;
    let selected = values
        .iter()
        .position(|value| *value == choice)
        .ok_or_else(|| malformed("decision choice is not a requested value"))?;
    probability(&answer["confidence"])?;
    if !choice_can_be_highest(selected, &probabilities) {
        return Err(malformed(
            "decision choice is not a highest-probability value",
        ));
    }
    Ok(())
}

/// A choice value is a string or a boolean, never coerced between the two.
fn typed_choice(value: &Value) -> bool {
    value.is_string() || value.is_boolean()
}

/// Validate a `[{value, probability}]` list covering exactly `values`, returning
/// probabilities in requested order.
fn typed_distribution(value: &Value, values: &[&Value]) -> Result<Vec<f64>, Failure> {
    let entries = value
        .as_array()
        .ok_or_else(|| malformed("decision omitted its probability list"))?;
    if entries.len() != values.len() {
        return Err(malformed(
            "decision probability values do not match the request",
        ));
    }
    let mut ordered: Vec<Option<f64>> = vec![None; values.len()];
    for entry in entries {
        let key = entry
            .get("value")
            .filter(|value| typed_choice(value))
            .ok_or_else(|| malformed("decision probability omitted its value"))?;
        let index = values
            .iter()
            .position(|value| *value == key)
            .ok_or_else(|| malformed("decision probability values do not match the request"))?;
        if ordered[index].is_some() {
            return Err(malformed("decision probability repeats a value"));
        }
        ordered[index] = Some(probability(&entry["probability"])?);
    }
    let probabilities: Vec<f64> = ordered
        .into_iter()
        .map(|value| value.unwrap_or(0.0))
        .collect();
    unit_total(&probabilities)?;
    Ok(probabilities)
}

/// Each score probability names its level's label at that level's position,
/// with consecutive integer values starting at 0 or 1.
fn validate_score(question: &Value, answer: &Value) -> Result<(), Failure> {
    let levels = question
        .get("levels")
        .and_then(Value::as_array)
        .ok_or_else(wire_failure)?;
    let score = answer
        .get("score")
        .and_then(Value::as_f64)
        .filter(|value| value.is_finite())
        .ok_or_else(|| malformed("decision score is not finite numeric data"))?;
    probability(&answer["confidence"])?;
    let entries = answer
        .get("probabilities")
        .and_then(Value::as_array)
        .ok_or_else(|| malformed("decision omitted its probability list"))?;
    if entries.len() != levels.len() {
        return Err(malformed(
            "decision score probabilities do not match the requested levels",
        ));
    }
    let mut probabilities = Vec::with_capacity(entries.len());
    let mut first = None;
    for (position, (entry, level)) in entries.iter().zip(levels).enumerate() {
        if entry.get("label").is_none() || entry.get("label") != level.get("label") {
            return Err(malformed(
                "decision score probabilities do not match the requested levels",
            ));
        }
        let value = entry
            .get("value")
            .and_then(Value::as_i64)
            .ok_or_else(|| malformed("decision score probability omitted its level value"))?;
        let base = *first.get_or_insert(value);
        let offset = i64::try_from(position).map_err(|_| wire_failure())?;
        if !(base == 0 || base == 1) || value != base + offset {
            return Err(malformed(
                "decision score probability values are not the level positions",
            ));
        }
        probabilities.push(probability(&entry["probability"])?);
    }
    unit_total(&probabilities)?;
    // The score is the probability-weighted average of the level values, and
    // values are base + position, so the score less its base is the
    // position-weighted average SystemOne scores already satisfy.
    let base = first.ok_or_else(wire_failure)?;
    if !score_matches_distribution(score - base as f64, &probabilities) {
        return Err(malformed("decision score does not match its probabilities"));
    }
    Ok(())
}

/// The provider's billing evidence, with its cache read and write subsets.
pub(super) fn openai_decision_usage(payload: &Value) -> Result<Usage, Failure> {
    let mut usage = decision_usage(payload)?;
    let details = &payload["usage"]["input_tokens_details"];
    let subset = |key: &str| -> Result<Option<u64>, Failure> {
        match details.get(key) {
            None | Some(Value::Null) => Ok(None),
            Some(value) => token_count(value).map(Some),
        }
    };
    let cached = subset("cached_tokens")?;
    let written = subset("cache_write_tokens")?;
    let input = usage.input_tokens.unwrap_or(0);
    if cached.unwrap_or(0).saturating_add(written.unwrap_or(0)) > input {
        return Err(malformed(
            "decision cached token counts exceed its input tokens",
        ));
    }
    usage.cached_input_tokens = cached;
    usage.cache_creation_input_tokens = written;
    Ok(usage)
}
