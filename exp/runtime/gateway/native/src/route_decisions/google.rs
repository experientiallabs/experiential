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
    score_mismatch, token_count, wire_failure, DecisionsAdmission, PROBABILITY_TOLERANCE,
};
use crate::errors::{Failure, FailureClass};
use crate::events::Usage;

/// Confidence is the winning outcome's probability (`max(noul, 1 - noul)` for
/// a yes/no answer), rounded independently to the same hundredths, so it may
/// sit half a hundredth from the bin it restates.
const CONFIDENCE_TOLERANCE: f64 = 0.005 + PROBABILITY_TOLERANCE;

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

/// Google's input content filters answered instead of the model: a
/// `prompt_feedback.block_reason` with no answers. The reason is Google's own
/// bounded token (`safety`, `blocklist`, `prohibited_content`, `image_safety`,
/// `other`), never its free-text message.
fn blocked(reason: &str) -> Failure {
    let token = if !reason.is_empty()
        && reason.len() <= 32
        && reason
            .chars()
            .all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == '_')
    {
        reason
    } else {
        "unspecified"
    };
    Failure::new(
        FailureClass::Refusal,
        &format!("the decision model's input filter blocked the request ({token})"),
    )
    .with_retry(false, false)
}

/// Confidence restates the winning outcome's probability.
fn consistent_confidence(confidence: f64, expected: f64) -> Result<(), Failure> {
    if (confidence - expected).abs() > CONFIDENCE_TOLERANCE {
        return Err(malformed(
            "decision confidence does not match its winning probability",
        ));
    }
    Ok(())
}

/// Validate a Google Decisions API answer against the admitted questions and
/// return it in the OpenAI shape.
///
/// A `prompt_feedback.block_reason` is the model's refusal of the whole input.
/// Otherwise every admitted question needs its positional key answered with
/// the question's Google type. Probabilities must be finite, within `[0, 1]`,
/// keyed by exactly the requested option keys (choice) or level indices
/// (score), and sum to one within the published-rounding envelope; confidence
/// must restate the winning probability; a score's legend must be the admitted
/// criteria and the score the probability-weighted level average. Usage is
/// Google's `total_input_tokens` with zero output.
pub(super) fn public_google_decisions(
    payload: Value,
    admission: &DecisionsAdmission,
) -> Result<(Value, Usage), Failure> {
    if let Some(reason) = payload
        .get("prompt_feedback")
        .and_then(|feedback| feedback.get("block_reason"))
        .filter(|reason| !reason.is_null())
    {
        return Err(blocked(reason.as_str().unwrap_or_default()));
    }
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
            // Google's contract clamps noul to [0.01, 0.99]; an endpoint
            // value is a shape the model never emits.
            let noul = probability(&answer["noul"])?;
            if !(0.01..=0.99).contains(&noul) {
                return Err(malformed(
                    "decision noul probability is outside Google's 0.01 to 0.99 range",
                ));
            }
            consistent_confidence(probability(&answer["confidence"])?, noul.max(1.0 - noul))?;
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
    consistent_confidence(probability(&answer["confidence"])?, probabilities[selected])?;
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
    let highest = probabilities.iter().copied().fold(0.0_f64, f64::max);
    consistent_confidence(probability(&answer["confidence"])?, highest)?;
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
/// always zero and, on Google's shared usage contract, optional: an absent
/// count is zero. No cache legs exist on this wire.
pub(super) fn google_decision_usage(payload: &Value) -> Result<Usage, Failure> {
    let input_tokens = token_count(&payload["usage"]["total_input_tokens"])?;
    let output_tokens = match &payload["usage"]["total_output_tokens"] {
        Value::Null => 0,
        reported => token_count(reported)?,
    };
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
