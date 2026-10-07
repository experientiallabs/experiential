//! OpenAI Decisions API wire: answer validation and admission decoding.

use serde_json::json;

use super::*;

fn openai_admission() -> DecisionsAdmission {
    DecisionsAdmission {
        wire: DecisionWire::Openai,
        questions: Map::new(),
        openai_questions: vec![
            json!({"type": "choice", "name": "intent", "instructions": "What does the customer want?",
                "choices": [{"value": "refund", "description": "wants money back"},
                    {"value": "support", "description": "needs help"}]}),
            json!({"type": "predicate", "name": "angry", "instructions": "Is the customer angry?"}),
            json!({"type": "score", "name": "urgency", "instructions": "How urgent?",
                "levels": [{"label": "low", "description": "can wait"},
                    {"label": "high", "description": "now"}]}),
        ],
        request_id: "request-1".into(),
        alias: "gpt-6-luna-decisions".into(),
        maximum_total_attempts: 1,
        maximum_same_deployment_attempts: 1,
        ..Default::default()
    }
}

fn openai_payload() -> Value {
    json!({
        "model": "gpt-6-luna",
        "answers": [
            {"type": "choice", "name": "intent", "choice": "refund",
                "probabilities": [{"value": "refund", "probability": 1}, {"value": "support", "probability": 0}],
                "confidence": 1},
            {"type": "predicate", "name": "angry", "probability": 0.31},
            {"type": "score", "name": "urgency", "score": 0.7,
                "probabilities": [{"value": 0, "label": "low", "probability": 0.3},
                    {"value": 1, "label": "high", "probability": 0.7}],
                "confidence": 0.7},
        ],
        "usage": {"input_tokens": 275, "input_tokens_details": {"cached_tokens": 64, "cache_write_tokens": 128},
            "output_tokens": 0, "output_tokens_details": {"reasoning_tokens": 0}, "total_tokens": 275},
    })
}

#[test]
fn openai_answer_passes_through_in_question_order_with_alias_and_usage() {
    let (body, usage) = public_openai_decisions(openai_payload(), &openai_admission()).unwrap();
    assert_eq!(body["model"], "gpt-6-luna-decisions");
    let names: Vec<&str> = body["answers"]
        .as_array()
        .unwrap()
        .iter()
        .map(|answer| answer["name"].as_str().unwrap())
        .collect();
    assert_eq!(names, ["intent", "angry", "urgency"]);
    assert_eq!(body["answers"][0]["choice"], "refund");
    assert_eq!(body["usage"]["input_tokens"], 275);
    assert_eq!(usage.input_tokens, Some(275));
    assert_eq!(usage.output_tokens, Some(0));
    assert_eq!(usage.cached_input_tokens, Some(64));
    assert_eq!(usage.cache_creation_input_tokens, Some(128));
}

#[test]
fn openai_answers_that_do_not_match_the_questions_are_malformed() {
    let mut missing = openai_payload();
    missing["answers"].as_array_mut().unwrap().pop();
    let mut unknown_choice = openai_payload();
    unknown_choice["answers"][0]["choice"] = json!("cancel");
    let mut wrong_type = openai_payload();
    wrong_type["answers"][1]["type"] = json!("choice");
    let mut bad_total = openai_payload();
    bad_total["answers"][0]["probabilities"][1]["probability"] = json!(0.5);
    let mut out_of_range = openai_payload();
    out_of_range["answers"][1]["probability"] = json!(1.5);
    let mut not_highest = openai_payload();
    not_highest["answers"][0]["choice"] = json!("support");
    let mut unbilled = openai_payload();
    unbilled["usage"] = json!({"input_tokens": 0, "output_tokens": 0});
    let mut out_of_order = openai_payload();
    out_of_order["answers"].as_array_mut().unwrap().swap(0, 1);
    let mut renamed = openai_payload();
    renamed["answers"][1]["name"] = json!("calm");
    let mut stringly_boolean = openai_payload();
    stringly_boolean["answers"][0]["choice"] = json!(true);
    let mut wrong_label = openai_payload();
    wrong_label["answers"][2]["probabilities"][0]["label"] = json!("high");
    let mut wrong_index = openai_payload();
    wrong_index["answers"][2]["probabilities"][1]["value"] = json!(5);
    let mut score_outside = openai_payload();
    score_outside["answers"][2]["score"] = json!(3.0);
    let mut score_mismatch = openai_payload();
    score_mismatch["answers"][2]["score"] = json!(0.1);
    let mut overcached = openai_payload();
    overcached["usage"]["input_tokens_details"]["cached_tokens"] = json!(275);
    for payload in [
        missing,
        unknown_choice,
        wrong_type,
        bad_total,
        out_of_range,
        not_highest,
        unbilled,
        out_of_order,
        renamed,
        stringly_boolean,
        wrong_label,
        wrong_index,
        score_outside,
        score_mismatch,
        overcached,
    ] {
        let failure = public_openai_decisions(payload, &openai_admission()).expect_err("malformed");
        assert_eq!(failure.failure_class, FailureClass::MalformedResponse);
        assert!(!failure.retryable_same_deployment);
    }
}

#[test]
fn openai_wire_admission_deserializes_its_question_list() {
    let admission: DecisionsAdmission = serde_json::from_value(json!({
        "request_id": "request-1", "alias": "a", "alias_revision_id": "r", "exact_model_id": "m",
        "route_reason": "direct", "route": [], "wire": "openai",
        "openai_questions": [{"type": "predicate", "name": "p", "instructions": "i"}],
        "maximum_total_attempts": 1, "maximum_same_deployment_attempts": 1,
    }))
    .unwrap();
    assert_eq!(admission.wire, DecisionWire::Openai);
    assert_eq!(admission.openai_questions.len(), 1);
    assert!(admission.questions.is_empty());
}

#[test]
fn openai_refusals_and_unnamed_questions_pass_through_positionally() {
    let mut admission = openai_admission();
    admission.openai_questions[1] =
        json!({"type": "predicate", "instructions": "Is the customer angry?"});
    let mut payload = openai_payload();
    payload["answers"][1] = json!({"type": "predicate", "probability": 0.2});
    payload["answers"][2] = json!({"type": "refusal", "name": "urgency"});
    let (body, _usage) = public_openai_decisions(payload, &admission).unwrap();
    assert_eq!(
        body["answers"][1],
        json!({"type": "predicate", "probability": 0.2})
    );
    assert_eq!(body["answers"][2]["type"], "refusal");
}

#[test]
fn openai_boolean_choice_values_stay_typed() {
    let mut admission = openai_admission();
    admission.openai_questions[0] = json!({"type": "choice", "name": "intent",
        "instructions": "Refund?", "choices": [{"value": true}, {"value": "true"}]});
    let mut payload = openai_payload();
    payload["answers"][0] = json!({"type": "choice", "name": "intent", "choice": true,
        "probabilities": [{"value": "true", "probability": 0.25}, {"value": true, "probability": 0.75}],
        "confidence": 0.75});
    let (body, _usage) = public_openai_decisions(payload.clone(), &admission).unwrap();
    assert_eq!(body["answers"][0]["choice"], json!(true));
    payload["answers"][0]["choice"] = json!("true");
    let failure = public_openai_decisions(payload, &admission).expect_err("not highest");
    assert_eq!(failure.failure_class, FailureClass::MalformedResponse);
}

#[test]
fn openai_one_based_score_values_match_their_weighted_average() {
    let mut payload = openai_payload();
    payload["answers"][2]["score"] = json!(1.7);
    payload["answers"][2]["probabilities"][0]["value"] = json!(1);
    payload["answers"][2]["probabilities"][1]["value"] = json!(2);
    public_openai_decisions(payload, &openai_admission()).unwrap();
}

#[test]
fn an_admission_without_its_wire_is_refused_as_drift() {
    let missing = serde_json::from_value::<DecisionsAdmission>(json!({
        "request_id": "request-1", "alias": "a", "alias_revision_id": "r", "exact_model_id": "m",
        "route_reason": "direct", "route": [], "questions": {},
        "maximum_total_attempts": 1, "maximum_same_deployment_attempts": 1,
    }));
    assert!(missing.is_err());
}
