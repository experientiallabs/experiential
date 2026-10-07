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
            {"type": "predicate", "name": "angry", "probability": 0.31},
            {"type": "choice", "name": "intent", "choice": "refund",
                "probabilities": [{"value": "refund", "probability": 1}, {"value": "support", "probability": 0}],
                "confidence": 1},
            {"type": "score", "name": "urgency", "score": 0.7,
                "probabilities": [{"value": 0, "label": "low", "probability": 0.3},
                    {"value": 1, "label": "high", "probability": 0.7}],
                "confidence": 0.7},
        ],
        "usage": {"input_tokens": 275, "input_tokens_details": {"cached_tokens": 0, "cache_write_tokens": 0},
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
}

#[test]
fn openai_answers_that_do_not_match_the_questions_are_malformed() {
    let mut missing = openai_payload();
    missing["answers"].as_array_mut().unwrap().pop();
    let mut unknown_choice = openai_payload();
    unknown_choice["answers"][1]["choice"] = json!("cancel");
    let mut wrong_type = openai_payload();
    wrong_type["answers"][0]["type"] = json!("choice");
    let mut bad_total = openai_payload();
    bad_total["answers"][1]["probabilities"][1]["probability"] = json!(0.5);
    let mut out_of_range = openai_payload();
    out_of_range["answers"][0]["probability"] = json!(1.5);
    let mut not_highest = openai_payload();
    not_highest["answers"][1]["choice"] = json!("support");
    let mut unbilled = openai_payload();
    unbilled["usage"] = json!({"input_tokens": 0, "output_tokens": 0});
    for payload in [
        missing,
        unknown_choice,
        wrong_type,
        bad_total,
        out_of_range,
        not_highest,
        unbilled,
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
