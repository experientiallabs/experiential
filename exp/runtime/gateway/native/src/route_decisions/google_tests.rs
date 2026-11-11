//! Google Decisions API wire: answer translation back to the OpenAI shape.

use std::time::{Duration, Instant};

use serde_json::json;

use super::tests::{block_on, calls, guarded, response};
use super::*;

fn google_admission() -> DecisionsAdmission {
    DecisionsAdmission {
        wire: DecisionWire::Google,
        questions: Map::new(),
        openai_questions: vec![
            json!({"type": "choice", "name": "intent", "instructions": "What does the customer want?",
                "choices": [{"value": "refund", "description": "wants money back"},
                    {"value": "support", "description": "needs help"}, {"value": true}]}),
            json!({"type": "predicate", "name": "angry", "instructions": "Is the customer angry?"}),
            json!({"type": "score", "instructions": "How urgent?",
                "levels": [{"label": "low", "description": "can wait"}, {"label": "high"}]}),
        ],
        request_id: "request-1".into(),
        alias: "gemini-decision-1".into(),
        maximum_total_attempts: 1,
        maximum_same_deployment_attempts: 1,
        ..Default::default()
    }
}

/// A recorded answer in Google's documented shape (keys sorted as Google
/// sorts option keys; hundredth-rounded probabilities).
fn google_payload() -> Value {
    json!({
        "answers": {
            "q0": {"type": "choice", "choice": "refund",
                "probabilities": {"refund": 0.9, "support": 0.07, "true": 0.03},
                "confidence": 0.9},
            "q1": {"type": "noul", "noul": 0.31, "confidence": 0.69},
            "q2": {"type": "score", "score": 0.7,
                "legend": {"0": "low: can wait", "1": "high"},
                "probabilities": {"0": 0.3, "1": 0.7}, "confidence": 0.7},
        },
        "usage": {
            "total_input_tokens": 324,
            "input_tokens_by_modality": [{"modality": "text", "tokens": 324}],
            "total_output_tokens": 0,
            "total_tokens": 324,
        },
    })
}

#[test]
fn google_answers_come_back_in_the_openai_shape_in_question_order() {
    let (body, usage) = public_google_decisions(google_payload(), &google_admission()).unwrap();
    assert_eq!(body["model"], "gemini-decision-1");
    assert_eq!(
        body["answers"],
        json!([
            {"type": "choice", "name": "intent", "choice": "refund",
                "probabilities": [{"value": "refund", "probability": 0.9},
                    {"value": "support", "probability": 0.07}, {"value": true, "probability": 0.03}],
                "confidence": 0.9},
            {"type": "predicate", "name": "angry", "probability": 0.31},
            {"type": "score", "score": 0.7,
                "probabilities": [{"value": 0, "label": "low", "probability": 0.3},
                    {"value": 1, "label": "high", "probability": 0.7}],
                "confidence": 0.7},
        ])
    );
    assert_eq!(
        body["usage"],
        json!({"input_tokens": 324, "output_tokens": 0, "total_tokens": 324})
    );
    assert_eq!(usage.input_tokens, Some(324));
    assert_eq!(usage.output_tokens, Some(0));
    assert_eq!(usage.cached_input_tokens, None);
}

#[test]
fn an_input_block_is_the_models_refusal_not_a_malformed_answer() {
    for (reason, token) in [
        (json!("safety"), "safety"),
        (json!("prohibited_content"), "prohibited_content"),
        (json!("Request blocked: <html>"), "unspecified"),
        (json!(7), "unspecified"),
    ] {
        let payload = json!({
            "prompt_feedback": {"block_reason": reason, "block_reason_message": "private-text-canary"},
            "usage": {"total_input_tokens": 12, "total_output_tokens": 0},
        });
        let failure = public_google_decisions(payload, &google_admission()).expect_err("blocked");
        assert_eq!(failure.failure_class, FailureClass::Refusal);
        assert!(!failure.retryable_same_deployment);
        assert!(!failure.failover_eligible);
        assert!(failure.safe_message.ends_with(&format!("({token})")));
        assert!(!failure.safe_message.contains("canary"));
    }
    let mut unblocked = google_payload();
    unblocked["prompt_feedback"] = json!({"block_reason": null});
    public_google_decisions(unblocked, &google_admission()).unwrap();
}

#[test]
fn an_omitted_output_count_is_zero_and_a_reported_one_passes_through() {
    let mut omitted = google_payload();
    omitted["usage"]
        .as_object_mut()
        .unwrap()
        .remove("total_output_tokens");
    let (body, usage) = public_google_decisions(omitted, &google_admission()).unwrap();
    assert_eq!(usage.output_tokens, Some(0));
    assert_eq!(body["usage"]["output_tokens"], 0);
    let mut reported = google_payload();
    reported["usage"]["total_output_tokens"] = json!(3);
    let (_body, usage) = public_google_decisions(reported, &google_admission()).unwrap();
    assert_eq!(usage.output_tokens, Some(3));
}

#[test]
fn confidence_must_restate_the_winning_probability_within_a_half_hundredth() {
    let mut near = google_payload();
    near["answers"]["q0"]["confidence"] = json!(0.895);
    near["answers"]["q1"]["confidence"] = json!(0.694);
    near["answers"]["q2"]["confidence"] = json!(0.705);
    public_google_decisions(near, &google_admission()).unwrap();
    for (key, confidence) in [("q0", 0.2), ("q1", 0.31), ("q2", 0.3)] {
        let mut payload = google_payload();
        payload["answers"][key]["confidence"] = json!(confidence);
        let failure =
            public_google_decisions(payload, &google_admission()).expect_err("inconsistent");
        assert_eq!(failure.failure_class, FailureClass::MalformedResponse);
    }
}

#[test]
fn a_boolean_choice_is_read_back_from_its_spelled_key() {
    let mut payload = google_payload();
    payload["answers"]["q0"]["choice"] = json!("true");
    payload["answers"]["q0"]["probabilities"] = json!({"refund": 0.1, "support": 0.1, "true": 0.8});
    payload["answers"]["q0"]["confidence"] = json!(0.8);
    let (body, _usage) = public_google_decisions(payload, &google_admission()).unwrap();
    assert_eq!(body["answers"][0]["choice"], json!(true));
}

#[test]
fn google_answers_that_do_not_match_the_questions_are_malformed() {
    let mut missing = google_payload();
    missing["answers"].as_object_mut().unwrap().remove("q2");
    let mut renamed = google_payload();
    let answer = renamed["answers"]
        .as_object_mut()
        .unwrap()
        .remove("q1")
        .unwrap();
    renamed["answers"]["angry"] = answer;
    let mut wrong_type = google_payload();
    wrong_type["answers"]["q1"]["type"] = json!("choice");
    let mut openai_type = google_payload();
    openai_type["answers"]["q1"]["type"] = json!("predicate");
    let mut unknown_choice = google_payload();
    unknown_choice["answers"]["q0"]["choice"] = json!("cancel");
    let mut not_highest = google_payload();
    not_highest["answers"]["q0"]["choice"] = json!("support");
    let mut extra_option = google_payload();
    extra_option["answers"]["q0"]["probabilities"]["other"] = json!(0.0);
    let mut bad_total = google_payload();
    bad_total["answers"]["q0"]["probabilities"]["support"] = json!(0.5);
    let mut out_of_range = google_payload();
    out_of_range["answers"]["q1"]["noul"] = json!(1.5);
    let mut endpoint_noul = google_payload();
    endpoint_noul["answers"]["q1"]["noul"] = json!(1.0);
    endpoint_noul["answers"]["q1"]["confidence"] = json!(1.0);
    let mut wrong_legend = google_payload();
    wrong_legend["answers"]["q2"]["legend"]["1"] = json!("high: now");
    let mut score_mismatch = google_payload();
    score_mismatch["answers"]["q2"]["score"] = json!(0.1);
    let mut score_outside = google_payload();
    score_outside["answers"]["q2"]["score"] = json!(3.0);
    let mut unbilled = google_payload();
    unbilled["usage"] = json!({"total_input_tokens": 0, "total_output_tokens": 0});
    let mut openai_usage = google_payload();
    openai_usage["usage"] = json!({"input_tokens": 324, "output_tokens": 0});
    for payload in [
        missing,
        renamed,
        wrong_type,
        openai_type,
        unknown_choice,
        not_highest,
        extra_option,
        bad_total,
        out_of_range,
        endpoint_noul,
        wrong_legend,
        score_mismatch,
        score_outside,
        unbilled,
        openai_usage,
        json!({"answers": []}),
    ] {
        let failure = public_google_decisions(payload, &google_admission()).expect_err("malformed");
        assert_eq!(failure.failure_class, FailureClass::MalformedResponse);
        assert!(!failure.retryable_same_deployment);
        assert!(!failure.failover_eligible);
    }
}

#[test]
fn google_wire_admission_deserializes_and_is_accepted_on_the_openai_route_only() {
    let text = json!({
        "request_id": "request-1", "alias": "a", "alias_revision_id": "r", "exact_model_id": "m",
        "route_reason": "direct", "route": [], "wire": "google",
        "openai_questions": [{"type": "predicate", "instructions": "i"}],
        "maximum_total_attempts": 1, "maximum_same_deployment_attempts": 1,
        "ignored_parameters": ["safety_identifier->dropped(google_decisions)"],
    })
    .to_string();
    let admission = parse_admission(&text, &[DecisionWire::Openai, DecisionWire::Google])
        .ok()
        .expect("google is an upstream of the OpenAI-shaped route");
    assert_eq!(admission.wire, DecisionWire::Google);
    assert_eq!(admission.ignored_parameters.len(), 1);
    assert!(matches!(
        parse_admission(&text, &[DecisionWire::Systemone]),
        Err(Parsed::Drift(_))
    ));
}

#[test]
fn a_buffered_google_answer_settles_googles_input_only_usage() {
    block_on(async {
        let (mut guard, observer, _) = guarded();
        let response = response(serde_json::to_vec(&google_payload()).unwrap());
        let (body, usage) = collect_response(
            response,
            Instant::now() + Duration::from_secs(1),
            Duration::from_secs(1),
            &google_admission(),
            &mut guard,
        )
        .await
        .unwrap();
        assert_eq!(body["model"], "gemini-decision-1");
        assert!(
            guard
                .settle("completed", Some(&usage), &[], None, true)
                .await
        );
        drop(guard);
        let written = calls(&observer);
        assert_eq!(written[0]["opened"], true);
        assert_eq!(written[0]["outcome"], "completed");
        assert_eq!(written[0]["usage"]["input_tokens"], 324);
        assert_eq!(written[0]["usage"]["output_tokens"], 0);
    });
}
