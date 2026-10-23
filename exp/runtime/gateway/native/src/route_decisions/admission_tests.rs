//! The admission bridge argument and the clone-free admission parse.

use axum::http::HeaderMap;
use serde_json::{json, Value};

use super::*;

#[test]
fn the_argument_carries_the_body_and_its_utf8_byte_length() {
    let body = r#"{"state":"café"}"#;
    let argument: Value =
        serde_json::from_str(&admission_argument("gateway-key", body, &HeaderMap::new())).unwrap();
    assert_eq!(argument["body"], body);
    assert_eq!(argument["body_bytes"], body.len());
    assert_ne!(body.len(), body.chars().count());
    assert_eq!(argument["raw_key"], "gateway-key");
}

fn admission(wire: &str) -> String {
    json!({
        "request_id": "request-1",
        "alias": "decisions",
        "alias_revision_id": "revision-1",
        "exact_model_id": "systemone-exact",
        "route_reason": "direct",
        "route": [],
        "wire": wire,
        "maximum_total_attempts": 1,
        "maximum_same_deployment_attempts": 1,
    })
    .to_string()
}

#[test]
fn a_matching_admission_parses_without_an_intermediate_value() {
    let parsed = parse_admission(&admission("systemone"), DecisionWire::Systemone);
    assert!(matches!(parsed, Ok(ref admission) if admission.request_id == "request-1"));
}

#[test]
fn escalation_drift_and_malformed_answers_stay_distinct() {
    let escalated = parse_admission(r#"{"escalate":"unavailable"}"#, DecisionWire::Systemone);
    assert!(matches!(escalated, Err(Parsed::Escalate(ref reason)) if reason == "unavailable"));
    let drifted = parse_admission(&admission("openai"), DecisionWire::Systemone);
    assert!(matches!(drifted, Err(Parsed::Drift(ref value)) if value["wire"] == "openai"));
    assert!(matches!(
        parse_admission("not json", DecisionWire::Systemone),
        Err(Parsed::Malformed)
    ));
}
