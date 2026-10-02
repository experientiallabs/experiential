//! Synthetic SystemOne response validation and settlement-callback regressions.

use std::sync::atomic::AtomicUsize;
use std::sync::Arc;

use pyo3::prelude::*;

use super::*;
use crate::bridge::Bridge;

fn admission() -> DecisionsAdmission {
    DecisionsAdmission {
        request_id: "request-1".into(),
        alias: "decider".into(),
        alias_revision_id: "revision-1".into(),
        exact_model_id: "typesafe/jev-1".into(),
        route_reason: "direct".into(),
        route: Vec::new(),
        questions: json!({
            "paid": {"type": "noul", "instructions": "Was payment received?"},
            "department": {"type": "choice", "instructions": "Choose the department",
                "criteria": {"billing": "Invoices", "technical": "Bugs", "sales": null}},
            "quantity": {"type": "score", "instructions": "Count items",
                "criteria": ["No items", "One or two items", "Three or more items"]},
        })
        .as_object()
        .unwrap()
        .clone(),
        maximum_total_attempts: 1,
        maximum_same_deployment_attempts: 1,
    }
}

fn payload() -> Value {
    json!({
        "model": "jev-1.13.0",
        "answers": {
            "paid": {"type": "noul", "noul": 0.99},
            "department": {"type": "choice", "choice": "billing", "confidence": 1.0,
                "probabilities": {"billing": 1.0, "technical": 0.0, "sales": 0.0}},
            "quantity": {"type": "score", "score": 2.0, "confidence": 1.0,
                "legend": {"0": "No items", "1": "One or two items", "2": "Three or more items"},
                "probabilities": {"0": 0.0, "1": 0.0, "2": 1.0}},
        },
        "usage": {"input_tokens": 451, "output_tokens": 68},
    })
}

fn assert_malformed(value: Value) {
    let failure = public_decisions(value, &admission()).expect_err("malformed answer");
    assert_eq!(failure.failure_class, FailureClass::MalformedResponse);
    assert!(!failure.retryable_same_deployment);
    assert!(!failure.failover_eligible);
}

#[test]
fn all_three_answers_are_preserved_and_only_public_fields_are_returned() {
    let mut provider = payload();
    provider["id"] = json!("provider-private-id");
    provider["provider_private"] = json!("must not escape");
    provider["answers"]["paid"]["provider_private"] = json!("must not escape");
    provider["usage"]["provider_private"] = json!(9000);
    let (public, usage) = public_decisions(provider, &admission()).unwrap();
    let mut expected = payload();
    expected["id"] = json!(stable_public_id("decision", "request-1"));
    expected["model"] = json!("decider");
    assert_eq!(public, expected);
    assert_eq!(usage.input_tokens, Some(451));
    assert_eq!(usage.output_tokens, Some(68));
    assert_eq!(usage.cached_input_tokens, None);
}

#[test]
fn missing_additional_and_mismatched_question_ids_are_rejected() {
    let mut missing = payload();
    missing["answers"].as_object_mut().unwrap().remove("paid");
    let mut additional = payload();
    additional["answers"]["extra"] = json!({"type": "noul", "noul": 1});
    let mut renamed = payload();
    let paid = renamed["answers"]
        .as_object_mut()
        .unwrap()
        .remove("paid")
        .unwrap();
    renamed["answers"]["renamed"] = paid;
    for value in [missing, additional, renamed] {
        assert_malformed(value);
    }
}

#[test]
fn mismatched_types_and_all_invalid_noul_values_are_rejected() {
    for value in [
        Value::Null,
        json!(true),
        json!("0.9"),
        json!(-0.1),
        json!(1.1),
        json!([]),
    ] {
        let mut provider = payload();
        provider["answers"]["paid"]["noul"] = value;
        assert_malformed(provider);
    }
    for id in ["paid", "department", "quantity"] {
        let mut provider = payload();
        provider["answers"][id]["type"] = json!("unknown");
        assert_malformed(provider);
    }
}

#[test]
fn choices_require_exact_category_keys_and_normalized_probabilities() {
    for probabilities in [
        json!({"billing": 1.0, "technical": 0.0}),
        json!({"billing": 1.0, "technical": 0.0, "extra": 0.0}),
        json!({"billing": 1.0, "technical": 0.0, "sales": 0.0, "extra": 0.0}),
        json!({"billing": 0.7, "technical": 0.2, "sales": 0.0}),
        json!({"billing": 1.2, "technical": -0.2, "sales": 0.0}),
        json!({"billing": true, "technical": 0.0, "sales": 0.0}),
    ] {
        let mut provider = payload();
        provider["answers"]["department"]["probabilities"] = probabilities;
        assert_malformed(provider);
    }
    for choice in [json!("unknown"), json!(0), Value::Null] {
        let mut provider = payload();
        provider["answers"]["department"]["choice"] = choice;
        assert_malformed(provider);
    }
}

#[test]
fn choice_requires_a_maximum_probability_and_allows_tied_winners() {
    let mut provider = payload();
    provider["answers"]["department"]["choice"] = json!("technical");
    assert_malformed(provider.clone());
    provider["answers"]["department"]["probabilities"] =
        json!({"billing": 0.5000001, "technical": 0.4999999, "sales": 0.0});
    assert_malformed(provider.clone());
    provider["answers"]["department"]["probabilities"] =
        json!({"billing": 0.0, "technical": 0.5, "sales": 0.5});
    provider["answers"]["department"]["confidence"] = json!(0.4);
    for choice in ["technical", "sales"] {
        provider["answers"]["department"]["choice"] = json!(choice);
        let (public, _) = public_decisions(provider.clone(), &admission())
            .expect("either tied maximum is a valid choice, including a null-described category");
        assert_eq!(public["answers"], provider["answers"]);
    }
}

fn choice_payload(probabilities: Value, choice: &str) -> (DecisionsAdmission, Value) {
    let mut admitted = admission();
    let criteria = probabilities.as_object().unwrap().keys();
    admitted.questions["department"]["criteria"] = json!(criteria
        .map(|key| (key.clone(), Value::Null))
        .collect::<Map<_, _>>());
    let mut provider = payload();
    provider["answers"]["department"] = json!({"type": "choice", "choice": choice,
        "confidence": 0.3, "probabilities": probabilities});
    (admitted, provider)
}

#[test]
fn independently_rounded_live_choices_preserve_provider_answers_and_usage() {
    // Numeric values from bounded synthetic jev-1.13.0 calls, never customer data.
    for (probabilities, choice) in [
        // Hundredth bins totalling 0.99 (the last with a published top tie), then a unit
        // total whose rounded winner sits one hundredth below another bin.
        (
            json!({"black": 0.17, "orange": 0.02, "green": 0.07, "purple": 0.03,
                "red": 0.41000000000000003, "blue": 0.19, "yellow": 0.1}),
            "red",
        ),
        (
            json!({"black": 0.01, "green": 0.1, "purple": 0.01, "pink": 0.0, "blue": 0.11,
                "white": 0.02, "brown": 0.01, "orange": 0.01, "red": 0.67, "yellow": 0.05}),
            "red",
        ),
        (
            json!({"orange": 0.01, "purple": 0.03, "yellow": 0.04, "white": 0.4, "black": 0.08,
                "red": 0.28, "green": 0.04, "pink": 0.01, "blue": 0.1}),
            "white",
        ),
        (
            json!({"brown": 0.01, "red": 0.34, "blue": 0.1, "yellow": 0.04, "orange": 0.01,
                "black": 0.08, "white": 0.34, "purple": 0.02, "green": 0.04, "pink": 0.01}),
            "white",
        ),
        (
            json!({"green": 0.05, "white": 0.3, "black": 0.1, "orange": 0.01, "brown": 0.01,
                "red": 0.31, "yellow": 0.04, "blue": 0.14, "pink": 0.01, "purple": 0.03}),
            "white",
        ),
    ] {
        let (admitted, provider) = choice_payload(probabilities, choice);
        let (public, usage) = public_decisions(provider.clone(), &admitted)
            .expect("independently rounded choice compatible with a unit distribution");
        assert_eq!(public["answers"], provider["answers"]);
        assert_eq!(usage.input_tokens, Some(451));
        assert_eq!(usage.output_tokens, Some(68));
    }
}

#[test]
fn rounding_never_excuses_infeasible_choices_or_high_precision_disagreement() {
    for (probabilities, choice) in [
        // Infeasible or finer-precision totals, then winner gaps no unit distribution closes.
        (json!({"a": 0.33, "b": 0.33, "c": 0.32}), "a"),
        (json!({"a": 0.34, "b": 0.34, "c": 0.34}), "a"),
        (json!({"a": 1.0, "b": 0.01, "c": 0.01}), "a"),
        (json!({"a": 0.335, "b": 0.33, "c": 0.33}), "a"),
        (json!({"a": 0.5 - 1e-12, "b": 0.49, "c": 0.0}), "a"),
        (json!({"a": 0.31, "b": 0.33, "c": 0.36}), "b"),
        (json!({"a": 0.0, "b": 0.49, "c": 0.5}), "b"),
        (json!({"a": 0.0, "b": 0.495, "c": 0.505}), "b"),
        (json!({"a": 0.0, "b": 0.5, "c": 0.51}), "b"),
    ] {
        let (admitted, provider) = choice_payload(probabilities, choice);
        let failure = public_decisions(provider, &admitted).expect_err("inconsistent choice");
        assert_eq!(failure.failure_class, FailureClass::MalformedResponse);
        assert!(!failure.retryable_same_deployment);
        assert!(!failure.failover_eligible);
    }
    // Feasible 0.99 and 1.01 totals; a one-hundredth gap closes only with spare mass.
    for (probabilities, choice) in [
        (json!({"a": 0.34, "b": 0.33, "c": 0.32}), "a"),
        (json!({"a": 0.34, "b": 0.34, "c": 0.33}), "b"),
        (json!({"a": 0.0, "b": 0.49, "c": 0.5}), "c"),
        (json!({"a": 0.01, "b": 0.49, "c": 0.5}), "b"),
    ] {
        let (admitted, provider) = choice_payload(probabilities, choice);
        public_decisions(provider, &admitted).expect("feasible rounded choice");
    }
}

#[test]
fn choice_rounding_matches_an_independent_three_bin_enumeration() {
    for probabilities in [
        [0.33, 0.33, 0.33],
        [0.34, 0.33, 0.32],
        [0.0, 0.49, 0.5],
        [0.01, 0.49, 0.5],
        [0.3, 0.31, 0.38],
        [0.0, 0.0, 1.0],
    ] {
        let bins = probabilities.map(|p| (p * 400.0_f64).round() as i64);
        let range = |bin: i64| (bin - 2).max(0)..=(bin + 2).min(400);
        for selected in 0..3 {
            let mut feasible = false;
            for a in range(bins[0]) {
                for b in range(bins[1]) {
                    let c = 400 - a - b;
                    let values = [a, b, c];
                    if range(bins[2]).contains(&c)
                        && values.iter().all(|value| *value <= values[selected])
                    {
                        feasible = true;
                    }
                }
            }
            assert_eq!(
                choice_can_be_highest(selected, &probabilities),
                feasible,
                "{probabilities:?}, selected={selected}"
            );
        }
    }
}

#[test]
fn score_requires_exact_legend_index_set_and_expected_value() {
    for (field, value) in [
        (
            "legend",
            json!({"0": "Wrong", "1": "One or two items", "2": "Three or more items"}),
        ),
        ("legend", json!({"0": "No items", "1": "One or two items"})),
        (
            "legend",
            json!(["No items", "One or two items", "Three or more items"]),
        ),
        ("probabilities", json!({"0": 0.0, "1": 0.0, "3": 1.0})),
        ("probabilities", json!({"0": 0.0, "1": 0.0, "2": 0.5})),
        ("score", json!(1.0)),
        ("score", json!(3.0)),
        ("score", json!(true)),
        ("score", Value::Null),
    ] {
        let mut provider = payload();
        provider["answers"]["quantity"][field] = value;
        assert_malformed(provider);
    }
    let mut provider = payload();
    provider["answers"]["quantity"]["probabilities"] = json!({"0": 0.1, "1": 0.3, "2": 0.6});
    provider["answers"]["quantity"]["score"] = json!(1.5);
    public_decisions(provider.clone(), &admission()).expect("expected fractional score");
    provider["answers"]["quantity"]["score"] = json!(1.50001);
    assert_malformed(provider);
}

fn score_payload(probabilities: &[f64], score: f64) -> (DecisionsAdmission, Value) {
    let mut admitted = admission();
    let criteria: Vec<Value> = (0..probabilities.len())
        .map(|i| json!(format!("Level {i}")))
        .collect();
    admitted.questions["quantity"]["criteria"] = json!(criteria);
    let legend: Map<String, Value> = criteria
        .into_iter()
        .enumerate()
        .map(|(i, value)| (i.to_string(), value))
        .collect();
    let probabilities: Map<String, Value> = probabilities
        .iter()
        .enumerate()
        .map(|(i, value)| (i.to_string(), json!(value)))
        .collect();
    let mut provider = payload();
    provider["answers"]["quantity"]["legend"] = json!(legend);
    provider["answers"]["quantity"]["probabilities"] = json!(probabilities);
    provider["answers"]["quantity"]["score"] = json!(score);
    (admitted, provider)
}

#[test]
fn independently_rounded_live_scores_preserve_provider_answers_and_usage() {
    // Numeric values from bounded synthetic jev-1.13.0 calls, never customer data.
    for (probabilities, score) in [
        (vec![0.0, 0.87, 0.13, 0.0, 0.0], 1.12),
        (vec![0.01, 0.0, 0.01, 0.07, 0.91], 3.88),
        (
            vec![0.84, 0.12, 0.02, 0.01, 0.01, 0.0, 0.0, 0.0, 0.0, 0.0],
            0.26,
        ),
        (
            vec![0.0, 0.07, 0.31, 0.43, 0.15, 0.03, 0.01, 0.0, 0.0, 0.0],
            2.78,
        ),
    ] {
        let (admitted, provider) = score_payload(&probabilities, score);
        let (public, usage) = public_decisions(provider.clone(), &admitted)
            .expect("independently rounded score compatible with a unit distribution");
        assert_eq!(public["answers"], provider["answers"]);
        assert_eq!(usage.input_tokens, Some(451));
        assert_eq!(usage.output_tokens, Some(68));
    }
}

#[test]
fn rounding_never_excuses_infeasible_scores_or_high_precision_disagreement() {
    for (probabilities, score) in [
        (vec![1.0, 0.0, 0.0, 0.0, 0.0], 0.03),
        (vec![0.0, 0.0, 0.0, 0.0, 1.0], 3.97),
        (vec![0.0, 0.87, 0.13, 0.0, 0.0], 1.5),
        (vec![0.0, 0.87, 0.13, 0.0, 0.0], 1.12001),
        (vec![0.0, 0.87, 0.13, 0.0, 0.0], 1.120000000005),
        (vec![0.0, 0.870001, 0.129999, 0.0, 0.0], 1.12),
        (vec![1e-12, 0.87 - 1e-12, 0.13, 0.0, 0.0], 1.12),
        (vec![0.33, 0.33, 0.32], 0.99),
        (vec![0.34, 0.34, 0.34, 0.0, 0.0], 1.02),
        (vec![0.0, 0.87, 0.13, 0.0, 0.0], -0.01),
        (vec![0.0, 0.87, 0.13, 0.0, 0.0], 4.01),
    ] {
        let (admitted, provider) = score_payload(&probabilities, score);
        let failure = public_decisions(provider, &admitted).expect_err("inconsistent score");
        assert_eq!(failure.failure_class, FailureClass::MalformedResponse);
        assert!(!failure.retryable_same_deployment);
        assert!(!failure.failover_eligible);
    }
    let (admitted, provider) = score_payload(&[0.0, 0.870001, 0.129999], 1.129999);
    public_decisions(provider, &admitted).expect("consistent higher precision is still valid");
    // Three rounded thirds total 0.99, yet (0.33, 0.335, 0.335) explains both them
    // and a 1.0 score, so the total no longer rejects this hundredth answer.
    let (admitted, provider) = score_payload(&[0.33, 0.33, 0.33], 1.0);
    public_decisions(provider, &admitted).expect("feasible rounded total and score");
}

#[test]
fn score_rounding_bounds_respect_clipping_and_shared_probability_mass() {
    // Only half a percent may leave a unit bin, not half a percent per zero bin.
    let (admitted, provider) = score_payload(&[1.0, 0.0, 0.0, 0.0, 0.0], 0.02);
    public_decisions(provider, &admitted).expect("feasible clipped rounding envelope");
    let (admitted, provider) = score_payload(&[0.5, 0.5], 0.51);
    public_decisions(provider, &admitted).expect("closed rounding envelopes may touch");
    let (admitted, provider) = score_payload(&[0.5, 0.5], 0.52);
    assert!(public_decisions(provider, &admitted).is_err());
}

#[test]
fn rounding_extrema_match_an_independent_three_bin_enumeration() {
    for probabilities in [
        [1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
        [0.33, 0.33, 0.34],
        [0.2, 0.3, 0.5],
    ] {
        // Twice as fine as the production half-cent grid: include interior and
        // boundary candidates without sharing the greedy allocation algorithm.
        let bins = probabilities.map(|p| (p * 400.0_f64).round() as i64);
        let mut possible = Vec::new();
        for a in (bins[0] - 2).max(0)..=(bins[0] + 2).min(400) {
            for b in (bins[1] - 2).max(0)..=(bins[1] + 2).min(400) {
                let c = 400 - a - b;
                if c >= (bins[2] - 2).max(0) && c <= (bins[2] + 2).min(400) {
                    possible.push(b + 2 * c);
                }
            }
        }
        let minimum = *possible.iter().min().unwrap();
        let maximum = *possible.iter().max().unwrap();
        for score_hundredths in 0..=200 {
            let feasible =
                score_hundredths * 4 + 2 >= minimum && score_hundredths * 4 - 2 <= maximum;
            assert_eq!(
                score_matches_distribution(score_hundredths as f64 / 100.0, &probabilities),
                feasible,
                "{probabilities:?}, score={score_hundredths}"
            );
        }
    }
    // Every permitted score hundredth survives ordinary binary representation.
    for value in 0..=900 {
        assert_eq!(hundredths(value as f64 / 100.0), Some(value));
    }
    assert_eq!(hundredths(0.8200000000000001), Some(82));
    assert_eq!(hundredths(0.82000001), None);
    assert_eq!(hundredths(1.120000000005), None);
    assert_eq!(hundredths(1e-12), None);
    assert_eq!(hundredths(f64::MIN_POSITIVE), None);
}

#[test]
fn structured_score_levels_require_exact_deep_legend_equality() {
    let mut admitted = admission();
    admitted.questions["quantity"]["criteria"] = json!([
        "No items",
        {"label": "Small", "examples": [1, 2], "nested": {"minimum": i64::MIN}},
        ["Large", {"minimum": 3, "maximum": u64::MAX}],
    ]);
    let mut provider = payload();
    provider["answers"]["quantity"]["legend"] = json!({
        "0": "No items",
        "1": {"label": "Small", "examples": [1, 2], "nested": {"minimum": i64::MIN}},
        "2": ["Large", {"minimum": 3, "maximum": u64::MAX}],
    });
    let parsed = strict_json(&serde_json::to_vec(&provider).unwrap()).unwrap();
    let (public, _) =
        public_decisions(parsed, &admitted).expect("structured levels survive the JSON wire");
    assert_eq!(public["answers"], provider["answers"]);
    for wrong_level in [
        json!(["Large", {"minimum": 4, "maximum": u64::MAX}]),
        json!([{"minimum": 3, "maximum": u64::MAX}, "Large"]),
        json!(["Large", {"minimum": 3, "maximum": u64::MAX, "extra": null}]),
        json!("Large"),
    ] {
        let mut wrong = provider.clone();
        wrong["answers"]["quantity"]["legend"]["2"] = wrong_level;
        let failure = public_decisions(wrong, &admitted).expect_err("deep mismatch");
        assert_eq!(failure.failure_class, FailureClass::MalformedResponse);
    }
}

#[test]
fn structured_choice_descriptions_do_not_change_the_selected_category() {
    let mut admitted = admission();
    admitted.questions["department"]["criteria"] = json!({
        "billing": {"examples": ["Invoice", "Charge"]},
        "technical": ["Bugs", {"severity": "high"}],
        "sales": null,
    });
    let provider = payload();
    let (public, _) =
        public_decisions(provider.clone(), &admitted).expect("structured choice criteria");
    assert_eq!(public["answers"], provider["answers"]);
}

#[test]
fn confidence_must_be_finite_unit_interval_for_choice_and_score() {
    for id in ["department", "quantity"] {
        for value in [
            Value::Null,
            json!(true),
            json!("1"),
            json!(-0.01),
            json!(1.01),
        ] {
            let mut provider = payload();
            provider["answers"][id]["confidence"] = value;
            assert_malformed(provider);
        }
    }
}

#[test]
fn usage_requires_both_counts_as_nonnegative_i64_integers() {
    for name in ["input_tokens", "output_tokens"] {
        for value in [
            Value::Null,
            json!(true),
            json!(-1),
            json!(1.5),
            json!(1.0),
            json!("1"),
            json!(u64::MAX),
        ] {
            let mut provider = payload();
            provider["usage"][name] = value;
            assert_malformed(provider);
        }
        let mut provider = payload();
        provider["usage"].as_object_mut().unwrap().remove(name);
        assert_malformed(provider);
    }
    let mut provider = payload();
    provider["usage"] = json!({"input_tokens": 0, "output_tokens": i64::MAX});
    let (_, usage) = public_decisions(provider, &admission()).unwrap();
    assert_eq!(usage.input_tokens, Some(0));
    assert_eq!(usage.output_tokens, Some(i64::MAX as u64));
}

#[test]
fn all_zero_usage_is_rejected_but_each_count_can_individually_be_zero() {
    let mut provider = payload();
    provider["usage"] = json!({"input_tokens": 0, "output_tokens": 0});
    assert_malformed(provider.clone());
    for (input, output) in [(0, 1), (1, 0)] {
        provider["usage"] = json!({"input_tokens": input, "output_tokens": output});
        let (public, usage) = public_decisions(provider.clone(), &admission()).unwrap();
        assert_eq!(public["usage"], provider["usage"]);
        assert_eq!(usage.input_tokens, Some(input));
        assert_eq!(usage.output_tokens, Some(output));
    }
}

#[test]
fn response_json_rejects_duplicates_nonfinite_and_trailing_data() {
    for bytes in [
        br#"{"answers":{"paid":{},"paid":{}}}"#.as_slice(),
        br#"{"usage":{"input_tokens":1,"input_tokens":2}}"#.as_slice(),
        br#"{"x":NaN}"#.as_slice(),
        br#"{"x":Infinity}"#.as_slice(),
        br#"{"x":1e999}"#.as_slice(),
        br#"{} {}"#.as_slice(),
    ] {
        let failure = strict_json(bytes).expect_err("invalid strict JSON");
        assert_eq!(failure.failure_class, FailureClass::MalformedResponse);
    }
    let value = json!({"string": "text", "escaped": "a\nb", "array": [null, true, -1, 1.5]});
    assert_eq!(
        strict_json(&serde_json::to_vec(&value).unwrap()).unwrap(),
        value
    );
}

#[test]
fn normalization_tolerance_is_small_and_does_not_change_provider_numbers() {
    let mut provider = payload();
    provider["answers"]["department"]["probabilities"]["billing"] = json!(0.9999995);
    let (public, _) = public_decisions(provider.clone(), &admission()).unwrap();
    assert_eq!(public["answers"], provider["answers"]);
    provider["answers"]["department"]["probabilities"]["billing"] = json!(0.99999);
    assert_malformed(provider);
}

#[test]
fn default_single_attempt_policy_never_redials_unknown_idempotency() {
    let failure = rejected_http(401);
    assert_eq!(failure.failure_class, FailureClass::ProviderAuthentication);
    assert!(!failure.retryable_same_deployment);
    let mut admitted = admission();
    admitted.maximum_total_attempts = 2;
    assert!(!successor_possible(
        admitted.policy(),
        &[wire(false)],
        Instant::now() + Duration::from_secs(1),
        1,
        1,
        0,
        &failure,
        false
    ));
    assert!(successor_possible(
        admitted.policy(),
        &[wire(false), wire(false)],
        Instant::now() + Duration::from_secs(1),
        1,
        1,
        0,
        &failure,
        false
    ));
}

#[test]
fn admission_forwards_attribution_and_trusted_ip_but_not_idempotency() {
    let mut headers = HeaderMap::new();
    headers.insert(
        "http-referer",
        "https://app.example.invalid".parse().unwrap(),
    );
    headers.insert("x-title", "Decision app".parse().unwrap());
    headers.insert("x-forwarded-for", "203.0.113.7, 192.0.2.9".parse().unwrap());
    headers.insert("idempotency-key", "caller-key".parse().unwrap());
    let argument: Value =
        serde_json::from_str(&admission_argument("gateway-key", "{}", &headers)).unwrap();
    assert_eq!(argument["app_referer"], "https://app.example.invalid");
    assert_eq!(argument["app_title"], "Decision app");
    assert_eq!(argument["client_ip"], "192.0.2.9");
    assert_eq!(argument["raw_key"], "gateway-key");
    assert!(argument.get("idempotency_key").is_none());
    headers.insert("x-real-ip", "198.51.100.4".parse().unwrap());
    let argument: Value =
        serde_json::from_str(&admission_argument("gateway-key", "{}", &headers)).unwrap();
    assert_eq!(argument["client_ip"], "198.51.100.4");
    let argument: Value =
        serde_json::from_str(&admission_argument("gateway-key", "{}", &HeaderMap::new())).unwrap();
    assert_eq!(argument["client_ip"], Value::Null);
}

fn rejected_http(status: u16) -> Failure {
    crate::upstream::decision_http_failure(crate::upstream::transport_failure(Some(status)), status)
}

fn wire(customer_managed: bool) -> DeploymentWire {
    serde_json::from_value(json!({
        "provider": "typesafe", "deployment_id": "typesafe-1", "dialect": "typesafe_systemone",
        "url": "https://example.invalid/v1/systemone", "headers": {}, "timeout_seconds": 1.0,
        "idempotency_key": "attempt-1", "billing_customer_managed": customer_managed,
    }))
    .unwrap()
}

#[test]
fn failed_rung_never_requests_a_disallowed_same_deployment_reservation() {
    let policy = admission().policy();
    let failure = route_failure(rejected_http(401), &wire(false), policy, 1);
    assert!(!failure.retryable_same_deployment);
    assert!(failure.failover_eligible);
    assert_eq!(failure.failure_class, FailureClass::ProviderAuthentication);
}

#[test]
fn uncertain_outcomes_never_advance_even_with_an_unused_second_rung() {
    let mut admitted = admission();
    admitted.maximum_total_attempts = 2;
    // Even an inconsistent wider retry policy cannot replay an uncertain call.
    admitted.maximum_same_deployment_attempts = 2;
    let policy = admitted.policy();
    for failure in [
        crate::upstream::transport_failure(None),
        rejected_http(402),
        rejected_http(429),
        rejected_http(529),
        Failure::new(FailureClass::Timeout, "uncertain open").with_retry(true, true),
        Failure::new(FailureClass::MalformedResponse, "uncertain answer").with_retry(true, true),
        timeout_failure(),
        malformed("invalid response"),
    ] {
        let failure = route_failure(failure, &wire(false), policy, 1);
        assert!(!failure.retryable_same_deployment);
        assert!(!failure.failover_eligible);
        assert!(!successor_possible(
            policy,
            &[wire(false), wire(false)],
            Instant::now() + Duration::from_secs(1),
            1,
            1,
            0,
            &failure,
            false
        ));
    }
    for status in [401, 403, 404] {
        let failure = route_failure(rejected_http(status), &wire(false), admission().policy(), 1);
        assert!(failure.decision_provider_rejected);
        assert_eq!(failure.failover_eligible, status == 401);
        assert_eq!(
            successor_possible(
                policy,
                &[wire(false), wire(false)],
                Instant::now() + Duration::from_secs(1),
                1,
                1,
                0,
                &failure,
                false
            ),
            status == 401
        );
    }
}

#[test]
fn customer_credential_and_quota_failures_keep_caller_ownership() {
    let policy = admission().policy();
    for status in [401, 403, 402] {
        let failure = route_failure(rejected_http(status), &wire(true), policy, 1);
        assert!(failure.customer_owned);
        assert!(!failure.retryable_same_deployment);
        // Only the explicit 401 credential rejection can advance.
        assert_eq!(failure.failover_eligible, status == 401);
        assert_eq!(failure.clone().boundary().public_error().status_code, 400);
        let house = route_failure(rejected_http(status), &wire(false), policy, 1);
        assert!(!house.customer_owned);
        assert_eq!(house.failover_eligible, status == 401);
    }
    let failure = route_failure(rejected_http(529), &wire(true), policy, 1);
    assert!(!failure.customer_owned);
    assert!(!failure.failover_eligible);
}

/// Only synthetic callback recording, not a claim about a durable ledger.
const PLANE_SOURCE: &std::ffi::CStr = cr#"
import json

class Plane:
    def __init__(self):
        self.calls = []

    def settle(self, argument):
        self.calls.append(json.loads(argument))
        return '{}'

    def close_thread_resources(self, argument):
        return '{}'

    def snapshot(self):
        return json.dumps(self.calls)
"#;

fn guarded() -> (AttemptGuard, Py<PyAny>, Arc<AtomicUsize>) {
    Python::initialize();
    let object = Python::attach(|py| {
        pyo3::types::PyModule::from_code(py, PLANE_SOURCE, c"decision_plane.py", c"decision_plane")
            .unwrap()
            .getattr("Plane")
            .unwrap()
            .call0()
            .unwrap()
            .unbind()
    });
    let observer = Python::attach(|py| object.clone_ref(py));
    let bridge = Arc::new(Bridge::new(object, 1).unwrap());
    let pending = Arc::new(AtomicUsize::new(0));
    let mut guard = AttemptGuard::new(bridge, pending.clone(), "request-1".into(), Instant::now());
    guard.rebind("attempt-1".into());
    (guard, observer, pending)
}

fn calls(object: &Py<PyAny>) -> Value {
    Python::attach(|py| {
        let text = object
            .bind(py)
            .call_method0("snapshot")
            .unwrap()
            .extract::<String>()
            .unwrap();
        serde_json::from_str(&text).unwrap()
    })
}

fn block_on<F: std::future::Future>(future: F) -> F::Output {
    tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap()
        .block_on(future)
}

fn response(body: impl Into<reqwest::Body>) -> reqwest::Response {
    http::Response::builder()
        .status(200)
        .header("retry-after", "2")
        .body(body.into())
        .unwrap()
        .into()
}

#[test]
fn settlement_rejection_marker_comes_from_definitive_http_status_only() {
    for status in [
        400, 401, 402, 403, 404, 422, 429, 529, 408, 500, 502, 503, 504,
    ] {
        block_on(async {
            let (mut guard, observer, _) = guarded();
            let failure = rejected_http(status);
            let expected = matches!(status, 400 | 401 | 403 | 404 | 422);
            assert_eq!(failure.decision_provider_rejected, expected);
            assert_eq!(failure.failover_eligible, status == 401);
            assert!(
                guard
                    .settle("failed", None, &[], Some(&failure), true)
                    .await
            );
            drop(guard);
            let written = calls(&observer);
            assert_eq!(written[0]["decision_provider_rejected"], expected);
            assert_eq!(written[0]["opened"], false);
            assert_eq!(written[0]["usage"], Value::Null);
        });
    }
}

#[test]
fn buffered_success_reports_opened_and_both_exact_usage_counts() {
    block_on(async {
        let (mut guard, observer, _) = guarded();
        let response = response(serde_json::to_vec(&payload()).unwrap());
        let (body, usage) = collect_response(
            response,
            Instant::now() + Duration::from_secs(1),
            Duration::from_secs(1),
            &admission(),
            &mut guard,
        )
        .await
        .unwrap();
        assert_eq!(body["model"], "decider");
        assert!(
            guard
                .settle("completed", Some(&usage), &[], None, true)
                .await
        );
        drop(guard);
        let written = calls(&observer);
        assert_eq!(written.as_array().unwrap().len(), 1);
        assert_eq!(written[0]["opened"], true);
        assert_eq!(written[0]["outcome"], "completed");
        assert_eq!(written[0]["decision_provider_rejected"], false);
        assert_eq!(written[0]["finalize"], true);
        assert_eq!(written[0]["usage"]["input_tokens"], 451);
        assert_eq!(written[0]["usage"]["output_tokens"], 68);
        assert_eq!(written[0]["rate_limit_headers"]["retry-after"], "2");
    });
}

#[test]
fn malformed_buffer_preserves_credible_usage_without_inventing_missing_counts() {
    let mut zero_usage = payload();
    zero_usage["usage"] = json!({"input_tokens": 0, "output_tokens": 0});
    let mut wrong_choice = payload();
    wrong_choice["answers"]["department"]["choice"] = json!("technical");
    let mut wrong_score = payload();
    wrong_score["answers"]["quantity"]["score"] = json!(1.9);
    let mut wrong_distribution = payload();
    wrong_distribution["answers"]["quantity"]["probabilities"]["2"] = json!(0.8);
    let mut partial_usage = wrong_choice.clone();
    partial_usage["usage"]
        .as_object_mut()
        .unwrap()
        .remove("output_tokens");
    let mut negative_usage = wrong_choice.clone();
    negative_usage["usage"]["input_tokens"] = json!(-1);
    for malformed_payload in [
        json!({"answers": {}}),
        zero_usage,
        wrong_choice,
        wrong_score,
        wrong_distribution,
        partial_usage,
        negative_usage,
    ] {
        let expected_usage = decision_usage(&malformed_payload).ok();
        block_on(async {
            let (mut guard, observer, _) = guarded();
            let response = response(serde_json::to_vec(&malformed_payload).unwrap());
            let failure = collect_response(
                response,
                Instant::now() + Duration::from_secs(1),
                Duration::from_secs(1),
                &admission(),
                &mut guard,
            )
            .await
            .expect_err("malformed");
            assert!(
                guard
                    .settle("failed", None, &[], Some(&failure), true)
                    .await
            );
            drop(guard);
            let written = calls(&observer);
            assert_eq!(written.as_array().unwrap().len(), 1);
            assert_eq!(written[0]["opened"], true);
            assert_eq!(written[0]["outcome"], "failed");
            assert_eq!(written[0]["failure"]["failure_class"], "malformed_response");
            assert_eq!(written[0]["decision_provider_rejected"], false);
            match &expected_usage {
                Some(usage) => {
                    assert_eq!(
                        written[0]["usage"]["input_tokens"],
                        usage.input_tokens.unwrap()
                    );
                    assert_eq!(
                        written[0]["usage"]["output_tokens"],
                        usage.output_tokens.unwrap()
                    );
                }
                None => assert_eq!(written[0]["usage"], Value::Null),
            }
            assert_eq!(written[0]["finalize"], true);
        });
    }
}

#[test]
fn cancelling_a_buffered_read_settles_the_opened_attempt_once() {
    block_on(async {
        let (mut guard, observer, pending) = guarded();
        let stream = futures_util::stream::pending::<Result<bytes::Bytes, std::io::Error>>();
        let response = response(reqwest::Body::wrap_stream(stream));
        let deadline = Instant::now() + Duration::from_secs(30);
        assert!(tokio::time::timeout(
            Duration::from_millis(10),
            collect_response(
                response,
                deadline,
                Duration::from_secs(30),
                &admission(),
                &mut guard
            )
        )
        .await
        .is_err());
        drop(guard);
        tokio::time::timeout(Duration::from_secs(2), async {
            while pending.load(Ordering::SeqCst) != 0 {
                tokio::task::yield_now().await;
            }
        })
        .await
        .expect("cancel settlement drains");
        let written = calls(&observer);
        assert_eq!(written.as_array().unwrap().len(), 1);
        assert_eq!(written[0]["opened"], true);
        assert_eq!(written[0]["outcome"], "failed");
        assert_eq!(written[0]["failure"]["failure_class"], "cancelled");
        assert_eq!(written[0]["decision_provider_rejected"], false);
        assert_eq!(written[0]["usage"], Value::Null);
        assert_eq!(written[0]["finalize"], true);
    });
}

#[test]
fn bounded_reader_enforces_phase_deadline_and_size_without_content_length() {
    block_on(async {
        let never = futures_util::stream::pending::<Result<bytes::Bytes, std::io::Error>>();
        let failure = read_bounded_body(
            response(reqwest::Body::wrap_stream(never)),
            Instant::now() + Duration::from_secs(1),
            Duration::from_millis(5),
        )
        .await
        .unwrap_err();
        assert_eq!(failure.failure_class, FailureClass::Timeout);
        let bytes = vec![b' '; MAXIMUM_RETAINED_OUTPUT_BYTES + 1];
        let failure = read_bounded_body(
            response(bytes),
            Instant::now() + Duration::from_secs(1),
            Duration::from_secs(1),
        )
        .await
        .unwrap_err();
        assert_eq!(failure.failure_class, FailureClass::MalformedResponse);
        assert_eq!(failure.safe_message, OUTPUT_OVERFLOW_MESSAGE);
    });
}

#[test]
fn independently_observed_usage_survives_cancellation_but_not_rebinding() {
    for rebind in [false, true] {
        block_on(async {
            let (mut guard, observer, pending) = guarded();
            guard.mark_opened();
            guard.record_decision_usage(decision_usage(&payload()).unwrap());
            if rebind {
                guard.rebind("attempt-2".into());
            }
            drop(guard);
            tokio::time::timeout(Duration::from_secs(2), async {
                while pending.load(Ordering::SeqCst) != 0 {
                    tokio::task::yield_now().await;
                }
            })
            .await
            .expect("cancellation settlement drains");
            let written = calls(&observer);
            assert_eq!(written.as_array().unwrap().len(), 1);
            assert_eq!(written[0]["failure"]["failure_class"], "cancelled");
            if rebind {
                assert_eq!(written[0]["usage"], Value::Null);
                assert_eq!(written[0]["attempt_id"], "attempt-2");
            } else {
                assert_eq!(written[0]["usage"]["input_tokens"], 451);
                assert_eq!(written[0]["usage"]["output_tokens"], 68);
            }
        });
    }
}
