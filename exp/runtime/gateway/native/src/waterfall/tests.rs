//! Unit tests for the waterfall's pure successor and allowance rules.

use std::collections::HashMap;

use super::*;

fn wire(base: Option<f64>, slope: Option<f64>) -> DeploymentWire {
    DeploymentWire {
        native_tool_translation: Default::default(),
        provider: "openai".to_string(),
        deployment_id: "d".to_string(),
        exact_model_id: "fixture-model".into(),
        dialect: "openai_compatible".to_string(),
        url: "https://provider.test".to_string(),
        headers: HashMap::new(),
        timeout_seconds: 60.0,
        upstream_payload: Value::Null,
        upstream_body: None,
        fireworks_reasoning_route_sha256: None,
        hunyuan_reasoning_route_sha256: None,
        reasoning_output_exposed: false,
        stop_sequences: Vec::new(),
        serialize_tool_calls: false,
        image_output: false,
        model_id: String::new(),
        billing_customer_managed: false,
        idempotency_key: "op".to_string(),
        time_to_first_byte_base_seconds: base,
        time_to_first_byte_seconds_per_million_input_tokens: slope,
        time_to_first_token_base_seconds: None,
        throttle_redial_budget: 0,
        throttle_redial: None,
        failover_only_on: None,
        zdr_constrained: false,
    }
}

/// `length` unrestricted rungs: the historical route shape.
fn plain(length: usize) -> Vec<DeploymentWire> {
    (0..length).map(|_| wire(None, None)).collect()
}

#[test]
fn first_byte_allowance_scales_with_input_and_honors_overrides() {
    let default_base = Duration::from_secs(15);
    // No overrides, tiny request: effectively the flat default.
    let flat = first_byte_allowance(&wire(None, None), default_base, 240.0, 100.0);
    assert!((flat.as_secs_f64() - 15.024).abs() < 1e-6);
    // No overrides, one million approximate tokens: base plus the
    // full default slope.
    let scaled = first_byte_allowance(&wire(None, None), default_base, 240.0, 1_000_000.0);
    assert!((scaled.as_secs_f64() - 255.0).abs() < 1e-6);
    // Deployment overrides replace both the base and the slope.
    let overridden = first_byte_allowance(
        &wire(Some(30.0), Some(60.0)),
        default_base,
        240.0,
        500_000.0,
    );
    assert!((overridden.as_secs_f64() - 60.0).abs() < 1e-6);
    // A zero slope pins the flat bound regardless of input size.
    let pinned = first_byte_allowance(&wire(None, Some(0.0)), default_base, 240.0, 9e9);
    assert!((pinned.as_secs_f64() - 15.0).abs() < 1e-6);
}

#[test]
fn open_phase_bound_ignores_the_per_chunk_timeout() {
    // A deployment authored with a 120 s first-byte allowance on a 60 s
    // per-chunk wire waits the full 120 s for headers; only the request
    // deadline can cut it shorter.
    let allowance = first_byte_allowance(
        &wire(Some(120.0), Some(0.0)),
        Duration::from_secs(15),
        240.0,
        0.0,
    );
    assert!((allowance.as_secs_f64() - 120.0).abs() < 1e-6);
    assert_eq!(
        open_phase_bound(Duration::from_secs(600), allowance),
        Duration::from_secs(120)
    );
    assert_eq!(
        open_phase_bound(Duration::from_secs(45), allowance),
        Duration::from_secs(45)
    );
    // The default allowance stays the fail-fast bound for a small prompt.
    let small = first_byte_allowance(&wire(None, None), Duration::from_secs(15), 240.0, 1_000.0);
    assert!(open_phase_bound(Duration::from_secs(600), small) < Duration::from_secs(16));
}

fn policy(refusal_failover: bool) -> RoutePolicy {
    RoutePolicy {
        maximum_total_attempts: 8,
        maximum_same_deployment_attempts: 2,
        refusal_failover,
        throttle_redial: None,
        physical_route_cap: None,
        backoff: None,
    }
}

fn far_deadline() -> Instant {
    Instant::now() + Duration::from_secs(60)
}

#[test]
fn successor_requires_capacity_and_an_eligible_class() {
    let retryable = Failure::new(FailureClass::ProviderInternal, "boom").with_retry(true, true);
    // Same-deployment retry within the per-deployment cap.
    assert!(successor_possible(
        policy(false),
        &plain(1),
        far_deadline(),
        1,
        1,
        0,
        &retryable,
        false,
    ));
    // The per-deployment cap forbids a redial but failover still runs.
    assert!(successor_possible(
        policy(false),
        &plain(2),
        far_deadline(),
        2,
        2,
        0,
        &retryable,
        false,
    ));
    // A single-deployment route with the redial cap reached is exhausted.
    assert!(!successor_possible(
        policy(false),
        &plain(1),
        far_deadline(),
        2,
        2,
        0,
        &retryable,
        false,
    ));
    // The hard total cap ends the ladder regardless of class.
    assert!(!successor_possible(
        policy(false),
        &plain(4),
        far_deadline(),
        8,
        1,
        0,
        &retryable,
        false,
    ));
    // An expired deadline ends the ladder.
    assert!(!successor_possible(
        policy(false),
        &plain(4),
        Instant::now(),
        1,
        1,
        0,
        &retryable,
        false,
    ));
}

#[test]
fn ineligible_classes_never_advance_without_refusal_opt_in() {
    let invalid = Failure::new(FailureClass::InvalidRequest, "bad request");
    assert!(!successor_possible(
        policy(false),
        &plain(4),
        far_deadline(),
        1,
        1,
        0,
        &invalid,
        false,
    ));
    let refusal = Failure::new(FailureClass::Refusal, "provider refused the request");
    assert!(!successor_possible(
        policy(false),
        &plain(4),
        far_deadline(),
        1,
        1,
        0,
        &refusal,
        false,
    ));
    // The refusal advances only when the alias revision opted in.
    assert!(successor_possible(
        policy(true),
        &plain(4),
        far_deadline(),
        1,
        1,
        0,
        &refusal,
        true,
    ));
    // Refusal failover cannot pass the last deployment.
    assert!(!successor_possible(
        policy(true),
        &plain(1),
        far_deadline(),
        1,
        1,
        0,
        &refusal,
        true,
    ));
}

#[test]
fn failover_only_classes_skip_the_redial_and_advance() {
    let throttled = Failure::new(FailureClass::Throttled, "throttled").with_retry(false, true);
    assert!(successor_possible(
        policy(false),
        &plain(2),
        far_deadline(),
        1,
        1,
        0,
        &throttled,
        false,
    ));
    assert!(!successor_possible(
        policy(false),
        &plain(1),
        far_deadline(),
        1,
        1,
        0,
        &throttled,
        false,
    ));
}

fn usage(output_tokens: Option<u64>, reasoning_tokens: Option<u64>) -> Usage {
    Usage {
        input_tokens: Some(9),
        output_tokens,
        cached_input_tokens: None,
        cache_creation_input_tokens: None,
        cache_creation_1h_input_tokens: None,
        reasoning_tokens,
    }
}

#[test]
fn a_billed_stop_with_no_output_is_an_empty_completion() {
    // The live OpenRouter DeepSeek shape: reasoning billed, nothing sent.
    assert!(billed_empty_completion(
        &Event::Completed,
        Some(&usage(Some(147), Some(148)))
    ));
    // One EOS token and nothing else is still a paid-for empty answer.
    assert!(billed_empty_completion(
        &Event::Completed,
        Some(&usage(Some(1), Some(0)))
    ));
    // A wire that reports thinking outside the output leg still counts it.
    assert!(billed_empty_completion(
        &Event::Completed,
        Some(&usage(Some(0), Some(30)))
    ));
    let failure = Failure::empty_completion();
    // The model's answer was nothing: its own class (never the health
    // circuit's operational set), a 400 the SDKs do not auto-retry, and the
    // pre-commit redial + ladder kept.
    assert_eq!(failure.failure_class, FailureClass::EmptyCompletion);
    assert!(failure.retryable_same_deployment && failure.failover_eligible);
    let public = failure.public_error();
    assert_eq!(public.status_code, 400);
    assert_eq!(public.code, "empty_completion");
    assert_eq!(public.error_type, "invalid_request_error");
    assert_eq!(FailureClass::EmptyCompletion.as_str(), "empty_completion");
}

#[test]
fn honest_output_less_endings_are_not_empty_completions() {
    // A zero-token stop is the provider saying nothing, not billing for it.
    assert!(!billed_empty_completion(
        &Event::Completed,
        Some(&usage(Some(0), None))
    ));
    // No usage report proves nothing was spent.
    assert!(!billed_empty_completion(&Event::Completed, None));
    // Truncation, a stop sequence, and a paused turn keep their own shapes.
    let billed = usage(Some(16), None);
    assert!(!billed_empty_completion(&Event::Incomplete, Some(&billed)));
    assert!(!billed_empty_completion(
        &Event::StoppedAtSequence("END".to_string()),
        Some(&billed)
    ));
    assert!(!billed_empty_completion(&Event::PausedTurn, Some(&billed)));
}

#[test]
fn a_stop_with_no_usage_report_and_no_output_is_an_unreported_empty_completion() {
    // The live Meta muse-spark shape: empty stop, no usage frame at all.
    assert!(unreported_empty_completion(&Event::Completed, None));
    // A report of zero tokens is the provider accounting for "nothing".
    assert!(!unreported_empty_completion(
        &Event::Completed,
        Some(&usage(Some(0), None))
    ));
    // A billed stop is the billed twin's case, never this one.
    assert!(!unreported_empty_completion(
        &Event::Completed,
        Some(&usage(Some(12), None))
    ));
    // Truncation, a stop sequence and a paused turn keep their own shapes.
    assert!(!unreported_empty_completion(&Event::Incomplete, None));
    assert!(!unreported_empty_completion(
        &Event::StoppedAtSequence("END".to_string()),
        None
    ));
    assert!(!unreported_empty_completion(&Event::PausedTurn, None));
}

#[test]
fn an_image_output_rung_answers_its_empty_completion_without_redial_or_ladder() {
    // The chat normalizers carry no image event, so an image generation is
    // always an empty completion: a redial would bill the house a second
    // whole image for the same nothing.
    let mut image = wire(None, None);
    image.image_output = true;
    let failure = empty_completion_failure(&image);
    assert_eq!(failure.failure_class, FailureClass::EmptyCompletion);
    assert!(!failure.retryable_same_deployment && !failure.failover_eligible);
    // Every other rung keeps the redial and the ladder: the empty answer is
    // not deterministic there.
    let text = wire(None, None);
    let failure = empty_completion_failure(&text);
    assert!(failure.retryable_same_deployment && failure.failover_eligible);
}

#[test]
fn first_token_allowance_has_its_own_base_and_shares_the_input_slope() {
    // The first-token base is independent of the header base (a thinking
    // model on a chat wire answers its headers at once and its first token a
    // minute later), and the deployment override on it wins; the slope is
    // the header allowance's, override or default.
    let plain = wire(None, None);
    let allowance = first_token_allowance(&plain, Duration::from_secs(120), 240.0, 1_000_000.0);
    assert_eq!(allowance, Duration::from_secs_f64(360.0));
    let mut overridden = wire(None, Some(0.0));
    overridden.time_to_first_token_base_seconds = Some(30.0);
    let allowance =
        first_token_allowance(&overridden, Duration::from_secs(120), 240.0, 1_000_000.0);
    assert_eq!(allowance, Duration::from_secs_f64(30.0));
    // The header allowance is untouched by the token base.
    let header = first_byte_allowance(&overridden, Duration::from_secs(15), 240.0, 0.0);
    assert_eq!(header, Duration::from_secs_f64(15.0));
}
