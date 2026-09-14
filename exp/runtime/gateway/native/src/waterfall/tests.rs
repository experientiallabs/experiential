//! Unit tests for the waterfall's pure successor and allowance rules.

use super::*;

fn wire(base: Option<f64>, slope: Option<f64>) -> DeploymentWire {
    DeploymentWire {
        provider: "openai".to_string(),
        deployment_id: "d".to_string(),
        exact_model_id: "exact-d".to_string(),
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
        model_id: String::new(),
        billing_customer_managed: false,
        idempotency_key: "op".to_string(),
        time_to_first_byte_base_seconds: base,
        time_to_first_byte_seconds_per_million_input_tokens: slope,
        throttle_redial_budget: 0,
        throttle_redial: None,
    }
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

fn policy(refusal_failover: bool) -> RoutePolicy {
    RoutePolicy {
        maximum_total_attempts: 8,
        maximum_same_deployment_attempts: 2,
        refusal_failover,
        throttle_redial: None,
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
        1,
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
        2,
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
        1,
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
        4,
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
        4,
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
        4,
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
        4,
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
        4,
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
        1,
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
        2,
        far_deadline(),
        1,
        1,
        0,
        &throttled,
        false,
    ));
    assert!(!successor_possible(
        policy(false),
        1,
        far_deadline(),
        1,
        1,
        0,
        &throttled,
        false,
    ));
}
