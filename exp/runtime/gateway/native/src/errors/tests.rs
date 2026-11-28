//! Public error envelope and sanitization regressions.

use super::*;

#[test]
fn continuation_receipt_cannot_be_supplied_by_bridge_or_provider() {
    let error: PublicError = serde_json::from_value(json!({
        "status_code": 400, "code": "content_filter", "message": "denied",
        "response_id": "resp_forged"
    }))
    .unwrap();
    assert!(error.response_id.is_none());
    assert!(error.json_body()["error"].get("response_id").is_none());
    let native = error.with_response_id("resp_native".into());
    assert_eq!(native.json_body()["error"]["response_id"], "resp_native");
}

#[test]
fn a_wire_copy_carries_the_tenant_origin_only_when_set() {
    let failure = Failure::new(FailureClass::ProviderAuthentication, "rejected");
    let wire = json!({"failure_class": "provider_authentication"});
    assert_eq!(failure.clone().with_wire_origin(wire.clone()), wire);
    let tenant = failure.with_tenant_origin(true).with_wire_origin(wire);
    assert_eq!(tenant["tenant_origin"], json!(true));
    // The echoed copy deserializes back into an opaque failure.
    let echoed: Failure = serde_json::from_value(json!({
        "failure_class": "provider_authentication",
        "safe_message": "rejected",
        "tenant_origin": true,
    }))
    .unwrap();
    assert_eq!(echoed.public_error().code, "custom_endpoint_failed");
}

#[test]
fn a_tenant_origin_collapses_every_upstream_class_and_keeps_the_rest() {
    let upstream = [
        FailureClass::InvalidRequest,
        FailureClass::Throttled,
        FailureClass::Transport,
        FailureClass::Timeout,
        FailureClass::ProviderAuthentication,
        FailureClass::ProviderNotFound,
        FailureClass::ProviderQuota,
        FailureClass::Refusal,
        FailureClass::MalformedResponse,
        FailureClass::ProviderInternal,
    ];
    for class in upstream {
        let mut failure = Failure::new(class, "connect failed: Connection refused")
            .with_provider_detail(Some("http 404".to_string()))
            .with_rejected_parameter(Some("model".to_string()))
            .with_tenant_origin(true);
        failure.customer_owned = true;
        failure.retry_after_seconds = Some(9);
        let error = failure.public_error();
        assert_eq!(error.status_code, 502, "{class:?}");
        assert_eq!(error.code, "custom_endpoint_failed");
        assert_eq!(error.message, TENANT_ORIGIN_MESSAGE);
        assert_eq!(error.param, None);
        assert_eq!(error.retry_after_seconds, None);
        assert_eq!(error.refusal_reason, None);
        assert_eq!(failure.caller_message(), TENANT_ORIGIN_MESSAGE);
        // Settlement still reads the precise cause.
        assert_eq!(failure.safe_message, "connect failed: Connection refused");
        assert_eq!(failure.failure_class, class);
    }
    for class in [
        FailureClass::Authentication,
        FailureClass::Authorization,
        FailureClass::QuotaExceeded,
        FailureClass::UnsupportedCapability,
        FailureClass::Cancelled,
        FailureClass::Guardrail,
        FailureClass::EmptyCompletion,
        FailureClass::Internal,
        FailureClass::Unavailable,
    ] {
        let failure = Failure::new(class, "gateway's own answer").with_tenant_origin(true);
        assert_ne!(failure.public_error().code, "custom_endpoint_failed");
        assert_eq!(failure.caller_message(), "gateway's own answer");
    }
    // An ordinary rung keeps its distinct answer, in caller wording.
    let house = Failure::new(FailureClass::Transport, "provider transport failed");
    assert_eq!(
        house.with_tenant_origin(false).caller_message(),
        "The gateway could not reach the model's provider. Retry after a short delay; \
             if it keeps failing, choose another model."
    );
    // A relayed provider detail reaches every surface, not only HTTP.
    let rejected = Failure::new(
        FailureClass::InvalidRequest,
        "provider rejected the request; verify the request fields",
    )
    .with_provider_detail(Some("stop must be at most 4 sequences".to_string()));
    assert_eq!(rejected.caller_message(), rejected.public_error().message);
    assert!(rejected
        .caller_message()
        .ends_with(": stop must be at most 4 sequences"));
}

#[test]
fn the_malformed_boundary_keeps_a_tenant_origin_opaque() {
    let failure = Failure::new(FailureClass::MalformedResponse, "unexpected item type")
        .with_tenant_origin(true)
        .boundary();
    assert!(failure.tenant_origin);
    assert_eq!(failure.public_error().message, TENANT_ORIGIN_MESSAGE);
    assert_eq!(
        failure.provider_detail.as_deref(),
        Some("unexpected item type")
    );
}

#[test]
fn a_refusal_is_a_request_error_with_its_own_code_never_a_routing_failure() {
    // The provider processed (and billed) the prompt and answered with a
    // refusal; `all_routes_failed` would file the model's verdict as an
    // infrastructure fault. Mirrors `public_failure_error`.
    let refused = Failure::new(FailureClass::Refusal, "provider refused the request");
    let error = refused.public_error();
    assert_eq!(error.status_code, 400);
    assert_eq!(error.code, "refusal");
    assert_eq!(error.error_type, "invalid_request_error");
    assert_eq!(error.message, "The model's provider declined to answer this request. Rephrase the request or choose another model.");
    assert_eq!(error.retry_after_seconds, None);
    // A refusal built without a reason still names its category.
    assert_eq!(error.refusal_reason, Some(RefusalReason::Unspecified));
    assert_eq!(error.json_body()["error"]["refusal_reason"], "unspecified");
    // Untyped provider failures keep the routing-failure shape.
    let internal = Failure::new(FailureClass::ProviderInternal, "provider failed");
    assert_eq!(internal.public_error().status_code, 502);
    assert_eq!(internal.public_error().code, "all_routes_failed");
}

#[test]
fn every_refusal_reason_renders_its_fixed_phrase_and_wire_name() {
    // (reason, wire name, public message): the message is the caller
    // sentence plus the reason's fixed phrase, never provider prose.
    let table = [
            (
                RefusalReason::CyberPolicy,
                "cyber_policy",
                "The model's provider declined this request (cybersecurity policy). Rephrase the request or \
                 choose another model.",
            ),
            (
                RefusalReason::Cbrn,
                "cbrn",
                "The model's provider declined this request (biological/chemical safety policy). Rephrase the request or \
                 choose another model.",
            ),
            (
                RefusalReason::ContentPolicy,
                "content_policy",
                "The model's provider declined this request (content policy). Rephrase the request or \
                 choose another model.",
            ),
            (
                RefusalReason::Recitation,
                "recitation",
                "The model's provider declined this request (recitation of copyrighted material). Rephrase the request or \
                 choose another model.",
            ),
            (
                RefusalReason::DataInspection,
                "data_inspection",
                "The model's provider declined this request (data inspection). Rephrase the request or \
                 choose another model.",
            ),
            (
                RefusalReason::Unspecified,
                "unspecified",
                "The model's provider declined to answer this request. Rephrase the request or choose another model.",
            ),
        ];
    for (reason, wire, message) in table {
        let failure = Failure::refusal(reason)
            .with_provider_detail(Some("cyber_policy: raw provider prose".into()));
        assert_eq!(failure.failure_class, FailureClass::Refusal);
        assert_eq!(failure.refusal_reason, Some(reason));
        assert_eq!(reason.as_str(), wire);
        let error = failure.public_error();
        // Status, code, and type are unchanged so existing clients behave
        // the same; the reason is the one additive field.
        assert_eq!(error.status_code, 400);
        assert_eq!(error.code, "refusal");
        assert_eq!(error.error_type, "invalid_request_error");
        assert_eq!(error.message, message);
        assert_eq!(error.refusal_reason, Some(reason));
        let body = error.json_body();
        assert_eq!(body["error"]["refusal_reason"], wire);
        assert_eq!(body["error"]["code"], "refusal");
        // The provider's own detail never reaches the caller.
        assert!(!body.to_string().contains("raw provider prose"));
        // The enum round-trips through its snake_case wire name.
        let back: RefusalReason =
            serde_json::from_value(serde_json::Value::String(wire.to_string()))
                .expect("round trip");
        assert_eq!(back, reason);
    }
    // Every other class stays free of the field, on the struct and in
    // the envelope, so non-refusal payloads are byte-identical.
    let throttled = Failure::new(FailureClass::Throttled, "slow down").public_error();
    assert_eq!(throttled.refusal_reason, None);
    assert!(throttled.json_body()["error"]
        .get("refusal_reason")
        .is_none());
    assert!(serde_json::to_value(&throttled)
        .expect("serializable")
        .get("refusal_reason")
        .is_none());
    // A boundary payload without the field still deserializes.
    let legacy: PublicError = serde_json::from_value(serde_json::json!({
        "status_code": 400, "code": "refusal", "message": "m"
    }))
    .expect("field is optional");
    assert_eq!(legacy.refusal_reason, None);
}

#[test]
fn rejected_parameter_reaches_the_public_error_only_for_invalid_requests() {
    let attributed = Failure::new(FailureClass::InvalidRequest, "provider rejected")
        .with_rejected_parameter(Some("input[1].status".to_string()));
    assert_eq!(
        attributed.public_error().param.as_deref(),
        Some("input[1].status")
    );
    // Any other class stays param-free even if a parameter leaked in.
    let internal = Failure::new(FailureClass::ProviderInternal, "provider failed")
        .with_rejected_parameter(Some("input[1].status".to_string()));
    assert_eq!(internal.public_error().param, None);
    // Serde omits the field when absent, so boundary payloads are unchanged.
    let bare = serde_json::to_value(Failure::new(FailureClass::InvalidRequest, "x"))
        .expect("serializable");
    assert!(bare.get("rejected_parameter").is_none());
    let carried = serde_json::to_value(attributed).expect("serializable");
    assert_eq!(
        carried["rejected_parameter"].as_str(),
        Some("input[1].status")
    );
}

#[test]
fn provider_detail_replaces_the_generic_advice_only_for_invalid_requests() {
    let explained = Failure::new(
        FailureClass::InvalidRequest,
        "provider rejected the request; verify the request fields against \
             the model alias capabilities",
    )
    .with_provider_detail(Some("`top_p` is deprecated for this model.".to_string()));
    assert_eq!(
        explained.public_error().message,
        "provider rejected the request: `top_p` is deprecated for this model."
    );
    // Any other class keeps its own message even if a detail leaked in.
    let internal = Failure::new(FailureClass::ProviderInternal, "provider failed")
        .with_provider_detail(Some("account 4711 is over its map".to_string()));
    assert_eq!(internal.public_error().message, "provider failed");
    let bare = serde_json::to_value(Failure::new(FailureClass::InvalidRequest, "x"))
        .expect("serializable");
    assert!(bare.get("provider_detail").is_none());
    let carried = serde_json::to_value(explained).expect("serializable");
    assert_eq!(
        carried["provider_detail"].as_str(),
        Some("`top_p` is deprecated for this model.")
    );
}

#[test]
fn boundary_replaces_malformed_detail_with_the_generic_message() {
    let coerced = Failure::new(FailureClass::MalformedResponse, "specific detail").boundary();
    assert_eq!(
        coerced.safe_message,
        "provider returned a malformed response; retry the request"
    );
    // The specific reason survives as the settlement-facing detail so the
    // ledger names the wire shape that failed, while the public error
    // stays generic (detail relay is scoped to invalid requests).
    assert_eq!(coerced.provider_detail.as_deref(), Some("specific detail"));
    assert_eq!(
        coerced.public_error().message,
        "The model's provider sent an incomplete or invalid response, so the gateway \
             discarded it. Retry the request; if it keeps happening, choose another model."
    );
    let transport = Failure::new(FailureClass::Transport, "kept").boundary();
    assert_eq!(transport.safe_message, "kept");
    assert_eq!(transport.provider_detail, None);
}

#[test]
fn failure_classes_round_trip_through_their_wire_names() {
    for class in [
        FailureClass::InvalidRequest,
        FailureClass::QuotaExceeded,
        FailureClass::MalformedResponse,
        FailureClass::Cancelled,
        FailureClass::Unavailable,
    ] {
        let wire = serde_json::to_value(class).expect("serializable");
        let back: FailureClass = serde_json::from_value(wire).expect("round trip");
        assert_eq!(back.as_str(), class.as_str());
    }
}

#[test]
fn a_throttled_failure_advertises_its_known_window_floored_at_the_default() {
    // The throttled-exhaustion message names the remaining window, so the
    // Retry-After header must advertise the same wait: a fixed 5s header
    // beside a "retry in 30s" body sends honoring clients straight back
    // into the window.
    let mut throttled = Failure::new(FailureClass::Throttled, "retry in 30s");
    throttled.retry_after_seconds = Some(30);
    assert_eq!(throttled.public_error().retry_after_seconds, Some(30));
    // A shorter-than-default window keeps the default floor.
    throttled.retry_after_seconds = Some(2);
    assert_eq!(throttled.public_error().retry_after_seconds, Some(5));
    // Without a known window the fixed default backoff applies.
    let bare = Failure::new(FailureClass::Throttled, "provider throttled the request");
    assert_eq!(bare.public_error().retry_after_seconds, Some(5));
}

#[test]
fn unavailable_maps_to_a_retryable_503() {
    // Parity with the python control plane's UNAVAILABLE mapping: a
    // transient roll condition is a retryable 503, not a closed 500/502.
    let failure = Failure::new(FailureClass::Unavailable, "the gateway is updating");
    let error = failure.public_error();
    assert_eq!(error.status_code, 503);
    assert_eq!(error.code, "gateway_unavailable");
    assert_eq!(error.error_type, "api_error");
    assert_eq!(error.retry_after_seconds, Some(2));
    // The wire name matches the python GatewayFailureClass member so a
    // failure serialized on either side deserializes on the other.
    assert_eq!(FailureClass::Unavailable.as_str(), "unavailable");
}
