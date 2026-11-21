//! Answer-shape tests for the native images surface (`route_images.rs`).

use super::*;

fn admission(image_count: usize) -> ImagesAdmission {
    ImagesAdmission {
        request_id: "request-1".to_string(),
        alias: "painter".to_string(),
        alias_revision_id: "revision-one".to_string(),
        exact_model_id: "model-exact".to_string(),
        route_reason: "direct".to_string(),
        route: Vec::new(),
        image_count,
        unit_billing: Vec::new(),
        wire_shapes: Vec::new(),
        maximum_total_attempts: 8,
        maximum_same_deployment_attempts: 2,
    }
}

#[test]
fn provider_answer_keeps_images_and_bills_token_usage() {
    let payload = json!({
        "created": 1_700_000_000,
        "background": "opaque",
        "output_format": "png",
        "quality": "low",
        "size": "1024x1024",
        "data": [
            {"b64_json": "aW1hZ2U=", "revised_prompt": "a cat"},
            {"url": "https://example.invalid/img.png"},
        ],
        "usage": {
            "input_tokens": 11,
            "output_tokens": 272,
            "total_tokens": 283,
            "input_tokens_details": {"text_tokens": 11, "image_tokens": 0},
        },
    });
    let (body, usage) = public_images(payload.clone(), &admission(2), None).expect("valid answer");
    assert_eq!(body["data"], payload["data"]);
    assert_eq!(body["usage"], payload["usage"]);
    assert_eq!(body["created"], json!(1_700_000_000));
    assert_eq!(body["quality"], json!("low"));
    assert_eq!(usage.input_tokens, Some(11));
    assert_eq!(usage.output_tokens, Some(272));
}

#[test]
fn provider_answer_carries_the_image_output_and_cache_read_subsets() {
    // gpt-image-1.5 answer shape (live, 2026-11-21): generated images and
    // text both inside output_tokens, priced at different rates.
    let payload = json!({
        "data": [{"b64_json": "aW1hZ2U="}],
        "usage": {
            "input_tokens": 1_554,
            "output_tokens": 403,
            "input_tokens_details": {"text_tokens": 18, "image_tokens": 1_536, "cached_tokens": 1_024},
            "output_tokens_details": {"image_tokens": 272, "text_tokens": 131},
        },
    });
    let (_, usage) = public_images(payload, &admission(1), None).expect("valid answer");
    assert_eq!(usage.image_output_tokens, Some(272));
    assert_eq!(usage.image_input_tokens, Some(1_536));
    assert_eq!(usage.cached_input_tokens, Some(1_024));

    // No breakdown: the subsets stay unknown, never an invented 0.
    let bare = json!({
        "data": [{"b64_json": "aW1hZ2U="}],
        "usage": {"input_tokens": 11, "output_tokens": 272},
    });
    let (_, usage) = public_images(bare, &admission(1), None).expect("valid answer");
    assert_eq!(usage.image_output_tokens, None);
    assert_eq!(usage.cached_input_tokens, None);

    // A subset above its total is clamped to the total.
    let over = json!({
        "data": [{"b64_json": "aW1hZ2U="}],
        "usage": {
            "input_tokens": 10,
            "output_tokens": 100,
            "input_tokens_details": {"cached_tokens": 50},
            "output_tokens_details": {"image_tokens": 500},
        },
    });
    let (_, usage) = public_images(over, &admission(1), None).expect("valid answer");
    assert_eq!(usage.image_output_tokens, Some(100));
    assert_eq!(usage.cached_input_tokens, Some(10));
}

#[test]
fn unbilled_short_or_imageless_answers_are_malformed() {
    let unbilled = json!({"created": 1, "data": [{"b64_json": "aW1hZ2U="}]});
    let short =
        json!({"data": [{"b64_json": "x"}], "usage": {"input_tokens": 1, "output_tokens": 1}});
    let imageless = json!({
        "data": [{"revised_prompt": "only text"}, {"b64_json": "x"}],
        "usage": {"input_tokens": 1, "output_tokens": 1},
    });
    for payload in [unbilled, short, imageless] {
        let failure = public_images(payload, &admission(2), None).expect_err("malformed");
        assert_eq!(failure.failure_class, FailureClass::MalformedResponse);
        assert!(failure.failover_eligible);
    }
}

#[test]
fn a_per_image_rung_bills_the_images_returned_not_tokens() {
    let payload = json!({
        "created": 1,
        "data": [{"url": "https://example.invalid/a.png"}, {"b64_json": "x"}],
        "usage": {"input_tokens": 9, "output_tokens": 0},
    });
    let units = ImageUnitBilling {
        variant: "hd/1024x1024".to_string(),
        input_images: None,
    };
    let (body, usage) =
        public_images(payload.clone(), &admission(2), Some(&units)).expect("valid answer");
    assert_eq!(body["usage"], payload["usage"]);
    assert_eq!(usage.input_tokens, None);
    assert_eq!(usage.output_tokens, None);
    let billed = usage.billed_units.expect("image units");
    assert_eq!(billed.kind, "image");
    assert_eq!(billed.variant, "hd/1024x1024");
    assert_eq!(billed.quantity_milli, 2_000);
    let usageless = json!({"data": [{"url": "https://example.invalid/a.png"}]});
    let (body, usage) =
        public_images(usageless, &admission(1), Some(&units)).expect("usageless answer");
    assert!(body.get("usage").is_none());
    assert_eq!(usage.billed_units.expect("units").quantity_milli, 1_000);
}

fn stability_shape() -> WireShape {
    WireShape {
        shape: "stability".to_string(),
        output_format: Some("jpeg".to_string()),
        multipart: false,
    }
}

fn flat_units() -> ImageUnitBilling {
    ImageUnitBilling {
        variant: String::new(),
        input_images: None,
    }
}

#[test]
fn a_stability_answer_becomes_one_billed_b64_image() {
    let payload = json!({
        "seeds": [2_130_420_379_u64],
        "finish_reasons": [null],
        "images": ["aW1hZ2U="],
    });
    let (body, usage) = public_stability_images(
        payload,
        &admission(1),
        Some(&flat_units()),
        &stability_shape(),
    )
    .expect("valid answer");
    assert_eq!(body["data"], json!([{"b64_json": "aW1hZ2U="}]));
    assert_eq!(body["output_format"], "jpeg");
    assert!(body["created"].as_u64().is_some_and(|created| created > 0));
    assert!(
        body.get("seeds").is_none(),
        "provider-only fields stay private"
    );
    let billed = usage.billed_units.expect("image units");
    assert_eq!(billed.kind, "image");
    assert_eq!(billed.variant, "");
    assert_eq!(billed.quantity_milli, 1_000);
    assert_eq!(usage.input_tokens, None);
}

#[test]
fn a_stability_content_filter_is_a_refusal_naming_only_its_subject() {
    for (reason, subject) in [
        ("Filter reason: prompt", "prompt"),
        ("Filter reason: output image", "output image"),
        ("Filter reason: input image", "input image"),
        ("Filter reason: something new", "content"),
    ] {
        let payload = json!({"finish_reasons": [reason]});
        let failure = public_stability_images(
            payload,
            &admission(1),
            Some(&flat_units()),
            &stability_shape(),
        )
        .expect_err("filtered");
        assert_eq!(failure.failure_class, FailureClass::Refusal);
        assert!(failure.safe_message.ends_with(&format!("({subject})")));
        assert!(!failure.retryable_same_deployment);
        assert!(!failure.failover_eligible);
    }
}

#[test]
fn a_stability_inference_error_is_a_retryable_provider_failure() {
    let payload = json!({"finish_reasons": ["Inference error"], "images": []});
    let failure = public_stability_images(
        payload,
        &admission(1),
        Some(&flat_units()),
        &stability_shape(),
    )
    .expect_err("inference error");
    assert_eq!(failure.failure_class, FailureClass::ProviderInternal);
    assert!(failure.retryable_same_deployment);
    assert!(failure.failover_eligible);
}

#[test]
fn a_stability_answer_without_its_image_or_price_is_malformed() {
    for payload in [
        json!({"finish_reasons": [null]}),
        json!({"finish_reasons": [null], "images": []}),
        json!({"finish_reasons": [null], "images": [""]}),
        json!({"finish_reasons": [null], "images": [7]}),
        json!(["not", "an", "object"]),
    ] {
        let failure = public_stability_images(
            payload,
            &admission(1),
            Some(&flat_units()),
            &stability_shape(),
        )
        .expect_err("malformed");
        assert_eq!(failure.failure_class, FailureClass::MalformedResponse);
    }
    let unpriced = public_stability_images(
        json!({"finish_reasons": [null], "images": ["aW1hZ2U="]}),
        &admission(1),
        None,
        &stability_shape(),
    )
    .expect_err("no per-image price");
    assert_eq!(unpriced.failure_class, FailureClass::MalformedResponse);
}

#[test]
fn admission_wire_shapes_default_to_the_openai_answer() {
    let without: ImagesAdmission = serde_json::from_value(json!({
        "request_id": "r", "alias": "a", "alias_revision_id": "v",
        "exact_model_id": "m", "route_reason": "direct", "route": [],
        "image_count": 1, "maximum_total_attempts": 1,
        "maximum_same_deployment_attempts": 1,
    }))
    .expect("older admission shape");
    assert!(without.wire_shapes.is_empty());
    let with: ImagesAdmission = serde_json::from_value(json!({
        "request_id": "r", "alias": "a", "alias_revision_id": "v",
        "exact_model_id": "m", "route_reason": "direct", "route": [],
        "image_count": 1, "maximum_total_attempts": 1,
        "maximum_same_deployment_attempts": 1,
        "wire_shapes": [{"shape": "stability", "output_format": "png"}, {"shape": "openai"}],
    }))
    .expect("shaped admission");
    assert!(with.wire_shapes[0].stability());
    assert!(!with.wire_shapes[1].stability());
    assert_eq!(with.wire_shapes[1].output_format, None);
}
