//! Image-edit answers: the source-image token subset and the consumed-image leg.

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

fn answer(usage: Value) -> Value {
    json!({"created": 1, "data": [{"b64_json": "aW1hZ2U="}], "usage": usage})
}

#[test]
fn an_openai_edit_answer_carries_its_source_image_tokens() {
    // The shape a GPT image edit answers with (measured live).
    let usage = json!({
        "input_tokens": 1554,
        "input_tokens_details": {"image_tokens": 1536, "text_tokens": 18},
        "output_tokens": 196,
        "output_tokens_details": {"image_tokens": 196, "text_tokens": 0},
        "total_tokens": 1750,
    });
    let (body, settled) =
        public_images(answer(usage.clone()), &admission(1), None).expect("valid answer");
    assert_eq!(body["usage"], usage);
    assert_eq!(settled.input_tokens, Some(1554));
    assert_eq!(settled.image_input_tokens, Some(1536));
    assert_eq!(settled.output_tokens, Some(196));
}

#[test]
fn an_absent_breakdown_is_unknown_and_a_reported_zero_is_zero() {
    let absent = json!({"input_tokens": 11, "output_tokens": 272});
    let (_, settled) = public_images(answer(absent), &admission(1), None).expect("absent");
    assert_eq!(settled.image_input_tokens, None);
    let zero = json!({
        "input_tokens": 11,
        "input_tokens_details": {"text_tokens": 11, "image_tokens": 0},
        "output_tokens": 272,
    });
    let (_, settled) = public_images(answer(zero), &admission(1), None).expect("zero");
    assert_eq!(settled.image_input_tokens, Some(0));
}

#[test]
fn an_image_subset_past_its_input_total_is_held_inside_it() {
    let over = json!({
        "input_tokens": 10,
        "input_tokens_details": {"image_tokens": 50},
        "output_tokens": 1,
    });
    let (_, settled) = public_images(answer(over), &admission(1), None).expect("clamped");
    assert_eq!(settled.image_input_tokens, Some(10));
}

#[test]
fn a_per_image_edit_bills_its_source_images_beside_its_output() {
    let payload = json!({
        "data": [{"b64_json": "x", "mime_type": "image/jpeg"}, {"b64_json": "y"}],
        "usage": {"cost_in_usd_ticks": 1_300_000_000_u64},
    });
    let units = ImageUnitBilling {
        variant: "edit".to_string(),
        input_images: Some(3),
    };
    let (_, settled) = public_images(payload, &admission(2), Some(&units)).expect("valid answer");
    let billed = settled.billed_units.expect("image units");
    assert_eq!(billed.variant, "edit");
    assert_eq!(billed.quantity_milli, 2_000);
    assert_eq!(billed.input_quantity_milli, Some(3_000));
    let wire = serde_json::to_value(&billed).expect("serializes");
    assert_eq!(wire["input_quantity_milli"], json!(3_000));
}

#[test]
fn a_generation_bills_no_consumed_leg_and_serializes_without_it() {
    let units = ImageUnitBilling {
        variant: String::new(),
        input_images: None,
    };
    let payload = json!({"data": [{"b64_json": "x"}]});
    let (_, settled) = public_images(payload, &admission(1), Some(&units)).expect("valid");
    let billed = settled.billed_units.expect("image units");
    assert_eq!(billed.input_quantity_milli, None);
    let wire = serde_json::to_value(&billed).expect("serializes");
    assert!(wire.get("input_quantity_milli").is_none());
}

#[test]
fn admission_names_the_multipart_wire_and_the_billed_source_images() {
    let shaped: ImagesAdmission = serde_json::from_value(json!({
        "request_id": "r", "alias": "a", "alias_revision_id": "v",
        "exact_model_id": "m", "route_reason": "direct", "route": [],
        "image_count": 1, "maximum_total_attempts": 1,
        "maximum_same_deployment_attempts": 1,
        "wire_shapes": [{"shape": "openai", "multipart": true}, {"shape": "openai"}],
        "unit_billing": [null, {"variant": "edit", "input_images": 2}],
    }))
    .expect("edit admission");
    assert!(shaped.wire_shapes[0].multipart);
    assert!(!shaped.wire_shapes[1].multipart);
    let billing = shaped.unit_billing[1].as_ref().expect("per-image rung");
    assert_eq!(billing.input_images, Some(2));
}
