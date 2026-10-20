//! Cache TTL usage survives Responses gateways without losing presence.

use super::*;

#[test]
fn responses_unreported_meters_survive_two_hops_with_required_integers() {
    for count in [None, Some(0)] {
        let mut usage = Usage {
            input_tokens: Some(100),
            output_tokens: Some(10),
            cached_input_tokens: count,
            cache_creation_input_tokens: count,
            reasoning_tokens: count,
            ..Usage::default()
        };
        for _ in 0..2 {
            let wire = responses_usage(Some(&usage));
            assert_eq!(wire["input_tokens_details"]["cached_tokens"], 0);
            assert_eq!(wire["input_tokens_details"]["cache_write_tokens"], 0);
            assert_eq!(wire["output_tokens_details"]["reasoning_tokens"], 0);
            assert_eq!(
                wire.get("unreported_token_details").is_some(),
                count.is_none()
            );
            usage = crate::events::openai_usage(Some(&wire)).unwrap().unwrap();
            assert_eq!(usage.cached_input_tokens, count);
            assert_eq!(usage.cache_creation_input_tokens, count);
            assert_eq!(usage.reasoning_tokens, count);
        }
    }
}

#[test]
fn responses_cache_ttl_survives_two_gateway_hops_without_inventing_zero() {
    for hour in [None, Some(0), Some(60)] {
        let mut usage = Usage {
            input_tokens: Some(120),
            output_tokens: Some(10),
            cached_input_tokens: Some(10),
            cache_creation_input_tokens: Some(100),
            cache_creation_1h_input_tokens: hour,
            cache_creation_12h_input_tokens: None,
            reasoning_tokens: Some(5),
            billed_units: None,
            cache_writes_read_back: false,
            image_output_tokens: None,
        };
        for _ in 0..2 {
            let wire = responses_usage(Some(&usage));
            let details = wire["input_tokens_details"].as_object().unwrap();
            assert_eq!(
                details.get("cache_write_1h_tokens"),
                hour.map(|n| json!(n)).as_ref()
            );
            usage = crate::events::openai_usage(Some(&wire)).unwrap().unwrap();
            assert_eq!(usage.cache_creation_input_tokens, Some(100));
            assert_eq!(usage.cache_creation_1h_input_tokens, hour);
            assert_eq!(usage.reasoning_tokens, Some(5));
        }
    }
}

#[test]
fn responses_keep_writes_read_back_disjoint_for_two_gateway_hops() {
    // Responses parsers have no overlapping mode, so a gateway hop must see
    // disjoint legs; only Chat folds read-back writes into cached_tokens.
    let read_back = Usage {
        input_tokens: Some(11_933),
        output_tokens: Some(9),
        cached_input_tokens: Some(0),
        cache_creation_input_tokens: Some(10_663),
        cache_writes_read_back: true,
        ..Usage::default()
    };
    let wire = responses_usage(Some(&read_back));
    assert_eq!(wire["input_tokens_details"]["cached_tokens"], 0);
    assert_eq!(wire["input_tokens_details"]["cache_write_tokens"], 10_663);
    let parsed = crate::events::openai_usage(Some(&wire)).unwrap().unwrap();
    assert_eq!(parsed.cache_creation_input_tokens, Some(10_663));
}

/// Responses reports the same estimate in `output_tokens_details`.
#[test]
fn responses_reports_an_estimated_reasoning_count_when_the_provider_publishes_none() {
    let usage = Usage {
        input_tokens: Some(48),
        output_tokens: Some(40),
        ..Usage::default()
    };
    let events = vec![
        Event::ThinkingDelta {
            index: 0,
            delta: "1001 is 7 x 11 x 13.".into(),
        },
        Event::TextDelta("No.".into()),
        Event::Usage(usage),
        Event::Completed,
    ];
    let completed = completed_responses_body(
        "request-1",
        "claude",
        1,
        ResponsesEnvelope::default(),
        &events,
    )
    .expect("completed body must encode");
    let wire = &completed.body["usage"];
    assert_eq!(wire["output_tokens_details"]["reasoning_tokens"], 39);
    assert_eq!(wire["estimated_token_details"], json!(["reasoning_tokens"]));
    assert!(wire["unreported_token_details"]
        .as_array()
        .is_none_or(|fields| !fields.contains(&json!("reasoning_tokens"))));
    assert_eq!(
        completed.usage.expect("usage aggregates").reasoning_tokens,
        None
    );
}

#[test]
fn the_twelve_hour_write_split_survives_two_gateway_hops() {
    let mut usage = Usage {
        input_tokens: Some(3250),
        output_tokens: Some(5),
        cached_input_tokens: Some(0),
        cache_creation_input_tokens: Some(3247),
        cache_creation_12h_input_tokens: Some(2000),
        reasoning_tokens: Some(0),
        ..Usage::default()
    };
    for _ in 0..2 {
        let wire = responses_usage(Some(&usage));
        assert_eq!(
            wire["input_tokens_details"]["cache_write_tokens_details"],
            json!({"ttl_30m_tokens": 1247, "ttl_12h_tokens": 2000})
        );
        usage = crate::events::openai_usage(Some(&wire)).unwrap().unwrap();
        assert_eq!(usage.cache_creation_input_tokens, Some(3247));
        assert_eq!(usage.cache_creation_12h_input_tokens, Some(2000));
    }
    // A usage with no observed 12-hour count names no split.
    usage.cache_creation_12h_input_tokens = None;
    let wire = responses_usage(Some(&usage));
    assert!(wire["input_tokens_details"]
        .get("cache_write_tokens_details")
        .is_none());
}
