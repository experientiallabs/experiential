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
            reasoning_tokens: Some(5),
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
