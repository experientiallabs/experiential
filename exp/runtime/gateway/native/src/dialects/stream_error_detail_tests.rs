//! Stream-error detail: a provider's in-stream failure carries its bounded diagnosis.

use super::*;

fn frame(payload: serde_json::Value) -> SseEvent {
    SseEvent {
        event: None,
        data: payload.to_string(),
    }
}

fn failed_detail(events: &[Event]) -> Option<String> {
    match events.last() {
        Some(Event::Failed(failure)) => failure.provider_detail.clone(),
        other => panic!("expected a failed terminal, got {other:?}"),
    }
}

#[test]
fn secret_shaped_words_are_masked_out_of_the_detail_line() {
    // The identifier screen treats any letter+digit label as a handle,
    // which covers key and token shapes: the word is masked and the
    // sentence around it survives, for every dialect that feeds the
    // shared detail path, Bedrock exception messages included.
    for (message, masked) in [
        (
            "Invalid key sk-abc123def provided.",
            "Invalid key [redacted] provided.",
        ),
        (
            "The access key AKIA9X7EXAMPLE is not authorized for this model.",
            "The access key [redacted] is not authorized for this model.",
        ),
        (
            "Bearer eyJhbGciOi9 was rejected.",
            "Bearer [redacted] was rejected.",
        ),
    ] {
        assert_eq!(
            provider_error_detail(None, Some(message), &[]).as_deref(),
            Some(masked),
            "a credential-shaped word must be masked: {message}"
        );
        assert_eq!(
            provider_error_detail(Some("validation_error"), Some(message), &[]).as_deref(),
            Some(format!("validation_error: {masked}").as_str()),
            "the code rides with the masked sentence: {message}"
        );
    }
}

#[test]
fn provider_error_detail_is_one_bounded_line() {
    assert_eq!(provider_error_detail(None, None, &[]), None);
    assert_eq!(
        provider_error_detail(Some("server_error"), None, &[]).as_deref(),
        Some("server_error")
    );
    assert_eq!(
        provider_error_detail(
            Some("server_error"),
            Some("The model failed  to respond."),
            &[]
        )
        .as_deref(),
        Some("server_error: The model failed to respond.")
    );
    // The line cuts at the first control character: a payload dump never
    // rides past its first row.
    assert_eq!(
        provider_error_detail(None, Some("first line\nsecond line"), &[]).as_deref(),
        Some("first line")
    );
    // A hostile code reduces to the shared identifier token.
    assert_eq!(
        provider_error_detail(Some("weird code!{}"), None, &[]).as_deref(),
        Some("non-identifier")
    );
    // The composed detail never exceeds the python provider_detail bound.
    let long = "x".repeat(400);
    let bounded = provider_error_detail(Some("code"), Some(&long), &[]).expect("bounded");
    assert_eq!(bounded.chars().count(), 240);
}

/// Every dialect's provider-declared stream failure carries its bounded
/// mechanism into `provider_detail` (ledger-only for these classes), so
/// the next post-open kill is diagnosable from the ledger instead of an
/// opaque "provider stream failed" (2026-09-05: gpt-6-astra kills ~120ms
/// post-dispatch across 20 orgs with the reason discarded on the wire).
#[test]
fn responses_failed_terminal_carries_its_error_detail() {
    let mut normalizer = Normalizer::new(Dialect::OpenAiResponses);
    let events = normalizer
        .feed(&frame(serde_json::json!({
            "type": "response.failed",
            "response": {
                "status": "failed",
                "error": {"code": "server_error", "message": "The model failed."},
            },
        })))
        .expect("failed terminal normalizes");
    assert_eq!(
        failed_detail(&events).as_deref(),
        Some("server_error: The model failed.")
    );
}

#[test]
fn responses_error_frame_carries_its_error_detail() {
    let mut normalizer = Normalizer::new(Dialect::OpenAiResponses);
    let events = normalizer
        .feed(&frame(serde_json::json!({
            "type": "error",
            "code": "rate_limit_exceeded",
            "message": "Rate limit reached for gpt-6-astra.",
            "param": null,
            "sequence_number": 1,
        })))
        .expect("error frame normalizes");
    // The model id trips the identifier screen (letters+digits label), so
    // it is masked while the code and the sentence around it survive: the
    // mechanism stays named without relaying a label-shaped word.
    assert_eq!(
        failed_detail(&events).as_deref(),
        Some("rate_limit_exceeded: Rate limit reached for [redacted].")
    );
}

#[test]
fn compatible_error_frame_carries_its_error_detail() {
    let mut normalizer = Normalizer::new(Dialect::OpenAiCompatible);
    let events = normalizer
        .feed(&frame(serde_json::json!({
            "error": {"code": 502, "message": "upstream connect error"},
        })))
        .expect("error frame normalizes");
    assert_eq!(
        failed_detail(&events).as_deref(),
        Some("502: upstream connect error")
    );
}

#[test]
fn anthropic_error_frame_carries_its_error_detail() {
    let mut normalizer = Normalizer::new(Dialect::AnthropicMessages);
    let events = normalizer
        .feed(&frame(serde_json::json!({
            "type": "error",
            "error": {"type": "overloaded_error", "message": "Overloaded"},
        })))
        .expect("error frame normalizes");
    assert_eq!(
        failed_detail(&events).as_deref(),
        Some("overloaded_error: Overloaded")
    );
}

#[test]
fn gemini_error_envelope_carries_its_error_detail() {
    let mut normalizer = Normalizer::new(Dialect::GeminiGenerateContent);
    let events = normalizer
        .feed(&frame(serde_json::json!({
            "error": {"code": 503, "status": "UNAVAILABLE", "message": "Overloaded."},
        })))
        .expect("error envelope normalizes");
    assert_eq!(
        failed_detail(&events).as_deref(),
        Some("UNAVAILABLE: Overloaded.")
    );
}

#[test]
fn unparsable_stream_frames_name_size_and_parse_position() {
    let mut normalizer = Normalizer::new(Dialect::OpenAiResponses);
    let failure = normalizer
        .feed(&SseEvent {
            event: None,
            data: "<html>502 Bad Gateway</html>".to_string(),
        })
        .expect_err("a non-JSON frame is malformed");
    assert!(
        failure.safe_message.contains("not valid JSON")
            && failure.safe_message.contains("(28 bytes)"),
        "reason must carry the parse diagnosis: {}",
        failure.safe_message
    );
}
