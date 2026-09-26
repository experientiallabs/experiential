//! `Normalizer::recover_abnormal_end` scoping: a Gemini stream that emitted content
//! and then ended abnormally settles Incomplete; every other dialect keeps the failure.

use super::*;

fn feed_text(normalizer: &mut Normalizer, text: &str) {
    let frame = SseEvent {
        event: None,
        data: serde_json::json!({
            "candidates": [{"content": {"parts": [{"text": text}]}}]
        })
        .to_string(),
    };
    let events = normalizer.feed(&frame).expect("content frame normalizes");
    assert!(events.iter().any(Event::is_output_token));
}

fn incoming() -> Failure {
    Failure::new(FailureClass::MalformedResponse, "boom").with_retry(false, true)
}

#[test]
fn gemini_after_content_recovers_incomplete_and_folds_usage() {
    let mut normalizer = Normalizer::new(Dialect::GeminiGenerateContent);
    feed_text(&mut normalizer, "hi");
    normalizer.usage = Some(Usage {
        input_tokens: Some(5),
        output_tokens: Some(2),
        ..Usage::default()
    });
    let recovered = normalizer
        .recover_abnormal_end(incoming())
        .expect("a partial answer recovers instead of failing");
    assert!(matches!(recovered.first(), Some(Event::Usage(_))));
    assert!(matches!(recovered.last(), Some(Event::Incomplete)));
    assert!(normalizer.saw_terminal());
}

#[test]
fn an_output_overflow_is_never_recovered_even_after_content() {
    // The retained-output ceiling is a deliberate gateway limit: a Gemini
    // stream that emitted content and then overflowed must still surface
    // `provider_output_too_large`, not be delivered and billed as a partial.
    let mut normalizer = Normalizer::new(Dialect::GeminiGenerateContent);
    feed_text(&mut normalizer, "hi");
    let overflow = Failure::new(FailureClass::ProviderInternal, OUTPUT_OVERFLOW_MESSAGE);
    let failure = normalizer
        .recover_abnormal_end(overflow)
        .expect_err("an overflow is not an abnormal end to salvage");
    assert_eq!(failure.safe_message, OUTPUT_OVERFLOW_MESSAGE);
    assert_eq!(failure.failure_class, FailureClass::ProviderInternal);
    assert!(!normalizer.saw_terminal());
}

#[test]
fn gemini_before_content_reclassifies_to_retryable_transport() {
    let mut normalizer = Normalizer::new(Dialect::GeminiGenerateContent);
    let failure = normalizer
        .recover_abnormal_end(incoming())
        .expect_err("nothing to salvage before content");
    assert_eq!(failure.failure_class, FailureClass::Transport);
    assert!(failure.retryable_same_deployment);
    assert!(failure.failover_eligible);
    assert!(!normalizer.saw_terminal());
}

#[test]
fn non_gemini_keeps_the_original_failure_even_after_content() {
    // Recovery is scoped to Gemini; an OpenAI-compatible stream that emitted
    // content and then broke keeps its original malformed classification.
    let mut normalizer = Normalizer::new(Dialect::OpenAiCompatible);
    let frame = SseEvent {
        event: None,
        data: serde_json::json!({"choices": [{"delta": {"content": "hi"}}]}).to_string(),
    };
    normalizer.feed(&frame).expect("content normalizes");
    let failure = normalizer
        .recover_abnormal_end(incoming())
        .expect_err("non-gemini keeps the original failure");
    assert_eq!(failure.failure_class, FailureClass::MalformedResponse);
    assert!(!normalizer.saw_terminal());
}

#[test]
fn an_already_terminal_stream_keeps_the_original_failure() {
    let mut normalizer = Normalizer::new(Dialect::GeminiGenerateContent);
    feed_text(&mut normalizer, "hi");
    let terminal = SseEvent {
        event: None,
        data: serde_json::json!({"candidates": [{"finishReason": "STOP"}]}).to_string(),
    };
    normalizer.feed(&terminal).expect("finish normalizes");
    normalizer
        .on_stream_end()
        .expect("transport finalizes the declared finish");
    assert!(normalizer.saw_terminal());
    let failure = normalizer
        .recover_abnormal_end(incoming())
        .expect_err("a terminated stream does not re-recover");
    assert_eq!(failure.failure_class, FailureClass::MalformedResponse);
}
