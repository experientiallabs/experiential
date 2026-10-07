//! Capture-only probability buffers, refusal memory and payload injection.
use super::*;
use serde_json::json;

fn delta(token: &str) -> ChoiceLogprobsDelta {
    crate::logprobs::parse(
        Some(&json!({"content":[{"token":token,"logprob":-1.0,"bytes":null,"top_logprobs":[]}]})),
        0,
    )
    .unwrap()
    .unwrap()
}

#[test]
fn injection_adds_only_logprobs_and_respects_caller_controls() {
    let payload = json!({"model":"m","messages":[],"stream":true});
    let injected = with_logprobs(&payload).unwrap();
    assert_eq!(injected["logprobs"], json!(true));
    assert!(injected.get("top_logprobs").is_none());
    assert_eq!(injected["model"], payload["model"]);
    assert!(with_logprobs(&json!({"logprobs": true})).is_none());
    assert!(with_logprobs(&json!({"top_logprobs": 3})).is_none());
    assert!(with_logprobs(&json!("not an object")).is_none());
}

#[test]
fn only_request_shaped_refusals_trigger_a_plain_redial() {
    for (class, expected) in [
        (FailureClass::InvalidRequest, true),
        (FailureClass::UnsupportedCapability, true),
        (FailureClass::Throttled, false),
        (FailureClass::ProviderAuthentication, false),
        (FailureClass::Transport, false),
        (FailureClass::ProviderInternal, false),
    ] {
        assert_eq!(may_be_refusal(&Failure::new(class, "x")), expected);
    }
}

#[test]
fn refusal_memory_is_per_rung() {
    assert!(!refused("memory-unit-a"));
    remember_refusal("memory-unit-a");
    assert!(refused("memory-unit-a"));
    assert!(!refused("memory-unit-b"));
}

#[test]
fn the_attempt_buffer_bound_is_cumulative_and_truncates() {
    let mut buffer = Buffer::default();
    let size = delta("a").retained_bytes();
    let fits = MAXIMUM_CAPTURED_LOGPROB_BYTES / size;
    for _ in 0..fits {
        buffer.push(delta("a"));
    }
    let (taken, truncated) = buffer.take();
    assert_eq!(taken.len(), fits);
    assert!(!truncated);
    // Taking never resets the attempt's budget.
    buffer.push(delta("a"));
    let (taken, truncated) = buffer.take();
    assert!(taken.is_empty());
    assert!(truncated);
}

#[test]
fn captured_extend_stops_at_the_admitted_size() {
    let mut captured = Captured::injected();
    let size = delta("a").retained_bytes();
    let added = captured.extend(vec![delta("a"), delta("b"), delta("c")], false, |extra| {
        extra <= size * 2
    });
    assert_eq!(added, size * 2);
    assert_eq!(captured.content.len(), 2);
    assert!(captured.truncated);
    assert_eq!(captured.bytes, size * 2);
}

#[test]
fn an_upstream_truncation_keeps_the_prefix_that_fits() {
    let mut captured = Captured::injected();
    captured.extend(vec![delta("a"), delta("b")], true, |_| true);
    assert_eq!(captured.content.len(), 2);
    assert!(captured.truncated);
    // Nothing is appended once the record is truncated.
    captured.extend(vec![delta("c")], false, |_| true);
    assert_eq!(captured.content.len(), 2);
}

#[test]
fn a_full_refusal_memory_still_remembers_the_latest_rung() {
    for index in 0..MAXIMUM_REFUSED_RUNGS + 1 {
        remember_refusal(&format!("eviction-{index}"));
    }
    assert!(refused(&format!("eviction-{MAXIMUM_REFUSED_RUNGS}")));
}

#[test]
fn nul_in_token_text_is_replaced_and_flagged() {
    let value = json!({"content":[{"token":"a\0b","logprob":-1.0,"bytes":[97,0,98],
        "top_logprobs":[{"token":"\0","logprob":-2.0,"bytes":[0]}]}]});
    let delta = crate::logprobs::parse(Some(&value), 0).unwrap().unwrap();
    let mut captured = Captured::injected();
    captured.extend(vec![delta], false, |_| true);
    assert!(captured.nul_replaced);
    assert_eq!(captured.content[0].token, "a\u{fffd}b");
    assert_eq!(captured.content[0].top_logprobs[0].token, "\u{fffd}");
    assert_eq!(captured.content[0].bytes, Some(vec![97, 0, 98]));
    assert!(!serde_json::to_string(&captured)
        .unwrap()
        .contains("\\u0000"));
}

#[test]
fn a_malformed_channel_marks_the_retained_prefix_truncated() {
    let mut probabilities = ChatProbabilities::Capture(Buffer::default());
    probabilities.retain(
        Some(&json!({"content":[{"token":"a","logprob":-1.0,"bytes":null,"top_logprobs":[]}]})),
        0,
        true,
    );
    probabilities.retain(
        Some(&json!({"content":[{"token":"b","logprob":"bad"}]})),
        0,
        true,
    );
    let (taken, truncated) = probabilities.take().unwrap();
    assert_eq!(taken.len(), 1);
    assert!(truncated);
}

#[test]
fn rung_identity_separates_connections_without_keeping_header_values() {
    let mut wire: crate::waterfall::DeploymentWire = serde_json::from_value(json!({
        "provider":"p","deployment_id":"d","dialect":"openai_compatible","url":"https://x",
        "headers":{"authorization":"Bearer one"},"timeout_seconds":1.0,"idempotency_key":"k"
    }))
    .unwrap();
    let first = rung_key(&wire);
    assert!(!first.contains("Bearer"));
    wire.headers
        .insert("authorization".into(), "Bearer two".into());
    assert_ne!(rung_key(&wire), first);
}

#[test]
fn an_output_frame_without_probabilities_marks_the_sequence_incomplete() {
    for missing in [None, Some(json!(null)), Some(json!({"content": []}))] {
        let mut probabilities = ChatProbabilities::Capture(Buffer::default());
        probabilities.retain(missing.as_ref(), 0, true);
        assert!(probabilities.take().unwrap().1);
    }
    // A frame with no output text (role, finish, usage) leaves it complete.
    let mut probabilities = ChatProbabilities::Capture(Buffer::default());
    probabilities.retain(None, 0, false);
    assert!(!probabilities.take().unwrap().1);
}
