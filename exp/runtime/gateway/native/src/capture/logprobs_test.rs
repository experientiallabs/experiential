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
