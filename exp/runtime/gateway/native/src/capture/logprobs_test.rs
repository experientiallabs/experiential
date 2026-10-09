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
        &FrameOutput {
            content: "a".into(),
            refusal: String::new(),
        },
    );
    probabilities.retain(
        Some(&json!({"content":[{"token":"b","logprob":"bad"}]})),
        0,
        &FrameOutput {
            content: "a".into(),
            refusal: String::new(),
        },
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
        probabilities.retain(
            missing.as_ref(),
            0,
            &FrameOutput {
                content: "a".into(),
                refusal: String::new(),
            },
        );
        assert!(probabilities.take().unwrap().1);
    }
    // A frame with no output text (role, finish, usage) leaves it complete.
    let mut probabilities = ChatProbabilities::Capture(Buffer::default());
    probabilities.retain(None, 0, &FrameOutput::default());
    assert!(!probabilities.take().unwrap().1);
}

#[test]
fn clearing_a_sidecar_returns_everything_it_was_charged() {
    let mut captured = Captured::injected();
    captured.extend(vec![delta("a"), delta("b")], false, |_| true);
    let charged = captured.heap_bytes();
    assert!(charged > std::mem::size_of::<Captured>());
    let mut sidecar = Some(captured);
    assert_eq!(release(&mut sidecar), charged);
    assert!(sidecar.is_none());
    assert_eq!(release(&mut sidecar), 0);
}

#[test]
fn only_a_route_with_an_injectable_rung_reserves_a_sidecar() {
    let rung = |id: &str, eligible: bool, byok: bool, dialect: &str| {
        let mut wire: crate::waterfall::DeploymentWire = serde_json::from_value(json!({
            "provider":"p","deployment_id":id,"dialect":dialect,"url":"https://route",
            "headers":{},"timeout_seconds":1.0,"idempotency_key":"k"
        }))
        .unwrap();
        wire.capture_logprobs = eligible;
        wire.billing_customer_managed = byok;
        wire
    };
    assert!(!route_may_inject(&[
        rung("route-a", false, false, "openai_compatible"),
        rung("route-b", true, true, "openai_compatible"),
        rung("route-c", true, false, "openai_responses"),
    ]));
    let eligible = rung("route-d", true, false, "openai_compatible");
    assert!(route_may_inject(std::slice::from_ref(&eligible)));
    remember_refusal(&rung_key(&eligible));
    assert!(!route_may_inject(&[eligible]));
}

#[test]
fn records_on_the_wrong_channel_mark_the_sequence_incomplete() {
    let mut probabilities = ChatProbabilities::Capture(Buffer::default());
    probabilities.retain(
        Some(&json!({"refusal":[{"token":"no","logprob":-1.0,"bytes":null,"top_logprobs":[]}]})),
        0,
        &FrameOutput {
            content: "a".into(),
            refusal: String::new(),
        },
    );
    // The misaligned frame is never retained; retention ends before it.
    let (taken, truncated) = probabilities.take().unwrap();
    assert!(taken.is_empty());
    assert!(truncated);
}

#[test]
fn a_relay_holds_only_a_bounded_prefix_before_its_attempt_wins() {
    let mut held = Held::default();
    let size = delta("a").retained_bytes();
    let many: Vec<_> = (0..MAXIMUM_HELD_LOGPROB_BYTES / size + 4)
        .map(|_| delta("a"))
        .collect();
    held.forward(Some((many, false)), None);
    assert!(held.pending.len() <= MAXIMUM_HELD_LOGPROB_BYTES / size);
    assert!(held.truncated);
}

#[test]
fn records_must_reproduce_the_emitted_text_exactly() {
    let record = |token: &str, bytes: Option<Vec<u8>>| json!({"token": token, "logprob": -1.0, "bytes": bytes, "top_logprobs": []});
    let frame = |records: Vec<serde_json::Value>, text: &str| {
        let mut probabilities = ChatProbabilities::Capture(Buffer::default());
        probabilities.retain(
            Some(&json!({ "content": records })),
            0,
            &FrameOutput {
                content: text.into(),
                refusal: String::new(),
            },
        );
        probabilities.take().unwrap().1
    };
    // Full coverage by bytes and by token text.
    assert!(!frame(
        vec![record("a", Some(vec![97])), record("b", Some(vec![98]))],
        "ab"
    ));
    assert!(!frame(vec![record("a", None), record("b", None)], "ab"));
    // A partial channel is a hole.
    assert!(frame(vec![record("a", Some(vec![97]))], "ab"));
    assert!(frame(vec![record("a", None)], "ab"));
}

#[test]
fn a_held_batch_that_ends_in_a_hole_keeps_its_prefix() {
    let mut held = Held::default();
    held.forward(Some((vec![delta("a")], true)), None);
    assert_eq!(held.pending.len(), 1);
    assert!(held.truncated);
}

#[test]
fn misaligned_and_refusal_frames_end_retention_without_entering_it() {
    let one =
        |token: &str| json!({"token": token, "logprob": -1.0, "bytes": null, "top_logprobs": []});
    let mut probabilities = ChatProbabilities::Capture(Buffer::default());
    let output = |content: &str, refusal: &str| FrameOutput {
        content: content.into(),
        refusal: refusal.into(),
    };
    probabilities.retain(Some(&json!({"content":[one("a")]})), 0, &output("a", ""));
    // Records spelling more than the frame emitted are not retained.
    probabilities.retain(
        Some(&json!({"content":[one("b"), one("c")]})),
        0,
        &output("b", ""),
    );
    let (taken, truncated) = probabilities.take().unwrap();
    assert_eq!(taken.len(), 1);
    assert!(truncated);
    let mut probabilities = ChatProbabilities::Capture(Buffer::default());
    probabilities.retain(Some(&json!({"refusal":[one("no")]})), 0, &output("", "no"));
    let (taken, truncated) = probabilities.take().unwrap();
    assert!(taken.is_empty());
    assert!(truncated);
}

#[test]
fn a_refusal_attributed_elsewhere_is_not_redialed() {
    let mut failure = Failure::new(FailureClass::InvalidRequest, "x");
    assert!(may_be_refusal(&failure));
    failure.rejected_parameter = Some("logprobs".into());
    assert!(may_be_refusal(&failure));
    failure.rejected_parameter = Some("tools".into());
    assert!(!may_be_refusal(&failure));
    let mut replay = Failure::new(FailureClass::InvalidRequest, "x");
    replay.encrypted_reasoning_rejected = true;
    assert!(!may_be_refusal(&replay));
}

#[test]
fn each_record_is_checked_by_its_own_representation() {
    let frame = |records: serde_json::Value, text: &str| {
        let mut probabilities = ChatProbabilities::Capture(Buffer::default());
        probabilities.retain(
            Some(&json!({ "content": records })),
            0,
            &FrameOutput {
                content: text.into(),
                refusal: String::new(),
            },
        );
        probabilities.take().unwrap().1
    };
    // Mixed: bytes on the first, token text on the second.
    assert!(!frame(
        json!([
            {"token":"a","logprob":-1.0,"bytes":[97],"top_logprobs":[]},
            {"token":"b","logprob":-1.0,"bytes":null,"top_logprobs":[]}
        ]),
        "ab"
    ));
    // Bytes that contradict the token text are what count.
    assert!(frame(
        json!([
            {"token":"a","logprob":-1.0,"bytes":[120],"top_logprobs":[]},
            {"token":"b","logprob":-1.0,"bytes":null,"top_logprobs":[]}
        ]),
        "ab"
    ));
}

#[test]
fn off_and_caller_states_never_capture() {
    assert!(!ChatProbabilities::Off.captures());
    assert!(!ChatProbabilities::Caller.captures());
    assert!(ChatProbabilities::Capture(Buffer::default()).captures());
}
