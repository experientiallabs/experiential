//! Cancellation retains normalized facts without another provider read.

use super::*;
use crate::settlement::Observation;
use std::sync::{
    atomic::{AtomicBool, AtomicUsize, Ordering},
    Arc,
};

struct Source {
    chunk: Option<Bytes>,
    polls: Arc<AtomicUsize>,
    dropped: Arc<AtomicBool>,
}
impl futures_util::Stream for Source {
    type Item = reqwest::Result<Bytes>;
    fn poll_next(
        mut self: std::pin::Pin<&mut Self>,
        _cx: &mut std::task::Context<'_>,
    ) -> std::task::Poll<Option<Self::Item>> {
        self.polls.fetch_add(1, Ordering::SeqCst);
        match self.chunk.take() {
            Some(chunk) => std::task::Poll::Ready(Some(Ok(chunk))),
            None => std::task::Poll::Pending,
        }
    }
}
impl Drop for Source {
    fn drop(&mut self) {
        self.dropped.store(true, Ordering::SeqCst);
    }
}

#[test]
fn repeated_close_meters_each_drained_delta_once() {
    let deadline = Instant::now() + Duration::from_secs(5);
    let mut relay = UpstreamRelay::from_stream(
        futures_util::stream::pending().boxed(),
        Dialect::OpenAiCompatible,
        deadline,
    );
    let observed = Observation::default();
    relay.set_observation(observed.clone());
    relay.ready.push_back(Event::TextDelta("once".into()));
    relay.close_transport();
    relay.close_transport();
    drop(relay);
    assert_eq!(observed.snapshot().streamed_output.text, "once");
}

#[test]
fn private_gemini_meter_is_separate_from_visible_signatures_and_terminal() {
    let deadline = Instant::now() + Duration::from_secs(5);
    let mut relay = UpstreamRelay::from_stream(
        futures_util::stream::pending().boxed(),
        Dialect::GeminiGenerateContent,
        deadline,
    );
    let observed = Observation::default();
    relay.set_observation(observed.clone());
    let events = relay.normalizer.feed(&crate::sse::SseEvent {
        event: None,
        data: r#"{"candidates":[{"content":{"parts":[{"thought":true,"text":"private"},{"text":"visible","thoughtSignature":"not tokens"},{"thoughtSignature":"also not tokens"}]}}]}"#.into(),
    }).unwrap();
    assert_eq!(events.len(), 1);
    assert!(matches!(&events[0], Event::TextDelta(text) if text == "visible"));
    relay.queue_events(events);
    relay.close_transport();
    let snapshot = observed.snapshot();
    assert_eq!(snapshot.streamed_output.reasoning, "private");
    assert_eq!(snapshot.streamed_output.text, "visible");
    assert!(snapshot.first_token_at.is_none());
    observed.record(&Event::Usage(Usage {
        input_tokens: Some(13),
        output_tokens: Some(7),
        ..Usage::default()
    }));
    observed.record(&Event::Completed);
    observed.record_gemini_reasoning("must not count after terminal");
    assert_eq!(observed.snapshot().streamed_output.reasoning, "private");
    assert_eq!(observed.snapshot().usage.unwrap().output_tokens, Some(7));
}

#[tokio::test]
async fn buffered_usage_and_terminal_win_without_reading_after_close() {
    let polls = Arc::new(AtomicUsize::new(0));
    let dropped = Arc::new(AtomicBool::new(false));
    let wire = concat!(
        "data: {\"choices\":[{\"delta\":{\"content\":\"answer\"}}]}\n\n",
        "data: {\"choices\":[],\"usage\":{\"prompt_tokens\":19,\"completion_tokens\":7}}\n\n",
        "data: [DONE]\n\n"
    );
    let source = Source {
        chunk: Some(Bytes::from_static(wire.as_bytes())),
        polls: polls.clone(),
        dropped: dropped.clone(),
    };
    let deadline = Instant::now() + Duration::from_secs(5);
    let mut relay = UpstreamRelay::from_stream(source.boxed(), Dialect::OpenAiCompatible, deadline);
    let observed = Observation::default();
    relay.set_observation(observed.clone());
    assert!(matches!(
        relay
            .next_event(deadline, Duration::from_secs(5), Instant::now())
            .await
            .unwrap(),
        Some(Event::TextDelta(_))
    ));
    let snapshot = observed.snapshot();
    assert!(matches!(snapshot.terminal, Some(Event::Completed)));
    assert_eq!(snapshot.usage.unwrap().output_tokens, Some(7));
    relay.close_transport();
    assert!(dropped.load(Ordering::SeqCst));
    assert_eq!(polls.load(Ordering::SeqCst), 1);
}

#[tokio::test]
async fn effective_stop_terminal_wins_raw_incomplete_on_disconnect() {
    let wire = concat!(
        "data: {\"choices\":[{\"delta\":{\"content\":\"answer STOP discarded\"}}]}\n\n",
        "data: {\"choices\":[{\"delta\":{},\"finish_reason\":\"length\"}]}\n\n",
        "data: [DONE]\n\n"
    );
    let source = futures_util::stream::iter([Ok(Bytes::from_static(wire.as_bytes()))]);
    let deadline = Instant::now() + Duration::from_secs(5);
    let mut relay = UpstreamRelay::from_stream(source.boxed(), Dialect::OpenAiCompatible, deadline);
    relay.set_stop_sequences(["STOP"]);
    let observed = Observation::default();
    relay.set_observation(observed.clone());
    relay
        .next_event(deadline, Duration::from_secs(5), Instant::now())
        .await
        .unwrap();
    relay.close_transport();
    assert!(matches!(
        observed.snapshot().terminal,
        Some(Event::StoppedAtSequence(_))
    ));
}

#[tokio::test]
async fn a_missing_input_leg_stays_unknown() {
    let wire = concat!(
        "data: {\"choices\":[],\"usage\":{\"completion_tokens\":2}}\n\n",
        "data: {\"choices\":[{\"delta\":{\"content\":\"answer\"}}]}\n\n",
        "data: [DONE]\n\n"
    );
    let source = futures_util::stream::iter([Ok(Bytes::from_static(wire.as_bytes()))]);
    let deadline = Instant::now() + Duration::from_secs(5);
    let mut relay = UpstreamRelay::from_stream(source.boxed(), Dialect::OpenAiCompatible, deadline);
    let observed = Observation::default();
    relay.set_observation(observed.clone());
    relay
        .next_event(deadline, Duration::from_secs(5), Instant::now())
        .await
        .unwrap();
    relay.close_transport();
    let usage = observed.snapshot().usage.unwrap();
    assert_eq!(usage.input_tokens, None);
    assert_eq!(usage.output_tokens, Some(2));
}

#[tokio::test]
async fn abnormal_end_keeps_unreported_usage_unknown() {
    let deadline = Instant::now() + Duration::from_secs(5);
    let mut relay = UpstreamRelay::from_stream(
        futures_util::stream::empty().boxed(),
        Dialect::OpenAiCompatible,
        deadline,
    );
    assert!(relay.usage_before_failure(None).is_none());

    let events = relay
        .normalizer
        .feed(&crate::sse::SseEvent {
            event: None,
            data: r#"{"choices":[],"usage":{"completion_tokens":2}}"#.into(),
        })
        .unwrap();
    relay.queue_events(events);
    let partial = relay.usage_before_failure(None).unwrap();
    assert_eq!(partial.input_tokens, None);
    assert_eq!(partial.output_tokens, Some(2));
}

#[test]
fn close_preserves_latest_cumulative_usage() {
    let deadline = Instant::now() + Duration::from_secs(5);
    let mut relay = UpstreamRelay::from_stream(
        futures_util::stream::pending().boxed(),
        Dialect::AnthropicMessages,
        deadline,
    );
    let observed = Observation::default();
    relay.set_observation(observed.clone());
    for data in [
        r#"{"type":"message_start","message":{"usage":{"input_tokens":13,"output_tokens":2}}}"#,
        r#"{"type":"message_delta","delta":{"stop_reason":null},"usage":{"output_tokens":4}}"#,
    ] {
        let events = relay
            .normalizer
            .feed(&crate::sse::SseEvent {
                event: None,
                data: data.into(),
            })
            .unwrap();
        relay.queue_events(events);
    }
    let before = observed.snapshot().usage.unwrap();
    assert_eq!(before.input_tokens, Some(13));
    assert_eq!(before.output_tokens, Some(4));
    relay.close_transport();
    let after = observed.snapshot().usage.unwrap();
    assert_eq!(after.input_tokens, Some(13));
    assert_eq!(after.output_tokens, Some(4));
}

#[tokio::test]
async fn malformed_sparse_frame_retains_meter_through_relay_failure() {
    let wire = concat!(
        "data: {\"choices\":[],\"usage\":{\"prompt_tokens\":13,\"completion_tokens\":7}}\n\n",
        "data: {\"choices\":\"bad\",\"usage\":{}}\n\n"
    );
    let deadline = Instant::now() + Duration::from_secs(5);
    let mut relay = UpstreamRelay::from_stream(
        futures_util::stream::iter([Ok(Bytes::from_static(wire.as_bytes()))]).boxed(),
        Dialect::OpenAiCompatible,
        deadline,
    );
    let observed = Observation::default();
    relay.set_observation(observed.clone());
    assert!(relay
        .next_event(deadline, Duration::from_secs(2), Instant::now())
        .await
        .is_err());
    let usage = relay.usage_before_failure(None).unwrap();
    assert_eq!(usage.input_tokens, Some(13));
    assert_eq!(usage.output_tokens, Some(7));
    relay.close_transport();
    assert_eq!(observed.snapshot().usage.unwrap().output_tokens, Some(7));
}

#[tokio::test]
async fn parsed_compatible_meter_survives_before_its_deferred_usage_event() {
    let wire = concat!(
        "data: {\"choices\":[],\"usage\":{\"prompt_tokens\":19,\"completion_tokens\":7}}\n\n",
        "data: {\"choices\":[{\"delta\":{\"content\":\"answer\"}}]}\n\n"
    );
    let source = futures_util::stream::iter([Ok(Bytes::from_static(wire.as_bytes()))]);
    let deadline = Instant::now() + Duration::from_secs(5);
    let mut relay = UpstreamRelay::from_stream(source.boxed(), Dialect::OpenAiCompatible, deadline);
    let observed = Observation::default();
    relay.set_observation(observed.clone());
    relay
        .next_event(deadline, Duration::from_secs(5), Instant::now())
        .await
        .unwrap();
    relay.close_transport();
    let snapshot = observed.snapshot();
    assert!(snapshot.terminal.is_none());
    let usage = snapshot
        .usage
        .expect("parsed usage is not deferred until terminal delivery");
    assert_eq!(usage.input_tokens, Some(19));
    assert_eq!(usage.output_tokens, Some(7));
}

#[tokio::test]
async fn close_drains_custom_input_without_inventing_provider_terminality() {
    use pyo3::prelude::*;

    Python::initialize();
    let plane = Python::attach(|py| {
        pyo3::types::PyModule::from_code(py, c"import json\nclass Plane:\n def __init__(self): self.writes = []\n def settle(self, argument):\n  self.writes.append(json.loads(argument))\n  return '{}'\n def close_thread_resources(self, argument): return '{}'\n", c"drain_plane.py", c"drain_plane")
            .unwrap().getattr("Plane").unwrap().call0().unwrap().unbind()
    });
    let bridge = std::sync::Arc::new(
        crate::bridge::Bridge::new(Python::attach(|py| plane.clone_ref(py)), 1).unwrap(),
    );
    let mut settled = 0;
    for valid in [false, true] {
        for completed in [false, true] {
            let mut relay = UpstreamRelay::from_stream(
                futures_util::stream::pending().boxed(),
                Dialect::OpenAiCompatible,
                Instant::now() + Duration::from_secs(5),
            );
            let mut guard = crate::settlement::AttemptGuard::new(
                bridge.clone(),
                std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0)),
                format!("request-{valid}-{completed}"),
                Instant::now(),
            );
            guard.rebind(format!("attempt-{valid}-{completed}"));
            guard.mark_dispatched();
            guard.mark_opened();
            let observation = guard.capture_observation();
            relay.set_observation(observation.clone());
            relay.set_native_tool_translation(NativeToolTranslation::from([(
                "patch".into(),
                ("patch".into(), None, true),
            )]));
            let mut events = vec![
                Event::ToolCallStarted {
                    index: 0,
                    call_id: "call".into(),
                    name: "patch".into(),
                    namespace: None,
                    caller: None,
                    custom: false,
                },
                Event::Usage(Usage {
                    input_tokens: Some(7),
                    output_tokens: Some(2),
                    ..Usage::default()
                }),
                Event::ToolCallCompleted {
                    index: 0,
                    call: crate::events::CompletedToolCall {
                        call_id: "call".into(),
                        name: "patch".into(),
                        namespace: None,
                        caller: None,
                        provider_item_id: None,
                        provider_status: None,
                        raw_arguments: if valid {
                            r#"{"input":"patch text"}"#
                        } else {
                            r#"{"input":7}"#
                        }
                        .into(),
                        custom: false,
                    },
                },
            ];
            if completed {
                events.push(Event::Completed);
            }
            relay.queue_events(events);
            relay.close_transport();
            let observed = observation.snapshot();
            let usage = observed.usage.expect("parsed usage is retained");
            assert_eq!(usage.input_tokens, Some(7));
            assert_eq!(usage.output_tokens, Some(2));
            assert_eq!(observed.terminal.is_some(), completed);
            if completed {
                assert!(matches!(observed.terminal, Some(Event::Completed)));
            }
            if valid {
                assert!(relay.ready.iter().any(|e| matches!(e, Event::ToolArgumentsDelta { delta, .. } if delta == "patch text")));
                assert!(!relay.ready.iter().any(|e| matches!(e, Event::Failed(_))));
            } else {
                assert!(
                    matches!(relay.ready.back(), Some(Event::Failed(f)) if f.failure_class == FailureClass::MalformedResponse)
                );
            }
            assert!(relay.pending.is_empty());
            assert!(relay.eof);
            assert!(guard.settle_cancelled(None, &[]).await);
            assert!(guard.settle_cancelled(None, &[]).await);
            drop(guard);
            settled += 1;
            let writes: String = Python::attach(|py| {
                py.import("json")
                    .unwrap()
                    .call_method1("dumps", (plane.bind(py).getattr("writes").unwrap(),))
                    .unwrap()
                    .extract()
                    .unwrap()
            });
            let writes: serde_json::Value = serde_json::from_str(&writes).unwrap();
            assert_eq!(writes.as_array().unwrap().len(), settled);
            let latest = &writes[settled - 1];
            assert_eq!(latest["usage"]["input_tokens"], 7);
            assert_eq!(latest["usage"]["output_tokens"], 2);
            assert_eq!(latest["usage_incomplete_due_to_disconnect"], !completed);
            assert_eq!(latest["finalize"], true);
            assert_eq!(latest["dispatched"], true);
            assert_eq!(
                latest["outcome"],
                if completed { "completed" } else { "failed" }
            );
            if completed {
                assert!(latest["failure"].is_null());
            } else {
                assert_eq!(latest["failure"]["failure_class"], "cancelled");
            }
        }
    }
}
