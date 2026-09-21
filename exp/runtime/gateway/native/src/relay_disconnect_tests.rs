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
async fn across_dial_unknown_leg_cannot_reuse_known_prior_subtotal() {
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
    relay.set_carried_usage(Some(Usage {
        input_tokens: Some(13),
        output_tokens: Some(7),
        ..Usage::default()
    }));
    relay
        .next_event(deadline, Duration::from_secs(5), Instant::now())
        .await
        .unwrap();
    relay.close_transport();
    let usage = observed.snapshot().usage.unwrap();
    assert_eq!(usage.input_tokens, None);
    assert_eq!(usage.output_tokens, Some(9));
}

#[tokio::test]
async fn abnormal_end_does_not_promote_an_earlier_dial_subtotal() {
    let deadline = Instant::now() + Duration::from_secs(5);
    let mut relay = UpstreamRelay::from_stream(
        futures_util::stream::empty().boxed(),
        Dialect::OpenAiCompatible,
        deadline,
    );
    relay.set_carried_usage(Some(Usage {
        input_tokens: Some(13),
        output_tokens: Some(7),
        ..Usage::default()
    }));
    let unknown = relay.usage_before_failure(None).unwrap();
    assert_eq!(unknown.input_tokens, None);
    assert_eq!(unknown.output_tokens, None);

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
    assert_eq!(partial.output_tokens, Some(9));
}

#[test]
fn close_does_not_merge_raw_current_usage_into_an_across_dial_total() {
    let deadline = Instant::now() + Duration::from_secs(5);
    let mut relay = UpstreamRelay::from_stream(
        futures_util::stream::pending().boxed(),
        Dialect::AnthropicMessages,
        deadline,
    );
    let observed = Observation::default();
    relay.set_observation(observed.clone());
    relay.set_carried_usage(Some(Usage {
        input_tokens: None,
        output_tokens: Some(7),
        ..Usage::default()
    }));
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
    assert_eq!(before.input_tokens, None);
    assert_eq!(before.output_tokens, Some(11));
    relay.close_transport();
    let after = observed.snapshot().usage.unwrap();
    assert_eq!(after.input_tokens, None);
    assert_eq!(after.output_tokens, Some(11));
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
