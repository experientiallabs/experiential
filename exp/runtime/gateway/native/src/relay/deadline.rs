//! The messages a caller reads when the request's total deadline runs out.
//! They name what happened (still writing, or nothing yet) and the remedy,
//! never the internal mechanism.

use std::collections::VecDeque;
use std::time::Instant;

use super::remaining;
use crate::errors::{Failure, FailureClass};

/// The request deadline ran out while the model was still producing output.
/// A model that streams for the whole deadline is almost always looping, so
/// the message names that and the two remedies a caller controls.
pub const DEADLINE_WHILE_GENERATING_MESSAGE: &str = "the response was cut off because the \
request reached the gateway's time limit while the model was still writing; this usually \
means the model got stuck repeating itself. Retry, or set a lower max_tokens so long \
answers stop on their own";

/// The request deadline ran out before the provider produced any output.
pub const DEADLINE_BEFORE_OUTPUT_MESSAGE: &str = "the request reached the gateway's time \
limit before the provider started answering; retry the request";

/// The terminal failure of a committed stream whose total deadline ran out.
pub fn deadline_while_generating() -> Failure {
    Failure::new(FailureClass::Timeout, DEADLINE_WHILE_GENERATING_MESSAGE)
}

/// Classify one mid-stream chunk timeout the way the python transport does:
/// a stalled provider read is a retryable transport failure unless the
/// request's own deadline is exhausted.
pub fn stream_timeout_failure(deadline: Instant) -> Failure {
    if remaining(deadline).is_zero() {
        Failure::new(FailureClass::Timeout, DEADLINE_BEFORE_OUTPUT_MESSAGE)
    } else {
        Failure::new(
            FailureClass::Transport,
            "provider transport failed; retry the request",
        )
        .with_retry(true, true)
    }
}

impl super::UpstreamRelay {
    /// The top-of-loop deadline check every committed stream runs. Returns
    /// true when the request deadline ran out while the provider is still
    /// writing, so the stream must end with [`deadline_while_generating`]
    /// rather than a close that reads as a clean end. When the provider
    /// already declared its terminal (in the committed prefix or relay queues, or a
    /// declared finish is draining its trailer) the stream runs on to that
    /// terminal, which the relay yields at once past the deadline, and `send`
    /// gains one terminal grace for the frames still queued ahead of it.
    /// Arm it as soon as the terminal is observed, so an in-flight send may
    /// cross the request deadline; the bound never renews per frame.
    pub fn deadline_cut(
        &self,
        deadline: Instant,
        send: &mut Instant,
        committed_prefix: &VecDeque<crate::events::Event>,
    ) -> bool {
        // Commitment may have moved a private reasoning carrier and its terminal
        // out of the relay into the caller's prefix. All three queues count.
        let terminal_pending = committed_prefix
            .iter()
            .any(crate::events::Event::is_terminal)
            || self.normalizer.metadata_drain_started().is_some()
            || self.ready.iter().any(crate::events::Event::is_terminal)
            || self.pending.iter().any(crate::events::Event::is_terminal);
        if !terminal_pending {
            return remaining(deadline).is_zero();
        }
        crate::respond::arm_terminal_send_deadline(deadline, send);
        false
    }
}

#[cfg(test)]
mod tests {
    use std::collections::VecDeque;
    use std::time::{Duration, Instant};

    use bytes::Bytes;
    use futures_util::{stream, StreamExt};

    use super::super::UpstreamRelay;
    use crate::dialects::Dialect;
    use crate::events::Event;

    fn relay(frames: Vec<&'static str>) -> UpstreamRelay {
        let frames = stream::iter(
            frames
                .into_iter()
                .map(|frame| Ok::<_, reqwest::Error>(Bytes::from(frame))),
        )
        .chain(stream::pending())
        .boxed();
        UpstreamRelay::from_stream(
            frames,
            Dialect::OpenAiCompatible,
            Instant::now() + Duration::from_secs(5),
        )
    }

    #[tokio::test]
    async fn an_expired_deadline_cuts_a_stream_still_writing() {
        let relay = relay(Vec::new());
        let deadline = Instant::now();
        let mut send = deadline;
        assert!(relay.deadline_cut(deadline, &mut send, &VecDeque::new()));
        assert_eq!(send, deadline);
    }

    #[tokio::test]
    async fn a_declared_terminal_outruns_the_deadline_with_one_send_grace() {
        // Content and its terminal arrive in one read; the terminal is queued
        // behind the content when the deadline runs out.
        let mut relay = relay(vec![
            "data: {\"choices\":[{\"delta\":{\"content\":\"hi\"}}]}\n\ndata: [DONE]\n\n",
        ]);
        let open = Instant::now() + Duration::from_secs(30);
        let first = relay
            .next_event(open, Duration::from_secs(5), Instant::now())
            .await
            .unwrap()
            .unwrap();
        assert!(matches!(first, Event::TextDelta(_)));
        let deadline = Instant::now();
        let mut send = deadline;
        assert!(!relay.deadline_cut(deadline, &mut send, &VecDeque::new()));
        let grace = send;
        assert!(grace > deadline);
        // A later check never renews the terminal's first send bound.
        assert!(!relay.deadline_cut(deadline, &mut send, &VecDeque::new()));
        assert_eq!(send, grace);
        let terminal = relay
            .next_event(deadline, Duration::from_secs(5), Instant::now())
            .await
            .unwrap();
        assert!(matches!(terminal, Some(Event::Completed)));
    }

    #[tokio::test]
    async fn a_terminal_in_the_committed_prefix_keeps_its_success_and_one_grace() {
        let relay = relay(Vec::new());
        let mut prefix = VecDeque::from([
            Event::ReasoningContentDelta {
                route_sha256: "a".repeat(64),
                delta: "private reasoning".to_owned(),
            },
            Event::Usage(crate::events::Usage::default()),
            Event::Completed,
        ]);
        let deadline = Instant::now();
        let mut send = deadline;
        assert!(!relay.deadline_cut(deadline, &mut send, &prefix));
        let grace = send;
        assert!(grace > deadline);
        while !prefix.is_empty() {
            assert!(!relay.deadline_cut(deadline, &mut send, &prefix));
            assert_eq!(send, grace);
            prefix.pop_front();
        }
    }

    #[tokio::test]
    async fn a_nonterminal_committed_prefix_does_not_extend_the_deadline() {
        let relay = relay(Vec::new());
        let prefix = VecDeque::from([Event::TextDelta("still writing".to_owned())]);
        let open = Instant::now() + Duration::from_secs(60);
        let mut send = open;
        assert!(!relay.deadline_cut(open, &mut send, &prefix));
        assert_eq!(send, open);
        let deadline = Instant::now();
        let mut send = deadline;
        assert!(relay.deadline_cut(deadline, &mut send, &prefix));
        assert_eq!(send, deadline);
    }

    #[tokio::test]
    async fn queued_terminals_arm_one_grace_before_the_request_deadline() {
        let relay = relay(Vec::new());
        let prefix = VecDeque::from([Event::Completed]);
        let deadline = Instant::now() + Duration::from_secs(60);
        let mut send = deadline;
        assert!(!relay.deadline_cut(deadline, &mut send, &prefix));
        assert_eq!(send, deadline + crate::respond::TERMINAL_FRAME_GRACE);
        let grace = send;
        assert!(!relay.deadline_cut(deadline, &mut send, &prefix));
        assert_eq!(send, grace);
    }

    #[tokio::test]
    async fn queued_completions_survive_backpressure_across_the_request_deadline() {
        let relay = relay(Vec::new());
        let prefix = VecDeque::from([Event::Completed]);
        let deadline = Instant::now() + Duration::from_millis(50);
        let mut send = deadline;
        assert!(!relay.deadline_cut(deadline, &mut send, &prefix));
        let (sender, mut receiver) = tokio::sync::mpsc::channel(1);
        sender.send(Ok(Bytes::from_static(b"start"))).await.unwrap();
        let receive = tokio::spawn(async move {
            tokio::time::sleep_until((deadline + Duration::from_millis(25)).into()).await;
            assert_eq!(receiver.recv().await.unwrap().unwrap(), b"start"[..]);
            receiver.recv().await.unwrap().unwrap()
        });
        let mut delivery = crate::respond::stream_delivery::Delivery::new(sender, false);
        assert!(delivery.send(send, Bytes::from_static(b"done")).await);
        assert_eq!(receive.await.unwrap(), b"done"[..]);
    }
}
