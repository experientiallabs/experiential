//! Subscriber liveness and SSE keepalives, independent of provider read clocks.

use std::future::Future;
use std::time::{Duration, Instant};

use bytes::Bytes;
use tokio::sync::mpsc;

use crate::respond::send_bounded;

const HEARTBEAT_PERIOD: Duration = Duration::from_secs(1);
const HEARTBEAT: &[u8] = b": keepalive\n\n";

/// A keyed owner keeps producing its bounded replay when its subscriber leaves.
/// An unkeyed stream stops work as soon as its body receiver is dropped.
pub(crate) struct Delivery {
    sender: mpsc::Sender<Result<Bytes, std::io::Error>>,
    keyed: bool,
    last_public: Instant,
}

impl Delivery {
    pub(crate) fn new(sender: mpsc::Sender<Result<Bytes, std::io::Error>>, keyed: bool) -> Self {
        Self {
            sender,
            keyed,
            last_public: Instant::now(),
        }
    }

    /// Wait on ONE provider future. Keepalives never recreate that future,
    /// reset its chunk timeout, become output tokens, or enter replay capture.
    pub(crate) async fn next<F: Future>(&mut self, future: F) -> Option<F::Output> {
        tokio::pin!(future);
        loop {
            // Also checked between immediately ready hidden events: a busy
            // reasoning stream must not starve subscriber liveness or pings.
            if self.sender.is_closed() && !self.keyed {
                return None;
            }
            if self.last_public.elapsed() >= HEARTBEAT_PERIOD {
                self.heartbeat();
            }
            tokio::select! {
                biased;
                _ = self.sender.closed(), if !self.keyed => return None,
                result = &mut future => return Some(result),
                _ = tokio::time::sleep_until((self.last_public + HEARTBEAT_PERIOD).into()) => {
                    self.heartbeat();
                }
            }
        }
    }

    /// An overflowed capture cannot outlive its subscriber, even with a key.
    /// The caller retains the replay lease until the attempt has settled.
    pub(crate) fn retain_replay(&mut self, replayable: bool) {
        self.keyed &= replayable;
    }

    fn heartbeat(&mut self) {
        // Heartbeats never block the provider reader behind a full channel.
        // Public frame sends remain bounded by the original request deadline.
        let _ = self.sender.try_send(Ok(Bytes::from_static(HEARTBEAT)));
        self.last_public = Instant::now();
    }

    pub(crate) async fn send(&mut self, deadline: Instant, data: Bytes) -> bool {
        if self.keyed && self.sender.is_closed() {
            return true;
        }
        let delivered = send_bounded(&self.sender, deadline, data).await;
        if delivered {
            self.last_public = Instant::now();
        }
        delivered || (self.keyed && self.sender.is_closed())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::{
        atomic::{AtomicUsize, Ordering},
        Arc,
    };

    struct Dropped(Arc<AtomicUsize>);
    impl Drop for Dropped {
        fn drop(&mut self) {
            self.0.fetch_add(1, Ordering::SeqCst);
        }
    }

    #[tokio::test]
    async fn quiet_disconnect_drops_one_pending_read_promptly() {
        let (sender, receiver) = mpsc::channel(1);
        let mut delivery = Delivery::new(sender, false);
        let drops = Arc::new(AtomicUsize::new(0));
        let held = Dropped(drops.clone());
        let read = async move {
            let _held = held;
            std::future::pending::<()>().await
        };
        drop(receiver);
        assert!(delivery.next(read).await.is_none());
        assert_eq!(drops.load(Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn heartbeat_does_not_restart_read_or_extend_its_timeout() {
        let (sender, mut receiver) = mpsc::channel(4);
        let mut delivery = Delivery::new(sender, false);
        let drops = Arc::new(AtomicUsize::new(0));
        let held = Dropped(drops.clone());
        let started = Instant::now();
        let read = async move {
            let _held = held;
            tokio::time::timeout(Duration::from_millis(1250), std::future::pending::<()>()).await
        };
        assert!(delivery.next(read).await.unwrap().is_err());
        assert_eq!(drops.load(Ordering::SeqCst), 1);
        assert!(started.elapsed() < Duration::from_secs(2));
        assert_eq!(receiver.try_recv().unwrap().unwrap(), HEARTBEAT);
    }

    #[tokio::test]
    async fn replay_overflow_stops_detached_work_but_not_connected_delivery() {
        let (sender, mut receiver) = mpsc::channel(2);
        let mut delivery = Delivery::new(sender, true);
        let mut capture = Vec::new();
        assert!(super::super::capture_frame_bounded(
            &mut capture,
            b"abcd",
            true,
            4
        ));
        let replayable = super::super::capture_frame_bounded(&mut capture, b"e", true, 4);
        assert!(!replayable);
        delivery.retain_replay(replayable);
        assert!(
            delivery
                .send(
                    Instant::now() + Duration::from_secs(1),
                    Bytes::from_static(b"connected")
                )
                .await
        );
        assert_eq!(receiver.recv().await.unwrap().unwrap(), b"connected"[..]);
        drop(receiver);
        assert!(tokio::time::timeout(
            Duration::from_millis(50),
            delivery.next(std::future::pending::<()>())
        )
        .await
        .expect("overflowed owner stops on loss")
        .is_none());
        assert_eq!(capture.capacity(), 0);
    }

    #[tokio::test]
    async fn keyed_owner_survives_subscriber_loss() {
        let (sender, receiver) = mpsc::channel(1);
        let mut delivery = Delivery::new(sender, true);
        drop(receiver);
        assert_eq!(delivery.next(async { 42 }).await, Some(42));
        assert!(
            delivery
                .send(Instant::now(), Bytes::from_static(b"answer"))
                .await
        );
    }
}
