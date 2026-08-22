//! Bounded in-process keyed-response replay, mirroring the python engine's
//! `BoundedReplayStore`: exactly one owner per unpublished operation, joiners
//! that wait on the owner's published result, exact completed-response replay,
//! and fail-closed conflict and abandonment semantics under the same count,
//! byte, and TTL bounds.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};

use tokio::sync::{watch, Mutex};

use crate::errors::PublicError;

const DEFAULT_CAPACITY: usize = 4_096;
const DEFAULT_BYTE_CAP: usize = 64 * 1024 * 1024;
const DEFAULT_TTL: Duration = Duration::from_secs(24 * 60 * 60);

/// Content-free replay identity computed by the shared python control plane.
#[derive(Debug, Clone, PartialEq, Eq, Hash, serde::Deserialize)]
pub struct ReplayKey {
    pub organization_id: String,
    pub identity_id: String,
    pub alias_revision_id: String,
    pub surface: String,
    pub caller_operation_sha256: String,
    pub canonical_request_sha256: String,
}

impl ReplayKey {
    /// Whether another live entry reuses this caller operation for a
    /// different canonical body within the same namespace and surface.
    fn conflicts_with(&self, other: &ReplayKey) -> bool {
        self.organization_id == other.organization_id
            && self.identity_id == other.identity_id
            && self.alias_revision_id == other.alias_revision_id
            && self.surface == other.surface
            && self.caller_operation_sha256 == other.caller_operation_sha256
            && self.canonical_request_sha256 != other.canonical_request_sha256
    }
}

/// Exact bounded HTTP result retained only for in-process replay. Header
/// values hold latin-1 decoded text so any HTTP-legal byte round-trips.
#[derive(Debug, Clone)]
pub struct CachedResponse {
    pub status_code: u16,
    pub media_type: String,
    pub headers: Vec<(String, String)>,
    pub body: Vec<u8>,
}

impl CachedResponse {
    /// Retained body plus metadata bytes, matching the python accounting.
    fn size_bytes(&self) -> usize {
        let metadata: usize = self.media_type.len()
            + self
                .headers
                .iter()
                .map(|(name, value)| name.len() + value.len())
                .sum::<usize>();
        self.body.len() + metadata
    }
}

/// Publication state observed by joiners through the entry's watch channel.
#[derive(Debug, Clone)]
enum PublishState {
    Pending,
    Published(Arc<CachedResponse>),
    Abandoned,
}

struct Entry {
    publish: watch::Sender<PublishState>,
    response: Option<Arc<CachedResponse>>,
    expires_at: Instant,
    size_bytes: usize,
    /// Recency counter for least-recently-used eviction of completed work.
    order: u64,
    /// Claim identity so a stale lease cannot publish over a newer entry.
    epoch: u64,
}

struct Inner {
    entries: HashMap<ReplayKey, Entry>,
    response_bytes: usize,
    order_counter: u64,
    epoch_counter: u64,
}

/// Single-process duplicate joining and exact completed-response replay.
pub struct ReplayStore {
    inner: Mutex<Inner>,
    capacity: usize,
    byte_cap: usize,
    ttl: Duration,
}

/// One caller's disposition for a keyed request.
pub enum Claim {
    /// This caller runs the provider work and must complete or abandon.
    Owner(OwnerLease),
    /// A matching request is in flight; wait for the owner's result.
    Join(Joiner),
    /// The owner already published; replay the exact stored response.
    Replay(Arc<CachedResponse>),
}

/// Ownership handle for one claimed replay key. Dropping the lease without a
/// successful `complete` abandons the entry so joiners fail closed instead of
/// waiting forever, mirroring the python engine's abandon-on-error paths.
pub struct OwnerLease {
    store: Arc<ReplayStore>,
    key: ReplayKey,
    epoch: u64,
    settled: bool,
}

/// Join handle waiting on the owner's published result.
pub struct Joiner {
    receiver: watch::Receiver<PublishState>,
}

fn conflict_error() -> PublicError {
    let mut error = PublicError::new(
        409,
        "idempotency_conflict",
        "The caller operation was reused with a different request body.",
        "invalid_request_error",
    );
    error.param = Some("Idempotency-Key".to_string());
    error
}

fn overloaded_error() -> PublicError {
    PublicError::new(
        429,
        "gateway_overloaded",
        "The bounded in-process replay window is full.",
        "api_error",
    )
}

fn replay_unavailable_error() -> PublicError {
    let mut error = PublicError::new(
        409,
        "idempotency_replay_unavailable",
        "The original keyed request ended before publishing a replayable result.",
        "api_error",
    );
    error.param = Some("Idempotency-Key".to_string());
    error
}

fn oversize_error() -> PublicError {
    PublicError::new(
        500,
        "idempotency_replay_unavailable",
        "The completed response exceeds the bounded replay cache.",
        "api_error",
    )
}

fn lost_ownership_error() -> PublicError {
    let mut error = PublicError::new(
        409,
        "idempotency_conflict",
        "The keyed operation no longer belongs to this request.",
        "invalid_request_error",
    );
    error.param = Some("Idempotency-Key".to_string());
    error
}

impl ReplayStore {
    pub fn new() -> Self {
        Self::with_bounds(DEFAULT_CAPACITY, DEFAULT_BYTE_CAP, DEFAULT_TTL)
    }

    pub fn with_bounds(capacity: usize, byte_cap: usize, ttl: Duration) -> Self {
        Self {
            inner: Mutex::new(Inner {
                entries: HashMap::new(),
                response_bytes: 0,
                order_counter: 0,
                epoch_counter: 0,
            }),
            capacity: capacity.max(1),
            byte_cap: byte_cap.max(1),
            ttl,
        }
    }

    /// Claim original work, join an in-flight duplicate, or replay completion.
    pub async fn claim(self: &Arc<Self>, key: ReplayKey) -> Result<Claim, PublicError> {
        let mut inner = self.inner.lock().await;
        let now = Instant::now();
        Self::expire(&mut inner, now);
        if inner
            .entries
            .keys()
            .any(|existing| existing.conflicts_with(&key))
        {
            return Err(conflict_error());
        }
        if inner.entries.contains_key(&key) {
            inner.order_counter += 1;
            let order = inner.order_counter;
            let entry = inner
                .entries
                .get_mut(&key)
                .expect("entry present under lock");
            entry.order = order;
            if let Some(response) = &entry.response {
                return Ok(Claim::Replay(response.clone()));
            }
            return Ok(Claim::Join(Joiner {
                receiver: entry.publish.subscribe(),
            }));
        }
        Self::make_capacity(&mut inner, self.capacity)?;
        inner.order_counter += 1;
        inner.epoch_counter += 1;
        let order = inner.order_counter;
        let epoch = inner.epoch_counter;
        let (publish, _) = watch::channel(PublishState::Pending);
        inner.entries.insert(
            key.clone(),
            Entry {
                publish,
                response: None,
                expires_at: now + self.ttl,
                size_bytes: 0,
                order,
                epoch,
            },
        );
        Self::evict_completed(&mut inner, self.capacity, self.byte_cap);
        Ok(Claim::Owner(OwnerLease {
            store: self.clone(),
            key,
            epoch,
            settled: false,
        }))
    }

    async fn complete_entry(
        &self,
        key: &ReplayKey,
        epoch: u64,
        response: CachedResponse,
    ) -> Result<(), PublicError> {
        let size = response.size_bytes();
        if size > self.byte_cap {
            self.abandon_entry(key, epoch).await;
            return Err(oversize_error());
        }
        let mut inner = self.inner.lock().await;
        let now = Instant::now();
        let published = {
            let Some(entry) = inner.entries.get(key) else {
                return Err(lost_ownership_error());
            };
            if entry.epoch != epoch || entry.response.is_some() {
                return Err(lost_ownership_error());
            }
            let shared = Arc::new(response);
            let entry = inner
                .entries
                .get_mut(key)
                .expect("entry present under lock");
            entry.size_bytes = size;
            entry.expires_at = now + self.ttl;
            entry.response = Some(shared.clone());
            let _ = entry.publish.send(PublishState::Published(shared));
            size
        };
        inner.response_bytes += published;
        inner.order_counter += 1;
        let order = inner.order_counter;
        if let Some(entry) = inner.entries.get_mut(key) {
            entry.order = order;
        }
        Self::evict_completed(&mut inner, self.capacity, self.byte_cap);
        Ok(())
    }

    async fn abandon_entry(&self, key: &ReplayKey, epoch: u64) {
        let mut inner = self.inner.lock().await;
        let matches = inner
            .entries
            .get(key)
            .is_some_and(|entry| entry.epoch == epoch && entry.response.is_none());
        if matches {
            if let Some(entry) = inner.entries.remove(key) {
                let _ = entry.publish.send(PublishState::Abandoned);
            }
        }
    }

    fn expire(inner: &mut Inner, now: Instant) {
        let expired: Vec<ReplayKey> = inner
            .entries
            .iter()
            .filter(|(_, entry)| entry.response.is_some() && entry.expires_at <= now)
            .map(|(key, _)| key.clone())
            .collect();
        for key in expired {
            if let Some(entry) = inner.entries.remove(&key) {
                inner.response_bytes -= entry.size_bytes;
            }
        }
    }

    /// Evict oldest completed entries until count and byte bounds hold.
    fn evict_completed(inner: &mut Inner, capacity: usize, byte_cap: usize) {
        while inner.entries.len() > capacity || inner.response_bytes > byte_cap {
            let Some(key) = Self::oldest_completed(inner) else {
                return;
            };
            if let Some(entry) = inner.entries.remove(&key) {
                inner.response_bytes -= entry.size_bytes;
            }
        }
    }

    /// Evict completed work or reject when every bounded slot is in flight.
    fn make_capacity(inner: &mut Inner, capacity: usize) -> Result<(), PublicError> {
        while inner.entries.len() >= capacity {
            let Some(key) = Self::oldest_completed(inner) else {
                return Err(overloaded_error());
            };
            if let Some(entry) = inner.entries.remove(&key) {
                inner.response_bytes -= entry.size_bytes;
            }
        }
        Ok(())
    }

    fn oldest_completed(inner: &Inner) -> Option<ReplayKey> {
        inner
            .entries
            .iter()
            .filter(|(_, entry)| entry.response.is_some())
            .min_by_key(|(_, entry)| entry.order)
            .map(|(key, _)| key.clone())
    }
}

impl OwnerLease {
    /// Publish one exact successful response from the unique owner.
    pub async fn complete(&mut self, response: CachedResponse) -> Result<(), PublicError> {
        self.settled = true;
        self.store
            .complete_entry(&self.key, self.epoch, response)
            .await
    }

    /// Remove failed owner work so no joiner receives invented content.
    pub async fn abandon(&mut self) {
        self.settled = true;
        self.store.abandon_entry(&self.key, self.epoch).await;
    }
}

impl Drop for OwnerLease {
    fn drop(&mut self) {
        if self.settled {
            return;
        }
        let Ok(handle) = tokio::runtime::Handle::try_current() else {
            return;
        };
        let store = self.store.clone();
        let key = self.key.clone();
        let epoch = self.epoch;
        handle.spawn(async move {
            store.abandon_entry(&key, epoch).await;
        });
    }
}

impl Joiner {
    /// Wait for the owner's published result or fail closed on abandonment.
    pub async fn result(mut self) -> Result<Arc<CachedResponse>, PublicError> {
        loop {
            let state = self.receiver.borrow_and_update().clone();
            match state {
                PublishState::Published(response) => return Ok(response),
                PublishState::Abandoned => return Err(replay_unavailable_error()),
                PublishState::Pending => {
                    if self.receiver.changed().await.is_err() {
                        return Err(replay_unavailable_error());
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn key(operation: &str, body: &str) -> ReplayKey {
        ReplayKey {
            organization_id: "org".to_string(),
            identity_id: "id".to_string(),
            alias_revision_id: "rev".to_string(),
            surface: "chat_completions".to_string(),
            caller_operation_sha256: operation.to_string(),
            canonical_request_sha256: body.to_string(),
        }
    }

    fn response(size: usize) -> CachedResponse {
        CachedResponse {
            status_code: 200,
            media_type: "application/json".to_string(),
            headers: vec![],
            body: vec![b'x'; size],
        }
    }

    #[tokio::test]
    async fn owner_publishes_and_duplicates_replay_the_exact_response() {
        let store = Arc::new(ReplayStore::new());
        let Claim::Owner(mut lease) = store.claim(key("op", "body")).await.expect("claim") else {
            panic!("first claim must own");
        };
        lease.complete(response(8)).await.expect("complete");
        match store.claim(key("op", "body")).await.expect("claim") {
            Claim::Replay(cached) => assert_eq!(cached.body.len(), 8),
            _ => panic!("second claim must replay"),
        }
    }

    #[tokio::test]
    async fn same_operation_different_body_conflicts() {
        let store = Arc::new(ReplayStore::new());
        let _lease = store.claim(key("op", "one")).await.expect("claim");
        let Err(error) = store.claim(key("op", "two")).await else {
            panic!("must conflict");
        };
        assert_eq!(error.status_code, 409);
        assert_eq!(error.code, "idempotency_conflict");
    }

    #[tokio::test]
    async fn joiner_receives_the_owner_result() {
        let store = Arc::new(ReplayStore::new());
        let Claim::Owner(mut lease) = store.claim(key("op", "body")).await.expect("claim") else {
            panic!("first claim must own");
        };
        let Claim::Join(joiner) = store.claim(key("op", "body")).await.expect("claim") else {
            panic!("second claim must join");
        };
        let waiting = tokio::spawn(joiner.result());
        lease.complete(response(4)).await.expect("complete");
        let cached = waiting.await.expect("join").expect("published");
        assert_eq!(cached.body.len(), 4);
    }

    #[tokio::test]
    async fn concurrent_identical_claims_produce_one_owner_and_identical_replies() {
        let store = Arc::new(ReplayStore::new());
        let mut owner = None;
        let mut joiners = Vec::new();
        for _ in 0..8 {
            match store.claim(key("op", "body")).await.expect("claim") {
                Claim::Owner(lease) => {
                    assert!(owner.is_none(), "exactly one claim may own");
                    owner = Some(lease);
                }
                Claim::Join(joiner) => joiners.push(joiner),
                Claim::Replay(_) => panic!("nothing is published yet"),
            }
        }
        assert_eq!(joiners.len(), 7);
        let waiting: Vec<_> = joiners
            .into_iter()
            .map(|joiner| tokio::spawn(joiner.result()))
            .collect();
        let mut lease = owner.expect("one owner");
        lease.complete(response(16)).await.expect("complete");
        for handle in waiting {
            let cached = handle.await.expect("join").expect("published");
            assert_eq!(cached.body, vec![b'x'; 16]);
        }
        match store.claim(key("op", "body")).await.expect("claim") {
            Claim::Replay(cached) => assert_eq!(cached.body, vec![b'x'; 16]),
            _ => panic!("later duplicates replay the stored response"),
        }
    }

    #[tokio::test]
    async fn abandoned_owner_fails_joiners_closed() {
        let store = Arc::new(ReplayStore::new());
        let Claim::Owner(mut lease) = store.claim(key("op", "body")).await.expect("claim") else {
            panic!("first claim must own");
        };
        let Claim::Join(joiner) = store.claim(key("op", "body")).await.expect("claim") else {
            panic!("second claim must join");
        };
        lease.abandon().await;
        let error = joiner.result().await.expect_err("must fail closed");
        assert_eq!(error.status_code, 409);
        assert_eq!(error.code, "idempotency_replay_unavailable");
    }

    #[tokio::test]
    async fn dropped_owner_abandons_without_explicit_settlement() {
        let store = Arc::new(ReplayStore::new());
        let Claim::Owner(lease) = store.claim(key("op", "body")).await.expect("claim") else {
            panic!("first claim must own");
        };
        let Claim::Join(joiner) = store.claim(key("op", "body")).await.expect("claim") else {
            panic!("second claim must join");
        };
        drop(lease);
        let error = joiner.result().await.expect_err("must fail closed");
        assert_eq!(error.code, "idempotency_replay_unavailable");
    }

    #[tokio::test]
    async fn oversize_result_abandons_and_fails_closed() {
        let store = Arc::new(ReplayStore::with_bounds(4, 16, DEFAULT_TTL));
        let Claim::Owner(mut lease) = store.claim(key("op", "body")).await.expect("claim") else {
            panic!("first claim must own");
        };
        let Err(error) = lease.complete(response(64)).await else {
            panic!("must reject oversize");
        };
        assert_eq!(error.status_code, 500);
        assert_eq!(error.code, "idempotency_replay_unavailable");
        match store.claim(key("op", "body")).await.expect("claim") {
            Claim::Owner(_) => {}
            _ => panic!("abandoned entry must reopen ownership"),
        }
    }

    #[tokio::test]
    async fn full_inflight_window_overloads() {
        let store = Arc::new(ReplayStore::with_bounds(1, 1024, DEFAULT_TTL));
        let _lease = store.claim(key("op-a", "body")).await.expect("claim");
        let Err(error) = store.claim(key("op-b", "body")).await else {
            panic!("must overload");
        };
        assert_eq!(error.status_code, 429);
        assert_eq!(error.code, "gateway_overloaded");
    }
}
