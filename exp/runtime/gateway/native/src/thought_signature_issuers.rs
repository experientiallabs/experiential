//! Which rung minted a Gemini thought signature this worker relayed.
//!
//! The Gemini API and Vertex AI each verify only the thought signatures they
//! issued. When a later turn of a conversation lands on the other backend
//! (latency demotion, ordering, failover), the replayed signature is refused
//! and `thought_signature_repair` swaps in Google's bypass value. That keeps
//! the turn alive, but the model loses its signed reasoning state and a long
//! agent conversation is then far more likely to end on a malformed function
//! call. Admission prefers the rung that minted the newest replayed signature
//! instead, so the turn reaches a backend that can verify it.
//!
//! The memory holds a digest of each signature a Gemini-dialect rung returned
//! and the deployment that returned it, per worker, for a bounded time and
//! count. No signature text is kept. A signature is unforgeable and unique to
//! its turn, so the digest needs no caller salt: a caller can only look up a
//! signature it already holds, and the answer is a rung of its own route.
//! A conversation whose turn reaches another worker has no entry there, and
//! routes and repairs exactly as before (the serving edge's keep-alives pin
//! most clients to one worker).

use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, LazyLock, Mutex};
use std::time::{Duration, Instant};

use sha2::{Digest, Sha256};

use crate::events::Event;
use crate::thought_signature_repair::THOUGHT_SIGNATURE_BYPASS;

/// How long after its last use a signature's issuer is remembered: longer
/// than an agent's pause between turns, short of an abandoned session.
const ISSUER_TTL: Duration = Duration::from_secs(2 * 60 * 60);

/// Signatures remembered per worker; the oldest insertion goes first.
const ISSUER_CAPACITY: usize = 100_000;

/// Distinct deployment ids shared before the share table is rebuilt; entries
/// keep their own handle, so a rebuild only drops ids no longer remembered.
const SHARED_DEPLOYMENTS_CAPACITY: usize = 1_024;

pub static ISSUERS: LazyLock<SignatureIssuers> = LazyLock::new(SignatureIssuers::default);

/// Signature digest to the deployment that minted it.
#[derive(Default)]
pub struct SignatureIssuers {
    state: Mutex<IssuerState>,
}

struct Issued {
    deployment_id: Arc<str>,
    last_seen: Instant,
    sequence: u64,
}

#[derive(Default)]
struct IssuerState {
    entries: HashMap<[u8; 32], Issued>,
    /// Insertion order as `(digest, sequence)`; an entry learned again gets a
    /// new sequence, so its stale queue entry is skipped by the sweep.
    insertion_order: VecDeque<([u8; 32], u64)>,
    next_sequence: u64,
    /// One shared string per deployment id, so entries cost no allocation.
    deployments: HashMap<String, Arc<str>>,
}

fn digest(signature: &str) -> [u8; 32] {
    Sha256::digest(signature.as_bytes()).into()
}

impl SignatureIssuers {
    /// Remember that `deployment_id` minted `signature`, as of now.
    pub fn remember(&self, signature: &str, deployment_id: &str) {
        if signature.is_empty() || signature == THOUGHT_SIGNATURE_BYPASS {
            return;
        }
        let key = digest(signature);
        let now = Instant::now();
        let mut state = self
            .state
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        if state.deployments.len() >= SHARED_DEPLOYMENTS_CAPACITY
            && !state.deployments.contains_key(deployment_id)
        {
            state.deployments.clear();
        }
        let deployment = match state.deployments.get(deployment_id) {
            Some(shared) => shared.clone(),
            None => {
                let shared: Arc<str> = Arc::from(deployment_id);
                state
                    .deployments
                    .insert(deployment_id.to_string(), shared.clone());
                shared
            }
        };
        let sequence = state.next_sequence;
        state.next_sequence += 1;
        state.entries.insert(
            key,
            Issued {
                deployment_id: deployment,
                last_seen: now,
                sequence,
            },
        );
        state.insertion_order.push_back((key, sequence));
        while state.entries.len() > ISSUER_CAPACITY
            || state.insertion_order.len() > 2 * ISSUER_CAPACITY
        {
            let Some((oldest, sequence)) = state.insertion_order.pop_front() else {
                break;
            };
            if state
                .entries
                .get(&oldest)
                .is_some_and(|entry| entry.sequence == sequence)
            {
                state.entries.remove(&oldest);
            }
        }
    }

    /// The deployment that minted `signature`, when remembered and unexpired;
    /// a hit refreshes it.
    pub fn issuer(&self, signature: &str) -> Option<String> {
        let key = digest(signature);
        let now = Instant::now();
        let mut state = self
            .state
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        match state.entries.get_mut(&key) {
            Some(entry) if now.duration_since(entry.last_seen) <= ISSUER_TTL => {
                entry.last_seen = now;
                Some(entry.deployment_id.to_string())
            }
            Some(_) => {
                state.entries.remove(&key);
                None
            }
            None => None,
        }
    }

    /// Remember every signature one relayed event carries for `deployment_id`.
    pub(crate) fn observe(&self, event: &Event, deployment_id: &str) {
        let signature = match event {
            Event::ToolCallStarted {
                thought_signature: Some(signature),
                ..
            } => Some(signature.as_str()),
            Event::GeminiThoughtPart(part) => part
                .get("thoughtSignature")
                .and_then(|value| value.as_str()),
            _ => None,
        };
        if let Some(signature) = signature {
            self.remember(signature, deployment_id);
        }
    }

    #[cfg(test)]
    fn len(&self) -> usize {
        self.state
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .entries
            .len()
    }
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;

    #[test]
    fn a_relayed_signature_names_the_rung_that_minted_it() {
        let issuers = SignatureIssuers::default();
        issuers.observe(
            &Event::ToolCallStarted {
                custom: false,
                index: 0,
                call_id: "call-1".into(),
                name: "read".into(),
                namespace: None,
                caller: None,
                thought_signature: Some("c2lnLWE=".into()),
            },
            "gemini-api",
        );
        issuers.observe(
            &Event::GeminiThoughtPart(
                json!({"thought": true, "thoughtSignature": "c2lnLWI="}).into(),
            ),
            "vertex",
        );
        assert_eq!(issuers.issuer("c2lnLWE=").as_deref(), Some("gemini-api"));
        assert_eq!(issuers.issuer("c2lnLWI=").as_deref(), Some("vertex"));
        assert_eq!(issuers.issuer("unknown"), None);
        // A later turn minted elsewhere moves the signature's issuer.
        issuers.remember("c2lnLWE=", "vertex");
        assert_eq!(issuers.issuer("c2lnLWE=").as_deref(), Some("vertex"));
    }

    #[test]
    fn deployment_churn_never_grows_the_shared_id_table_past_its_bound() {
        let issuers = SignatureIssuers::default();
        for index in 0..(3 * SHARED_DEPLOYMENTS_CAPACITY) {
            issuers.remember(&format!("sig-{index}"), &format!("deployment-{index}"));
        }
        let shared = issuers.state.lock().unwrap().deployments.len();
        assert!(shared <= SHARED_DEPLOYMENTS_CAPACITY);
        // Entries keep their issuer across a rebuild.
        assert_eq!(issuers.issuer("sig-0").as_deref(), Some("deployment-0"));
    }

    #[test]
    fn the_bypass_value_and_unsigned_events_are_never_remembered() {
        let issuers = SignatureIssuers::default();
        issuers.remember(THOUGHT_SIGNATURE_BYPASS, "gemini-api");
        issuers.remember("", "gemini-api");
        issuers.observe(&Event::TextDelta("hi".into()), "gemini-api");
        issuers.observe(
            &Event::GeminiThoughtPart(json!({"text": "plan"}).into()),
            "vertex",
        );
        assert_eq!(issuers.len(), 0);
    }

    #[test]
    fn an_issuer_expires_after_its_window_and_the_oldest_goes_first_at_capacity() {
        let issuers = SignatureIssuers::default();
        issuers.remember("old", "gemini-api");
        issuers
            .state
            .lock()
            .unwrap()
            .entries
            .values_mut()
            .for_each(|entry| {
                entry.last_seen = Instant::now() - ISSUER_TTL - Duration::from_secs(1)
            });
        assert_eq!(issuers.issuer("old"), None);
        for index in 0..=ISSUER_CAPACITY {
            issuers.remember(&format!("sig-{index}"), "vertex");
        }
        assert_eq!(issuers.len(), ISSUER_CAPACITY);
        assert_eq!(issuers.issuer("sig-0"), None);
        assert_eq!(
            issuers.issuer(&format!("sig-{ISSUER_CAPACITY}")).as_deref(),
            Some("vertex")
        );
    }
}
