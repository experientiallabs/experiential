//! Tenant scheduling before a callback occupies a Python worker.

use std::collections::HashMap;
use std::sync::{Arc, Mutex, Weak};
use std::time::{Duration, Instant};

use serde::Deserialize;
use tokio::sync::{OwnedSemaphorePermit, Semaphore};

use crate::errors::PublicError;

/// An admitted request's scope comes only from the control plane's answer.
#[derive(Default, Deserialize)]
struct AdmissionScope {
    request_id: Option<String>,
    caller_scope: Option<String>,
}

/// Scheduling reads engine-owned identity, never caller headers or body fields.
#[derive(Default, Deserialize)]
struct RequestIdentity {
    request_id: Option<String>,
    raw_key: Option<String>,
    finalize: Option<bool>,
}

/// Independent tenant shares for serving and durable cleanup callbacks.
pub(crate) struct CallbackCapacity {
    requests: Mutex<HashMap<String, (String, Instant)>>,
    pub(crate) tokens: Arc<Mutex<HashMap<String, String>>>,
    last_cleanup: Mutex<Instant>,
    pub(crate) request_timeout: Duration,
    serving: Mutex<HashMap<String, Weak<Semaphore>>>,
    settlement: Mutex<HashMap<String, Weak<Semaphore>>>,
    unscoped_serving: Arc<Semaphore>,
    unscoped_settlement: Arc<Semaphore>,
    serving_share: usize,
    settlement_share: usize,
}

impl CallbackCapacity {
    pub(crate) fn new(serving: usize, settlement: usize) -> Self {
        let serving_share = (serving / 2).max(1);
        let settlement_share = (settlement / 2).max(1);
        Self {
            requests: Mutex::new(HashMap::new()),
            tokens: Arc::new(Mutex::new(HashMap::new())),
            last_cleanup: Mutex::new(Instant::now()),
            request_timeout: Duration::from_secs(3700),
            serving: Mutex::new(HashMap::new()),
            settlement: Mutex::new(HashMap::new()),
            unscoped_serving: Arc::new(Semaphore::new(serving_share)),
            unscoped_settlement: Arc::new(Semaphore::new(settlement_share)),
            serving_share,
            settlement_share,
        }
    }

    /// Cleanup never competes for the serving pool's workers or permits.
    pub(crate) fn is_settlement(method: &str) -> bool {
        matches!(method, "settle" | "abandon")
    }

    /// Acquire tenant capacity before taking shared callback capacity.
    pub(crate) async fn acquire(
        &self,
        method: &str,
        argument: &str,
    ) -> Result<OwnedSemaphorePermit, PublicError> {
        let identity = serde_json::from_str::<RequestIdentity>(argument).unwrap_or_default();
        let scope = identity
            .raw_key
            .as_ref()
            .and_then(|token| {
                self.tokens
                    .lock()
                    .ok()
                    .and_then(|items| items.get(token).cloned())
            })
            .or_else(|| {
                identity.request_id.as_ref().and_then(|id| {
                    self.requests
                        .lock()
                        .ok()
                        .and_then(|items| items.get(id).map(|(scope, _)| scope.clone()))
                })
            });
        let settlement = Self::is_settlement(method);
        let pool = if let Some(scope) = scope {
            let (registry, share) = if settlement {
                (&self.settlement, self.settlement_share)
            } else {
                (&self.serving, self.serving_share)
            };
            let mut registry = registry.lock().map_err(|_| PublicError::internal())?;
            registry.retain(|_, item| item.strong_count() != 0);
            match registry.get(&scope).and_then(Weak::upgrade) {
                Some(pool) => pool,
                None => {
                    let pool = Arc::new(Semaphore::new(share));
                    registry.insert(scope, Arc::downgrade(&pool));
                    pool
                }
            }
        } else if settlement {
            self.unscoped_settlement.clone()
        } else {
            self.unscoped_serving.clone()
        };
        pool.acquire_owned()
            .await
            .map_err(|_| PublicError::internal())
    }

    /// Record only host-vouched scope; remove it after a successful terminal write.
    pub(crate) fn completed(&self, method: &str, argument: &str, reply: &str) {
        if let Ok(mut last) = self.last_cleanup.lock() {
            if last.elapsed() >= Duration::from_secs(1) {
                if let Ok(mut requests) = self.requests.lock() {
                    let now = Instant::now();
                    requests.retain(|_, (_, deadline)| *deadline > now);
                }
                *last = Instant::now();
            }
        }
        if method == "admit" || method.starts_with("admit_") {
            if let Ok(AdmissionScope {
                request_id: Some(id),
                caller_scope: Some(scope),
            }) = serde_json::from_str(reply)
            {
                let organization = scope.split_once(':').map_or(scope.as_str(), |(org, _)| org);
                if let Ok(mut requests) = self.requests.lock() {
                    // A host that fails to finalize cannot grow scheduler state forever.
                    // Overflow stays in the bounded unscoped share; it never gains capacity.
                    if requests.len() < 65_536 && id.len() <= 128 && organization.len() <= 128 {
                        requests.insert(
                            id,
                            (
                                organization.to_owned(),
                                Instant::now() + self.request_timeout,
                            ),
                        );
                    }
                }
            }
        } else if Self::is_settlement(method) {
            let identity = serde_json::from_str::<RequestIdentity>(argument).unwrap_or_default();
            let terminalized = method == "settle" && identity.finalize.unwrap_or(true)
                || method == "abandon"
                    && serde_json::from_str::<serde_json::Value>(reply)
                        .ok()
                        .and_then(|value| {
                            value
                                .get("terminalized")
                                .and_then(serde_json::Value::as_bool)
                        })
                        == Some(true);
            if terminalized {
                if let Some(id) = identity.request_id {
                    if let Ok(mut requests) = self.requests.lock() {
                        requests.remove(&id);
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn runtime() -> tokio::runtime::Runtime {
        tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap()
    }

    #[test]
    fn busy_tenant_waits_before_shared_capacity_and_cleanup_has_its_own_share() {
        runtime().block_on(async {
            let capacity = CallbackCapacity::new(4, 2);
            capacity.completed(
                "admit",
                "{}",
                r#"{"request_id":"a","caller_scope":"hot:one"}"#,
            );
            capacity.completed(
                "admit",
                "{}",
                r#"{"request_id":"b","caller_scope":"hot:two"}"#,
            );
            capacity.completed(
                "admit",
                "{}",
                r#"{"request_id":"c","caller_scope":"quiet:one"}"#,
            );
            let first = capacity
                .acquire("start_attempt", r#"{"request_id":"a"}"#)
                .await
                .unwrap();
            let second = capacity
                .acquire("start_attempt", r#"{"request_id":"b"}"#)
                .await
                .unwrap();
            assert!(tokio::time::timeout(
                std::time::Duration::from_millis(10),
                capacity.acquire("start_attempt", r#"{"request_id":"a"}"#)
            )
            .await
            .is_err());
            let quiet = capacity
                .acquire("start_attempt", r#"{"request_id":"c"}"#)
                .await
                .unwrap();
            let cleanup = capacity
                .acquire("settle", r#"{"request_id":"a"}"#)
                .await
                .unwrap();
            drop((first, second, quiet, cleanup));
            capacity
                .acquire("start_attempt", r#"{"request_id":"a"}"#)
                .await
                .unwrap();
        });
    }

    #[test]
    fn supplied_scope_cannot_claim_tenant_capacity_and_nonfinal_settlement_keeps_identity() {
        runtime().block_on(async {
            let capacity = CallbackCapacity::new(4, 2);
            let first = capacity
                .acquire("admit", r#"{"caller_scope":"fake:a"}"#)
                .await
                .unwrap();
            let second = capacity
                .acquire("admit", r#"{"caller_scope":"fake:b"}"#)
                .await
                .unwrap();
            assert!(tokio::time::timeout(
                std::time::Duration::from_millis(10),
                capacity.acquire("admit", r#"{"caller_scope":"fake:c"}"#)
            )
            .await
            .is_err());
            drop((first, second));
            capacity.completed(
                "admit",
                "{}",
                r#"{"request_id":"a","caller_scope":"org:key"}"#,
            );
            capacity.completed("settle", r#"{"request_id":"a","finalize":false}"#, "{}");
            assert!(capacity.requests.lock().unwrap().contains_key("a"));
            capacity.completed("settle", r#"{"request_id":"a","finalize":true}"#, "{}");
            assert!(capacity.requests.lock().unwrap().is_empty());
        });
    }

    #[test]
    fn trusted_relay_tokens_isolate_the_first_authentication_call() {
        runtime().block_on(async {
            let capacity = CallbackCapacity::new(4, 2);
            capacity.tokens.lock().unwrap().extend([
                ("hot-1".to_owned(), "hot".to_owned()),
                ("hot-2".to_owned(), "hot".to_owned()),
                ("quiet-1".to_owned(), "quiet".to_owned()),
            ]);
            let first = capacity
                .acquire("authenticate", r#"{"raw_key":"hot-1"}"#)
                .await
                .unwrap();
            let second = capacity
                .acquire("authenticate", r#"{"raw_key":"hot-2"}"#)
                .await
                .unwrap();
            assert!(tokio::time::timeout(
                Duration::from_millis(10),
                capacity.acquire("authenticate", r#"{"raw_key":"hot-1"}"#)
            )
            .await
            .is_err());
            let quiet = capacity
                .acquire("authenticate", r#"{"raw_key":"quiet-1"}"#)
                .await
                .unwrap();
            drop((first, second, quiet));
        });
    }

    #[test]
    fn alternate_surfaces_and_busy_abandon_preserve_the_trusted_scope() {
        let capacity = CallbackCapacity::new(4, 2);
        for surface in [
            "admit_embeddings",
            "admit_images",
            "admit_speech",
            "admit_transcription",
            "admit_decision",
            "admit_openai_decision",
        ] {
            capacity.completed(
                surface,
                "{}",
                r#"{"request_id":"a","caller_scope":"org:identity"}"#,
            );
            assert_eq!(capacity.requests.lock().unwrap()["a"].0, "org");
            capacity.completed(
                "abandon",
                r#"{"request_id":"a"}"#,
                r#"{"terminalized":false}"#,
            );
            assert!(capacity.requests.lock().unwrap().contains_key("a"));
            capacity.completed(
                "abandon",
                r#"{"request_id":"a"}"#,
                r#"{"terminalized":true}"#,
            );
            assert!(capacity.requests.lock().unwrap().is_empty());
        }
    }
}
