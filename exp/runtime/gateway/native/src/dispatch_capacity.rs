//! Tenant shares held for the entire lifetime of an active request or upload.

use std::collections::HashMap;
use std::sync::{Arc, Mutex, Weak};

use tokio::sync::{OwnedSemaphorePermit, Semaphore};

use crate::errors::PublicError;
use crate::permit_queue::{PermitQueue, ScopedPermit};

/// Both capacities travel with a response stream and survive handler cancellation.
pub(crate) struct DispatchPermit {
    _global: Option<OwnedSemaphorePermit>,
    _scoped: Option<ScopedPermit>,
    _organization_pool: Option<Arc<OrganizationPool>>,
}

/// The global queue grants this share together with its own capacity.
struct OrganizationPool {
    semaphore: Arc<Semaphore>,
}

impl OrganizationPool {
    fn new(limit: usize) -> Self {
        Self {
            semaphore: Arc::new(Semaphore::new(limit)),
        }
    }
}

pub(crate) struct DispatchCapacity {
    limit: Option<usize>,
    scopes: Mutex<HashMap<String, Weak<OrganizationPool>>>,
}

impl DispatchCapacity {
    pub(crate) fn new(limit: Option<usize>, _aging: Option<f64>) -> Self {
        Self {
            limit,
            scopes: Mutex::new(HashMap::new()),
        }
    }

    pub(crate) async fn acquire(
        &self,
        organization: Option<&str>,
        global: &Arc<Semaphore>,
        queue: &PermitQueue,
        class: Option<u8>,
    ) -> Result<DispatchPermit, PublicError> {
        if let Some(limit) = self.limit {
            let organization = organization.ok_or_else(PublicError::invalid_key)?;
            let pool = {
                let mut scopes = self.scopes.lock().map_err(|_| PublicError::internal())?;
                scopes.retain(|_, pool| pool.strong_count() > 0);
                match scopes.get(organization).and_then(Weak::upgrade) {
                    Some(pool) => pool,
                    None => {
                        let pool = Arc::new(OrganizationPool::new(limit));
                        scopes.insert(organization.to_owned(), Arc::downgrade(&pool));
                        pool
                    }
                }
            };
            let permit = queue
                .acquire_scoped(global, &pool.semaphore, class)
                .await
                .map_err(|_| PublicError::draining())?;
            return Ok(DispatchPermit {
                _global: None,
                _scoped: Some(permit),
                _organization_pool: Some(pool),
            });
        }
        let global = queue
            .acquire(global, class)
            .await
            .map_err(|_| PublicError::draining())?;
        Ok(DispatchPermit {
            _global: Some(global),
            _scoped: None,
            _organization_pool: None,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    #[test]
    fn a_high_class_waiter_overtakes_a_low_class_waiter_inside_one_organization() {
        tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap()
            .block_on(async {
                let shares = DispatchCapacity::new(Some(1), Some(10.0));
                let global = Arc::new(Semaphore::new(4));
                let queue = PermitQueue::new(Some(10.0));
                let held = shares
                    .acquire(Some("org"), &global, &queue, None)
                    .await
                    .unwrap();
                let order = Mutex::new(Vec::new());
                tokio::join!(
                    async {
                        let _permit = shares
                            .acquire(Some("org"), &global, &queue, Some(0))
                            .await
                            .unwrap();
                        order.lock().unwrap().push("low");
                    },
                    async {
                        tokio::task::yield_now().await;
                        let _permit = shares
                            .acquire(Some("org"), &global, &queue, Some(8))
                            .await
                            .unwrap();
                        order.lock().unwrap().push("high");
                    },
                    async {
                        tokio::time::sleep(Duration::from_millis(10)).await;
                        drop(held);
                    }
                );
                assert_eq!(*order.lock().unwrap(), ["high", "low"]);
            });
    }

    #[test]
    fn hot_streams_and_cancelled_waiters_leave_capacity_for_other_organizations() {
        tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap()
            .block_on(async {
                let shares = DispatchCapacity::new(Some(2), None);
                let global = Arc::new(Semaphore::new(4));
                let queue = PermitQueue::new(None);
                let first = shares
                    .acquire(Some("hot"), &global, &queue, None)
                    .await
                    .unwrap();
                let second = shares
                    .acquire(Some("hot"), &global, &queue, None)
                    .await
                    .unwrap();
                assert!(tokio::time::timeout(
                    Duration::from_millis(10),
                    shares.acquire(Some("hot"), &global, &queue, Some(8))
                )
                .await
                .is_err());
                assert_eq!(global.available_permits(), 2);
                let quiet = shares
                    .acquire(Some("quiet"), &global, &queue, None)
                    .await
                    .unwrap();
                assert_eq!(global.available_permits(), 1);
                drop(first);
                let retry = shares
                    .acquire(Some("hot"), &global, &queue, None)
                    .await
                    .unwrap();
                drop((second, quiet, retry));
                assert_eq!(global.available_permits(), 4);
            });
    }

    #[test]
    fn globally_waiting_low_priority_does_not_reserve_the_tenant_share() {
        tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap()
            .block_on(async {
                let shares = DispatchCapacity::new(Some(1), Some(10.0));
                let global = Arc::new(Semaphore::new(1));
                let queue = PermitQueue::new(Some(10.0));
                let held = shares
                    .acquire(Some("other"), &global, &queue, None)
                    .await
                    .unwrap();
                let order = Mutex::new(Vec::new());
                tokio::join!(
                    async {
                        let _permit = shares
                            .acquire(Some("org"), &global, &queue, Some(0))
                            .await
                            .unwrap();
                        order.lock().unwrap().push("low");
                    },
                    async {
                        tokio::task::yield_now().await;
                        let _permit = shares
                            .acquire(Some("org"), &global, &queue, Some(8))
                            .await
                            .unwrap();
                        order.lock().unwrap().push("high");
                    },
                    async {
                        tokio::time::sleep(Duration::from_millis(10)).await;
                        drop(held);
                    }
                );
                assert_eq!(*order.lock().unwrap(), ["high", "low"]);
                assert_eq!(global.available_permits(), 1);
            });
    }

    #[test]
    fn shutdown_rejects_waiters_and_missing_scope_without_taking_capacity() {
        tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap()
            .block_on(async {
                let shares = DispatchCapacity::new(Some(1), None);
                let global = Arc::new(Semaphore::new(3));
                let queue = PermitQueue::new(None);
                assert!(shares.acquire(None, &global, &queue, None).await.is_err());
                assert_eq!(global.available_permits(), 3);
                let held = shares
                    .acquire(Some("org"), &global, &queue, None)
                    .await
                    .unwrap();
                tokio::join!(
                    async {
                        assert!(shares
                            .acquire(Some("org"), &global, &queue, None)
                            .await
                            .is_err());
                    },
                    async {
                        tokio::task::yield_now().await;
                        global.close();
                        queue.close();
                    }
                );
                assert!(shares
                    .acquire(Some("new-org"), &global, &queue, None)
                    .await
                    .is_err());
                assert_eq!(global.available_permits(), 2);
                drop(held);
                assert_eq!(global.available_permits(), 3);
                let legacy = DispatchCapacity::new(None, None);
                let global = Arc::new(Semaphore::new(1));
                let queue = PermitQueue::new(None);
                let legacy_permit = legacy.acquire(None, &global, &queue, None).await.unwrap();
                assert_eq!(global.available_permits(), 0);
                drop(legacy_permit);
                assert_eq!(global.available_permits(), 1);
            });
    }
}
