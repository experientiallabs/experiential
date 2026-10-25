//! Tenant shares held for the entire lifetime of an active request or upload.

use std::collections::HashMap;
use std::sync::{Arc, Mutex, Weak};

use tokio::sync::{OwnedSemaphorePermit, Semaphore};

use crate::errors::PublicError;
use crate::permit_queue::PermitQueue;

/// Both capacities travel with a response stream and survive handler cancellation.
pub(crate) struct DispatchPermit {
    _global: OwnedSemaphorePermit,
    _organization: Option<OwnedSemaphorePermit>,
    _organization_pool: Option<Arc<OrganizationPool>>,
}

/// Organization queues cannot occupy the global queue before getting their share.
struct OrganizationPool {
    semaphore: Arc<Semaphore>,
    queue: PermitQueue,
}

impl OrganizationPool {
    fn new(limit: usize, aging: Option<f64>) -> Self {
        Self {
            semaphore: Arc::new(Semaphore::new(limit)),
            queue: PermitQueue::new(aging),
        }
    }
}

pub(crate) struct DispatchCapacity {
    limit: Option<usize>,
    scopes: Mutex<HashMap<String, Weak<OrganizationPool>>>,
    unscoped: Arc<OrganizationPool>,
    aging: Option<f64>,
}

impl DispatchCapacity {
    pub(crate) fn new(limit: Option<usize>, aging: Option<f64>) -> Self {
        Self {
            limit,
            scopes: Mutex::new(HashMap::new()),
            unscoped: Arc::new(OrganizationPool::new(limit.unwrap_or(1), aging)),
            aging,
        }
    }

    pub(crate) async fn acquire(
        &self,
        organization: Option<&str>,
        global: &Arc<Semaphore>,
        queue: &PermitQueue,
        class: Option<u8>,
    ) -> Result<DispatchPermit, PublicError> {
        let (organization_permit, organization_pool) = if let Some(limit) = self.limit {
            let pool = if let Some(organization) = organization {
                let mut scopes = self.scopes.lock().map_err(|_| PublicError::internal())?;
                scopes.retain(|_, pool| pool.strong_count() > 0);
                match scopes.get(organization).and_then(Weak::upgrade) {
                    Some(pool) => pool,
                    None => {
                        let pool = Arc::new(OrganizationPool::new(limit, self.aging));
                        scopes.insert(organization.to_owned(), Arc::downgrade(&pool));
                        pool
                    }
                }
            } else {
                self.unscoped.clone()
            };
            let permit = pool
                .queue
                .acquire(&pool.semaphore, class)
                .await
                .map_err(|_| PublicError::draining())?;
            (Some(permit), Some(pool))
        } else {
            (None, None)
        };
        let global = queue
            .acquire(global, class)
            .await
            .map_err(|_| PublicError::draining())?;
        Ok(DispatchPermit {
            _global: global,
            _organization: organization_permit,
            _organization_pool: organization_pool,
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
}
