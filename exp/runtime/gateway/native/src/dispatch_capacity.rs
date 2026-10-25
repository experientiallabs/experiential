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
}

/// Organization queues cannot occupy the global queue before getting their share.
pub(crate) struct DispatchCapacity {
    limit: Option<usize>,
    scopes: Mutex<HashMap<String, Weak<Semaphore>>>,
    unscoped: Arc<Semaphore>,
}

impl DispatchCapacity {
    pub(crate) fn new(limit: Option<usize>) -> Self {
        Self {
            limit,
            scopes: Mutex::new(HashMap::new()),
            unscoped: Arc::new(Semaphore::new(limit.unwrap_or(1))),
        }
    }

    pub(crate) async fn acquire(
        &self,
        organization: Option<&str>,
        global: &Arc<Semaphore>,
        queue: &PermitQueue,
        class: Option<u8>,
    ) -> Result<DispatchPermit, PublicError> {
        let organization_permit = if let Some(limit) = self.limit {
            let pool = if let Some(organization) = organization {
                let mut scopes = self.scopes.lock().map_err(|_| PublicError::internal())?;
                scopes.retain(|_, pool| pool.strong_count() > 0);
                match scopes.get(organization).and_then(Weak::upgrade) {
                    Some(pool) => pool,
                    None => {
                        let pool = Arc::new(Semaphore::new(limit));
                        scopes.insert(organization.to_owned(), Arc::downgrade(&pool));
                        pool
                    }
                }
            } else {
                self.unscoped.clone()
            };
            Some(
                pool.acquire_owned()
                    .await
                    .map_err(|_| PublicError::draining())?,
            )
        } else {
            None
        };
        let global = queue
            .acquire(global, class)
            .await
            .map_err(|_| PublicError::draining())?;
        Ok(DispatchPermit {
            _global: global,
            _organization: organization_permit,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    #[test]
    fn hot_streams_and_cancelled_waiters_leave_capacity_for_other_organizations() {
        tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap()
            .block_on(async {
                let shares = DispatchCapacity::new(Some(2));
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
