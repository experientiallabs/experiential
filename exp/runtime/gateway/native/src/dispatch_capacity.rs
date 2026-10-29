//! Explicit execution resources acquired together with process capacity.

use std::sync::Arc;

use tokio::sync::{OwnedSemaphorePermit, Semaphore};

use crate::errors::PublicError;
use crate::execution_binding::ExecutionLease;
use crate::permit_queue::{PermitQueue, ScopedPermit};

/// Capacity survives all clones, including a request's detached final accounting.
#[derive(Clone)]
pub(crate) struct DispatchPermit {
    _held: Arc<HeldCapacity>,
}

struct HeldCapacity {
    _global: Option<OwnedSemaphorePermit>,
    _scoped: Option<ScopedPermit>,
    _execution: Option<Arc<ExecutionLease>>,
}

pub(crate) struct DispatchCapacity;

impl DispatchCapacity {
    pub(crate) fn new() -> Self {
        Self
    }

    pub(crate) async fn acquire(
        &self,
        execution: Option<&Arc<ExecutionLease>>,
        global: &Arc<Semaphore>,
        queue: &PermitQueue,
        class: Option<u8>,
    ) -> Result<DispatchPermit, PublicError> {
        if let Some(execution) = execution {
            let permit = queue
                .acquire_scoped(global, &execution.resources.active, class)
                .await
                .map_err(|_| PublicError::draining())?;
            return Ok(DispatchPermit {
                _held: Arc::new(HeldCapacity {
                    _global: None,
                    _scoped: Some(permit),
                    _execution: Some(execution.clone()),
                }),
            });
        }
        let global = queue
            .acquire(global, class)
            .await
            .map_err(|_| PublicError::draining())?;
        Ok(DispatchPermit {
            _held: Arc::new(HeldCapacity {
                _global: Some(global),
                _scoped: None,
                _execution: None,
            }),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::execution_binding::tests::lease;
    use std::sync::Mutex;
    use std::time::Duration;

    #[test]
    fn a_high_class_waiter_overtakes_a_low_class_waiter_inside_one_binding() {
        tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap()
            .block_on(async {
                let shares = DispatchCapacity::new();
                let first_binding = lease(1, 2, 1);
                let global = Arc::new(Semaphore::new(4));
                let queue = PermitQueue::new(Some(10.0));
                let held = shares
                    .acquire(Some(&first_binding), &global, &queue, None)
                    .await
                    .unwrap();
                let order = Mutex::new(Vec::new());
                tokio::join!(
                    async {
                        let _permit = shares
                            .acquire(Some(&first_binding), &global, &queue, Some(0))
                            .await
                            .unwrap();
                        order.lock().unwrap().push("low");
                    },
                    async {
                        tokio::task::yield_now().await;
                        let _permit = shares
                            .acquire(Some(&first_binding), &global, &queue, Some(8))
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
    fn hot_streams_and_cancelled_waiters_leave_capacity_for_other_bindings() {
        tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap()
            .block_on(async {
                let shares = DispatchCapacity::new();
                let first_binding = lease(2, 2, 1);
                let second_binding = lease(2, 2, 1);
                let global = Arc::new(Semaphore::new(4));
                let queue = PermitQueue::new(None);
                let first = shares
                    .acquire(Some(&first_binding), &global, &queue, None)
                    .await
                    .unwrap();
                let second = shares
                    .acquire(Some(&first_binding), &global, &queue, None)
                    .await
                    .unwrap();
                assert!(tokio::time::timeout(
                    Duration::from_millis(10),
                    shares.acquire(Some(&first_binding), &global, &queue, Some(8))
                )
                .await
                .is_err());
                assert_eq!(global.available_permits(), 2);
                let quiet = shares
                    .acquire(Some(&second_binding), &global, &queue, None)
                    .await
                    .unwrap();
                assert_eq!(global.available_permits(), 1);
                drop(first);
                let retry = shares
                    .acquire(Some(&first_binding), &global, &queue, None)
                    .await
                    .unwrap();
                drop((second, quiet, retry));
                assert_eq!(global.available_permits(), 4);
            });
    }

    #[test]
    fn globally_waiting_low_priority_does_not_reserve_supplied_capacity() {
        tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap()
            .block_on(async {
                let shares = DispatchCapacity::new();
                let first_binding = lease(1, 2, 1);
                let second_binding = lease(1, 2, 1);
                let global = Arc::new(Semaphore::new(1));
                let queue = PermitQueue::new(Some(10.0));
                let held = shares
                    .acquire(Some(&second_binding), &global, &queue, None)
                    .await
                    .unwrap();
                let order = Mutex::new(Vec::new());
                tokio::join!(
                    async {
                        let _permit = shares
                            .acquire(Some(&first_binding), &global, &queue, Some(0))
                            .await
                            .unwrap();
                        order.lock().unwrap().push("low");
                    },
                    async {
                        tokio::task::yield_now().await;
                        let _permit = shares
                            .acquire(Some(&first_binding), &global, &queue, Some(8))
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
    fn individual_execution_uses_the_full_process_capacity() {
        tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap()
            .block_on(async {
                let capacity = DispatchCapacity::new();
                let global = Arc::new(Semaphore::new(3));
                let queue = PermitQueue::new(None);
                let first = capacity.acquire(None, &global, &queue, None).await.unwrap();
                let second = capacity.acquire(None, &global, &queue, None).await.unwrap();
                let third = capacity.acquire(None, &global, &queue, None).await.unwrap();
                assert_eq!(global.available_permits(), 0);
                let cleanup = first.clone();
                drop(first);
                assert_eq!(global.available_permits(), 0);
                drop(cleanup);
                assert_eq!(global.available_permits(), 1);
                queue.close();
                global.close();
                assert!(capacity.acquire(None, &global, &queue, None).await.is_err());
                drop((second, third));
                assert_eq!(global.available_permits(), 3);
            });
    }
}
