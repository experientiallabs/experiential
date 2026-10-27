//! Calls into the Python control plane (`NativeControlPlane`).
//!
//! Every call crosses the boundary as one JSON string in and one JSON string
//! out, executed on a fixed pool of long-lived worker threads under a bounded
//! permit count so GIL contention stays fixed regardless of data-plane
//! concurrency.
//!
//! The pool is dedicated rather than tokio's blocking pool because the
//! control plane caches per-thread state (one SQLite connection per thread in
//! a `threading.local`). Tokio blocking threads retire after an idle timeout
//! without that cache being released, so descriptor and memory retention
//! would track every blocking thread that ever ran a callback. A fixed set of
//! workers pins the cache to exactly `worker_count` threads for the life of
//! the bridge, and each worker releases its cached resources through the
//! control plane's `close_thread_resources` callback before it exits. Bound
//! targets share these workers, so their thread-local resources close after
//! every callback, before its execution lease can retire.

use std::sync::{mpsc, Arc, Mutex};
use std::thread::JoinHandle;

use pyo3::prelude::*;
use tokio::sync::{oneshot, OwnedSemaphorePermit, Semaphore};

use crate::errors::PublicError;
use crate::execution_binding::ExecutionLease;

/// One queued control-plane call and the responder that hands its outcome
/// back to the awaiting request task.
struct Job {
    method: &'static str,
    argument: String,
    responder: oneshot::Sender<Result<String, PublicError>>,
    // Cancellation of the awaiting request cannot release queued/running capacity.
    _permit: OwnedSemaphorePermit,
    _execution_permit: Option<OwnedSemaphorePermit>,
    execution: Option<Arc<ExecutionLease>>,
}

/// Bounded bridge to one Python `NativeControlPlane` instance.
///
/// Dropping the bridge closes the job queue, waits for every worker to run
/// its `close_thread_resources` cleanup, and joins the threads, so a stopped
/// server leaves no cached per-thread connection behind.
pub struct Bridge {
    pool: Arc<WorkerPool>,
    pub(crate) execution: Option<Arc<ExecutionLease>>,
}

/// One process owns the workers; bound request views share them without adding threads.
struct WorkerPool {
    queue: Mutex<Option<mpsc::Sender<Job>>>,
    workers: Mutex<Vec<JoinHandle<()>>>,
    permits: Arc<Semaphore>,
    settlement_queue: Mutex<Option<mpsc::Sender<Job>>>,
    settlement_permits: Arc<Semaphore>,
}

impl Bridge {
    /// Start `maximum_concurrent_calls` named worker threads over one queue.
    ///
    /// The permit count equals the worker count, so an accepted call always
    /// has an idle worker and never queues behind another call after its
    /// permit is granted.
    pub fn new(object: Py<PyAny>, maximum_concurrent_calls: usize) -> Result<Self, String> {
        let worker_count = maximum_concurrent_calls.clamp(1, 256);
        let settlement_count = (worker_count / 4).max(2);
        let (sender, receiver) = mpsc::channel::<Job>();
        let receiver = Arc::new(Mutex::new(receiver));
        let mut workers = Vec::with_capacity(worker_count);
        for index in 0..worker_count {
            let receiver = receiver.clone();
            let object = Python::attach(|py| object.clone_ref(py));
            let handle = std::thread::Builder::new()
                .name(format!("gateway-bridge-{index}"))
                .spawn(move || worker_loop(&receiver, &object))
                .map_err(|error| format!("failed to start bridge worker {index}: {error}"))?;
            workers.push(handle);
        }
        let (settlement_sender, settlement_receiver) = mpsc::channel::<Job>();
        let settlement_receiver = Arc::new(Mutex::new(settlement_receiver));
        for index in 0..settlement_count {
            let receiver = settlement_receiver.clone();
            let object = Python::attach(|py| object.clone_ref(py));
            let handle = std::thread::Builder::new()
                .name(format!("gateway-settlement-{index}"))
                .spawn(move || worker_loop(&receiver, &object))
                .map_err(|error| format!("failed to start settlement worker {index}: {error}"))?;
            workers.push(handle);
        }
        Ok(Self {
            pool: Arc::new(WorkerPool {
                settlement_queue: Mutex::new(Some(settlement_sender)),
                settlement_permits: Arc::new(Semaphore::new(settlement_count)),
                queue: Mutex::new(Some(sender)),
                workers: Mutex::new(workers),
                permits: Arc::new(Semaphore::new(worker_count)),
            }),
            execution: None,
        })
    }

    /// Bind a request to the host's supplied target and resources, without new workers.
    pub(crate) fn bind(&self, execution: Arc<ExecutionLease>) -> Self {
        Self {
            pool: self.pool.clone(),
            execution: Some(execution),
        }
    }

    /// Call one control-plane method with a JSON-string argument.
    pub async fn call(
        &self,
        method: &'static str,
        argument: String,
    ) -> Result<String, PublicError> {
        let settlement = matches!(method, "settle" | "abandon");
        // Supplied capacity is acquired before a shared worker. Individual
        // execution has no additional pool, identity lookup or allocation policy.
        let execution_permit = if let Some(execution) = &self.execution {
            let capacity = if settlement {
                &execution.resources.settlement
            } else {
                &execution.resources.serving
            };
            Some(
                capacity
                    .clone()
                    .acquire_owned()
                    .await
                    .map_err(|_| PublicError::internal())?,
            )
        } else {
            None
        };
        let pool = if settlement {
            &self.pool.settlement_permits
        } else {
            &self.pool.permits
        };
        let queue = if settlement {
            &self.pool.settlement_queue
        } else {
            &self.pool.queue
        };
        let permit = pool
            .clone()
            .acquire_owned()
            .await
            .map_err(|_| PublicError::internal())?;
        // Latency is measured from permit grant so it reflects the python
        // callback itself, not queueing behind other bridge calls.
        let call_started = std::time::Instant::now();
        let (responder, outcome) = oneshot::channel();
        let submitted = match queue.lock() {
            Ok(guard) => match guard.as_ref() {
                Some(sender) => sender
                    .send(Job {
                        method,
                        argument,
                        responder,
                        _permit: permit,
                        _execution_permit: execution_permit,
                        execution: self.execution.clone(),
                    })
                    .is_ok(),
                None => false,
            },
            Err(_) => false,
        };
        if !submitted {
            return Err(PublicError::internal());
        }
        let outcome = outcome.await;
        crate::metrics::METRICS
            .bridge_call_ms
            .record(call_started.elapsed());
        match outcome {
            Ok(result) => result,
            Err(_) => Err(PublicError::internal()),
        }
    }
}

impl Drop for WorkerPool {
    /// Close the queue and join every worker after its per-thread cleanup.
    ///
    /// Joining waits for each worker to reacquire the interpreter and run its
    /// `close_thread_resources` cleanup, so the dropping thread must not hold
    /// an interpreter attachment. `serve` guarantees this: the bridge lives
    /// and dies inside its detached serving closure.
    fn drop(&mut self) {
        if let Ok(mut guard) = self.queue.lock() {
            guard.take();
        }
        if let Ok(mut guard) = self.settlement_queue.lock() {
            guard.take();
        }
        if let Ok(mut workers) = self.workers.lock() {
            for handle in workers.drain(..) {
                let _ = handle.join();
            }
        }
    }
}

/// Run queued control-plane calls until the queue closes, then release this
/// thread's cached python resources before exiting.
///
/// The worker holds one interpreter attachment for its whole life and only
/// detaches (releasing the GIL) while waiting for the next job. A fresh
/// attachment per call would register a fresh python thread identity each
/// time, so `threading.local` caches (the control plane's per-thread SQLite
/// connections) would miss on every call and accumulate one connection per
/// call instead of one per worker.
fn worker_loop(receiver: &Mutex<mpsc::Receiver<Job>>, object: &Py<PyAny>) {
    Python::attach(|py| {
        loop {
            // The GIL is released while idle, and the receive lock is held
            // only while waiting for the next job and released before the
            // job runs, so idle workers hand off the queue without
            // serializing the python calls themselves.
            let received = py.detach(|| match receiver.lock() {
                Ok(guard) => guard.recv().map_err(|_| ()),
                Err(_) => Err(()),
            });
            let Ok(job) = received else { break };
            // A panic maps to the shared internal error and never poisons
            // the receive lock, so one poisoned call cannot take down the
            // pool.
            let target = job
                .execution
                .as_ref()
                .map_or(object, |execution| &execution.resources.target);
            let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                control_plane_call(py, target, job.method, &job.argument)
            }))
            .unwrap_or_else(|_| Err(PublicError::internal()));
            // Shared workers must not retain a departed target's per-path
            // caches. Run cleanup even after a failed callback or cancelled
            // waiter, while the job still owns its target and both permits.
            // The individual process bridge retains its fixed-worker cache.
            if job.execution.is_some() {
                let cleanup = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    control_plane_call(py, target, "close_thread_resources", "{}")
                }))
                .unwrap_or_else(|_| Err(PublicError::internal()));
                if cleanup.is_err() {
                    // Cleanup cannot change a committed accounting outcome.
                    // Never print the callback's exception or target identity.
                    eprintln!("bound gateway callback thread-resource cleanup failed");
                }
            }
            let _ = job.responder.send(outcome);
        }
        // The control plane caches one SQLite connection per worker thread;
        // closing them here bounds a host that starts and stops many
        // gateways in one process to the live pool's connections. The thread
        // is exiting and has no caller to answer, so a cleanup failure is
        // deliberately ignored.
        let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _ = object
                .bind(py)
                .call_method1("close_thread_resources", ("{}",));
        }));
    });
}

/// Run one control-plane call on the attached worker and map its outcome.
fn control_plane_call(
    py: Python<'_>,
    object: &Py<PyAny>,
    method: &'static str,
    argument: &str,
) -> Result<String, PublicError> {
    let bound = object.bind(py);
    match bound.call_method1(method, (argument,)) {
        Ok(result) => result
            .extract::<String>()
            .map_err(|_| PublicError::internal()),
        Err(error) => Err(public_error_from_pyerr(py, &error)),
    }
}

/// Map one Python exception to a public error.
///
/// The control plane attaches a `public_error_json` attribute to every
/// sanitized boundary failure; anything without it is an internal error,
/// mirroring the Python engine's catch-all in `_exception_response`.
fn public_error_from_pyerr(py: Python<'_>, error: &PyErr) -> PublicError {
    let value = error.value(py);
    if let Ok(payload) = value.getattr("public_error_json") {
        if let Ok(text) = payload.extract::<String>() {
            if let Ok(public) = serde_json::from_str::<PublicError>(&text) {
                return public;
            }
        }
    }
    PublicError::internal()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// An instrumented control plane recording the threads that served it.
    const PLANE_SOURCE: &std::ffi::CStr = cr#"
import threading


class Plane:
    """Record which threads run callbacks and which release resources."""

    def __init__(self):
        self.lock = threading.Lock()
        self.call_threads = set()
        self.closed_threads = []
        self.barrier = threading.Barrier(2, timeout=10.0)
        self.started = threading.Event()
        self.release = threading.Event()
        self.fail_close = False

    def block(self, argument):
        self.started.set()
        if not self.release.wait(timeout=10.0):
            raise RuntimeError("blocked callback was not released")
        return argument

    def echo(self, argument):
        with self.lock:
            self.call_threads.add(threading.get_ident())
        return argument

    def rendezvous(self, argument):
        self.barrier.wait()
        return argument

    def boom(self, argument):
        raise RuntimeError("unsanitized failure")

    def settle(self, argument):
        return argument

    def close_thread_resources(self, argument):
        with self.lock:
            self.closed_threads.append(threading.get_ident())
        if self.fail_close:
            raise RuntimeError("cleanup failure with private target information")
        return "{}"
"#;

    /// Instantiate the instrumented python control plane.
    fn plane() -> Py<PyAny> {
        Python::initialize();
        Python::attach(|py| {
            pyo3::types::PyModule::from_code(py, PLANE_SOURCE, c"plane.py", c"plane")
                .expect("plane module compiles")
                .getattr("Plane")
                .expect("plane class exists")
                .call0()
                .expect("plane instantiates")
                .unbind()
        })
    }

    /// Read one integer-list attribute length from the plane.
    fn attribute_length(object: &Py<PyAny>, name: &str) -> usize {
        Python::attach(|py| {
            object
                .bind(py)
                .getattr(name)
                .expect("attribute exists")
                .len()
                .expect("attribute is sized")
        })
    }

    /// Run one future on a fresh runtime.
    fn block_on<F: std::future::Future>(future: F) -> F::Output {
        tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .expect("runtime builds")
            .block_on(future)
    }

    #[test]
    fn calls_stay_on_a_fixed_set_of_worker_threads() {
        let object = plane();
        let observer = Python::attach(|py| object.clone_ref(py));
        let bridge = Bridge::new(object, 2).expect("bridge starts");
        block_on(async {
            for index in 0..32 {
                let result = bridge.call("echo", format!("payload-{index}")).await;
                assert_eq!(result.expect("echo succeeds"), format!("payload-{index}"));
            }
        });
        assert!(attribute_length(&observer, "call_threads") <= 2);
        drop(bridge);
    }

    #[test]
    fn workers_serve_calls_concurrently() {
        let object = plane();
        let bridge = Arc::new(Bridge::new(object, 4).expect("bridge starts"));
        // Both calls block on a two-party barrier inside python, so they can
        // only complete if two workers run them at the same time.
        let (first, second) = block_on(async {
            let left = bridge.clone();
            let right = bridge.clone();
            tokio::join!(
                left.call("rendezvous", "left".to_string()),
                right.call("rendezvous", "right".to_string()),
            )
        });
        assert_eq!(first.expect("first call succeeds"), "left");
        assert_eq!(second.expect("second call succeeds"), "right");
    }

    #[test]
    fn cancelled_waiter_keeps_capacity_until_its_callback_finishes() {
        let object = plane();
        let observer = Python::attach(|py| object.clone_ref(py));
        let bridge = Arc::new(Bridge::new(object, 1).expect("bridge starts"));
        block_on(async {
            let caller = bridge.clone();
            let task = tokio::spawn(async move { caller.call("block", "first".into()).await });
            tokio::time::timeout(std::time::Duration::from_secs(5), async {
                loop {
                    let started = Python::attach(|py| {
                        observer
                            .bind(py)
                            .getattr("started")
                            .unwrap()
                            .call_method0("is_set")
                            .unwrap()
                            .extract::<bool>()
                            .unwrap()
                    });
                    if started {
                        break;
                    }
                    tokio::task::yield_now().await;
                }
            })
            .await
            .expect("callback starts");
            task.abort();
            assert!(task.await.unwrap_err().is_cancelled());
            let free_while_running = bridge.pool.permits.available_permits();
            Python::attach(|py| {
                observer
                    .bind(py)
                    .getattr("release")
                    .unwrap()
                    .call_method0("set")
                    .unwrap();
            });
            tokio::time::timeout(std::time::Duration::from_secs(5), async {
                while bridge.pool.permits.available_permits() == 0 {
                    tokio::task::yield_now().await;
                }
            })
            .await
            .expect("completed callback releases capacity");
            assert_eq!(
                free_while_running, 0,
                "the job, not its waiter, owns capacity"
            );
            assert_eq!(bridge.call("echo", "next".into()).await.unwrap(), "next");
        });
    }

    #[test]
    fn bound_job_keeps_its_target_after_the_request_and_front_leave() {
        use crate::execution_binding::{ExecutionBindings, ExecutionHandle};

        let process = plane();
        let process_observer = Python::attach(|py| process.clone_ref(py));
        let target = plane();
        let observer = Python::attach(|py| target.clone_ref(py));
        let handle = Python::attach(|py| {
            ExecutionHandle::new(target, 128, 1, 1, &pyo3::types::PyDict::new(py))
        })
        .unwrap();
        let bindings = ExecutionBindings::new(1).unwrap();
        let token = "x".repeat(32);
        bindings.register(token.clone(), &handle).unwrap();
        let bridge = Bridge::new(process, 2).unwrap();
        let bound = bridge.bind(bindings.claim(&token).unwrap());
        block_on(async {
            let request = tokio::spawn(async move { bound.call("block", "accepted".into()).await });
            tokio::time::timeout(std::time::Duration::from_secs(5), async {
                while !Python::attach(|py| {
                    observer
                        .bind(py)
                        .getattr("started")
                        .unwrap()
                        .call_method0("is_set")
                        .unwrap()
                        .extract::<bool>()
                        .unwrap()
                }) {
                    tokio::task::yield_now().await;
                }
            })
            .await
            .unwrap();
            bindings.remove(&token).unwrap();
            request.abort();
            assert!(request.await.unwrap_err().is_cancelled());
            assert!(handle.in_use());
            assert_eq!(bridge.pool.permits.available_permits(), 1);
            // The original process target remains available while another target blocks.
            assert_eq!(
                bridge.call("echo", "process".into()).await.unwrap(),
                "process"
            );
            Python::attach(|py| {
                observer
                    .bind(py)
                    .getattr("release")
                    .unwrap()
                    .call_method0("set")
                    .unwrap();
            });
            tokio::time::timeout(std::time::Duration::from_secs(5), async {
                while handle.in_use() {
                    tokio::task::yield_now().await;
                }
            })
            .await
            .unwrap();
        });
        // The callback is cancelled, but its target is cleaned before its
        // lease becomes idle, without stopping the shared process bridge.
        assert_eq!(attribute_length(&observer, "closed_threads"), 1);
        drop(bridge);
        assert_eq!(attribute_length(&process_observer, "closed_threads"), 4);
    }

    #[test]
    fn bound_target_churn_closes_sqlite_on_serving_and_cleanup_workers() {
        use crate::execution_binding::{ExecutionBindings, ExecutionHandle};

        const SOURCE: &std::ffi::CStr = cr#"
import sqlite3
import tempfile
import threading
from pathlib import Path


class DatabasePlane:
    def __init__(self):
        self.directory = tempfile.TemporaryDirectory()
        self.path = Path(self.directory.name) / "gateway.db"
        self.local = threading.local()
        self.closed = 0

    def echo(self, argument):
        if not hasattr(self.local, "connection"):
            self.local.connection = sqlite3.connect(self.path)
        self.local.connection.execute("SELECT 1").fetchone()
        return argument

    def settle(self, argument):
        return self.echo(argument)

    def boom(self, argument):
        self.echo(argument)
        raise RuntimeError("callback failed after opening its database")

    def close_thread_resources(self, argument):
        connection = self.local.connection
        connection.close()
        del self.local.connection
        try:
            connection.execute("SELECT 1")
        except sqlite3.ProgrammingError:
            self.closed += 1
        else:
            raise AssertionError("database descriptor remains usable")
        return "{}"
"#;
        let bridge = Bridge::new(plane(), 2).unwrap();
        let bindings = ExecutionBindings::new(1).unwrap();
        for index in 0..16 {
            let target = Python::attach(|py| {
                pyo3::types::PyModule::from_code(py, SOURCE, c"database.py", c"database")
                    .unwrap()
                    .getattr("DatabasePlane")
                    .unwrap()
                    .call0()
                    .unwrap()
                    .unbind()
            });
            let observer = Python::attach(|py| target.clone_ref(py));
            let handle = Python::attach(|py| {
                ExecutionHandle::new(target, 128, 1, 1, &pyo3::types::PyDict::new(py))
            })
            .unwrap();
            let token = format!("{index:032}");
            bindings.register(token.clone(), &handle).unwrap();
            let bound = bridge.bind(bindings.claim(&token).unwrap());
            block_on(async {
                assert_eq!(
                    bound.call("echo", "request".into()).await.unwrap(),
                    "request"
                );
                assert_eq!(
                    bound.call("settle", "receipt".into()).await.unwrap(),
                    "receipt"
                );
                assert!(bound.call("boom", "failure".into()).await.is_err());
            });
            Python::attach(|py| {
                assert_eq!(
                    observer
                        .bind(py)
                        .getattr("closed")
                        .unwrap()
                        .extract::<usize>()
                        .unwrap(),
                    3,
                    "every target's actual SQLite connection closes before its callback returns"
                );
            });
            drop(bound);
        }
        // Churn never stops the pool or consumes the process target's capacity.
        assert_eq!(
            block_on(bridge.call("echo", "alive".into())).unwrap(),
            "alive"
        );
    }

    #[test]
    fn cleanup_failure_cannot_change_an_acknowledged_accounting_result() {
        use crate::execution_binding::{ExecutionBindings, ExecutionHandle};

        let target = plane();
        Python::attach(|py| target.bind(py).setattr("fail_close", true).unwrap());
        let observer = Python::attach(|py| target.clone_ref(py));
        let handle = Python::attach(|py| {
            ExecutionHandle::new(target, 128, 1, 1, &pyo3::types::PyDict::new(py))
        })
        .unwrap();
        let bindings = ExecutionBindings::new(1).unwrap();
        let token = "c".repeat(32);
        bindings.register(token.clone(), &handle).unwrap();
        let bridge = Bridge::new(plane(), 2).unwrap();
        let bound = bridge.bind(bindings.claim(&token).unwrap());
        assert_eq!(
            block_on(bound.call("settle", "durable receipt".into())).unwrap(),
            "durable receipt"
        );
        assert_eq!(attribute_length(&observer, "closed_threads"), 1);
        assert!(block_on(bound.call("boom", "{}".into())).is_err());
        assert_eq!(attribute_length(&observer, "closed_threads"), 2);
    }

    #[test]
    fn dropping_the_bridge_releases_every_worker_thread() {
        let object = plane();
        let observer = Python::attach(|py| object.clone_ref(py));
        let bridge = Bridge::new(object, 3).expect("bridge starts");
        block_on(async {
            bridge
                .call("echo", "warm".to_string())
                .await
                .expect("echo succeeds");
        });
        drop(bridge);
        // Drop joins the workers, so every one of them has already run its
        // `close_thread_resources` cleanup, including idle workers.
        assert_eq!(attribute_length(&observer, "closed_threads"), 5);
    }

    #[test]
    fn an_unsanitized_python_failure_maps_to_the_internal_error() {
        let object = plane();
        let bridge = Bridge::new(object, 1).expect("bridge starts");
        let outcome = block_on(bridge.call("boom", "{}".to_string()));
        let error = outcome.expect_err("boom fails");
        assert_eq!(
            serde_json::to_value(&error).expect("error serializes"),
            serde_json::to_value(PublicError::internal()).expect("error serializes"),
        );
    }
}
