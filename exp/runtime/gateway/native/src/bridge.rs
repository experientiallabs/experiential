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
//! control plane's `close_thread_resources` callback before it exits.

use std::sync::{mpsc, Arc, Mutex};
use std::thread::JoinHandle;

use pyo3::prelude::*;
use tokio::sync::{oneshot, Semaphore};

use crate::errors::PublicError;

/// One queued control-plane call and the responder that hands its outcome
/// back to the awaiting request task.
struct Job {
    method: &'static str,
    argument: String,
    responder: oneshot::Sender<Result<String, PublicError>>,
}

/// Bounded bridge to one Python `NativeControlPlane` instance.
///
/// Dropping the bridge closes the job queue, waits for every worker to run
/// its `close_thread_resources` cleanup, and joins the threads, so a stopped
/// server leaves no cached per-thread connection behind.
pub struct Bridge {
    queue: Mutex<Option<mpsc::Sender<Job>>>,
    workers: Mutex<Vec<JoinHandle<()>>>,
    permits: Arc<Semaphore>,
}

impl Bridge {
    /// Start `maximum_concurrent_calls` named worker threads over one queue.
    ///
    /// The permit count equals the worker count, so an accepted call always
    /// has an idle worker and never queues behind another call after its
    /// permit is granted.
    pub fn new(object: Py<PyAny>, maximum_concurrent_calls: usize) -> Result<Self, String> {
        let worker_count = maximum_concurrent_calls.max(1);
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
        Ok(Self {
            queue: Mutex::new(Some(sender)),
            workers: Mutex::new(workers),
            permits: Arc::new(Semaphore::new(worker_count)),
        })
    }

    /// Call one control-plane method with a JSON-string argument.
    pub async fn call(
        &self,
        method: &'static str,
        argument: String,
    ) -> Result<String, PublicError> {
        let _permit = self
            .permits
            .acquire()
            .await
            .map_err(|_| PublicError::internal())?;
        // Latency is measured from permit grant so it reflects the python
        // callback itself, not queueing behind other bridge calls.
        let call_started = std::time::Instant::now();
        let (responder, outcome) = oneshot::channel();
        let submitted = match self.queue.lock() {
            Ok(guard) => match guard.as_ref() {
                Some(sender) => sender
                    .send(Job {
                        method,
                        argument,
                        responder,
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

impl Drop for Bridge {
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
            let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                control_plane_call(py, object, job.method, job.argument)
            }))
            .unwrap_or_else(|_| Err(PublicError::internal()));
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
    argument: String,
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

    def echo(self, argument):
        with self.lock:
            self.call_threads.add(threading.get_ident())
        return argument

    def rendezvous(self, argument):
        self.barrier.wait()
        return argument

    def boom(self, argument):
        raise RuntimeError("unsanitized failure")

    def close_thread_resources(self, argument):
        with self.lock:
            self.closed_threads.append(threading.get_ident())
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
        assert_eq!(attribute_length(&observer, "closed_threads"), 3);
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
