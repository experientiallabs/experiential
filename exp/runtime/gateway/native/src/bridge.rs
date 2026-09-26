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

use bytes::Bytes;
use pyo3::prelude::*;
use serde_json::json;
use tokio::sync::{oneshot, OwnedSemaphorePermit, Semaphore};

use crate::errors::PublicError;

/// One queued control-plane call and the responder that hands its outcome
/// back to the awaiting request task.
struct Job {
    operation: JobOperation,
    started_at: std::time::Instant,
    responder: oneshot::Sender<Result<String, PublicError>>,
    // Cancellation of the awaiting request cannot release queued/running capacity.
    _permit: OwnedSemaphorePermit,
}

/// One callback or one authentication-first Chat admission on a worker.
enum JobOperation {
    Callback {
        method: &'static str,
        argument: String,
    },
    AuthenticatedChatAdmission(ChatAdmission),
}

/// Chat body and trusted metadata held until the key has been authenticated.
pub(crate) struct ChatAdmission {
    pub(crate) raw_key: String,
    pub(crate) body: Bytes,
    pub(crate) client_request_id: Option<String>,
    pub(crate) client_ip: Option<String>,
    pub(crate) capture_session_id: Option<String>,
}

/// Record callback-pool wait time even when the waiting request is cancelled.
struct BridgePermitWaitTimer<'a> {
    started: std::time::Instant,
    histogram: &'a crate::metrics::Histogram,
}

impl<'a> BridgePermitWaitTimer<'a> {
    fn new(histogram: &'a crate::metrics::Histogram) -> Self {
        Self {
            started: std::time::Instant::now(),
            histogram,
        }
    }
}

impl Drop for BridgePermitWaitTimer<'_> {
    fn drop(&mut self) {
        self.histogram.record(self.started.elapsed());
    }
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
        self.dispatch(JobOperation::Callback { method, argument }, Some(method))
            .await
    }

    /// Authenticate one Chat key before converting or encoding its request body,
    /// then admit it on the same worker job and permit.
    pub async fn authenticate_then_admit_chat(
        &self,
        request: ChatAdmission,
    ) -> Result<String, PublicError> {
        self.dispatch(JobOperation::AuthenticatedChatAdmission(request), None)
            .await
    }

    async fn dispatch(
        &self,
        operation: JobOperation,
        measured_method: Option<&'static str>,
    ) -> Result<String, PublicError> {
        let permit_wait_timer =
            BridgePermitWaitTimer::new(&crate::metrics::METRICS.bridge_permit_wait_ms);
        let permit = self
            .permits
            .clone()
            .acquire_owned()
            .await
            .map_err(|_| PublicError::internal())?;
        drop(permit_wait_timer);
        // Latency is measured from permit grant so it reflects the python
        // callback itself, not queueing behind other bridge calls.
        let call_started = std::time::Instant::now();
        let (responder, outcome) = oneshot::channel();
        let submitted = match self.queue.lock() {
            Ok(guard) => match guard.as_ref() {
                Some(sender) => sender
                    .send(Job {
                        operation,
                        started_at: call_started,
                        responder,
                        _permit: permit,
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
        if let Some(method) = measured_method {
            crate::metrics::METRICS.record_bridge_call(method, call_started.elapsed());
        }
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
            let Job {
                operation,
                started_at,
                responder,
                _permit,
            } = job;
            // A panic maps to the shared internal error and never poisons
            // the receive lock, so one poisoned call cannot take down the
            // pool.
            let outcome =
                std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| match operation {
                    JobOperation::Callback { method, argument } => {
                        control_plane_call(py, object, method, argument)
                    }
                    JobOperation::AuthenticatedChatAdmission(request) => {
                        authenticate_then_admit_chat_call(
                            py, object, request, &responder, started_at,
                        )
                    }
                }))
                .unwrap_or_else(|_| Err(PublicError::internal()));
            let _ = responder.send(outcome);
            drop(_permit);
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

/// Authenticate a Chat key before body conversion, then call Python admission.
fn authenticate_then_admit_chat_call(
    py: Python<'_>,
    object: &Py<PyAny>,
    request: ChatAdmission,
    responder: &oneshot::Sender<Result<String, PublicError>>,
    started_at: std::time::Instant,
) -> Result<String, PublicError> {
    let authentication_argument = serde_json::to_string(&json!({"raw_key": request.raw_key}))
        .map_err(|_| PublicError::internal())?;
    let authentication = control_plane_call(
        py,
        object,
        "authenticate_for_chat_admission",
        authentication_argument,
    );
    crate::metrics::METRICS.record_bridge_call("authenticate", started_at.elapsed());
    authentication?;
    if responder.is_closed() {
        return Err(PublicError::internal());
    }

    // Body conversion and JSON encoding happen only after authentication, while
    // the GIL is released so another worker can run its bounded callback.
    let admission_started = std::time::Instant::now();
    let admission_argument = py.detach(move || {
        let body =
            String::from_utf8(request.body.to_vec()).map_err(|_| PublicError::invalid_json())?;
        serde_json::to_string(&json!({
            "raw_key": request.raw_key,
            "body": body,
            "idempotency_key": Option::<String>::None,
            "client_request_id": request.client_request_id,
            "client_ip": request.client_ip,
            "capture_session_id": request.capture_session_id,
        }))
        .map_err(|_| PublicError::internal())
    });
    match admission_argument {
        Ok(_) if responder.is_closed() => Err(PublicError::internal()),
        Ok(argument) => {
            let admission = control_plane_call(py, object, "admit", argument);
            crate::metrics::METRICS.record_bridge_call("admit", admission_started.elapsed());
            admission
        }
        Err(error) => Err(error),
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
import json
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
        self.authentication_calls = []
        self.admission_calls = []

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

    def authenticate(self, argument):
        data = json.loads(argument)
        with self.lock:
            self.authentication_calls.append(data["raw_key"])
        if data["raw_key"] == "block":
            self.started.set()
            if not self.release.wait(timeout=10.0):
                raise RuntimeError("authentication callback was not released")
        if data["raw_key"] == "invalid":
            error = RuntimeError("virtual key rejected")
            error.public_error_json = json.dumps({
                "status_code": 401,
                "code": "invalid_key",
                "message": "A valid gateway Bearer key is required.",
                "error_type": "authentication_error",
            })
            raise error
        return "{}"

    def authenticate_for_chat_admission(self, argument):
        return self.authenticate(argument)

    def admit(self, argument):
        with self.lock:
            self.admission_calls.append(json.loads(argument))
        return "{}"

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
            let free_while_running = bridge.permits.available_permits();
            Python::attach(|py| {
                observer
                    .bind(py)
                    .getattr("release")
                    .unwrap()
                    .call_method0("set")
                    .unwrap();
            });
            tokio::time::timeout(std::time::Duration::from_secs(5), async {
                while bridge.permits.available_permits() == 0 {
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
    fn cancelled_callback_permit_wait_is_recorded() {
        let metrics = crate::metrics::Metrics::new();
        let permits = Arc::new(Semaphore::new(0));
        block_on(async {
            let histogram = &metrics.bridge_permit_wait_ms;
            let waiting_permits = permits.clone();
            tokio::select! {
                _result = async move {
                    let _timer = BridgePermitWaitTimer::new(histogram);
                    waiting_permits.acquire_owned().await
                } => panic!("a closed-free empty semaphore must keep waiting"),
                () = tokio::time::sleep(std::time::Duration::from_millis(1)) => {}
            }
        });
        assert_eq!(metrics.snapshot()["bridge_permit_wait_ms"]["count"], 1);
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

    #[test]
    fn chat_admission_authenticates_before_body_conversion() {
        let object = plane();
        let observer = Python::attach(|py| object.clone_ref(py));
        let bridge = Bridge::new(object, 1).expect("bridge starts");

        let invalid_key = block_on(bridge.authenticate_then_admit_chat(ChatAdmission {
            raw_key: "invalid".to_string(),
            body: Bytes::from_static(&[0xff]),
            client_request_id: None,
            client_ip: None,
            capture_session_id: None,
        }))
        .expect_err("invalid key is rejected before invalid UTF-8 is decoded");
        assert_eq!(invalid_key.status_code, 401);
        assert_eq!(invalid_key.code, "invalid_key");
        assert_eq!(attribute_length(&observer, "authentication_calls"), 1);
        assert_eq!(attribute_length(&observer, "admission_calls"), 0);

        let invalid_body = block_on(bridge.authenticate_then_admit_chat(ChatAdmission {
            raw_key: "valid".to_string(),
            body: Bytes::from_static(&[0xff]),
            client_request_id: None,
            client_ip: None,
            capture_session_id: None,
        }))
        .expect_err("valid key still receives the invalid-body response");
        assert_eq!(invalid_body.status_code, 400);
        assert_eq!(invalid_body.code, "invalid_json");
        assert_eq!(attribute_length(&observer, "authentication_calls"), 2);
        assert_eq!(attribute_length(&observer, "admission_calls"), 0);

        let admitted = block_on(bridge.authenticate_then_admit_chat(ChatAdmission {
            raw_key: "valid".to_string(),
            body: Bytes::from_static(br#"{"model":"coding"}"#),
            client_request_id: Some("client-1".to_string()),
            client_ip: Some("127.0.0.1".to_string()),
            capture_session_id: Some("capture-1".to_string()),
        }))
        .expect("valid Chat admission succeeds");
        assert_eq!(admitted, "{}");
        assert_eq!(attribute_length(&observer, "authentication_calls"), 3);
        assert_eq!(attribute_length(&observer, "admission_calls"), 1);
        drop(bridge);
    }

    #[test]
    fn cancelled_chat_admission_stops_after_authentication() {
        let object = plane();
        let observer = Python::attach(|py| object.clone_ref(py));
        let bridge = Arc::new(Bridge::new(object, 1).expect("bridge starts"));

        block_on(async {
            let caller = bridge.clone();
            let task = tokio::spawn(async move {
                caller
                    .authenticate_then_admit_chat(ChatAdmission {
                        raw_key: "block".to_string(),
                        body: Bytes::from_static(&[0xff]),
                        client_request_id: None,
                        client_ip: None,
                        capture_session_id: None,
                    })
                    .await
            });
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
            .expect("authentication callback starts");
            task.abort();
            assert!(task.await.unwrap_err().is_cancelled());
            Python::attach(|py| {
                observer
                    .bind(py)
                    .getattr("release")
                    .unwrap()
                    .call_method0("set")
                    .unwrap();
            });
            tokio::time::timeout(std::time::Duration::from_secs(5), async {
                while bridge.permits.available_permits() == 0 {
                    tokio::task::yield_now().await;
                }
            })
            .await
            .expect("cancelled callback releases its permit");
        });

        assert_eq!(attribute_length(&observer, "authentication_calls"), 1);
        assert_eq!(attribute_length(&observer, "admission_calls"), 0);
    }
}
