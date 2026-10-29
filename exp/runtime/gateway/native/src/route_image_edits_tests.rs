//! Slow uploads consume only their supplied execution capacity and expire.

use super::*;
use std::sync::atomic::{AtomicBool, AtomicUsize};
use std::sync::Arc;
use std::time::Duration;

use axum::body::Body;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyModule};

use crate::execution_binding::{ExecutionBindings, ExecutionHandle};

fn plane() -> Py<PyAny> {
    Python::initialize();
    Python::attach(|py| {
        PyModule::from_code(
            py,
            cr#"
import json
class Control:
    def __init__(self):
        self.started = 0
    def authenticate(self, argument):
        return "{}"
    def admit_image_edits(self, argument):
        return json.dumps(dict(request_id="edit", alias="image", alias_revision_id="revision",
            exact_model_id="model", route_reason="direct", route=[], image_count=1,
            maximum_total_attempts=1, maximum_same_deployment_attempts=1))
    def start_attempt(self, argument):
        self.started += 1
        return '{"exhausted":true,"known_unbilled":true}'
    def close_thread_resources(self, argument):
        return "{}"
"#,
            c"image_upload_owner.py",
            c"image_upload_owner",
        )
        .unwrap()
        .getattr("Control")
        .unwrap()
        .call0()
        .unwrap()
        .unbind()
    })
}

fn state() -> AppState {
    AppState {
        bridge: Arc::new(crate::bridge::Bridge::new(plane(), 2).unwrap()),
        http: crate::upstream::build_client(Duration::from_secs(1), false).unwrap(),
        permits: Arc::new(tokio::sync::Semaphore::new(2)),
        dispatch_capacity: Arc::new(crate::dispatch_capacity::DispatchCapacity::new()),
        permit_queue: Arc::new(crate::permit_queue::PermitQueue::new(None)),
        request_timeout: Duration::from_secs(2),
        time_to_first_byte: Duration::from_secs(1),
        time_to_first_byte_slope_seconds_per_million_input_tokens: 0.0,
        time_to_first_token: Duration::from_secs(1),
        pre_first_byte_budget: None,
        pre_first_byte_reserve: Duration::ZERO,
        pre_first_byte_max: None,
        pending_settlements: Arc::new(AtomicUsize::new(0)),
        handled_requests: Arc::new(AtomicUsize::new(0)),
        replays: Arc::new(crate::replay::ReplayStore::new()),
        capture: None,
        guardrail_detectors: Arc::new(Default::default()),
    }
}

fn bound(base: &AppState, target: &Py<PyAny>) -> AppState {
    let handle = Python::attach(|py| {
        ExecutionHandle::new(target.clone_ref(py), 1, 2, 1, &PyDict::new(py), None)
    })
    .unwrap();
    let bindings = ExecutionBindings::new(1).unwrap();
    let token = "image-upload-bound-test-token".repeat(2);
    bindings.register(token.clone(), &handle).unwrap();
    let mut state = base.clone();
    state.bridge = Arc::new(base.bridge.bind(bindings.claim(&token).unwrap()));
    state
}

fn request(body: Body) -> axum::extract::Request {
    axum::extract::Request::builder()
        .method("POST")
        .header("authorization", "Bearer image-upload-fixture")
        .header("content-type", "multipart/form-data; boundary=B")
        .body(body)
        .unwrap()
}

fn slow_body(polled: Arc<AtomicBool>) -> Body {
    Body::from_stream(futures_util::stream::once(async move {
        polled.store(true, Ordering::SeqCst);
        std::future::pending::<Result<bytes::Bytes, std::io::Error>>().await
    }))
}

#[test]
fn slow_image_uploads_are_scoped_deadlined_and_do_not_reacquire_capacity() {
    tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap()
        .block_on(async {
            let base = state();
            let busy_target = plane();
            let quiet_target = plane();
            let busy = bound(&base, &busy_target);
            let quiet = bound(&base, &quiet_target);
            let first_polled = Arc::new(AtomicBool::new(false));
            let first = tokio::spawn(image_edits(
                State(busy.clone()),
                request(slow_body(first_polled.clone())),
            ));
            tokio::time::timeout(Duration::from_secs(1), async {
                while !first_polled.load(Ordering::SeqCst) {
                    tokio::task::yield_now().await;
                }
            })
            .await
            .expect("first upload reached its bounded read");
            assert_eq!(base.permits.available_permits(), 1);

            let second_polled = Arc::new(AtomicBool::new(false));
            let second = tokio::spawn(image_edits(
                State(busy),
                request(slow_body(second_polled.clone())),
            ));
            let complete = Body::from(
                "--B\r\nContent-Disposition: form-data; name=\"image\"; filename=\"x.png\"\r\n\r\nabc\r\n--B--\r\n",
            );
            // One remaining process slot and one quiet slot must suffice for
            // upload AND ladder. A second acquisition would deadlock here.
            let answer = tokio::time::timeout(
                Duration::from_secs(1),
                image_edits(State(quiet), request(complete)),
            )
            .await
            .expect("quiet execution reaches the exhausted fixture ladder");
            assert!(answer.status().is_server_error());
            Python::attach(|py| {
                assert_eq!(quiet_target.getattr(py, "started").unwrap().extract::<u32>(py).unwrap(), 1);
                assert_eq!(busy_target.getattr(py, "started").unwrap().extract::<u32>(py).unwrap(), 0);
            });
            assert!(!second_polled.load(Ordering::SeqCst));
            assert_eq!(base.permits.available_permits(), 1);
            second.abort();
            assert!(second.await.unwrap_err().is_cancelled());
            let expired = tokio::time::timeout(Duration::from_secs(3), first)
                .await
                .unwrap()
                .unwrap();
            assert_eq!(expired.status(), StatusCode::REQUEST_TIMEOUT);
            assert_eq!(base.permits.available_permits(), 2);
            assert_eq!(base.pending_settlements.load(Ordering::SeqCst), 0);
        });
}
