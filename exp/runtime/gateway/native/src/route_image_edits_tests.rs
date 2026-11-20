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

fn png(marker: u8) -> Vec<u8> {
    let mut bytes = b"\x89PNG\r\n\x1a\n".to_vec();
    bytes.push(marker);
    bytes
}

fn part(name: &str, filename: &str, bytes: &[u8]) -> Vec<u8> {
    let mut body = format!(
        "--B\r\nContent-Disposition: form-data; name=\"{name}\"; filename=\"{filename}\"\r\nContent-Type: application/octet-stream\r\n\r\n"
    )
    .into_bytes();
    body.extend_from_slice(bytes);
    body.extend_from_slice(b"\r\n");
    body
}

fn text(name: &str, value: &str) -> Vec<u8> {
    format!("--B\r\nContent-Disposition: form-data; name=\"{name}\"\r\n\r\n{value}\r\n")
        .into_bytes()
}

fn upload(parts: &[Vec<u8>]) -> axum::extract::Request {
    let mut body = parts.concat();
    body.extend_from_slice(b"--B--\r\n");
    request(Body::from(body))
}

#[test]
fn repeated_image_parts_keep_their_order_and_sniffed_types() {
    let state = state();
    tokio::runtime::Runtime::new().unwrap().block_on(async {
        let jpeg = [0xff, 0xd8, 0xff, 0x00];
        let request = upload(&[
            text("model", "gpt-image-2.5-sunburst"),
            part("image[]", "a.bin", &png(1)),
            part("image[]", "b.bin", &jpeg),
            part("mask", "m.png", &png(3)),
        ]);
        let (fields, files, parts) = parse_edit_upload(&state, request).await.expect("parsed");
        assert_eq!(fields["model"], "gpt-image-2.5-sunburst");
        let images = files["image[]"].as_array().expect("a list in upload order");
        assert_eq!(images.len(), 2);
        assert!(files["mask"].is_string());
        let names: Vec<&str> = parts.iter().map(|part| part.name.as_str()).collect();
        assert_eq!(names, ["image[]", "image[]", "mask"]);
        assert_eq!(parts[0].content_type, "image/png");
        assert_eq!(parts[1].content_type, "image/jpeg");
    });
}

#[test]
fn too_many_source_images_and_a_repeated_mask_are_refused_by_name() {
    let state = state();
    tokio::runtime::Runtime::new().unwrap().block_on(async {
        let many: Vec<Vec<u8>> = (0..=MAXIMUM_EDIT_IMAGES)
            .map(|index| part("image[]", "x.png", &png(index as u8)))
            .collect();
        let error = parse_edit_upload(&state, upload(&many)).await.unwrap_err();
        assert_eq!(error.param.as_deref(), Some("image"));
        let masks = [
            part("image", "x.png", &png(0)),
            part("mask", "m.png", &png(1)),
            part("mask", "m.png", &png(2)),
        ];
        let error = parse_edit_upload(&state, upload(&masks)).await.unwrap_err();
        assert_eq!(error.param.as_deref(), Some("mask"));
    });
}

#[test]
fn the_openai_edit_body_carries_fields_then_images_then_the_mask() {
    let part = |name: &str, marker: u8| EditPart {
        name: name.to_string(),
        bytes: png(marker),
        content_type: "image/png".to_string(),
    };
    let fields = json!({"model": "gpt-image-2.5-sunburst", "n": 2, "prompt": "watercolor"});
    let several = encode_edit_multipart(
        "B",
        &fields,
        &[
            part("image[]", 1),
            part("mask", 9),
            part("image[]", 2),
            part("style_image", 7),
        ],
    );
    let text = String::from_utf8_lossy(&several);
    assert!(text.contains("name=\"n\"\r\n\r\n2\r\n"));
    assert!(text.contains("name=\"prompt\"\r\n\r\nwatercolor\r\n"));
    assert_eq!(
        text.matches("name=\"image[]\"; filename=\"image-").count(),
        2
    );
    assert!(text.find("image-0.png").unwrap() < text.find("image-1.png").unwrap());
    assert!(text.find("image-1.png").unwrap() < text.find("name=\"mask\"").unwrap());
    assert!(
        !text.contains("style_image"),
        "a style image never rides this wire"
    );
    assert!(text.ends_with("--B--\r\n"));
    let one = encode_edit_multipart("B", &fields, &[part("image[]", 1)]);
    let text = String::from_utf8_lossy(&one);
    assert!(text.contains("name=\"image\"; filename=\"image-0.png\"\r\nContent-Type: image/png"));
    assert!(!text.contains("image[]"));
}
