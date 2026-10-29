//! Claim a host binding once and pass its bound bridge into ordinary route state.

use std::sync::Arc;

use axum::extract::{FromRef, Request, State};
use axum::middleware::Next;
use axum::response::Response;

use crate::execution_binding::{ExecutionBindings, ExecutionLease};
use crate::respond::{bearer_key, error_response};
use crate::server::AppState;

tokio::task_local! {
    static EXECUTION: Arc<ExecutionLease>;
}

/// Router state stays process-wide; each handler receives its own bound view.
#[derive(Clone)]
pub(crate) struct ServerState {
    pub(crate) base: AppState,
    pub(crate) bindings_enabled: bool,
}

impl FromRef<ServerState> for AppState {
    fn from_ref(server: &ServerState) -> Self {
        let mut state = server.base.clone();
        if server.bindings_enabled {
            if let Ok(execution) = EXECUTION.try_with(Arc::clone) {
                state.guardrail_detectors = execution.resources.guardrail_detectors.clone();
                state.capture = execution.resources.capture.clone();
                state.replays = Arc::new(state.replays.bound(execution.resources.replay_namespace));
                state.bridge = Arc::new(state.bridge.bind(execution));
            }
        }
        state
    }
}

/// Claimed work keeps its target even if its registration or HTTP front disappears.
pub(crate) async fn bind_execution(
    State(bindings): State<ExecutionBindings>,
    request: Request,
    next: Next,
) -> Response {
    let path = request.uri().path();
    // Diagnostics use the process owner and never claim a caller's resources.
    if matches!(
        path,
        "/health/live" | "/health/ready" | "/metrics" | "/metrics.json" | "/usage" | "/usage.json"
    ) {
        return next.run(request).await;
    }
    let messages = path == "/v1/messages" || path == "/v1/messages/count_tokens";
    let token = if messages {
        crate::route_messages::messages_api_key(request.headers())
    } else {
        bearer_key(request.headers())
    };
    let execution = match token.and_then(|token| bindings.claim(&token)) {
        Ok(execution) => execution,
        Err(error) if messages => return crate::route_messages::messages_error_response(&error),
        Err(error) => return error_response(&error),
    };
    // Task-local storage is needed only while the handler extracts its state.
    // The bound Arc<Bridge> then travels with streams, WebSockets, guards and jobs.
    EXECUTION.scope(execution, next.run(request)).await
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::execution_binding::ExecutionHandle;
    use axum::routing::post;
    use axum::Router;
    use pyo3::prelude::*;
    use pyo3::types::PyDict;
    use std::sync::atomic::AtomicUsize;
    use std::time::{Duration, Instant};
    use tokio::sync::Semaphore;

    fn handle(pattern: &str, replacement: &str) -> ExecutionHandle {
        Python::initialize();
        Python::attach(|py| {
            let mapping = PyDict::new(py);
            let detector = crate::RegexDetector::new(
                &serde_json::json!({"patterns": [pattern], "replacement": replacement}).to_string(),
            )
            .unwrap();
            mapping
                .set_item("same-adapter", Py::new(py, detector).unwrap())
                .unwrap();
            let handle = ExecutionHandle::new(py.None(), 2, 2, 1, &mapping, None).unwrap();
            // The handle owns a snapshot, not the caller's mutable Python dictionary.
            mapping.clear();
            handle
        })
    }

    fn server_state(process: &ExecutionHandle) -> ServerState {
        let bridge = crate::bridge::Bridge::new(Python::attach(|py| py.None()), 1).unwrap();
        ServerState {
            base: AppState {
                bridge: Arc::new(bridge),
                http: crate::upstream::build_client(Duration::from_secs(1), false).unwrap(),
                permits: Arc::new(Semaphore::new(4)),
                dispatch_capacity: Arc::new(crate::dispatch_capacity::DispatchCapacity::new()),
                permit_queue: Arc::new(crate::permit_queue::PermitQueue::new(None)),
                request_timeout: Duration::from_secs(30),
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
                guardrail_detectors: process.inner.guardrail_detectors.clone(),
            },
            bindings_enabled: true,
        }
    }

    fn enforce(
        state: &AppState,
        action: &str,
        text: &str,
    ) -> Result<String, crate::errors::Failure> {
        use crate::guardrails::plan::{OutputPlan, PlanCheck};

        let plan = OutputPlan {
            protected: true,
            max_response_bytes: 1024,
            policy_id: "rule".into(),
            organization_id: String::new(),
            identity_id: String::new(),
            checks: vec![PlanCheck {
                action: action.into(),
                adapter_id: "same-adapter".into(),
                check_id: "output".into(),
                capability: "regex".into(),
                timeout_ms: 5000,
            }],
        };
        let events = crate::guardrails::plan::enforce(
            &plan,
            &state.guardrail_detectors,
            vec![crate::events::Event::TextDelta(text.into())],
            Instant::now() + Duration::from_secs(30),
        )?;
        Ok(crate::guardrails::projection(&events).text)
    }

    #[test]
    fn bound_targets_keep_their_own_detectors_after_extraction() {
        tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap()
            .block_on(async {
                let process = handle("process", "[process]");
                let first = handle("alpha", "[first]");
                let second = handle("beta", "[second]");
                let empty = Python::attach(|py| {
                    ExecutionHandle::new(py.None(), 2, 2, 1, &PyDict::new(py), None).unwrap()
                });
                let server = server_state(&process);
                let bindings = ExecutionBindings::new(3).unwrap();
                let mut states = Vec::new();
                for (index, target) in [&first, &second, &empty].into_iter().enumerate() {
                    let token = format!("{index:032}");
                    bindings.register(token.clone(), target).unwrap();
                    let lease = bindings.claim(&token).unwrap();
                    let state = EXECUTION
                        .scope(lease, async { AppState::from_ref(&server) })
                        .await;
                    states.push(state);
                }
                // The task-local binding has ended; the extracted stream state still owns it.
                assert!(first.in_use() && second.in_use() && empty.in_use());
                assert_eq!(
                    enforce(&states[0], "modify", "alpha beta").unwrap(),
                    "[first] beta"
                );
                assert_eq!(
                    enforce(&states[1], "modify", "alpha beta").unwrap(),
                    "alpha [second]"
                );
                assert!(enforce(&states[0], "block", "alpha").is_err());
                assert_eq!(enforce(&states[1], "block", "alpha").unwrap(), "alpha");
                assert!(matches!(
                    enforce(&states[2], "block", "process")
                        .unwrap_err()
                        .failure_class,
                    crate::errors::FailureClass::Unavailable
                ));
                // Independent control planes can reuse every caller/store-local replay ID.
                let key: crate::replay::ReplayKey = serde_json::from_value(serde_json::json!({
                    "organization_id": "local", "identity_id": "key", "alias_revision_id": "rev",
                    "surface": "chat_completions", "caller_operation_sha256": "same-op",
                    "canonical_request_sha256": "same-body"
                }))
                .unwrap();
                let crate::replay::Claim::Owner(mut first_owner) =
                    states[0].replays.claim(key.clone()).await.unwrap()
                else {
                    panic!("first target must own its request");
                };
                let crate::replay::Claim::Owner(mut second_owner) =
                    states[1].replays.claim(key.clone()).await.unwrap()
                else {
                    panic!("another target must not join the first target");
                };
                let crate::replay::Claim::Join(first_joiner) =
                    states[0].replays.claim(key.clone()).await.unwrap()
                else {
                    panic!("same-target duplicates must still join");
                };
                for (owner, body) in [(&mut first_owner, "first"), (&mut second_owner, "second")] {
                    owner
                        .complete(crate::replay::CachedResponse {
                            status_code: 200,
                            media_type: "text/plain".into(),
                            headers: vec![],
                            body: body.as_bytes().to_vec(),
                        })
                        .await
                        .unwrap();
                }
                assert_eq!(first_joiner.result().await.unwrap().body, b"first");
                for (state, expected) in states
                    .iter()
                    .zip([b"first".as_slice(), b"second".as_slice()])
                {
                    let crate::replay::Claim::Replay(cached) =
                        state.replays.claim(key.clone()).await.unwrap()
                    else {
                        panic!("each target must replay only its own response");
                    };
                    assert_eq!(cached.body, expected);
                }
                // Individual execution keeps its own map, even inside a host binding's scope.
                let individual = ServerState {
                    bindings_enabled: false,
                    ..server.clone()
                };
                let token = "individual-check".repeat(3);
                bindings.register(token.clone(), &first).unwrap();
                let individual_state = EXECUTION
                    .scope(bindings.claim(&token).unwrap(), async {
                        AppState::from_ref(&individual)
                    })
                    .await;
                assert_eq!(
                    enforce(&individual_state, "modify", "process alpha").unwrap(),
                    "[process] alpha"
                );
                drop(states);
                assert!(!first.in_use() && !second.in_use() && !empty.in_use());
            });
    }

    #[test]
    fn missing_and_consumed_bindings_preserve_each_wire_error_envelope() {
        tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap()
            .block_on(async {
                Python::initialize();
                let handle = Python::attach(|py| {
                    ExecutionHandle::new(py.None(), 2, 2, 1, &pyo3::types::PyDict::new(py), None)
                })
                .unwrap();
                let bindings = ExecutionBindings::new(4).unwrap();
                let app = Router::new()
                    .route("/v1/messages", post(|| async { "ok" }))
                    .route("/v1/messages/count_tokens", post(|| async { "ok" }))
                    .route("/v1/chat/completions", post(|| async { "ok" }))
                    .layer(axum::middleware::from_fn_with_state(
                        bindings.clone(),
                        bind_execution,
                    ));
                let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
                let address = listener.local_addr().unwrap();
                let server = tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
                let client = reqwest::Client::new();
                for path in [
                    "/v1/messages",
                    "/v1/messages/count_tokens",
                    "/v1/chat/completions",
                ] {
                    let token = "x".repeat(32);
                    bindings.register(token.clone(), &handle).unwrap();
                    for (credential, status) in [
                        ("missing", 401),
                        (token.as_str(), 200),
                        (token.as_str(), 401),
                    ] {
                        let response = client
                            .post(format!("http://{address}{path}"))
                            .bearer_auth(credential)
                            .send()
                            .await
                            .unwrap();
                        assert_eq!(response.status().as_u16(), status);
                        if status == 401 {
                            let body: serde_json::Value = response.json().await.unwrap();
                            assert_eq!(
                                body.get("type").and_then(|value| value.as_str()),
                                path.contains("messages").then_some("error")
                            );
                            assert!(body.get("error").is_some());
                        }
                    }
                    assert!(!handle.in_use());
                }
                server.abort();
            });
    }
}
