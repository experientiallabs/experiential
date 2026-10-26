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
        "/health/live" | "/health/ready" | "/metrics" | "/metrics.json"
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

    #[test]
    fn missing_and_consumed_bindings_preserve_each_wire_error_envelope() {
        tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap()
            .block_on(async {
                Python::initialize();
                let handle = ExecutionHandle::new(Python::attach(|py| py.None()), 2, 2, 1).unwrap();
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
