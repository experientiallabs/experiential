//! Process-owned usage diagnostics must not claim a host's execution binding.

use super::*;
use crate::execution_binding::{ExecutionBindings, ExecutionHandle};
use crate::execution_context::{bind_execution, ServerState};
use axum::middleware;
use pyo3::types::{PyDict, PyModule};

fn plane(owner: &str) -> Py<PyAny> {
    Python::initialize();
    Python::attach(|py| {
        PyModule::from_code(
            py,
            cr#"
import json
class Owner:
    def __init__(self, name):
        self.name = name
    def usage_json(self, argument):
        return json.dumps({"owner": self.name, "argument": json.loads(argument)})
    def usage_page(self, argument):
        return json.dumps({"html": self.name + ":" + argument})
    def close_thread_resources(self, argument):
        return "{}"
"#,
            c"usage_owner.py",
            c"usage_owner",
        )
        .unwrap()
        .getattr("Owner")
        .unwrap()
        .call1((owner,))
        .unwrap()
        .unbind()
    })
}

fn state() -> ServerState {
    ServerState {
        base: AppState {
            bridge: Arc::new(crate::bridge::Bridge::new(plane("process"), 1).unwrap()),
            http: crate::upstream::build_client(Duration::from_secs(1), false).unwrap(),
            permits: Arc::new(tokio::sync::Semaphore::new(4)),
            dispatch_capacity: Arc::new(crate::dispatch_capacity::DispatchCapacity::new()),
            permit_queue: Arc::new(crate::permit_queue::PermitQueue::new(None)),
            request_timeout: Duration::from_secs(5),
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
        },
        bindings_enabled: true,
    }
}

#[test]
fn usage_uses_process_owner_and_preserves_one_use_binding() {
    tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap()
        .block_on(async {
            let target = plane("bound");
            let handle =
                Python::attach(|py| ExecutionHandle::new(target, 2, 2, 1, &PyDict::new(py), None))
                    .unwrap();
            let bindings = ExecutionBindings::new(1).unwrap();
            let token = "registered-relay-token".repeat(2);
            bindings.register(token.clone(), &handle).unwrap();
            let app = Router::new()
                .route("/usage.json", get(usage_json))
                .route("/usage", get(usage_page))
                .route("/v1/test-owner", get(usage_json))
                .layer(middleware::from_fn_with_state(bindings, bind_execution))
                .with_state(state());
            let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
            let address = listener.local_addr().unwrap();
            let server = tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
            let client = reqwest::Client::builder()
                .timeout(Duration::from_secs(5))
                .build()
                .unwrap();
            for path in ["/usage", "/usage.json"] {
                for authenticated in [false, true] {
                    let request = client.get(format!("http://{address}{path}"));
                    let request = if authenticated {
                        request.bearer_auth(&token)
                    } else {
                        request
                    };
                    let response = request.send().await.unwrap();
                    assert_eq!(response.status().as_u16(), 200);
                    let body = response.text().await.unwrap();
                    assert!(body.contains("process") && !body.contains("bound"));
                    assert!(handle.in_use());
                }
                let malformed = client
                    .get(format!("http://{address}{path}"))
                    .header("authorization", "Basic invalid")
                    .send()
                    .await
                    .unwrap();
                assert_eq!(malformed.status().as_u16(), 401);
            }
            let response = client
                .get(format!("http://{address}/v1/test-owner"))
                .bearer_auth(&token)
                .send()
                .await
                .unwrap();
            assert_eq!(response.status().as_u16(), 200);
            assert_eq!(response.json::<Value>().await.unwrap()["owner"], "bound");
            let replay = client
                .get(format!("http://{address}/v1/test-owner"))
                .bearer_auth(&token)
                .send()
                .await
                .unwrap();
            assert_eq!(replay.status().as_u16(), 401);
            server.abort();
            let _ = server.await;
        });
}
