//! Cross-language integration with real admission, callback authority, and loopback HTTP.

use super::*;
use crate::bridge::Bridge;
use pyo3::prelude::*;
use pyo3::types::PyModule;
use tokio::io::AsyncWriteExt;

/// The only URL substitution is private test fixture state after official admission.
const FIXTURE: &std::ffi::CStr = cr#"
import json
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory

from exp.runtime.gateway.native_explicit_cache_test import (
    _Host, _authority, _control, _vertex_control, _admission, _start_first, _wire, _RESOURCE,
)
from exp.runtime.models.providers.google_cache import VertexCacheProject


def setup(base, vertex):
    """Build actual admission/accounting with a fake atomic cache-spend authority."""
    temporary = TemporaryDirectory(prefix="exp-native-cache-bridge-")
    authority = _authority()
    if vertex:
        authority = replace(authority, vertex_project=VertexCacheProject("fruit-project", "123456789"))
    host = _Host(authority)
    control, key = (_vertex_control if vertex else _control)(Path(temporary.name), host)
    return temporary, host, control, key, base, vertex


def admit(fixture):
    """Admit and reserve on the real plane before substituting loopback-only URLs."""
    temporary, host, control, key, base, vertex = fixture
    admission = _admission(control, key)
    wire = dict(_wire(admission))
    assert wire["explicit_cache"] is True
    assert wire["url"].startswith("https://aiplatform.googleapis.com/" if vertex else "https://generativelanguage.googleapis.com/")
    entry = control._accounting.entry(admission["request_id"])
    assert entry is not None and entry.explicit_cache_state is not None
    state = entry.explicit_cache_state
    binding = state.bindings[0]
    assert binding is not None
    root = "/v1/projects/fruit-project/locations/global" if vertex else "/v1beta"
    models = "/publishers/google/models/" if vertex else "/models/"
    state.bindings = (replace(binding, plan=replace(binding.plan, create_url=base + root + "/cachedContents")),)
    wire["url"] = base + root + models + "gemini-2.5-pro:streamGenerateContent?alt=sse"
    _start_first(control, admission)
    return admission["request_id"], json.dumps(wire)


def assert_accounted(fixture, expected_claims):
    """Check real host callbacks recorded exactly one scoped reservation and create."""
    temporary, host, control, key, base, vertex = fixture
    assert host.claim_calls == expected_claims
    assert len(host.authority_calls) == expected_claims
    assert host.record_calls == 1 and len(host.offers) == len(host.results) == 1
    offer = next(iter(host.offers.values()))
    result = next(iter(host.results.values()))
    assert result.operation_id == offer.operation_id
    resource = "projects/123456789/locations/global/" + _RESOURCE if vertex else _RESOURCE
    assert result.outcome == "ready" and result.resource_name == resource
    assert result.total_tokens == 1536 and result.expire_time == offer.expires_at
    assert result.create_time is not None
    assert offer.requested_at - 2 <= result.create_time <= result.observed_at
    assert result.create_time <= result.expire_time
    assert host.reserved == offer.reservation_nano_usd > 0


def cleanup(fixture):
    """Close fixture-thread SQLite connections before removing its temporary directory."""
    temporary, host, control, key, base, vertex = fixture
    control.close_thread_resources("{}")
    temporary.cleanup()
"#;

fn fixture(base: &str, vertex: bool) -> (Py<PyModule>, Py<PyAny>, Py<PyAny>) {
    Python::initialize();
    Python::attach(|py| {
        let module = PyModule::from_code(
            py,
            FIXTURE,
            c"google_cache_bridge_fixture.py",
            c"google_cache_bridge_fixture",
        )
        .expect("project test dependencies must be available to the embedded interpreter");
        let fixture = module
            .getattr("setup")
            .unwrap()
            .call1((base, vertex))
            .unwrap();
        let control = fixture.get_item(2).unwrap().unbind();
        (module.unbind(), fixture.unbind(), control)
    })
}

fn admit(module: &Py<PyModule>, fixture: &Py<PyAny>) -> (String, DeploymentWire) {
    Python::attach(|py| {
        let (request_id, wire): (String, String) = module
            .bind(py)
            .getattr("admit")
            .unwrap()
            .call1((fixture.bind(py),))
            .unwrap()
            .extract()
            .unwrap();
        (request_id, serde_json::from_str(&wire).unwrap())
    })
}

fn assert_accounted(module: &Py<PyModule>, fixture: &Py<PyAny>, expected_claims: u64) {
    Python::attach(|py| {
        module
            .bind(py)
            .getattr("assert_accounted")
            .unwrap()
            .call1((fixture.bind(py), expected_claims))
            .unwrap();
    });
}

/// Server echoes the absolute expiry supplied by the real Python create callback.
async fn loopback(resource: &'static str) -> (String, tokio::task::JoinHandle<Vec<Value>>) {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let base = format!("http://{}", listener.local_addr().unwrap());
    let task = tokio::spawn(async move {
        let mut bodies = Vec::new();
        for index in 0..3 {
            let (mut socket, _) = listener.accept().await.unwrap();
            let body = crate::waterfall::ladder_tests::read_request_body(&mut socket).await;
            let body: Value = serde_json::from_str(&body).unwrap();
            let response = if index == 0 {
                assert!(body.get("ttl").is_none());
                assert!(body.get("expireTime").is_some());
                json!({
                    "name": resource,
                    "expireTime": body["expireTime"],
                    "createTime": tests::timestamp(epoch_now().floor() as u64).1,
                    "usageMetadata": {"totalTokenCount": 1536},
                    "private_provider_text": "must never cross the accounting bridge",
                })
                .to_string()
            } else {
                "data: {\"candidates\":[{\"content\":{\"parts\":[{\"text\":\"42\"}]},\"finishReason\":\"STOP\"}]}\n\n".to_string()
            };
            bodies.push(body);
            let response = format!(
                "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                response.len(), response,
            );
            socket.write_all(response.as_bytes()).await.unwrap();
            let _ = socket.shutdown().await;
        }
        bodies
    });
    (base, task)
}

#[tokio::test]
#[ignore = "requires project dependencies in the matching embedded Python; run with PYO3_PYTHON and PYTHONPATH"]
async fn real_python_control_plane_creates_and_reuses_through_bridge() {
    exercise_bridge(false).await;
    exercise_bridge(true).await;
}

async fn exercise_bridge(vertex: bool) {
    let resource = if vertex {
        "projects/123456789/locations/global/cachedContents/verified_test_resource"
    } else {
        "cachedContents/verified_test_resource"
    };
    let (base, server) = loopback(resource).await;
    let (module, fixture, control) = fixture(&base, vertex);
    let bridge = Bridge::new(control, 1).unwrap();
    let http = crate::upstream::build_client(Duration::from_secs(1)).unwrap();
    for expected_claims in 1..=2 {
        let (request_id, original) = admit(&module, &fixture);
        let cached = execute(
            &http,
            &original,
            &request_id,
            Instant::now() + Duration::from_secs(10),
            false,
            EndpointPolicy::Loopback,
            |method, argument| bridge.call(method, argument),
        )
        .await
        .unwrap()
        .expect("the real host authorizes the recorded resource");
        assert_accounted(&module, &fixture, expected_claims);
        assert!(original.upstream_payload.get("cachedContent").is_none());
        assert_eq!(cached.upstream_payload["cachedContent"], resource);
        assert_eq!(
            cached.upstream_payload["contents"],
            json!([
                {"role":"user", "parts":[{"text":"How many apples are available?"}]}
            ])
        );
        assert!(cached.upstream_payload.get("systemInstruction").is_none());
        assert_eq!(
            cached.upstream_payload["generationConfig"],
            original.upstream_payload["generationConfig"]
        );
        let response = crate::upstream::open_stream(
            &http,
            &cached.url,
            &cached.headers,
            &cached.idempotency_key,
            &cached.upstream_payload,
            None,
            Duration::from_secs(1),
            crate::dialects::Dialect::GeminiGenerateContent,
        )
        .await
        .unwrap();
        assert!(response.text().await.unwrap().contains("42"));
    }
    let bodies = tokio::time::timeout(Duration::from_secs(2), server)
        .await
        .unwrap()
        .unwrap();
    assert_eq!(bodies.len(), 3, "one cache create plus two generations");
    assert!(bodies[0].get("systemInstruction").is_some());
    assert!(bodies[0]
        .to_string()
        .contains("private-inventory-prefix-canary"));
    assert!(bodies[1..]
        .iter()
        .all(|body| !body.to_string().contains("private-inventory-prefix-canary")));
    drop(bridge);
    Python::attach(|py| {
        module
            .bind(py)
            .getattr("cleanup")
            .unwrap()
            .call1((fixture.bind(py),))
            .unwrap();
    });
}
