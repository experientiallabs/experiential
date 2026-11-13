//! Real cache HTTP through the native request owner and native financial authority.

use super::*;
use crate::accounting::cache_bridge_fixture as native;
use tokio::io::AsyncWriteExt;

fn configuration(base: &str, vertex: bool) -> (Value, Value, DeploymentWire) {
    // This fixture is exported from the real admission binder, without Python callbacks.
    let case: Value = include_str!("../../tests/generated/cache.jsonl")
        .lines()
        .map(|line| crate::lossless_json::from_str::<Value>(line).unwrap())
        .find(|case| case["kind"] == "callbacks" && case["host_kind"] == "creator")
        .unwrap();
    let mut configuration = case["configuration"].clone();
    let mut authority = case["authority"].clone();
    let root = if vertex {
        "/v1/projects/fruit-project/locations/global"
    } else {
        "/v1beta"
    };
    let models = if vertex {
        "/publishers/google/models/"
    } else {
        "/models/"
    };
    let plan = &mut configuration["bindings"][0]["plan"];
    plan["create_url"] = json!(format!("{base}{root}/cachedContents"));
    if vertex {
        plan["resource_prefix"] = json!("projects/fruit-project/locations/global/cachedContents/");
        authority["vertex_project"] =
            json!({"endpoint_project":"fruit-project", "project_number":"123456789"});
    }
    let mut wire = tests::wire(&format!(
        "{base}{root}{models}gemini-2.5-pro:streamGenerateContent?alt=sse"
    ));
    wire.deployment_id = "coding".into();
    wire.upstream_payload = plan["create_payload"].clone();
    wire.upstream_payload["generationConfig"] =
        plan["generation_payload"]["generationConfig"].clone();
    wire.upstream_payload["contents"]
        .as_array_mut()
        .unwrap()
        .extend(
            plan["generation_payload"]["contents"]
                .as_array()
                .unwrap()
                .iter()
                .cloned(),
        );
    (configuration, authority, wire)
}

/// Server echoes the absolute expiry supplied by the native spending authority.
async fn loopback(resource: &'static str) -> (String, tokio::task::JoinHandle<Vec<Value>>) {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let base = format!("http://{}", listener.local_addr().unwrap());
    let task = tokio::spawn(async move {
        let mut bodies = Vec::new();
        for index in 0..3 {
            let (mut socket, _) = listener.accept().await.unwrap();
            let body = crate::waterfall::ladder_tests::read_request_body(&mut socket).await;
            let body: Value = crate::lossless_json::from_str(&body).unwrap();
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
async fn native_request_owner_creates_and_reuses_cache_without_python() {
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
    let (configuration, authority, original) = configuration(&base, vertex);
    let host = native::host(authority);
    let http = crate::upstream::build_client(Duration::from_secs(1), false).unwrap();
    for expected_claims in 1..=2 {
        let request_id = format!("request-{expected_claims}");
        let fixture = native::admit(configuration.clone(), host.clone(), &request_id);
        let cached = execute(
            &http,
            &original,
            &request_id,
            Instant::now() + Duration::from_secs(10),
            false,
            EndpointPolicy::Loopback,
            |method, argument| {
                let fixture = fixture.clone();
                async move {
                    tokio::task::spawn_blocking(move || fixture.call(method, &argument))
                        .await
                        .unwrap()
                }
            },
        )
        .await
        .unwrap()
        .expect("the real host authorizes the recorded resource");
        native::assert_accounted(&host, expected_claims, resource);
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
}
