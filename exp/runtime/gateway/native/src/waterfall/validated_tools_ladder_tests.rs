//! Ladder tests for the schema-validated redial (`validated_tools`): after a
//! Gemini dial ends on MALFORMED_FUNCTION_CALL, the redial of the same rung
//! and a sibling Google rung both ask for `VALIDATED` function calling, and a
//! first dial never does.

use std::sync::{Arc, Mutex};
use std::time::Duration;

use serde_json::{json, Value};
use tokio::io::AsyncWriteExt;

use super::ladder_tests::{block_on, finish, read_request, wire, Harness};
use super::*;

/// Gemini's answer when its sampled function call does not parse: thought
/// only, then the malformed finish.
const MALFORMED_FRAME: &str = concat!(
    "{\"candidates\":[{\"content\":{\"parts\":[],\"role\":\"model\"},",
    "\"finishReason\":\"MALFORMED_FUNCTION_CALL\"}],\"usageMetadata\":{\"promptTokenCount\":9,",
    "\"candidatesTokenCount\":0,\"thoughtsTokenCount\":40,\"totalTokenCount\":49}}"
);

/// One whole Gemini answer: a valid call, a STOP finish and its meter.
const CALL_FRAME: &str = concat!(
    "{\"candidates\":[{\"content\":{\"parts\":[{\"functionCall\":{\"name\":\"read\",",
    "\"args\":{\"path\":\"a.py\"}}}],\"role\":\"model\"},\"finishReason\":\"STOP\"}],",
    "\"usageMetadata\":{\"promptTokenCount\":9,\"candidatesTokenCount\":5,\"totalTokenCount\":14}}"
);

fn streamed(frame: &str) -> String {
    let body = format!("data: {frame}\n\n");
    format!(
        "HTTP/1.1 200 OK\r\ncontent-type: text/event-stream\r\n\
         content-length: {}\r\nconnection: close\r\n\r\n{body}",
        body.len(),
    )
}

/// A scripted Google rung: answers its connections in order and records the
/// body each dial carried.
async fn spawn_google_rung(script: Vec<String>) -> (String, Arc<Mutex<Vec<String>>>) {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
        .await
        .expect("bind");
    let address = listener.local_addr().expect("address");
    let bodies = Arc::new(Mutex::new(Vec::new()));
    let recorder = bodies.clone();
    tokio::spawn(async move {
        for answer in script {
            let (mut socket, _) = listener.accept().await.expect("accept");
            let (_, body) = read_request(&mut socket).await;
            recorder.lock().expect("lock").push(body);
            socket.write_all(answer.as_bytes()).await.expect("write");
            let _ = socket.shutdown().await;
        }
    });
    let url = format!("http://{address}/v1beta/models/neon:streamGenerateContent?alt=sse");
    (url, bodies)
}

fn google_wire(provider: &str, deployment_id: &str, url: &str) -> DeploymentWire {
    let mut rung = wire(deployment_id, url, 0);
    rung.provider = provider.to_string();
    rung.dialect = "gemini_generate_content".to_string();
    rung.upstream_payload = json!({
        "contents": [{"role": "user", "parts": [{"text": "read a.py"}]}],
        "tools": [{"functionDeclarations": [{
            "name": "read",
            "parametersJsonSchema": {"type": "object", "properties": {"path": {"type": "string"}}},
        }]}],
    });
    rung
}

/// The function-calling mode one dial sent, if any.
fn sent_mode(body: &str) -> Value {
    let sent: Value = serde_json::from_str(body).expect("json body");
    sent["toolConfig"]["functionCallingConfig"]["mode"].clone()
}

#[test]
fn a_malformed_function_call_redials_the_rung_with_validated_calling() {
    block_on(async {
        let harness = Harness::new();
        harness
            .configure(json!({"rules": [null], "same_redial": 2}))
            .await;
        let (url, bodies) =
            spawn_google_rung(vec![streamed(MALFORMED_FRAME), streamed(CALL_FRAME)]).await;
        let rung = google_wire("gemini", "gemini-neon", &url);
        let (won, guard) = harness
            .run_as("key", None, &[rung], None, Duration::from_secs(10))
            .await;
        let won = finish(guard, won).await;
        let bodies = bodies.lock().unwrap();
        let Won::Committed(committed) = won else {
            panic!(
                "the validated redial must serve the turn: {} dials",
                bodies.len()
            );
        };
        drop(committed);
        assert_eq!(bodies.len(), 2);
        assert_eq!(sent_mode(&bodies[0]), Value::Null);
        assert_eq!(sent_mode(&bodies[1]), json!("VALIDATED"));
    });
}

#[test]
fn the_sibling_backend_after_a_malformed_function_call_dials_validated() {
    block_on(async {
        let harness = Harness::new();
        let (gemini_url, gemini_bodies) =
            spawn_google_rung(vec![streamed(MALFORMED_FRAME), streamed(MALFORMED_FRAME)]).await;
        let (vertex_url, vertex_bodies) = spawn_google_rung(vec![streamed(CALL_FRAME)]).await;
        let route = [
            google_wire("gemini", "gemini-neon", &gemini_url),
            google_wire("vertex", "vertex-neon", &vertex_url),
        ];
        let (won, guard) = harness
            .run_as("key", None, &route, None, Duration::from_secs(10))
            .await;
        let Won::Committed(committed) = finish(guard, won).await else {
            panic!("the validated sibling must serve the turn");
        };
        drop(committed);
        let gemini_bodies = gemini_bodies.lock().unwrap();
        let vertex_bodies = vertex_bodies.lock().unwrap();
        assert_eq!(sent_mode(&gemini_bodies[0]), Value::Null);
        assert!(gemini_bodies[1..]
            .iter()
            .all(|body| sent_mode(body) == json!("VALIDATED")));
        assert_eq!(vertex_bodies.len(), 1);
        assert_eq!(sent_mode(&vertex_bodies[0]), json!("VALIDATED"));
    });
}
