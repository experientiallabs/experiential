//! Ladder tests for the Gemini thought-signature replay repair
//! (`thought_signature_repair`): a Vertex rung that refuses a signature the
//! Gemini API minted ("Invalid thought signature.", production 2026-11-19) is
//! re-dialed once with the bypass value instead of failing the turn, a later
//! turn on that rung skips the refused dial, and the backend that minted the
//! signature still receives it verbatim.

use std::sync::{Arc, Mutex};
use std::time::Duration;

use serde_json::{json, Value};
use tokio::io::AsyncWriteExt;

use super::ladder_tests::{block_on, finish, read_request, wire, Harness};
use super::*;
use crate::thought_signature_repair::THOUGHT_SIGNATURE_BYPASS;

/// Vertex's answer to a replayed signature another backend minted.
const INVALID_SIGNATURE_BODY: &str = concat!(
    "{\"error\":{\"code\":400,\"message\":\"Invalid thought signature.\",",
    "\"status\":\"INVALID_ARGUMENT\"}}"
);

/// One whole Gemini answer: text, a STOP finish and its meter.
const GEMINI_TEXT_FRAME: &str = concat!(
    "{\"candidates\":[{\"content\":{\"parts\":[{\"text\":\"done\"}],\"role\":\"model\"},",
    "\"finishReason\":\"STOP\"}],\"usageMetadata\":{\"promptTokenCount\":9,",
    "\"candidatesTokenCount\":1,\"totalTokenCount\":10}}"
);

fn rejected() -> String {
    format!(
        "HTTP/1.1 400 Bad Request\r\ncontent-type: application/json\r\n\
         content-length: {}\r\nconnection: close\r\n\r\n{INVALID_SIGNATURE_BODY}",
        INVALID_SIGNATURE_BODY.len(),
    )
}

fn answered() -> String {
    let body = format!("data: {GEMINI_TEXT_FRAME}\n\n");
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

/// A Google rung replaying one tool turn whose call carries `signature`.
fn google_wire(provider: &str, deployment_id: &str, url: &str, signature: &str) -> DeploymentWire {
    let mut rung = wire(deployment_id, url, 0);
    rung.provider = provider.to_string();
    rung.dialect = "gemini_generate_content".to_string();
    rung.upstream_payload = json!({"contents": [
        {"role": "user", "parts": [{"text": "look it up"}]},
        {"role": "model", "parts": [
            {"functionCall": {"name": "lookup", "args": {}}, "thoughtSignature": signature},
        ]},
        {"role": "user", "parts": [{"functionResponse": {"name": "lookup", "response": {"ok": true}}}]},
    ]});
    rung
}

/// The replayed call's signature as one dial sent it.
fn sent_signature(body: &str) -> String {
    let sent: Value = serde_json::from_str(body).expect("json body");
    sent["contents"][1]["parts"][0]["thoughtSignature"]
        .as_str()
        .expect("signature")
        .to_string()
}

async fn serve(harness: &Harness, caller: &str, rung: DeploymentWire) -> (bool, Value) {
    let before = harness.story().await;
    let (won, guard) = harness
        .run_as("key", Some(caller), &[rung], None, Duration::from_secs(10))
        .await;
    let Won::Committed(committed) = finish(guard, won).await else {
        panic!("a refused foreign signature must not fail the turn");
    };
    let stripped = committed.encrypted_reasoning_stripped;
    drop(committed);
    let after = harness.story().await;
    let dials =
        after["starts"].as_array().unwrap().len() - before["starts"].as_array().unwrap().len();
    (stripped, json!(dials))
}

#[test]
fn a_foreign_thought_signature_is_replaced_and_redialed_not_failed() {
    block_on(async {
        let harness = Harness::new();
        let caller = "org:gemini-signature-caller";
        let signature = "minted-on-the-gemini-api-signature==";

        // The Vertex rung refuses the Gemini API's signature, then serves the
        // same turn with the bypass in its place, through a new reservation.
        let (url, bodies) = spawn_google_rung(vec![rejected(), answered()]).await;
        let rung = google_wire("vertex", "vertex-neon", &url, signature);
        let (stripped, dials) = serve(&harness, caller, rung).await;
        assert!(stripped);
        assert_eq!(dials, json!(2));
        {
            let bodies = bodies.lock().unwrap();
            assert_eq!(bodies.len(), 2);
            assert_eq!(sent_signature(&bodies[0]), signature);
            assert_eq!(sent_signature(&bodies[1]), THOUGHT_SIGNATURE_BYPASS);
        }

        // A later turn of the conversation on that rung skips the refused dial.
        let (url, bodies) = spawn_google_rung(vec![answered()]).await;
        let rung = google_wire("vertex", "vertex-neon", &url, signature);
        let (stripped, dials) = serve(&harness, caller, rung).await;
        assert!(stripped);
        assert_eq!(dials, json!(1));
        assert_eq!(
            sent_signature(&bodies.lock().unwrap()[0]),
            THOUGHT_SIGNATURE_BYPASS
        );

        // The backend that minted it still receives it verbatim.
        let (url, bodies) = spawn_google_rung(vec![answered()]).await;
        let rung = google_wire("gemini", "gemini-api-neon", &url, signature);
        let (stripped, dials) = serve(&harness, caller, rung).await;
        assert!(!stripped);
        assert_eq!(dials, json!(1));
        assert_eq!(sent_signature(&bodies.lock().unwrap()[0]), signature);
    });
}

#[test]
fn a_refusal_the_bypass_cannot_clear_is_the_callers_named_error() {
    block_on(async {
        let harness = Harness::new();
        // Already the bypass: nothing to replace, so no re-dial.
        let (url, bodies) = spawn_google_rung(vec![rejected()]).await;
        let rung = google_wire(
            "vertex",
            "vertex-unrepairable",
            &url,
            THOUGHT_SIGNATURE_BYPASS,
        );
        let (won, guard) = harness
            .run_as(
                "key",
                Some("org:gemini-signature-unrepairable"),
                &[rung],
                None,
                Duration::from_secs(10),
            )
            .await;
        let Won::Failed(error) = finish(guard, won).await else {
            panic!("an unrepairable refusal fails the turn");
        };
        assert!(
            error.message.contains("thought signature"),
            "{}",
            error.message
        );
        assert_eq!(bodies.lock().unwrap().len(), 1);
    });
}

#[test]
fn a_cache_overlay_refusal_strips_the_whole_history_and_later_dials_plain() {
    let signature = "cached-prefix-signature-from-the-gemini-api==";
    let scope = "org:gemini-signature-overlay";
    let plain = google_wire(
        "vertex",
        "vertex-overlay",
        "http://127.0.0.1:9/x",
        signature,
    );
    // The overlay moved the signed history into a cached prefix; only the
    // live suffix rides the dialed payload.
    let mut overlay = plain.clone();
    overlay.automatic_cache = true;
    overlay.upstream_payload = json!({
        "cachedContent": "cachedContents/prefix",
        "contents": [plain.upstream_payload["contents"][2].clone()],
    });
    let mut repaired = None;
    let mut repair = crate::replay_repair::AttemptRepair::begin(
        &overlay,
        &plain,
        Some(scope),
        &mut repaired,
        "r",
    );
    let refusal = crate::errors::Failure::new(FailureClass::InvalidRequest, "refused")
        .with_encrypted_reasoning_rejected(true);
    assert!(repair.repair_after(&refusal));
    drop(repair);
    let stripped = repaired.expect("the plain history, stripped");
    assert!(stripped.get("cachedContent").is_none());
    assert_eq!(
        sent_signature(&stripped.to_string()),
        THOUGHT_SIGNATURE_BYPASS
    );
    // The next turn on that rung skips the cache, so the repair reaches it.
    assert!(crate::thought_signature_repair::remembers_refused_signature(&plain, Some(scope)));
    let mut minted = plain.clone();
    minted.deployment_id = "gemini-api-overlay".to_string();
    assert!(!crate::thought_signature_repair::remembers_refused_signature(&minted, Some(scope)));
}

#[test]
fn a_relayed_signature_is_remembered_under_the_rung_that_minted_it() {
    block_on(async {
        let harness = Harness::new();
        let minted = "bWludGVkLWJ5LXZlcnRleC1sYWRkZXItdGVzdA==";
        let frame = format!(
            "{{\"candidates\":[{{\"content\":{{\"parts\":[{{\"functionCall\":{{\"name\":\"lookup\",\
             \"args\":{{}}}},\"thoughtSignature\":\"{minted}\"}}],\"role\":\"model\"}},\
             \"finishReason\":\"STOP\"}}],\"usageMetadata\":{{\"promptTokenCount\":9,\
             \"candidatesTokenCount\":1,\"totalTokenCount\":10}}}}"
        );
        let body = format!("data: {frame}\n\n");
        let answer = format!(
            "HTTP/1.1 200 OK\r\ncontent-type: text/event-stream\r\n\
             content-length: {}\r\nconnection: close\r\n\r\n{body}",
            body.len(),
        );
        let (url, _bodies) = spawn_google_rung(vec![answer]).await;
        let rung = google_wire("vertex", "vertex-neon-issuer", &url, "c2lnLW9sZA==");
        let (won, guard) = harness
            .run_as("key", None, &[rung], None, Duration::from_secs(10))
            .await;
        let Won::Committed(committed) = finish(guard, won).await else {
            panic!("the rung serves the turn");
        };
        drop(committed);
        assert_eq!(
            crate::thought_signature_issuers::ISSUERS
                .issuer(minted)
                .as_deref(),
            Some("vertex-neon-issuer")
        );
    });
}
