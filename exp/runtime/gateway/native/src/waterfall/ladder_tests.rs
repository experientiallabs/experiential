//! Ladder tests for throttle backoff-and-redial: a scripted python control
//! plane and scripted local rungs drive `acquire_attempt` end to end, so the
//! wait schedule, the `throttle_backoff` reservation flag, the commitment
//! boundary, the ladder advance after the redial budget, and the exhaustion
//! `Retry-After` are all observed on the real loop rather than on its parts.

use std::collections::HashMap;
use std::sync::atomic::AtomicUsize;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use pyo3::prelude::*;
use serde_json::{json, Value};
use tokio::io::{AsyncReadExt, AsyncWriteExt};

use super::*;
use crate::upstream::build_client;

/// A control plane that mirrors the python candidate policy for one two-rung
/// route: a `throttle_backoff` reservation redials the same depth while the
/// per-rung redial cap allows, a failover-eligible failure advances, and
/// anything else exhausts with the failure echoed back. Every call is
/// recorded so the test can read the ledger story.
const PLANE_SOURCE: &std::ffi::CStr = cr#"
import json
import threading


class Plane:
    """Scripted control plane recording every reservation and settlement."""

    def __init__(self):
        self.lock = threading.Lock()
        self.starts = []
        self.settles = []
        self.counts = [0, 0]
        self.max_redials = 2

    def start_attempt(self, argument):
        data = json.loads(argument)
        with self.lock:
            self.starts.append(data)
            depth = data.get("current_depth")
            failure = data.get("failure")
            if depth is None:
                candidate = 0
            elif (
                data.get("throttle_backoff")
                and failure["failure_class"] == "throttled"
                and self.counts[depth] <= self.max_redials
            ):
                candidate = depth
            elif failure.get("failover_eligible") and depth + 1 < len(self.counts):
                candidate = depth + 1
            else:
                return json.dumps({"exhausted": True, "failure": failure})
            self.counts[candidate] += 1
            ordinal = data["attempt_ordinal"]
            return json.dumps({"attempt_id": f"attempt-{ordinal}", "route_depth": candidate})

    def settle(self, argument):
        with self.lock:
            self.settles.append(json.loads(argument))
        return "{}"

    def abandon(self, argument):
        return "{}"

    def dump(self, argument):
        with self.lock:
            return json.dumps(
                {"starts": self.starts, "settles": self.settles, "counts": self.counts}
            )

    def close_thread_resources(self, argument):
        return "{}"
"#;

fn plane() -> Py<PyAny> {
    Python::initialize();
    Python::attach(|py| {
        pyo3::types::PyModule::from_code(py, PLANE_SOURCE, c"ladder_plane.py", c"ladder_plane")
            .expect("plane module compiles")
            .getattr("Plane")
            .expect("plane class exists")
            .call0()
            .expect("plane instantiates")
            .unbind()
    })
}

/// One scripted provider answer for one connection.
#[derive(Clone)]
enum Answer {
    /// A 429 with the optional stated wait.
    Throttle(Option<u32>),
    /// A 400 carrying this exact JSON body.
    Rejected(&'static str),
    /// A 200 event stream carrying these SSE frames, then `[DONE]`.
    Stream(&'static [&'static str]),
    /// A 200 native Responses event stream carrying these SSE frames and
    /// then a `response.completed` terminal (the Responses wire has no
    /// `[DONE]`).
    ResponsesStream(&'static [&'static str]),
}

fn render(answer: &Answer) -> String {
    match answer {
        Answer::Rejected(body) => format!(
            "HTTP/1.1 400 Bad Request\r\ncontent-type: application/json\r\n\
             content-length: {}\r\nconnection: close\r\n\r\n{body}",
            body.len(),
        ),
        Answer::Throttle(retry_after) => {
            let body = "{\"error\":{\"message\":\"We're currently processing too many requests - \
                        please try again later\",\"type\":\"server_error\",\"code\":null}}";
            let header = retry_after
                .map(|seconds| format!("retry-after: {seconds}\r\n"))
                .unwrap_or_default();
            format!(
                "HTTP/1.1 429 Too Many Requests\r\ncontent-type: application/json\r\n{header}\
                 content-length: {}\r\nconnection: close\r\n\r\n{body}",
                body.len(),
            )
        }
        Answer::Stream(frames) | Answer::ResponsesStream(frames) => {
            let mut body = String::new();
            for frame in frames.iter() {
                body.push_str("data: ");
                body.push_str(frame);
                body.push_str("\n\n");
            }
            match answer {
                Answer::ResponsesStream(_) => {
                    body.push_str("data: ");
                    body.push_str(RESPONSES_COMPLETED_FRAME);
                    body.push_str("\n\n");
                }
                _ => body.push_str("data: [DONE]\n\n"),
            }
            format!(
                "HTTP/1.1 200 OK\r\ncontent-type: text/event-stream\r\n\
                 content-length: {}\r\nconnection: close\r\n\r\n{body}",
                body.len(),
            )
        }
    }
}

/// One scripted rung: answers its connections in script order and records
/// when each was accepted and the request body it carried.
struct Rung {
    url: String,
    accepted: Arc<Mutex<Vec<Instant>>>,
    bodies: Arc<Mutex<Vec<String>>>,
}

/// Read one whole HTTP/1.1 request (headers, then `content-length` bytes of
/// body) and return the body text.
async fn read_request_body(socket: &mut tokio::net::TcpStream) -> String {
    let mut received: Vec<u8> = Vec::new();
    let mut chunk = [0u8; 16_384];
    loop {
        let header_end = received
            .windows(4)
            .position(|window| window == b"\r\n\r\n")
            .map(|at| at + 4);
        if let Some(header_end) = header_end {
            let headers = String::from_utf8_lossy(&received[..header_end]).to_ascii_lowercase();
            let content_length = headers
                .lines()
                .find_map(|line| line.strip_prefix("content-length:"))
                .and_then(|value| value.trim().parse::<usize>().ok())
                .unwrap_or(0);
            if received.len() >= header_end + content_length {
                return String::from_utf8_lossy(&received[header_end..header_end + content_length])
                    .into_owned();
            }
        }
        let read = socket.read(&mut chunk).await.unwrap_or(0);
        if read == 0 {
            return String::new();
        }
        received.extend_from_slice(&chunk[..read]);
    }
}

async fn spawn_rung(script: Vec<Answer>) -> Rung {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
        .await
        .expect("bind");
    let address = listener.local_addr().expect("address");
    let accepted = Arc::new(Mutex::new(Vec::new()));
    let bodies = Arc::new(Mutex::new(Vec::new()));
    let recorder = accepted.clone();
    let body_recorder = bodies.clone();
    tokio::spawn(async move {
        for answer in script {
            let (mut socket, _) = listener.accept().await.expect("accept");
            recorder.lock().expect("lock").push(Instant::now());
            let body = read_request_body(&mut socket).await;
            body_recorder.lock().expect("lock").push(body);
            socket
                .write_all(render(&answer).as_bytes())
                .await
                .expect("write");
            let _ = socket.shutdown().await;
        }
    });
    Rung {
        url: format!("http://{address}/v1/chat/completions"),
        accepted,
        bodies,
    }
}

fn wire(deployment_id: &str, url: &str, throttle_redial_budget: u32) -> DeploymentWire {
    DeploymentWire {
        provider: "openai".to_string(),
        deployment_id: deployment_id.to_string(),
        dialect: "openai_compatible".to_string(),
        url: url.to_string(),
        headers: HashMap::new(),
        model_id: "gpt-test".to_string(),
        billing_customer_managed: false,
        timeout_seconds: 10.0,
        upstream_payload: json!({
            "model": "gpt-test",
            "messages": [{"role": "user", "content": "hi"}],
            "stream": true,
        }),
        upstream_body: None,
        fireworks_reasoning_route_sha256: None,
        hunyuan_reasoning_route_sha256: None,
        reasoning_output_exposed: false,
        stop_sequences: Vec::new(),
        serialize_tool_calls: false,
        image_output: false,
        idempotency_key: format!("op-{deployment_id}"),
        time_to_first_byte_base_seconds: None,
        time_to_first_byte_seconds_per_million_input_tokens: None,
        throttle_redial_budget,
    }
}

/// One native Responses rung whose replayed input carries these reasoning
/// payloads (sealed elsewhere, as far as the scripted rung is concerned)
/// beside the caller's visible turns. Every test names its own payloads: the
/// per-worker repair memory is process-global, so a payload one test's
/// refusal remembers would be stripped proactively in another.
fn responses_wire(deployment_id: &str, url: &str, encrypted: &[&str]) -> DeploymentWire {
    // The Codex shape: the call replays with the provider id of its turn.
    let mut input = vec![
        json!({"role": "user", "content": "plan the change"}),
        json!({"type": "function_call", "id": "fc_turn_1", "call_id": "call_1", "name": "exec", "arguments": "{}"}),
        json!({"type": "function_call_output", "call_id": "call_1", "output": "ok"}),
        json!({"role": "user", "content": "now apply it"}),
    ];
    for (offset, content) in encrypted.iter().enumerate() {
        input.insert(
            1 + offset,
            json!({"type": "reasoning", "summary": [], "encrypted_content": content}),
        );
    }
    DeploymentWire {
        dialect: "openai_responses".to_string(),
        url: url.replace("/v1/chat/completions", "/v1/responses"),
        upstream_payload: json!({
            "model": "gpt-test",
            "input": input,
            "store": false,
            "stream": true,
            "include": ["reasoning.encrypted_content"],
        }),
        ..wire(deployment_id, url, 0)
    }
}

/// OpenAI's verdict on a replayed reasoning payload it cannot decrypt, as
/// answered live to a customer's stateless Responses turn (2026-09-15).
const INVALID_ENCRYPTED_CONTENT_BODY: &str = concat!(
    "{\"error\":{\"message\":\"The encrypted content rsn_...hA== could not be verified. ",
    "Reason: Encrypted content could not be decrypted or parsed.\",",
    "\"type\":\"invalid_request_error\",\"param\":null,\"code\":\"invalid_encrypted_content\"}}"
);

const RESPONSES_COMPLETED_FRAME: &str = concat!(
    "{\"type\":\"response.completed\",\"response\":{\"status\":\"completed\",",
    "\"usage\":{\"input_tokens\":12,\"output_tokens\":3,\"total_tokens\":15}}}"
);

const RESPONSES_TEXT_FRAME: &str = concat!(
    "{\"type\":\"response.output_text.delta\",\"item_id\":\"msg_1\",",
    "\"output_index\":0,\"content_index\":0,\"delta\":\"applied\"}"
);

const SCHEDULE: ThrottleRedial = ThrottleRedial {
    max_attempts: 2,
    base_delay_ms: 100,
    max_delay_ms: 2_000,
};

const TEXT_FRAME: &str = "{\"choices\":[{\"delta\":{\"content\":\"hi\"}}]}";
const THROTTLE_FRAME: &str = "{\"error\":{\"code\":\"rate_limit_exceeded\",\
                              \"message\":\"Rate limit reached\"}}";

/// Everything one ladder run needs, kept alive together.
struct Harness {
    bridge: Arc<Bridge>,
    http: reqwest::Client,
}

impl Harness {
    fn new() -> Self {
        Self {
            bridge: Arc::new(Bridge::new(plane(), 2).expect("bridge starts")),
            http: build_client(Duration::from_secs(2)).expect("client"),
        }
    }

    async fn run(
        &self,
        route: &[DeploymentWire],
        throttle_redial: Option<ThrottleRedial>,
        deadline: Duration,
    ) -> (Won, AttemptGuard) {
        self.run_as("key", route, throttle_redial, deadline).await
    }

    async fn run_as(
        &self,
        raw_key: &str,
        route: &[DeploymentWire],
        throttle_redial: Option<ThrottleRedial>,
        deadline: Duration,
    ) -> (Won, AttemptGuard) {
        let mut guard = AttemptGuard::new(
            self.bridge.clone(),
            Arc::new(AtomicUsize::new(0)),
            "request-throttle".to_string(),
            Instant::now(),
        );
        let context = WaterfallContext {
            bridge: &self.bridge,
            http: &self.http,
            request_id: "request-throttle",
            raw_key,
            route,
            policy: RoutePolicy {
                maximum_total_attempts: 8,
                maximum_same_deployment_attempts: 2,
                refusal_failover: false,
                throttle_redial,
            },
            deadline: Instant::now() + deadline,
            time_to_first_byte: Duration::from_secs(5),
            time_to_first_byte_slope_seconds_per_million_input_tokens: 0.0,
            approximate_input_tokens: 10.0,
            output_less_retention: None,
            output_token_cap: None,
        };
        let won = acquire_attempt(&context, &mut guard).await;
        (won, guard)
    }

    async fn story(&self) -> Value {
        let text = self
            .bridge
            .call("dump", "{}".to_string())
            .await
            .expect("dump succeeds");
        serde_json::from_str(&text).expect("story parses")
    }
}

fn block_on<F: std::future::Future>(future: F) -> F::Output {
    tokio::runtime::Builder::new_multi_thread()
        .worker_threads(2)
        .enable_all()
        .build()
        .expect("runtime builds")
        .block_on(future)
}

fn gaps(rung: &Rung) -> Vec<Duration> {
    let accepted = rung.accepted.lock().expect("lock");
    accepted.windows(2).map(|pair| pair[1] - pair[0]).collect()
}

async fn finish(mut guard: AttemptGuard, won: Won) -> Won {
    if let Won::Committed(_) = &won {
        guard.settle("completed", None, &[], None, true).await;
    }
    won
}

#[test]
fn a_throttled_rung_is_redialed_after_backoff_and_then_serves() {
    block_on(async {
        let harness = Harness::new();
        // Two throttles (the second stating a one-second wait), then service.
        let rung_a = spawn_rung(vec![
            Answer::Throttle(None),
            Answer::Throttle(Some(1)),
            Answer::Stream(&[TEXT_FRAME]),
        ])
        .await;
        let rung_b = spawn_rung(vec![Answer::Stream(&[TEXT_FRAME])]).await;
        let route = [wire("a", &rung_a.url, 2), wire("b", &rung_b.url, 0)];
        let (won, guard) = harness
            .run(&route, Some(SCHEDULE), Duration::from_secs(60))
            .await;
        let won = finish(guard, won).await;
        let Won::Committed(committed) = won else {
            panic!("the warm rung serves after its redials");
        };
        assert_eq!(committed.depth, 0);
        assert!(matches!(committed.prefix.first(), Some(Event::TextDelta(text)) if text == "hi"));
        drop(committed);

        // The rung saw three connections: the first redial after the
        // jittered base wait, the second after the stated Retry-After.
        let waits = gaps(&rung_a);
        assert_eq!(waits.len(), 2);
        assert!(waits[0] >= Duration::from_millis(50) && waits[0] < Duration::from_secs(1));
        assert!(waits[1] >= Duration::from_secs(1) && waits[1] < Duration::from_secs(2));
        assert!(rung_b.accepted.lock().expect("lock").is_empty());

        // Every redial was its own reservation, flagged as a post-backoff
        // redial of the same depth and carrying the throttle's stated wait.
        let story = harness.story().await;
        let starts = story["starts"].as_array().expect("starts");
        assert_eq!(starts.len(), 3);
        assert_eq!(starts[0]["throttle_backoff"], false);
        assert_eq!(starts[0]["failure"], Value::Null);
        for (ordinal, start) in starts.iter().enumerate().skip(1) {
            assert_eq!(start["attempt_ordinal"], ordinal);
            assert_eq!(start["current_depth"], 0);
            assert_eq!(start["throttle_backoff"], true);
            assert_eq!(start["failure"]["failure_class"], "throttled");
        }
        assert_eq!(starts[1]["failure"]["retry_after_seconds"], Value::Null);
        assert_eq!(starts[2]["failure"]["retry_after_seconds"], 1);
        // Both throttled attempts settled failed without finalizing the
        // request; the served attempt settled last.
        let settles = story["settles"].as_array().expect("settles");
        assert_eq!(settles.len(), 3);
        for settle in &settles[..2] {
            assert_eq!(settle["outcome"], "failed");
            assert_eq!(settle["finalize"], false);
            assert_eq!(settle["failure"]["failure_class"], "throttled");
        }
        assert_eq!(settles[2]["outcome"], "completed");
        assert_eq!(story["counts"], json!([3, 0]));
    });
}

#[test]
fn spent_redials_advance_the_ladder_and_exhaustion_carries_the_largest_retry_after() {
    block_on(async {
        let harness = Harness::new();
        // The warm rung throttles through its whole redial budget, its last
        // answer stating the longest wait; the cold rung (not worth waiting
        // for) throttles once with a shorter one.
        let rung_a = spawn_rung(vec![
            Answer::Throttle(None),
            Answer::Throttle(None),
            Answer::Throttle(Some(9)),
        ])
        .await;
        let rung_b = spawn_rung(vec![Answer::Throttle(Some(6))]).await;
        let route = [wire("a", &rung_a.url, 2), wire("b", &rung_b.url, 0)];
        let (won, guard) = harness
            .run(&route, Some(SCHEDULE), Duration::from_secs(60))
            .await;
        let won = finish(guard, won).await;
        let Won::Failed(error) = won else {
            panic!("an exhausted ladder answers with the typed error");
        };
        assert_eq!(error.status_code, 429);
        assert_eq!(error.code, "unavailable_route");
        // The longest wait any rung asked for, not the last rung's.
        assert_eq!(error.retry_after_seconds, Some(9));

        assert_eq!(gaps(&rung_a).len(), 2);
        assert_eq!(rung_b.accepted.lock().expect("lock").len(), 1);
        let story = harness.story().await;
        let starts = story["starts"].as_array().expect("starts");
        // Initial dispatch, two redials, one cold failover; the cold rung's
        // throttle exhausts on the data plane's own facts (no later rung),
        // so no further reservation is asked for.
        assert_eq!(starts.len(), 4);
        assert_eq!(starts[1]["throttle_backoff"], true);
        assert_eq!(starts[2]["throttle_backoff"], true);
        assert_eq!(starts[3]["throttle_backoff"], false);
        assert_eq!(starts[3]["current_depth"], 0);
        assert_eq!(starts[3]["failure"]["retry_after_seconds"], 9);
        assert_eq!(story["counts"], json!([3, 1]));
        let settles = story["settles"].as_array().expect("settles");
        assert_eq!(settles.len(), 4);
        assert!(settles[..3]
            .iter()
            .all(|settle| settle["finalize"] == false));
        assert_eq!(settles[3]["finalize"], true);
        assert_eq!(settles[3]["failure"]["retry_after_seconds"], 6);
    });
}

#[test]
fn commitment_ends_redials_even_when_the_committed_stream_then_throttles() {
    block_on(async {
        let harness = Harness::new();
        // The provider streams text, then declares a rate-limit error inside
        // the stream. The text committed the deployment, so the throttle is
        // the committed relay's to report: no redial, no failover.
        let rung_a = spawn_rung(vec![Answer::Stream(&[TEXT_FRAME, THROTTLE_FRAME])]).await;
        let rung_b = spawn_rung(vec![Answer::Stream(&[TEXT_FRAME])]).await;
        let route = [wire("a", &rung_a.url, 2), wire("b", &rung_b.url, 2)];
        let (won, guard) = harness
            .run(&route, Some(SCHEDULE), Duration::from_secs(60))
            .await;
        let won = finish(guard, won).await;
        let Won::Committed(committed) = won else {
            panic!("the first semantic event commits the deployment");
        };
        assert_eq!(committed.depth, 0);
        drop(committed);
        assert_eq!(rung_a.accepted.lock().expect("lock").len(), 1);
        assert!(rung_b.accepted.lock().expect("lock").is_empty());
        let story = harness.story().await;
        assert_eq!(story["starts"].as_array().expect("starts").len(), 1);
    });
}

#[test]
fn without_a_schedule_or_a_budget_a_throttle_stays_failover_only() {
    for (schedule, budget) in [(None, 2), (Some(SCHEDULE), 0)] {
        block_on(async {
            let harness = Harness::new();
            let rung_a = spawn_rung(vec![Answer::Throttle(Some(1))]).await;
            let rung_b = spawn_rung(vec![Answer::Stream(&[TEXT_FRAME])]).await;
            let route = [
                wire("a", &rung_a.url, budget),
                wire("b", &rung_b.url, budget),
            ];
            let (won, guard) = harness.run(&route, schedule, Duration::from_secs(60)).await;
            let won = finish(guard, won).await;
            let Won::Committed(committed) = won else {
                panic!("the throttle fails over to the next rung");
            };
            assert_eq!(committed.depth, 1);
            drop(committed);
            assert_eq!(rung_a.accepted.lock().expect("lock").len(), 1);
            let story = harness.story().await;
            let starts = story["starts"].as_array().expect("starts");
            assert_eq!(starts.len(), 2);
            assert_eq!(starts[1]["throttle_backoff"], false);
            assert_eq!(starts[1]["current_depth"], 0);
        });
    }
}

#[test]
fn a_wait_that_cannot_fit_the_deadline_advances_instead_of_waiting() {
    block_on(async {
        let harness = Harness::new();
        let rung_a = spawn_rung(vec![Answer::Throttle(None)]).await;
        let rung_b = spawn_rung(vec![Answer::Stream(&[TEXT_FRAME])]).await;
        let route = [wire("a", &rung_a.url, 2), wire("b", &rung_b.url, 2)];
        // Three seconds left, and the redial would need its own five-second
        // first-byte allowance after the wait: the ladder advances at once.
        let started = Instant::now();
        let (won, guard) = harness
            .run(&route, Some(SCHEDULE), Duration::from_secs(3))
            .await;
        let won = finish(guard, won).await;
        let Won::Committed(committed) = won else {
            panic!("the next rung serves");
        };
        assert_eq!(committed.depth, 1);
        drop(committed);
        assert!(started.elapsed() < Duration::from_secs(1));
        let story = harness.story().await;
        assert_eq!(story["starts"][1]["throttle_backoff"], false);
    });
}

#[test]
fn a_low_stake_request_gets_fewer_redials_than_a_high_stake_one() {
    block_on(async {
        let harness = Harness::new();
        // Both requests hit a rung that throttles three times in a row; the
        // schedule allows two redials. Admission sized the high-stake
        // request's budget at the full two (its cache meets the threshold)
        // and the low-stake request's at one (cache below it).
        let high_rung = spawn_rung(vec![
            Answer::Throttle(None),
            Answer::Throttle(None),
            Answer::Throttle(None),
        ])
        .await;
        let low_rung = spawn_rung(vec![
            Answer::Throttle(None),
            Answer::Throttle(None),
            Answer::Throttle(None),
        ])
        .await;
        let spill = spawn_rung(vec![
            Answer::Stream(&[TEXT_FRAME]),
            Answer::Stream(&[TEXT_FRAME]),
        ])
        .await;

        let high_route = [wire("a", &high_rung.url, 2), wire("b", &spill.url, 2)];
        let (won, guard) = harness
            .run(&high_route, Some(SCHEDULE), Duration::from_secs(60))
            .await;
        let Won::Committed(committed) = finish(guard, won).await else {
            panic!("the spill rung serves the high-stake request");
        };
        assert_eq!(committed.depth, 1);
        drop(committed);
        // Two redials of the warm rung before the cold advance.
        assert_eq!(high_rung.accepted.lock().expect("lock").len(), 3);

        let low_harness = Harness::new();
        let low_route = [wire("a", &low_rung.url, 1), wire("b", &spill.url, 2)];
        let (won, guard) = low_harness
            .run(&low_route, Some(SCHEDULE), Duration::from_secs(60))
            .await;
        let Won::Committed(committed) = finish(guard, won).await else {
            panic!("the spill rung serves the low-stake request");
        };
        assert_eq!(committed.depth, 1);
        drop(committed);
        // One redial only: the second throttle advances the ladder even
        // though the schedule itself allows two.
        assert_eq!(low_rung.accepted.lock().expect("lock").len(), 2);

        let high = harness.story().await;
        let low = low_harness.story().await;
        assert_eq!(high["counts"], json!([3, 1]));
        assert_eq!(low["counts"], json!([2, 1]));
        let flags = |story: &Value| -> Vec<bool> {
            story["starts"]
                .as_array()
                .expect("starts")
                .iter()
                .map(|start| start["throttle_backoff"] == true)
                .collect()
        };
        assert_eq!(flags(&high), vec![false, true, true, false]);
        assert_eq!(flags(&low), vec![false, true, false]);
    });
}

#[test]
fn a_refused_encrypted_reasoning_item_is_stripped_and_the_same_rung_redialed() {
    block_on(async {
        let harness = Harness::new();
        let before = METRICS.snapshot()["encrypted_reasoning_stripped"]
            .as_u64()
            .expect("counter");
        // The rung refuses the sealed-elsewhere payload, then serves the
        // stripped replay on its second connection.
        let rung_a = spawn_rung(vec![
            Answer::Rejected(INVALID_ENCRYPTED_CONTENT_BODY),
            Answer::ResponsesStream(&[RESPONSES_TEXT_FRAME]),
        ])
        .await;
        let rung_b = spawn_rung(vec![Answer::ResponsesStream(&[RESPONSES_TEXT_FRAME])]).await;
        let route = [
            responses_wire("a", &rung_a.url, &["rsn_a_refused_encrypted_reasoning_item_is_stripped_and_the_same_rung_redialed_hA=="]),
            responses_wire("b", &rung_b.url, &["rsn_a_refused_encrypted_reasoning_item_is_stripped_and_the_same_rung_redialed_hA=="]),
        ];
        let (won, guard) = harness.run(&route, None, Duration::from_secs(60)).await;
        let won = finish(guard, won).await;
        let Won::Committed(committed) = won else {
            panic!("the same rung serves the stripped replay");
        };
        assert_eq!(committed.depth, 0);
        assert!(committed.encrypted_reasoning_stripped);
        assert!(matches!(
            committed.prefix.first(),
            Some(Event::ProviderOutputItemStarted { .. })
        ));
        drop(committed);

        // Two dials of the same rung: the replay as sent, then without the
        // refused item; the visible turns and tool items travel both times.
        let bodies = rung_a.bodies.lock().expect("lock").clone();
        assert_eq!(bodies.len(), 2);
        let sent: Value = serde_json::from_str(&bodies[0]).expect("first body");
        let repaired: Value = serde_json::from_str(&bodies[1]).expect("second body");
        assert_eq!(sent["input"].as_array().expect("input").len(), 5);
        let repaired_input = repaired["input"].as_array().expect("input");
        assert_eq!(repaired_input.len(), 4);
        assert!(repaired_input
            .iter()
            .all(|item| item.get("encrypted_content").is_none()));
        // The call the stripped reasoning governed replays id-less (the
        // provider would demand the reasoning item back for `fc_turn_1`).
        assert_eq!(sent["input"][2]["id"], "fc_turn_1");
        assert_eq!(repaired_input[1]["type"], "function_call");
        assert!(repaired_input[1].get("id").is_none());
        assert_eq!(repaired_input[1]["call_id"], "call_1");
        assert_eq!(repaired_input[3]["content"], "now apply it");
        assert_eq!(repaired["include"], json!(["reasoning.encrypted_content"]));
        assert!(rung_b.accepted.lock().expect("lock").is_empty());

        // One reservation covers both dials: the ledger sees one attempt,
        // settled completed, never a failed 400 and never a failover.
        let story = harness.story().await;
        assert_eq!(story["starts"].as_array().expect("starts").len(), 1);
        let settles = story["settles"].as_array().expect("settles");
        assert_eq!(settles.len(), 1);
        assert_eq!(settles[0]["outcome"], "completed");
        assert_eq!(story["counts"], json!([1, 0]));
        let after = METRICS.snapshot()["encrypted_reasoning_stripped"]
            .as_u64()
            .expect("counter");
        assert!(after > before);
    });
}

#[test]
fn a_second_refusal_of_the_stripped_replay_surfaces_the_providers_400() {
    block_on(async {
        let harness = Harness::new();
        let rung_a = spawn_rung(vec![
            Answer::Rejected(INVALID_ENCRYPTED_CONTENT_BODY),
            Answer::Rejected(INVALID_ENCRYPTED_CONTENT_BODY),
        ])
        .await;
        let rung_b = spawn_rung(vec![Answer::ResponsesStream(&[RESPONSES_TEXT_FRAME])]).await;
        let route = [
            responses_wire(
                "a",
                &rung_a.url,
                &["rsn_a_second_refusal_of_the_stripped_replay_surfaces_the_providers_400_hA=="],
            ),
            responses_wire(
                "b",
                &rung_b.url,
                &["rsn_a_second_refusal_of_the_stripped_replay_surfaces_the_providers_400_hA=="],
            ),
        ];
        let (won, guard) = harness.run(&route, None, Duration::from_secs(60)).await;
        let won = finish(guard, won).await;
        let Won::Failed(error) = won else {
            panic!("a client error is the caller's, never walked down the ladder");
        };
        assert_eq!(error.status_code, 400, "{error:?}");
        // Exactly one repair: the rung was dialed twice and no other rung.
        assert_eq!(rung_a.accepted.lock().expect("lock").len(), 2);
        assert!(rung_b.accepted.lock().expect("lock").is_empty());
        // Both dials ran under the one reservation, and a client error asks
        // the control plane for no successor.
        let story = harness.story().await;
        assert_eq!(story["starts"].as_array().expect("starts").len(), 1);
        let settles = story["settles"].as_array().expect("settles");
        assert_eq!(settles.len(), 1);
        assert_eq!(settles[0]["outcome"], "failed");
        assert_eq!(settles[0]["failure"]["failure_class"], "invalid_request");
    });
}

#[test]
fn the_verdict_on_a_replay_with_nothing_to_strip_surfaces_at_once() {
    block_on(async {
        let harness = Harness::new();
        let rung_a = spawn_rung(vec![Answer::Rejected(INVALID_ENCRYPTED_CONTENT_BODY)]).await;
        let route = [responses_wire("a", &rung_a.url, &[])];
        let (won, guard) = harness.run(&route, None, Duration::from_secs(60)).await;
        let won = finish(guard, won).await;
        let Won::Failed(error) = won else {
            panic!("nothing to repair: the provider's 400 is the answer");
        };
        let story = harness.story().await;
        assert_eq!(error.status_code, 400, "{error:?} {story}");
        assert_eq!(rung_a.accepted.lock().expect("lock").len(), 1);
        assert_eq!(rung_a.bodies.lock().expect("lock").len(), 1);
    });
}

#[test]
fn a_remembered_repair_is_redialed_without_earning_the_refusal_again() {
    block_on(async {
        let harness = Harness::new();
        // The rung refuses the foreign payload, throttles the stripped
        // re-dial, and serves the post-backoff redial: that redial must
        // carry the stripped payload directly, not the refused original.
        let rung_a = spawn_rung(vec![
            Answer::Rejected(INVALID_ENCRYPTED_CONTENT_BODY),
            Answer::Throttle(None),
            Answer::ResponsesStream(&[RESPONSES_TEXT_FRAME]),
        ])
        .await;
        let rung_b = spawn_rung(vec![Answer::ResponsesStream(&[RESPONSES_TEXT_FRAME])]).await;
        let route = [
            DeploymentWire {
                throttle_redial_budget: 2,
                ..responses_wire("a", &rung_a.url, &["rsn_a_remembered_repair_is_redialed_without_earning_the_refusal_again_hA=="])
            },
            responses_wire("b", &rung_b.url, &["rsn_a_remembered_repair_is_redialed_without_earning_the_refusal_again_hA=="]),
        ];
        let (won, guard) = harness
            .run(&route, Some(SCHEDULE), Duration::from_secs(60))
            .await;
        let won = finish(guard, won).await;
        let Won::Committed(committed) = won else {
            panic!("the redialed rung serves the remembered stripped payload");
        };
        assert_eq!(committed.depth, 0);
        assert!(committed.encrypted_reasoning_stripped);
        drop(committed);

        // Three dials: the replay as sent, the stripped re-dial, and the
        // post-backoff redial that starts from the stripped payload.
        let bodies = rung_a.bodies.lock().expect("lock").clone();
        assert_eq!(bodies.len(), 3);
        let has_encrypted = |body: &str| {
            let value: Value = serde_json::from_str(body).expect("body");
            value["input"]
                .as_array()
                .expect("input")
                .iter()
                .any(|item| item.get("encrypted_content").is_some())
        };
        assert!(has_encrypted(&bodies[0]));
        assert!(!has_encrypted(&bodies[1]));
        assert!(!has_encrypted(&bodies[2]));
        assert!(rung_b.accepted.lock().expect("lock").is_empty());

        // Two reservations: the first covers the refusal and its stripped
        // re-dial (settled failed on the throttle), the second is the
        // post-backoff redial of the same depth that served.
        let story = harness.story().await;
        let starts = story["starts"].as_array().expect("starts");
        assert_eq!(starts.len(), 2);
        assert_eq!(starts[1]["throttle_backoff"], true);
        assert_eq!(starts[1]["current_depth"], 0);
        assert_eq!(starts[1]["failure"]["failure_class"], "throttled");
        let settles = story["settles"].as_array().expect("settles");
        assert_eq!(settles.len(), 2);
        assert_eq!(settles[0]["outcome"], "failed");
        assert_eq!(settles[1]["outcome"], "completed");
    });
}

#[test]
fn a_remembered_refused_payload_is_stripped_before_the_first_dial() {
    block_on(async {
        // Turn one: the rung refuses the foreign payload (the verdict quotes
        // its head and tail), the stripped re-dial serves. The local payload
        // of the same conversation is not what was refused.
        let foreign = "rsn_a_remembered_refused_payload_hA==";
        let local = "gAAA_a_remembered_refused_payload_local==";
        let harness = Harness::new();
        let rung = spawn_rung(vec![
            Answer::Rejected(INVALID_ENCRYPTED_CONTENT_BODY),
            Answer::ResponsesStream(&[RESPONSES_TEXT_FRAME]),
        ])
        .await;
        let route = [responses_wire("a", &rung.url, &[foreign, local])];
        let (won, guard) = harness.run(&route, None, Duration::from_secs(60)).await;
        let Won::Committed(committed) = finish(guard, won).await else {
            panic!("the stripped re-dial serves turn one");
        };
        assert!(committed.encrypted_reasoning_stripped);
        drop(committed);
        assert_eq!(rung.bodies.lock().expect("lock").len(), 2);

        // Turn two, same caller, same history: only the remembered payload
        // is stripped, before any dial, so the rung sees exactly one dial
        // that still carries the local payload; the disclosure holds.
        let later = Harness::new();
        let rung = spawn_rung(vec![Answer::ResponsesStream(&[RESPONSES_TEXT_FRAME])]).await;
        let route = [responses_wire("a", &rung.url, &[foreign, local])];
        let (won, guard) = later.run(&route, None, Duration::from_secs(60)).await;
        let Won::Committed(committed) = finish(guard, won).await else {
            panic!("the remembered strip serves turn two");
        };
        assert!(committed.encrypted_reasoning_stripped);
        drop(committed);
        let bodies = rung.bodies.lock().expect("lock").clone();
        assert_eq!(bodies.len(), 1);
        let sent: Value = serde_json::from_str(&bodies[0]).expect("body");
        let payloads: Vec<&str> = sent["input"]
            .as_array()
            .expect("input")
            .iter()
            .filter_map(|item| item.get("encrypted_content").and_then(Value::as_str))
            .collect();
        assert_eq!(payloads, vec![local]);
        let story = later.story().await;
        assert_eq!(story["starts"].as_array().expect("starts").len(), 1);
        assert_eq!(story["settles"][0]["outcome"], "completed");

        // Another caller replaying the same payload is not affected by this
        // caller's memory: its first dial carries both payloads.
        let stranger = Harness::new();
        let rung = spawn_rung(vec![Answer::ResponsesStream(&[RESPONSES_TEXT_FRAME])]).await;
        let route = [responses_wire("a", &rung.url, &[foreign, local])];
        let (won, guard) = stranger
            .run_as("other-key", &route, None, Duration::from_secs(60))
            .await;
        let Won::Committed(committed) = finish(guard, won).await else {
            panic!("the stranger's replay serves as sent");
        };
        assert!(!committed.encrypted_reasoning_stripped);
        drop(committed);
        let body: Value =
            serde_json::from_str(&rung.bodies.lock().expect("lock")[0]).expect("body");
        assert_eq!(body["input"].as_array().expect("input").len(), 6);
    });
}
