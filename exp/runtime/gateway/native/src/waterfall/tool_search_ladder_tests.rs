//! Ladder tests for gateway-executed tool search: a scripted python control
//! plane answering `tool_search_round` with a rebuilt wire and scripted local
//! rungs drive `acquire_attempt` end to end, so the withheld search-call
//! turn, the same-depth `tool_search_round` reservation, the rebuilt body
//! dialed, the rounds handed to the winning attempt, the after-output drop,
//! the round budget, and a failed control-plane round are all observed on
//! the real loop.

use std::sync::atomic::AtomicUsize;
use std::sync::Arc;
use std::time::{Duration, Instant};

use pyo3::prelude::*;
use serde_json::{json, Value};

use super::ladder_tests::{block_on, finish, spawn_rung, wire, Answer};
use super::*;
use crate::bridge::Bridge;
use crate::relay::collect_committed;
use crate::tool_search::ToolSearchAdmission;
use crate::upstream::build_client;

/// A control plane that reserves the first dial at depth 0, a
/// `tool_search_round` re-dial at the same depth, and exhausts anything else;
/// its `tool_search_round` extends the configured wire's conversation with
/// the call and a canned result. Every call is recorded for the test.
const PLANE_SOURCE: &std::ffi::CStr = cr#"
import json
import threading


class Plane:
    """Scripted control plane for the tool-search ladder."""

    def __init__(self):
        self.lock = threading.Lock()
        self.starts = []
        self.settles = []
        self.rounds = []
        self.abandons = []
        self.wire = None
        self.fail_round = False
        self.exhaust_at = 99

    def configure(self, argument):
        data = json.loads(argument)
        with self.lock:
            self.wire = data.get("wire", self.wire)
            self.fail_round = data.get("fail_round", self.fail_round)
            self.exhaust_at = data.get("exhaust_at", self.exhaust_at)
        return "{}"

    def start_attempt(self, argument):
        data = json.loads(argument)
        with self.lock:
            self.starts.append(data)
            depth = data.get("current_depth")
            if depth is None:
                candidate = 0
            elif data.get("tool_search_round"):
                candidate = depth
            else:
                return json.dumps({"exhausted": True, "failure": data.get("failure")})
            ordinal = data["attempt_ordinal"]
            return json.dumps({"attempt_id": f"attempt-{ordinal}", "route_depth": candidate})

    def tool_search_round(self, argument):
        data = json.loads(argument)
        with self.lock:
            self.rounds.append(data)
            if self.fail_round:
                raise RuntimeError("search unavailable")
            wire = json.loads(json.dumps(self.wire))
            rounds = []
            for call in data["calls"]:
                wire["upstream_payload"]["messages"].append({
                    "role": "assistant",
                    "tool_calls": [{
                        "id": call["call_id"],
                        "type": "function",
                        "function": {"name": call["name"], "arguments": call["arguments"]},
                    }],
                })
                wire["upstream_payload"]["messages"].append({
                    "role": "tool",
                    "tool_call_id": call["call_id"],
                    "content": json.dumps({
                        "matched": [{"name": "get_weather"}, {"name": "get_forecast"}],
                        "loaded": True,
                    }),
                })
                rounds.append({
                    "call_id": call["call_id"],
                    "query": json.loads(call["arguments"]).get("query"),
                    "pattern": None,
                    "matched": ["get_weather", "get_forecast"],
                })
            exhausted = data["round"] >= self.exhaust_at
            return json.dumps({"wire": wire, "rounds": rounds, "exhausted": exhausted})

    def settle(self, argument):
        with self.lock:
            self.settles.append(json.loads(argument))
        return "{}"

    def abandon(self, argument):
        with self.lock:
            self.abandons.append(json.loads(argument))
        return "{}"

    def dump(self, argument):
        with self.lock:
            return json.dumps({
                "starts": self.starts,
                "settles": self.settles,
                "rounds": self.rounds,
                "abandons": self.abandons,
            })

    def close_thread_resources(self, argument):
        return "{}"
"#;

fn plane() -> Py<PyAny> {
    Python::initialize();
    Python::attach(|py| {
        pyo3::types::PyModule::from_code(
            py,
            PLANE_SOURCE,
            c"tool_search_plane.py",
            c"tool_search_plane",
        )
        .expect("plane module compiles")
        .getattr("Plane")
        .expect("plane class exists")
        .call0()
        .expect("plane instantiates")
        .unbind()
    })
}

/// The wire the plane rebuilds from, as the control plane would serialize
/// it; mirrors `ladder_tests::wire` for the scripted rung at `url`.
fn wire_json(url: &str) -> Value {
    json!({
        "provider": "openai",
        "deployment_id": "a",
        "dialect": "openai_compatible",
        "url": url,
        "headers": {},
        "model_id": "gpt-test",
        "timeout_seconds": 10.0,
        "upstream_payload": {
            "model": "gpt-test",
            "messages": [{"role": "user", "content": "hi"}],
            "stream": true,
        },
        "idempotency_key": "op-a",
    })
}

const SEARCH_START_FRAME: &str = concat!(
    "{\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":0,\"id\":\"call_x\",",
    "\"type\":\"function\",\"function\":{\"name\":\"tool_search\",\"arguments\":\"\"}}]}}]}"
);
const SEARCH_ARGUMENTS_FRAME: &str = concat!(
    "{\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":0,\"function\":",
    "{\"arguments\":\"{\\\"query\\\":\\\"weather\\\"}\"}}]}}]}"
);
const USAGE_FRAME: &str =
    "{\"choices\":[],\"usage\":{\"prompt_tokens\":5,\"completion_tokens\":3}}";
const ZERO_USAGE_FRAME: &str =
    "{\"choices\":[],\"usage\":{\"prompt_tokens\":5,\"completion_tokens\":0}}";
const FINISH_TOOL_CALLS_FRAME: &str =
    "{\"choices\":[{\"delta\":{},\"finish_reason\":\"tool_calls\"}]}";
const FINISH_STOP_FRAME: &str = "{\"choices\":[{\"delta\":{},\"finish_reason\":\"stop\"}]}";
const TEXT_FRAME: &str = "{\"choices\":[{\"delta\":{\"content\":\"hi\"}}]}";

/// A turn whose only output is one call to the gateway's search tool.
const SEARCH_TURN: &[&str] = &[
    SEARCH_START_FRAME,
    SEARCH_ARGUMENTS_FRAME,
    USAGE_FRAME,
    FINISH_TOOL_CALLS_FRAME,
];

struct SearchHarness {
    bridge: Arc<Bridge>,
    http: crate::upstream::UpstreamClient,
}

impl SearchHarness {
    fn new() -> Self {
        Self {
            bridge: Arc::new(Bridge::new(plane(), 2).expect("bridge starts")),
            http: build_client(Duration::from_secs(2), false).expect("client"),
        }
    }

    async fn configure(&self, configuration: Value) {
        self.bridge
            .call("configure", configuration.to_string())
            .await
            .expect("configure succeeds");
    }

    async fn run(
        &self,
        route: &[DeploymentWire],
        tool_search: &ToolSearchAdmission,
    ) -> (Won, AttemptGuard) {
        let mut guard = AttemptGuard::new(
            self.bridge.clone(),
            Arc::new(AtomicUsize::new(0)),
            "request-search".to_string(),
            Instant::now(),
        );
        let context = WaterfallContext {
            bridge: &self.bridge,
            http: &self.http,
            request_id: "request-search",
            raw_key: "key",
            caller_scope: Some("org:key-holder"),
            route,
            policy: RoutePolicy {
                maximum_total_attempts: 8,
                maximum_same_deployment_attempts: 2,
                refusal_failover: false,
                throttle_redial: None,
                physical_route_cap: None,
                backoff: None,
            },
            deadline: Instant::now() + Duration::from_secs(60),
            time_to_first_byte: Duration::from_secs(5),
            time_to_first_byte_slope_seconds_per_million_input_tokens: 0.0,
            time_to_first_token: Duration::from_secs(120),
            approximate_input_tokens: 10.0,
            chat_logprobs: false,
            output_less_retention: None,
            output_token_cap: None,
            tool_search: Some(tool_search),
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

fn admission(max_rounds: u32) -> ToolSearchAdmission {
    ToolSearchAdmission {
        tool_name: "tool_search".to_string(),
        max_rounds,
        deferred: 42,
        surface_shape: "openrouter".to_string(),
    }
}

#[test]
fn a_search_call_turn_runs_a_round_and_the_same_rung_serves_the_answer() {
    block_on(async {
        let harness = SearchHarness::new();
        // First turn: the model searches. Second turn (the rebuilt wire):
        // it answers.
        let rung = spawn_rung(vec![
            Answer::Stream(SEARCH_TURN),
            Answer::Stream(&[TEXT_FRAME]),
        ])
        .await;
        harness
            .configure(json!({"wire": wire_json(&rung.url)}))
            .await;
        let route = [wire("a", &rung.url, 0)];
        let (won, guard) = harness.run(&route, &admission(3)).await;
        let won = finish(guard, won).await;
        let Won::Committed(committed) = won else {
            panic!("the rung serves the answer after the round");
        };
        assert_eq!(committed.depth, 0);
        assert!(matches!(committed.prefix.first(), Some(Event::TextDelta(text)) if text == "hi"));
        assert!(!committed.tool_search_dropped_after_output);
        assert_eq!(committed.tool_search_rounds.len(), 1);
        assert_eq!(committed.tool_search_rounds[0].call_id, "call_x");
        assert_eq!(
            committed.tool_search_rounds[0].query.as_deref(),
            Some("weather")
        );
        assert_eq!(
            committed.tool_search_rounds[0].matched,
            vec!["get_weather".to_string(), "get_forecast".to_string()]
        );
        drop(committed);

        // Two physical dials; the second carried the extended conversation
        // the control plane rebuilt, not the original payload.
        let bodies = rung.bodies.lock().expect("lock").clone();
        assert_eq!(bodies.len(), 2);
        assert!(!bodies[0].contains("tool_call_id"));
        assert!(bodies[1].contains("\"tool_call_id\":\"call_x\""));
        assert!(bodies[1].contains("\"name\":\"tool_search\""));

        let story = harness.story().await;
        let starts = story["starts"].as_array().expect("starts");
        assert_eq!(starts.len(), 2);
        assert_eq!(starts[0]["tool_search_round"], false);
        assert_eq!(starts[0]["throttle_backoff"], false);
        assert_eq!(starts[1]["attempt_ordinal"], 1);
        assert_eq!(starts[1]["current_depth"], 0);
        assert_eq!(starts[1]["tool_search_round"], true);
        assert_eq!(starts[1]["throttle_backoff"], false);
        assert_eq!(starts[1]["failure"], Value::Null);
        let rounds = story["rounds"].as_array().expect("rounds");
        assert_eq!(rounds.len(), 1);
        assert_eq!(rounds[0]["request_id"], "request-search");
        assert_eq!(rounds[0]["route_depth"], 0);
        assert_eq!(rounds[0]["round"], 1);
        assert_eq!(rounds[0]["calls"][0]["call_id"], "call_x");
        assert_eq!(rounds[0]["calls"][0]["name"], "tool_search");
        assert_eq!(
            rounds[0]["calls"][0]["arguments"],
            "{\"query\":\"weather\"}"
        );
        assert_eq!(rounds[0]["usage"]["input_tokens"], 5);
        assert_eq!(rounds[0]["usage"]["output_tokens"], 3);
        // The search-call turn settled completed without finalizing and
        // never billed the round; the serving attempt's finalizing
        // settlement bills it once.
        let settles = story["settles"].as_array().expect("settles");
        assert_eq!(settles.len(), 2);
        assert_eq!(settles[0]["attempt_id"], "attempt-0");
        assert_eq!(settles[0]["outcome"], "completed");
        assert_eq!(settles[0]["finalize"], false);
        assert_eq!(settles[0]["usage"]["input_tokens"], 5);
        assert!(settles[0].get("tool_search_requests").is_none());
        assert_eq!(settles[1]["attempt_id"], "attempt-1");
        assert_eq!(settles[1]["outcome"], "completed");
        assert_eq!(settles[1]["finalize"], true);
        assert_eq!(settles[1]["tool_search_requests"], 1);
        assert!(story["abandons"].as_array().expect("abandons").is_empty());
    });
}

#[test]
fn private_reasoning_does_not_hide_a_search_round() {
    block_on(async {
        let harness = SearchHarness::new();
        let rung = spawn_rung(vec![
            Answer::Stream(&[
                "{\"choices\":[{\"delta\":{\"reasoning_content\":\"private thought\"}}]}",
                SEARCH_START_FRAME,
                SEARCH_ARGUMENTS_FRAME,
                USAGE_FRAME,
                FINISH_TOOL_CALLS_FRAME,
            ]),
            Answer::Stream(&[TEXT_FRAME]),
        ])
        .await;
        harness
            .configure(json!({"wire": wire_json(&rung.url)}))
            .await;
        let mut deployment = wire("a", &rung.url, 0);
        deployment.fireworks_reasoning_route_sha256 = Some("a".repeat(64));
        let (won, guard) = harness.run(&[deployment], &admission(3)).await;
        let Won::Committed(committed) = finish(guard, won).await else {
            panic!("search must reach the answer turn");
        };
        assert_eq!(committed.tool_search_rounds.len(), 1);
        assert!(matches!(committed.prefix.first(), Some(Event::TextDelta(text)) if text == "hi"));
        assert_eq!(harness.story().await["rounds"].as_array().unwrap().len(), 1);
    });
}

#[test]
fn a_round_followed_by_an_output_less_turn_settles_with_its_rounds() {
    block_on(async {
        let harness = SearchHarness::new();
        let rung = spawn_rung(vec![
            Answer::Stream(SEARCH_TURN),
            Answer::Stream(&[ZERO_USAGE_FRAME, FINISH_STOP_FRAME]),
        ])
        .await;
        harness
            .configure(json!({"wire": wire_json(&rung.url)}))
            .await;
        let route = [wire("a", &rung.url, 0)];
        let (won, _guard) = harness.run(&route, &admission(3)).await;
        let Won::Settled(settled) = won else {
            panic!("a zero-token stop after the round settles output-less");
        };
        assert_eq!(settled.depth, 0);
        assert_eq!(settled.tool_search_rounds.len(), 1);
        assert!(matches!(settled.events.last(), Some(Event::Completed)));
        let story = harness.story().await;
        let settles = story["settles"].as_array().expect("settles");
        assert_eq!(settles.len(), 2);
        assert_eq!(settles[1]["finalize"], true);
        assert_eq!(settles[1]["tool_search_requests"], 1);
    });
}

#[test]
fn a_search_call_beside_other_output_is_dropped_and_the_rung_commits() {
    block_on(async {
        let harness = SearchHarness::new();
        // The model starts a search call, then answers in the same turn: the
        // text commits the rung and the call (completed at the finish chunk)
        // never reaches the caller.
        let rung = spawn_rung(vec![Answer::Stream(&[
            SEARCH_START_FRAME,
            SEARCH_ARGUMENTS_FRAME,
            TEXT_FRAME,
            USAGE_FRAME,
            FINISH_STOP_FRAME,
        ])])
        .await;
        harness
            .configure(json!({"wire": wire_json(&rung.url)}))
            .await;
        let route = [wire("a", &rung.url, 0)];
        let (won, guard) = harness.run(&route, &admission(3)).await;
        let won = finish(guard, won).await;
        let Won::Committed(mut committed) = won else {
            panic!("the text commits the rung");
        };
        assert!(committed.tool_search_dropped_after_output);
        assert!(committed.tool_search_rounds.is_empty());
        let events = collect_committed(
            &mut committed,
            Instant::now() + Duration::from_secs(30),
            Duration::from_secs(5),
            Instant::now(),
        )
        .await
        .expect("the committed stream drains");
        assert!(
            !events.iter().any(|event| matches!(
                event,
                Event::ToolCallStarted { .. }
                    | Event::ToolArgumentsDelta { .. }
                    | Event::ToolCallCompleted { .. }
            )),
            "the dropped call never reaches the caller: {events:?}"
        );
        assert!(matches!(events.first(), Some(Event::TextDelta(text)) if text == "hi"));
        assert_eq!(committed.relay.withheld_search_call_count(), 1);
        drop(committed);
        let story = harness.story().await;
        assert_eq!(story["starts"].as_array().expect("starts").len(), 1);
        assert!(story["rounds"].as_array().expect("rounds").is_empty());
        assert_eq!(rung.bodies.lock().expect("lock").len(), 1);
    });
}

#[test]
fn the_round_budget_fails_closed_once_exhausted() {
    block_on(async {
        let harness = SearchHarness::new();
        // The control plane marks round 1 as exhausted (the tool withdrawn),
        // yet the model calls the search tool again: no answer exists and
        // the request fails closed rather than looping or half-answering.
        let rung = spawn_rung(vec![
            Answer::Stream(SEARCH_TURN),
            Answer::Stream(SEARCH_TURN),
        ])
        .await;
        harness
            .configure(json!({"wire": wire_json(&rung.url), "exhaust_at": 1}))
            .await;
        let route = [wire("a", &rung.url, 0)];
        let (won, _guard) = harness.run(&route, &admission(3)).await;
        let Won::Failed(error) = won else {
            panic!("a search call past the budget fails the request closed");
        };
        assert_eq!(error.status_code, 500);
        let story = harness.story().await;
        assert_eq!(story["rounds"].as_array().expect("rounds").len(), 1);
        assert_eq!(story["starts"].as_array().expect("starts").len(), 2);
        let settles = story["settles"].as_array().expect("settles");
        assert_eq!(settles.len(), 2);
        assert_eq!(settles[0]["outcome"], "completed");
        assert_eq!(settles[0]["finalize"], false);
        assert_eq!(settles[1]["outcome"], "failed");
        assert_eq!(settles[1]["finalize"], true);
        assert_eq!(settles[1]["failure"]["failure_class"], "internal");
        assert_eq!(rung.bodies.lock().expect("lock").len(), 2);

        // The rust-side budget alone (max_rounds 1) fails the second search
        // call the same way without the control plane's help.
        let strict = SearchHarness::new();
        let rung = spawn_rung(vec![
            Answer::Stream(SEARCH_TURN),
            Answer::Stream(SEARCH_TURN),
        ])
        .await;
        strict
            .configure(json!({"wire": wire_json(&rung.url)}))
            .await;
        let route = [wire("a", &rung.url, 0)];
        let (won, _guard) = strict.run(&route, &admission(1)).await;
        assert!(matches!(won, Won::Failed(error) if error.status_code == 500));
        let story = strict.story().await;
        assert_eq!(story["rounds"].as_array().expect("rounds").len(), 1);
    });
}

#[test]
fn a_failed_round_callback_fails_the_request_closed_with_the_gateways_error() {
    block_on(async {
        let harness = SearchHarness::new();
        let rung = spawn_rung(vec![Answer::Stream(SEARCH_TURN)]).await;
        harness
            .configure(json!({"wire": wire_json(&rung.url), "fail_round": true}))
            .await;
        let route = [wire("a", &rung.url, 0)];
        let (won, _guard) = harness.run(&route, &admission(3)).await;
        let Won::Failed(error) = won else {
            panic!("a control plane that cannot run the round fails the request");
        };
        assert_eq!(error.status_code, 500);
        assert_eq!(error.code, "internal_error");
        let story = harness.story().await;
        // The round was asked for while the search-call attempt was still
        // open, so the failure settles THAT attempt as the request's
        // finalizing failure and the metered searches ride along; no
        // attempt-less abandon ever happens.
        let settles = story["settles"].as_array().expect("settles");
        assert_eq!(settles.len(), 1);
        assert_eq!(settles[0]["outcome"], "failed");
        assert_eq!(settles[0]["finalize"], true);
        assert_eq!(settles[0]["failure"]["failure_class"], "internal");
        assert!(settles[0].get("tool_search_requests").is_none());
        let abandons = story["abandons"].as_array().expect("abandons");
        assert_eq!(abandons.len(), 0);
        assert_eq!(story["starts"].as_array().expect("starts").len(), 1);
        assert_eq!(rung.bodies.lock().expect("lock").len(), 1);
    });
}
