use super::{input::InputGate, inspection::Check, resources::Resources};
use serde_json::{json, Value};
use std::io::{Read, Write};
use std::net::TcpListener;
use std::sync::mpsc;
use std::time::{Duration, Instant};

#[test]
fn cancelling_an_approved_input_preserves_its_metered_terminal() {
    let gate = InputGate::start(vec![], 30.0).unwrap();
    assert_eq!(gate.wait()["action"], "allow");
    gate.cancel(None);
    gate.cancel(None);
    assert_eq!(gate.decision()["action"], "allow");
    let resources = Resources::new(Some(gate), None, None, "request").unwrap();
    let mut terminal = json!({"state": "cancelled", "usage": {"input_tokens": 100},
        "failure": {"failure_class": "cancelled"}});
    let expected = terminal.clone();
    resources.terminal(&mut terminal);
    assert_eq!(terminal, expected);
}

#[test]
fn explicit_inspection_failure_remains_immutable_after_cancellation() {
    let gate = InputGate::start(vec![], 30.0).unwrap();
    assert_eq!(gate.wait()["action"], "allow");
    gate.cancel(Some(
        json!({"failure_class": "unavailable", "safe_message": "inspection failed",
        "safe_details": {"code": "inspection_unavailable"}}),
    ));
    let decision = gate.decision();
    gate.cancel(None);
    assert_eq!(gate.decision(), decision);
    assert_eq!(
        decision["failure"]["safe_details"]["input_guardrail_denied"],
        true
    );
    assert_eq!(
        decision["failure"]["safe_details"]["code"],
        "inspection_unavailable"
    );
}

#[test]
fn read_only_input_programs_match_python_order_and_coverage() {
    for (i, line) in include_str!("../../tests/generated/input.jsonl")
        .lines()
        .enumerate()
    {
        let case: Value = crate::lossless_json::from_str(line).unwrap();
        let gate = InputGate::start(
            crate::lossless_json::from_value(case["checks"].clone()).unwrap(),
            30.0,
        )
        .unwrap();
        let actual = gate.wait();
        assert_eq!(actual["action"], case["expected"]["action"], "case {i}");
        for key in ["failure_class", "safe_message", "safe_details"] {
            assert_eq!(
                actual["failure"][key], case["expected"]["failure"][key],
                "case {i} field {key}"
            );
        }
    }
}

#[test]
fn decision_category_scores_match_python_fsum_without_threshold_drift() {
    for (i, line) in include_str!("../../tests/generated/input_scores.jsonl")
        .lines()
        .enumerate()
    {
        let case: Value = crate::lossless_json::from_str(line).unwrap();
        assert_eq!(
            serde_json::to_value(
                super::input_decisions::score(&case["adapter"], &case["response"]).unwrap()
            )
            .unwrap(),
            case["expected"],
            "case {i}"
        );
    }
    for probability in [
        json!(null),
        json!(true),
        json!("0.9"),
        json!(-0.01),
        json!(1.001),
    ] {
        assert!(super::input_decisions::score(
            &json!({"categories": {"category": {"questions": {"q": {}}, "aggregate": "max"}}}),
            &json!({"answers": {"q": {"noul": probability}}})
        )
        .is_err());
    }
}

pub(super) fn server(
    body: &'static str,
    status: u16,
) -> (
    String,
    mpsc::Receiver<Value>,
    mpsc::Sender<()>,
    std::thread::JoinHandle<()>,
) {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    listener.set_nonblocking(true).unwrap();
    let url = format!("http://{}/input", listener.local_addr().unwrap());
    let (sent, received) = mpsc::channel();
    let (release, proceed) = mpsc::channel();
    let thread = std::thread::spawn(move || {
        let deadline = Instant::now() + Duration::from_secs(5);
        let mut stream = loop {
            if let Ok((stream, _)) = listener.accept() {
                break stream;
            }
            assert!(
                Instant::now() < deadline,
                "native classifier did not connect"
            );
            std::thread::sleep(Duration::from_millis(1));
        };
        stream.set_nonblocking(false).unwrap();
        stream
            .set_read_timeout(Some(Duration::from_secs(3)))
            .unwrap();
        let mut bytes = Vec::new();
        let end = loop {
            let mut buffer = [0; 512];
            let n = stream.read(&mut buffer).unwrap();
            assert_ne!(n, 0);
            bytes.extend_from_slice(&buffer[..n]);
            if let Some(end) = bytes.windows(4).position(|v| v == b"\r\n\r\n") {
                break end + 4;
            }
        };
        let headers = String::from_utf8(bytes[..end].to_vec())
            .unwrap()
            .to_lowercase();
        assert!(!headers.contains("\r\ncookie:"));
        let length: usize = headers
            .lines()
            .find_map(|line| line.strip_prefix("content-length: "))
            .unwrap()
            .parse()
            .unwrap();
        while bytes.len() < end + length {
            let mut buffer = [0; 512];
            let n = stream.read(&mut buffer).unwrap();
            assert_ne!(n, 0);
            bytes.extend_from_slice(&buffer[..n]);
        }
        sent.send(crate::lossless_json::from_slice(&bytes[end..end + length]).unwrap())
            .unwrap();
        proceed.recv_timeout(Duration::from_secs(5)).unwrap();
        let _ = write!(stream, "HTTP/1.1 {status} Test\r\nContent-Length: {}\r\nContent-Type: application/json\r\nConnection: close\r\n\r\n{body}", body.len());
    });
    (url, received, release, thread)
}

pub(super) fn check(url: &str, protected: bool) -> Check {
    crate::lossless_json::from_value(json!({"policy": {"protected": protected},
        "check": {"check_id": "check", "action": "block", "timeout_ms": 3000},
        "adapter": {"kind": "http_json", "url": url, "max_response_bytes": 1024,
            "transport": {"public_only": false, "headers": {}},
            "payload": {"stage": "input", "request": {"messages": [{"content": "fixture"}]}}}}))
    .unwrap()
}

#[tokio::test]
async fn asynchronous_verdict_wakes_every_waiter_and_denial_survives_release() {
    let (url, received, release, server) = server("{\"flagged\":true}", 200);
    let mut check = check(&url, true);
    check.policy["input_fail_open"] = json!(true);
    let gate = InputGate::start(vec![check], 5.0).unwrap();
    received.recv_timeout(Duration::from_secs(3)).unwrap();
    let first = gate.clone();
    let second = gate.clone();
    let first = tokio::spawn(async move { first.wait_async().await });
    let second = tokio::spawn(async move { second.wait_async().await });
    release.send(()).unwrap();
    for waiter in [first, second] {
        let decision = tokio::time::timeout(Duration::from_secs(2), waiter)
            .await
            .unwrap()
            .unwrap();
        assert_eq!(decision["action"], "error");
        assert_eq!(decision["moderation"]["status"], "denied");
    }
    gate.release_incomplete("response_ready_timeout");
    assert_eq!(gate.decision()["action"], "error");
    server.join().unwrap();
}

#[test]
fn expired_strict_inspection_records_its_unavailable_decision() {
    let mut check = check("http://127.0.0.1:1/inspect", true);
    check.policy["input_fail_open"] = json!(false);
    let gate = InputGate::start(vec![check], 0.0).unwrap();
    let decision = gate.wait();
    assert_eq!(decision["action"], "error");
    assert_eq!(decision["moderation"]["status"], "unavailable");
    assert_eq!(decision["moderation"]["observations"][0]["action"], "error");
}

#[tokio::test]
async fn response_timeout_releases_and_cancels_moderation_without_waiving_provider_usage() {
    let (url, received, release, server) = server("{\"flagged\":true}", 200);
    let mut check = check(&url, true);
    check.policy["input_fail_open"] = json!(true);
    check.policy["input_max_wait_ms"] = json!(0);
    let gate = InputGate::start(vec![check], 5.0).unwrap();
    received.recv_timeout(Duration::from_secs(3)).unwrap();
    assert_eq!(gate.grace(), Some(Duration::ZERO));
    let resources = Resources::new(Some(gate.clone()), None, None, "request").unwrap();
    resources.release_incomplete();
    let decision = tokio::time::timeout(Duration::from_millis(100), resources.wait())
        .await
        .unwrap();
    assert_eq!(decision["action"], "allow");
    assert_eq!(decision["moderation"]["status"], "failed_open");
    release.send(()).unwrap();
    server.join().unwrap();
    assert_eq!(gate.decision(), decision);
    let mut terminal = json!({"state": "completed", "usage": {"input_tokens": 11}});
    resources.terminal(&mut terminal);
    assert_eq!(terminal["state"], "completed");
    assert_eq!(terminal["usage"]["input_tokens"], 11);
    assert!(terminal["input_guardrail_denied"].is_null());
    assert_eq!(terminal["input_moderation"]["status"], "failed_open");
}

#[tokio::test]
async fn unavailable_classifier_and_observed_denial_cannot_refuse_delivery() {
    for observe in [false, true] {
        let (url, received, release, server) =
            server("{\"flagged\":true}", if observe { 200 } else { 503 });
        let mut check = check(&url, true);
        check.policy["input_fail_open"] = json!(true);
        check.policy["mode"] = json!(if observe { "observe" } else { "enforce" });
        let input = InputGate::start(vec![check], 5.0).unwrap();
        received.recv_timeout(Duration::from_secs(3)).unwrap();
        let resources = Resources::new(Some(input.clone()), None, None, "request").unwrap();
        if observe {
            assert_eq!(resources.wait().await["action"], "allow");
            resources.close("request");
            assert_eq!(input.decision()["action"], "pending");
        }
        release.send(()).unwrap();
        let decision = input.wait_async().await;
        assert_eq!(decision["action"], "allow");
        assert_eq!(
            decision["moderation"]["status"],
            if observe { "observed" } else { "failed_open" }
        );
        assert_eq!(
            decision["moderation"]["observations"][0]["outcome"],
            if observe { "flagged" } else { "unavailable" }
        );
        server.join().unwrap();
    }
}

#[test]
fn input_overlaps_dispatch_without_the_gil_and_cancellation_cannot_be_rescued() {
    for cancel in [false, true] {
        let (url, received, release, server) = server("{\"flagged\":false}", 200);
        pyo3::Python::initialize();
        pyo3::Python::attach(|_| {
            let gate = InputGate::start(vec![check(&url, true)], 5.0).unwrap();
            assert_eq!(
                received.recv_timeout(Duration::from_secs(3)).unwrap()["stage"],
                "input"
            );
            assert_eq!(gate.decision()["action"], "pending");
            if cancel {
                gate.cancel(None);
            }
            release.send(()).unwrap();
            let (tx, rx) = mpsc::channel();
            let worker_gate = gate.clone();
            let worker = std::thread::spawn(move || {
                tx.send(worker_gate.wait()).unwrap();
            });
            let result = rx
                .recv_timeout(Duration::from_secs(4))
                .expect("native input must never need the GIL");
            assert_eq!(result["action"], if cancel { "error" } else { "allow" });
            if cancel {
                assert_eq!(
                    result["failure"]["safe_details"]["input_guardrail_denied"],
                    true
                );
            }
            worker.join().unwrap();
            server.join().unwrap();
            assert_eq!(gate.decision(), result);
        });
    }
}

#[test]
fn decision_http_requires_exact_200_and_strict_json_and_reuses_definitive_scores() {
    for (body, status, expected) in [
        ("{\"answers\":{\"q\":{\"noul\":0.7}}}", 200, "allow"),
        ("{\"answers\":{\"q\":{\"noul\":0.7}}}", 201, "error"),
        (
            "{\"answers\":{\"q\":{\"noul\":0.1,\"noul\":0.9}}}",
            200,
            "error",
        ),
        ("{\"answers\":{\"q\":{\"noul\":NaN}}}", 200, "error"),
        ("{\"answers\":{\"q\":{\"noul\":true}}}", 200, "error"),
    ] {
        let (url, received, release, server) = server(body, status);
        let mut c = check(&url, true);
        c.adapter["kind"] = json!("decision_model");
        c.adapter["timeout_ms"] = json!(3000);
        c.adapter["cache"] = json!({"max_entries": 2, "ttl_seconds": 60});
        c.adapter["cache_scope"] = json!(url);
        c.adapter["cache_key"] = json!("fixture");
        c.adapter["categories"] =
            json!({"category": {"questions": {"q": {}}, "aggregate": "mean", "threshold": 0.8}});
        let gate = InputGate::start(vec![c.clone()], 5.0).unwrap();
        received.recv_timeout(Duration::from_secs(3)).unwrap();
        release.send(()).unwrap();
        assert_eq!(gate.wait()["action"], expected);
        server.join().unwrap();
        if expected == "allow" {
            // The listener is gone: a cache miss would fail closed.
            assert_eq!(
                InputGate::start(vec![c], 5.0).unwrap().wait()["action"],
                "allow"
            );
        }
    }
}

#[test]
fn an_unapproved_terminal_preserves_usage_while_explicitly_waiving_liability() {
    let (url, received, release, server) = server("{\"flagged\":false}", 200);
    let input = InputGate::start(vec![check(&url, true)], 5.0).unwrap();
    received.recv_timeout(Duration::from_secs(3)).unwrap();
    let resources = Resources::new(Some(input.clone()), None, None, "request").unwrap();
    let mut terminal = json!({"state": "cancelled", "usage": {"input_tokens": 11, "output_tokens": 5}, "usage_estimated": true});
    resources.terminal(&mut terminal);
    assert_eq!(terminal["input_guardrail_denied"], true);
    assert_eq!(
        terminal["usage"],
        json!({"input_tokens": 11, "output_tokens": 5})
    );
    assert_eq!(terminal["usage_estimated"], true);
    resources.close("request");
    release.send(()).unwrap();
    server.join().unwrap();
    assert_eq!(resources.status()["action"], "closed");
    assert_ne!(input.wait()["action"], "allow");
}

struct CaptureSink(mpsc::Sender<String>);
impl crate::capture::delivery::Sink for CaptureSink {
    type Prepared = String;
    fn preparation_bytes(maximum: usize) -> usize {
        maximum
    }
    fn prepare(
        &self,
        record: &crate::capture::record::Record,
        maximum: usize,
    ) -> Result<String, ()> {
        record.encode(maximum).ok_or(())
    }
    fn write(&mut self, prepared: &String) -> Result<(), ()> {
        self.0.send(prepared.clone()).map_err(|_| ())
    }
}

fn captured(id: &str) -> Vec<u8> {
    serde_json::to_vec(&json!({"request_id": id, "scope": {"organization_id": "org", "identity_id": "identity", "application_id": "application"},
        "protocol": "chat_completions", "model_id": "model",
        "context": {"schema_version": 1, "request": {"messages": [{"role": "user", "content": "fixture"}]}}})).unwrap()
}

#[test]
fn capture_is_registered_once_after_approval_and_never_after_close_or_capacity_failure() {
    use crate::capture::{admission, collector::Collector};
    use std::sync::Arc;
    for mode in ["allow", "closed", "full"] {
        let (sent, received) = mpsc::channel();
        let config = crate::lossless_json::from_value(json!({"delivery": {"maximum_records": 8, "maximum_bytes": 65536, "maximum_record_bytes": 8192},
            "maximum_pending_records": 1, "maximum_pending_bytes": 65536, "maximum_request_bytes": 4096,
            "maximum_response_bytes": 4096, "ttl_seconds": 30, "settlement_required": true, "asynchronous_delivery": true})).unwrap();
        let collector = Arc::new(Collector::new(config, CaptureSink(sent)).unwrap());
        let input = InputGate::start(vec![], 5.0).unwrap();
        assert_eq!(input.wait()["action"], "allow");
        let resources = Resources::new(
            Some(input),
            Some(collector.clone()),
            Some(captured("request")),
            "request",
        )
        .unwrap();
        assert_eq!(
            super::inspection_http::runtime()
                .unwrap()
                .block_on(resources.wait_for_tools())["action"],
            "allow"
        );
        assert!(resources.pending_capture.lock().unwrap().is_some());
        assert!(collector.attach("request").is_none());
        if mode == "full" {
            assert!(admission::begin(&collector, &captured("occupied")));
        }
        if mode == "closed" {
            resources.close("request");
        }
        let result = resources.status();
        assert_eq!(
            result["action"],
            if mode == "closed" { "closed" } else { "allow" }
        );
        assert_eq!(resources.status(), result);
        if mode == "full" {
            let mut terminal = json!({});
            resources.terminal(&mut terminal);
            assert_ne!(terminal["input_guardrail_denied"], true);
        }
        collector.settle("occupied", false, false);
        resources.settled(
            "request",
            &json!({"capture": {"keep_prompt": true, "keep_response": false}}),
        );
        assert!(collector.close_until(Instant::now() + Duration::from_secs(3)));
        let records: Vec<_> = received.try_iter().collect();
        assert_eq!(records.len(), usize::from(mode == "allow"));
        if mode == "allow" {
            let record: Value = crate::lossless_json::from_str(&records[0]).unwrap();
            assert_eq!(record["request"]["request_id"], "request");
        }
    }
}

#[test]
fn capture_activation_racing_request_close_never_registers_twice() {
    use crate::capture::collector::Collector;
    use std::sync::{Arc, Barrier};
    for _ in 0..32 {
        let (sent, received) = mpsc::channel();
        let config = crate::lossless_json::from_value(json!({"delivery": {"maximum_records": 8, "maximum_bytes": 65536, "maximum_record_bytes": 8192},
            "maximum_pending_records": 1, "maximum_pending_bytes": 65536, "maximum_request_bytes": 4096,
            "maximum_response_bytes": 4096, "ttl_seconds": 30, "settlement_required": true, "asynchronous_delivery": true})).unwrap();
        let collector = Arc::new(Collector::new(config, CaptureSink(sent)).unwrap());
        let input = InputGate::start(vec![], 5.0).unwrap();
        assert_eq!(input.wait()["action"], "allow");
        let resources = Resources::new(
            Some(input),
            Some(collector.clone()),
            Some(captured("request")),
            "request",
        )
        .unwrap();
        let start = Barrier::new(3);
        std::thread::scope(|scope| {
            let activate = scope.spawn(|| {
                start.wait();
                resources.status()
            });
            let close = scope.spawn(|| {
                start.wait();
                resources.close("request");
            });
            start.wait();
            let _ = activate.join().unwrap();
            close.join().unwrap();
        });
        assert_eq!(resources.status()["action"], "closed");
        assert!(resources.pending_capture.lock().unwrap().is_none());
        resources.settled(
            "request",
            &json!({"capture": {"keep_prompt": true, "keep_response": false}}),
        );
        assert!(collector.close_until(Instant::now() + Duration::from_secs(3)));
        assert!(received.try_iter().count() <= 1);
    }
}

#[tokio::test]
async fn successor_input_keeps_all_predecessors_live_and_races_their_denials() {
    for denied in [0, 1] {
        let mut gates = Vec::new();
        let mut releases = Vec::new();
        let mut servers = Vec::new();
        for _ in 0..2 {
            let (url, received, release, server) = server("{\"flagged\":true}", 200);
            let mut check = check(&url, true);
            check.policy["input_fail_open"] = json!(true);
            check.policy["input_max_wait_ms"] = json!(0);
            let gate = InputGate::start(vec![check], 5.0).unwrap();
            received.recv_timeout(Duration::from_secs(2)).unwrap();
            gates.push(gate);
            releases.push(release);
            servers.push(server);
        }
        let resources = Resources::new(Some(gates[0].clone()), None, None, "request").unwrap();
        resources.stage_reinspection(gates[1].clone()).unwrap();
        let mut terminal = json!({"state":"completed", "finalize":false});
        resources.terminal(&mut terminal);
        resources.activate_reinspection().unwrap();
        assert_eq!(resources.status()["action"], "pending");
        assert_eq!(gates[0].decision()["action"], "pending");
        releases[denied].send(()).unwrap();
        let decision = tokio::time::timeout(Duration::from_secs(1), resources.wait())
            .await
            .unwrap();
        assert_eq!(decision["action"], "error");
        assert_eq!(gates[1 - denied].decision()["action"], "pending");
        releases[1 - denied].send(()).unwrap();
        for server in servers {
            server.join().unwrap();
        }
    }
}

#[tokio::test]
async fn one_response_release_expires_every_pending_physical_input() {
    let mut gates = Vec::new();
    let mut servers = Vec::new();
    for _ in 0..2 {
        let (url, received, release, server) = server("{\"flagged\":true}", 200);
        let mut check = check(&url, true);
        check.policy["input_fail_open"] = json!(true);
        let gate = InputGate::start(vec![check], 5.0).unwrap();
        received.recv_timeout(Duration::from_secs(2)).unwrap();
        gates.push(gate);
        servers.push((release, server));
    }
    let resources = Resources::new(Some(gates[0].clone()), None, None, "request").unwrap();
    resources.stage_reinspection(gates[1].clone()).unwrap();
    resources.activate_reinspection().unwrap();
    assert_eq!(resources.grace(), Some(Duration::ZERO));
    resources.release_incomplete();
    let decision = tokio::time::timeout(Duration::from_millis(100), resources.wait())
        .await
        .unwrap();
    assert_eq!(decision["action"], "allow");
    assert_eq!(decision["moderation"]["status"], "failed_open");
    assert_eq!(
        decision["moderation"]["observations"]
            .as_array()
            .unwrap()
            .len(),
        2
    );
    for gate in gates {
        assert_eq!(
            gate.decision()["moderation"]["reason"],
            "response_ready_timeout"
        );
    }
    for (release, server) in servers {
        release.send(()).unwrap();
        server.join().unwrap();
    }
}

#[tokio::test]
async fn one_policy_cannot_extend_another_policys_response_grace() {
    let (url, received, release, server) = server("{\"flagged\":false}", 200);
    let mut slow = check(&url, true);
    slow.policy["input_fail_open"] = json!(true);
    slow.policy["input_max_wait_ms"] = json!(1000);
    slow.policy["policy_id"] = json!("long-grace");
    let mut zero = slow.clone();
    zero.policy["policy_id"] = json!("zero-grace");
    zero.policy["input_max_wait_ms"] = json!(0);
    zero.adapter = json!({"kind":"keyword","subjects":["denied"],"needles":["denied"]});
    let gate = InputGate::start(vec![slow, zero], 5.0).unwrap();
    received.recv_timeout(Duration::from_secs(2)).unwrap();
    gate.begin_response(Instant::now());
    release.send(()).unwrap();
    let decision = gate.wait_async().await;
    assert_eq!(decision["action"], "allow");
    assert_eq!(
        decision["moderation"]["observations"][0]["outcome"],
        "allow"
    );
    assert_eq!(
        decision["moderation"]["observations"][1]["outcome"],
        "response_ready_timeout"
    );
    server.join().unwrap();
}

#[test]
fn cancellation_keeps_completed_scores_and_each_remaining_policys_failure_mode() {
    let (first_url, first_received, first_release, first_server) =
        server("{\"answers\":{\"q\":{\"noul\":0.25}}}", 200);
    let (second_url, second_received, second_release, second_server) =
        server("{\"flagged\":true}", 200);
    let mut first = check(&first_url, true);
    first.policy["input_fail_open"] = json!(true);
    first.adapter["kind"] = json!("decision_model");
    first.adapter["timeout_ms"] = json!(3000);
    first.adapter["cache"] = json!({"max_entries":2,"ttl_seconds":60});
    first.adapter["cache_scope"] = json!(first_url);
    first.adapter["cache_key"] = json!("fixture-progress");
    first.adapter["categories"] =
        json!({"category":{"questions":{"q":{}},"aggregate":"mean","threshold":0.8}});
    let mut second = check(&second_url, true);
    second.policy["input_fail_open"] = json!(true);
    let mut strict = check("http://127.0.0.1:1/strict", true);
    strict.adapter = json!({"kind":"allow"});
    strict.policy["input_fail_open"] = json!(false);
    let gate = InputGate::start(vec![first, second, strict], 5.0).unwrap();
    first_received.recv_timeout(Duration::from_secs(2)).unwrap();
    first_release.send(()).unwrap();
    second_received
        .recv_timeout(Duration::from_secs(2))
        .unwrap();
    gate.cancel(None);
    let resources = Resources::new(Some(gate), None, None, "request").unwrap();
    let decision = resources.status();
    assert_eq!(decision["moderation"]["status"], "failed_open");
    assert_eq!(decision["moderation"]["reason"], "cancelled");
    let records = &decision["moderation"]["observations"];
    assert_eq!(records[1]["outcome"], "allow");
    assert_eq!(records[1]["scores"], json!([["category", 0.25]]));
    assert_eq!(records[1]["cached"], false);
    assert_eq!(records[2]["outcome"], "cancelled");
    assert_eq!(records[2]["action"], "allow");
    assert_eq!(records[0]["action"], "allow");
    second_release.send(()).unwrap();
    first_server.join().unwrap();
    second_server.join().unwrap();
}

#[tokio::test]
async fn mixed_observation_never_delays_enforcement_or_response_release() {
    let (url, received, release, server) = server("{\"flagged\":true}", 200);
    let mut observed = check(&url, true);
    observed.policy["mode"] = json!("observe");
    let mut enforced = observed.clone();
    enforced.policy["mode"] = json!("enforce");
    enforced.policy["input_fail_open"] = json!(true);
    enforced.adapter = json!({"kind":"allow"});
    let gate = InputGate::start(vec![observed, enforced], 5.0).unwrap();
    received.recv_timeout(Duration::from_secs(2)).unwrap();
    let decision = tokio::time::timeout(Duration::from_millis(100), gate.wait_async())
        .await
        .unwrap();
    assert_eq!(decision["action"], "allow");
    gate.begin_response(Instant::now());
    gate.cancel(None);
    release.send(()).unwrap();
    server.join().unwrap();
    assert_eq!(gate.wait_async().await["action"], "allow");
}

#[tokio::test]
async fn slow_fail_open_policy_does_not_gate_strict_search_approval() {
    for strict_first in [false, true] {
        let (url, received, release, server) = server("{\"flagged\":true}", 200);
        let mut permissive = check(&url, true);
        permissive.policy["input_fail_open"] = json!(true);
        let mut strict = check("http://127.0.0.1:1/unused", true);
        strict.adapter = json!({"kind":"allow"});
        let checks = if strict_first {
            vec![strict, permissive]
        } else {
            vec![permissive, strict]
        };
        let gate = InputGate::start(checks, 5.0).unwrap();
        received.recv_timeout(Duration::from_secs(2)).unwrap();
        let resources = Resources::new(Some(gate.clone()), None, None, "request").unwrap();
        let allowed = tokio::time::timeout(Duration::from_millis(100), resources.wait_for_tools())
            .await
            .unwrap();
        assert_eq!(allowed["action"], "pending");
        assert_eq!(gate.decision()["action"], "pending");
        release.send(()).unwrap();
        assert_eq!(resources.wait().await["action"], "error");
        server.join().unwrap();
    }
}

#[test]
fn request_deadline_labels_unstarted_sibling_checks_as_timed_out() {
    let (url, received, release, server) = server("{\"flagged\":false}", 200);
    let first = check(&url, true);
    let mut second = first.clone();
    second.check["check_id"] = json!("unstarted");
    second.adapter = json!({"kind":"allow"});
    let gate = InputGate::start(vec![first, second], 0.2).unwrap();
    received.recv_timeout(Duration::from_secs(2)).unwrap();
    // Let the inspection task reach its deadline without a caller racing its recovery.
    std::thread::sleep(Duration::from_millis(300));
    let decision = gate.wait();
    release.send(()).unwrap();
    server.join().unwrap();
    assert_eq!(decision["moderation"]["status"], "unavailable");
    let records = decision["moderation"]["observations"].as_array().unwrap();
    assert_eq!(records.len(), 2);
    assert!(records.iter().all(|record| record["outcome"] == "timeout"));
}
