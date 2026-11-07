use super::{input::InputGate, inspection::Check, resources::Resources};
use serde_json::{json, Value};
use std::io::{Read, Write};
use std::net::TcpListener;
use std::sync::mpsc;
use std::time::{Duration, Instant};

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

fn server(
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

fn check(url: &str, protected: bool) -> Check {
    crate::lossless_json::from_value(json!({"policy": {"protected": protected},
        "check": {"check_id": "check", "action": "block", "timeout_ms": 3000},
        "adapter": {"kind": "http_json", "url": url, "max_response_bytes": 1024,
            "transport": {"public_only": false, "headers": {}},
            "payload": {"stage": "input", "request": {"messages": [{"content": "fixture"}]}}}}))
    .unwrap()
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
        if mode == "full" {
            assert!(admission::begin(&collector, &captured("occupied")));
        }
        if mode == "closed" {
            resources.close("request");
        }
        let result = resources.status();
        assert_eq!(
            result["action"],
            if mode == "allow" {
                "allow"
            } else if mode == "closed" {
                "closed"
            } else {
                "error"
            }
        );
        assert_eq!(resources.status(), result);
        if mode == "full" {
            assert_eq!(
                result["failure"]["safe_details"]["code"],
                "capture_unavailable"
            );
            let mut terminal = json!({});
            resources.terminal(&mut terminal);
            assert_eq!(terminal["input_guardrail_denied"], true);
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
