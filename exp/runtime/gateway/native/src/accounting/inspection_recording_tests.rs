//! Native decisions retain uncertainty and streaming rollup semantics.

use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use serde_json::{json, Value};

use super::input_reports::{Factory, Reports};
use super::inspection::{Check, Inspection};
use crate::errors::PublicError;
use crate::native_callbacks::{CallbackTarget, NativeTarget};

#[derive(Default)]
struct Records(Mutex<Vec<Value>>);
impl CallbackTarget for Records {
    fn handles(&self, method: &str) -> bool {
        method == "record_input_moderation"
    }
    fn call(&self, _method: &str, argument: &str) -> Result<String, PublicError> {
        self.0
            .lock()
            .unwrap()
            .push(crate::lossless_json::from_str(argument).unwrap());
        Ok("{}".into())
    }
}

fn sink() -> (Factory, Arc<Records>) {
    let records = Arc::new(Records::default());
    (
        Factory {
            target: NativeTarget::new(records.clone()),
            request_id: "fixture".into(),
            reports: Arc::new(Reports::default()),
        },
        records,
    )
}

#[test]
fn optional_inspection_failure_allows_delivery_but_retains_an_error_record() {
    let (factory, records) = sink();
    let check: Check = crate::lossless_json::from_value(json!({
        "policy":{"policy_id":"fixture", "protected":false, "max_response_bytes":4096},
        "check":{"check_id":"check", "adapter_id":"fixture", "stage":"output", "capability":"pii", "action":"block", "timeout_ms":0},
        "adapter":{"kind":"unavailable"}
    })).unwrap();
    let mut inspection = Inspection::new(&[check], Instant::now() + Duration::from_secs(1));
    inspection.reporting = Some(factory.clone());
    assert_eq!(
        inspection.output(&json!({"text":"unchanged"})).unwrap()["action"],
        "allow"
    );
    let mut case: Value = crate::lossless_json::from_str(
        include_str!("../../tests/generated/input_programs.jsonl")
            .lines()
            .next()
            .unwrap(),
    )
    .unwrap();
    case["checks"][0]["policy"]["protected"] = json!(false);
    case["checks"][0]["check"]["timeout_ms"] = json!(0);
    case["checks"][0]["adapter"] = json!({"kind":"unavailable", "input_limit":1048576});
    let checks: Vec<Check> = crate::lossless_json::from_value(case["checks"].clone()).unwrap();
    let subject = crate::lossless_json::from_value(case["subject"].clone()).unwrap();
    super::inspection_http::runtime()
        .unwrap()
        .block_on(super::input_program::enforce(
            &checks[..1],
            &subject,
            Instant::now() + Duration::from_secs(1),
            Some(&factory),
        ))
        .unwrap();
    assert!(factory
        .reports
        .drain(Instant::now() + Duration::from_secs(2)));
    let rows = records.0.lock().unwrap();
    assert_eq!(rows.len(), 2);
    for row in rows.iter() {
        let observation = &row["moderation"]["observations"][0];
        assert_eq!(observation["action"], "error");
        assert_eq!(observation["outcome"], "unavailable");
    }
}

#[test]
fn clean_stream_segments_do_not_flood_the_decision_recorder() {
    let (factory, records) = sink();
    let case: Value = crate::lossless_json::from_str(
        include_str!("../../tests/generated/inspection_stream.jsonl")
            .lines()
            .next()
            .unwrap(),
    )
    .unwrap();
    let checks: Vec<Check> = crate::lossless_json::from_value(case["checks"].clone()).unwrap();
    let mut inspection = Inspection::new(&checks, Instant::now() + Duration::from_secs(10));
    inspection.reporting = Some(factory.clone());
    for _ in 0..300 {
        let output = inspection
            .segment(&json!({"pending":"clean text", "final":true, "settled_bytes":0}))
            .unwrap();
        assert_eq!(output["action"], "allow");
        assert_eq!(output["flagged"], false);
    }
    assert!(factory
        .reports
        .drain(Instant::now() + Duration::from_secs(2)));
    assert!(records.0.lock().unwrap().is_empty());
    let output = inspection
        .segment(&json!({"pending":"person@example.test", "final":true, "settled_bytes":0}))
        .unwrap();
    assert_eq!(output["flagged"], true);
    assert!(factory
        .reports
        .drain(Instant::now() + Duration::from_secs(2)));
    let rows = records.0.lock().unwrap();
    assert_eq!(rows.len(), 1);
    assert_eq!(rows[0]["moderation"]["observations"][0]["action"], "modify");
    assert_eq!(factory.reports.counters()["dropped"], 0);
}

#[test]
fn optional_initial_inspection_records_its_unavailable_verdict_without_blocking() {
    let (factory, records) = sink();
    let check: Check = crate::lossless_json::from_value(json!({
        "policy":{"policy_id":"fixture", "protected":false},
        "check":{"check_id":"check", "adapter_id":"fixture", "stage":"input", "capability":"pii", "action":"block", "timeout_ms":100},
        "adapter":{"kind":"unavailable"}
    })).unwrap();
    let gate = super::input::InputGate::start(vec![check], 1.0).unwrap();
    gate.bind_reporter(factory.reporter());
    assert_eq!(gate.wait()["action"], "allow");
    assert!(factory
        .reports
        .drain(Instant::now() + Duration::from_secs(2)));
    let rows = records.0.lock().unwrap();
    assert_eq!(rows.len(), 1);
    assert_eq!(rows[0]["moderation"]["observations"][0]["action"], "error");
    assert_eq!(
        rows[0]["moderation"]["observations"][0]["outcome"],
        "unavailable"
    );
}

#[test]
fn reinspection_retains_classifier_scores_cache_hits_and_failures() {
    let (factory, records) = sink();
    let (url, received, release, server) =
        super::input_tests::server("{\"answers\":{\"q\":{\"noul\":0.7}}}", 200);
    let mut check = super::input_tests::check(&url, false);
    check.policy["policy_id"] = json!("fixture");
    check.check["stage"] = json!("input");
    check.adapter["kind"] = json!("decision_model");
    check.adapter["timeout_ms"] = json!(3000);
    check.adapter["cache"] = json!({"max_entries":2,"ttl_seconds":60});
    check.adapter["cache_scope"] = json!(url);
    check.adapter["configuration_digest"] = json!("fixture");
    check.adapter["window"] =
        json!({"system_chars":0,"latest_user_chars":100,"prior_messages":0,"prior_chars":100});
    check.adapter["categories"] =
        json!({"category":{"questions":{"q":{}},"aggregate":"mean","threshold":0.8}});
    let case: Value = crate::lossless_json::from_str(
        include_str!("../../tests/generated/input_programs.jsonl")
            .lines()
            .next()
            .unwrap(),
    )
    .unwrap();
    let mut subject: super::search_subject::Subject =
        crate::lossless_json::from_value(case["subject"].clone()).unwrap();
    let runtime = super::inspection_http::runtime().unwrap();
    let mut server = Some(server);
    release.send(()).unwrap();
    for index in 0..3 {
        if index == 2 {
            subject.request["messages"][0]["content"] = json!("changed user input");
        }
        runtime
            .block_on(super::input_program::enforce(
                std::slice::from_ref(&check),
                &subject,
                Instant::now() + Duration::from_secs(5),
                Some(&factory),
            ))
            .unwrap();
        if index == 0 {
            assert_eq!(
                received.recv_timeout(Duration::from_secs(2)).unwrap()["state"],
                "[latest user]\nsecret"
            );
            server.take().unwrap().join().unwrap();
        }
    }
    assert!(factory
        .reports
        .drain(Instant::now() + Duration::from_secs(2)));
    let rows = records.0.lock().unwrap();
    assert_eq!(rows.len(), 3);
    let observations: Vec<_> = rows
        .iter()
        .map(|row| &row["moderation"]["observations"][0])
        .collect();
    assert!(observations.iter().all(|record| {
        record["classifier_kind"] == "decision_model"
            && record["subject_sha256"].as_str().unwrap().len() == 64
    }));
    assert_eq!(
        observations.iter().filter(|v| v["cached"] == true).count(),
        1
    );
    assert_eq!(
        observations
            .iter()
            .filter(|v| v["scores"] == json!([["category", 0.7]]))
            .count(),
        2
    );
    assert_eq!(
        observations
            .iter()
            .filter(|v| v["outcome"] == "unavailable")
            .count(),
        1
    );
    let telemetry = factory.reports.drain_telemetry();
    let sum = |key: &str| -> u64 {
        telemetry["aggregates"]
            .as_array()
            .unwrap()
            .iter()
            .map(|row| row[key].as_u64().unwrap())
            .sum()
    };
    assert_eq!(sum("cache_hits"), 1);
    assert_eq!(sum("cache_misses"), 2);
    assert_eq!(sum("errors"), 1);
    // An empty live window bypasses the classifier rather than sending its template payload.
    let (empty_factory, empty_records) = sink();
    subject.request["messages"][0]["content"] = json!("");
    runtime
        .block_on(super::input_program::enforce(
            &[check],
            &subject,
            Instant::now() + Duration::from_secs(5),
            Some(&empty_factory),
        ))
        .unwrap();
    assert!(empty_factory
        .reports
        .drain(Instant::now() + Duration::from_secs(2)));
    let rows = empty_records.0.lock().unwrap();
    assert_eq!(rows.len(), 1);
    assert!(rows[0]["moderation"]["observations"][0]["classifier_kind"].is_null());
}

#[test]
fn failed_open_modifications_record_the_effective_action_and_underlying_outcome() {
    for (adapter, timeout, outcome) in [
        (json!({"kind":"unavailable"}), 1000, "unavailable"),
        (json!({"kind":"coverage_error"}), 1000, "unsupported"),
        (
            json!({"kind":"keyword","needles":["secret"]}),
            1000,
            "uncertain",
        ),
        (json!({"kind":"allow"}), 0, "unavailable"),
    ] {
        let (factory, records) = sink();
        let mut case: Value = crate::lossless_json::from_str(
            include_str!("../../tests/generated/input_programs.jsonl")
                .lines()
                .next()
                .unwrap(),
        )
        .unwrap();
        case["checks"][0]["policy"]["protected"] = json!(true);
        case["checks"][0]["policy"]["input_fail_open"] = json!(true);
        case["checks"][0]["check"]["action"] = json!("modify");
        case["checks"][0]["check"]["timeout_ms"] = json!(timeout);
        case["checks"][0]["adapter"] = adapter;
        let checks: Vec<Check> = crate::lossless_json::from_value(case["checks"].clone()).unwrap();
        let subject: super::search_subject::Subject =
            crate::lossless_json::from_value(case["subject"].clone()).unwrap();
        let result = super::inspection_http::runtime()
            .unwrap()
            .block_on(super::input_program::enforce(
                &checks[..1],
                &subject,
                Instant::now() + Duration::from_secs(5),
                Some(&factory),
            ))
            .unwrap();
        assert_eq!(result.request, subject.request);
        assert!(factory
            .reports
            .drain(Instant::now() + Duration::from_secs(2)));
        let rows = records.0.lock().unwrap();
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0]["moderation"]["status"], "failed_open");
        let record = &rows[0]["moderation"]["observations"][0];
        assert_eq!(record["action"], "allow");
        assert_eq!(record["would_action"], "modify");
        assert_eq!(record["outcome"], outcome);
    }
}
