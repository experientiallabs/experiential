//! Behavioral checks for the complete native accounting callback walk.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use pyo3::Python;
use serde_json::{json, Value};

use crate::errors::PublicError;
use crate::native_callbacks::{CallbackTarget, NativeTarget};

use super::dispatch::Dispatch;
use super::health::Policy;
use super::loads::Reservation;
use super::ownership::Owner;
use super::route_config::Registration;
use super::shared::Shared;

#[derive(Default)]
struct Ledger {
    calls: Mutex<Vec<(String, Value)>>,
    fail_settle: AtomicBool,
    fail_reserve: AtomicBool,
}

impl CallbackTarget for Ledger {
    fn handles(&self, method: &str) -> bool {
        matches!(
            method,
            "start_attempt" | "settle" | "price" | "abandon" | "record_late_meter"
        )
    }
    fn call(&self, method: &str, argument: &str) -> Result<String, PublicError> {
        let data: Value = crate::lossless_json::from_str(argument).unwrap();
        self.calls
            .lock()
            .unwrap()
            .push((method.into(), data.clone()));
        if (matches!(method, "settle" | "record_late_meter")
            && self.fail_settle.load(Ordering::SeqCst))
            || (method == "start_attempt" && self.fail_reserve.load(Ordering::SeqCst))
        {
            return Err(PublicError::internal());
        }
        Ok(match method {
            "start_attempt" => json!({"attempt_id": format!("attempt-{}", data["dispatch"]["ordinal"]), "promo_funded": false}),
            "abandon" => json!({"terminalized": true, "known_unbilled": data["certify_no_effects"]}),
            "price" | "settle" | "record_late_meter" => json!({"charge": data["terminal"]["usage"]["input_tokens"]}),
            _ => panic!("unexpected method"),
        }.to_string())
    }
}

fn config() -> Registration {
    let lane = |name: &str| {
        json!({"key": ["revision", name, "connection"], "pool_id": "pool", "stage": {
        "lead": 0, "failover_mode": "maximize_cache_affinity", "throttle_cache_threshold": null, "throttle_redial": false},
        "policy": {"concurrency_bound": 2, "sticky_spill_seconds": 300.0},
        "prices": {"input_nano_usd_per_million_tokens": 1_000_000_000_u64, "output_nano_usd_per_million_tokens": 2_000_000_000_u64},
        "ceiling": {"kind": "tokens", "input": 100, "output": 50, "cached": false, "writes": false,
            "hour": false, "twelve": false, "reasoning": false},
        "output_tokens": 50, "output_ceiling": 50, "customer_managed": false, "forwards_tier": false,
        "route_reason": "direct", "keeps_pin": false, "cache_placed": false})
    };
    crate::lossless_json::from_value(json!({
        "request_id": "request", "organization_id": "tenant", "remaining_seconds": 60.0, "surface": "chat_completions",
        "lanes": [lane("lane-a"), lane("lane-b")], "rules": [null, null],
        "attempt_policy": {"maximum_total_attempts": 8, "maximum_same_deployment_attempts": 2, "physical_route_cap": 4},
        "refusal_failover": false, "selected_route": false, "priority": 0, "weight": 1, "fast": false,
        "class": null, "class_cap": null, "pace": null, "pace_saturation": null,
        "input_tokens": 100, "counted_input_tokens": 87, "output_ceiling": 50, "service_tier": null,
        "redial_budgets": [0,0], "fingerprint": "session", "bind_sticky": true, "sticky_preferred": false,
        "caching_off": false, "recovery_scoped": false, "verified_warm_deployment": null, "verified_warm_seconds": 0.0,
        "recovery_reason": null, "fallback_reason": null, "paid_prework": false, "capacity_message": "Model is at capacity. Please retry."
    })).unwrap()
}

fn setup(config: Registration) -> (Arc<Owner>, Arc<Shared>, Arc<Ledger>) {
    let ledger = Arc::new(Ledger::default());
    let shared = Arc::new(Shared::new(Policy::default(), None, 60.0, 100).unwrap());
    let driver = Dispatch::new(config, shared.clone(), NativeTarget::new(ledger.clone())).unwrap();
    let owner = Owner::new(
        "request".into(),
        "tenant".into(),
        Instant::now() + Duration::from_secs(60),
        driver,
    );
    (owner, shared, ledger)
}

fn terminal(input: u64, finalize: bool) -> Value {
    json!({"request_id": "request", "attempt_id": "attempt-0", "outcome": "completed", "finalize": finalize,
        "opened": true, "dispatched": true, "usage": {"input_tokens": input, "output_tokens": 15, "cached_input_tokens": 20}})
}

#[test]
fn native_recovery_evidence_commits_once_at_the_first_terminal_receipt() {
    let mut c = config();
    let scope = json!({"provider":"provider","exact_model_id":"model","endpoint_scope":"endpoint",
        "region_scope":"region","credential_scope":"generation","organization_id":"tenant"});
    let key = [
        "tenant".to_string(),
        "identity".into(),
        "session".into(),
        "prefix".into(),
    ];
    c.recovery = Some(
        crate::lossless_json::from_value(json!({"key":key,"lanes":[
        {"scope":scope,"retention_seconds":60.0,"sticky_seconds":30.0},null]}))
        .unwrap(),
    );
    let (owner, shared, ledger) = setup(c);
    owner
        .call(
            "start_attempt",
            json!({"request_id":"request","attempt_ordinal":0}),
        )
        .unwrap();
    let before = super::session_observations::now();
    owner.call("price", terminal(37, true)).unwrap();
    let after = super::session_observations::now();
    assert!(shared.recovery.lock().unwrap().inspect(&key).is_none());
    ledger.fail_settle.store(true, Ordering::SeqCst);
    assert!(owner.call("settle", terminal(37, true)).is_err());
    assert!(shared.recovery.lock().unwrap().inspect(&key).is_none());
    ledger.fail_settle.store(false, Ordering::SeqCst);
    assert!(owner.recover(Instant::now()).unwrap());
    let first = {
        let registry = shared.recovery.lock().unwrap();
        let history = registry.inspect(&key).unwrap();
        let evidence = &history.evidence["lane-a"];
        assert!((before..=after).contains(&evidence.observed_at));
        assert_eq!(history.retained.as_deref(), Some("lane-a"));
        assert_eq!(history.retained_until, evidence.observed_at + 30.0);
        serde_json::to_value(history).unwrap()
    };
    owner.call("settle", terminal(999, true)).unwrap();
    assert_eq!(
        serde_json::to_value(shared.recovery.lock().unwrap().inspect(&key)).unwrap(),
        first
    );
}

#[test]
fn search_freezes_recovery_identity_for_each_physical_attempt() {
    let fixture: Value = crate::lossless_json::from_str(
        include_str!("../../tests/generated/search_dispatches.jsonl")
            .lines()
            .next()
            .unwrap(),
    )
    .unwrap();
    let mut c = config();
    c.lanes.truncate(1);
    c.rules.truncate(1);
    c.redial_budgets.truncate(1);
    c.features.search = Some(crate::lossless_json::from_value(fixture["config"].clone()).unwrap());
    let key = [
        "tenant".to_owned(),
        "identity".into(),
        "session".into(),
        "initial-prefix".into(),
    ];
    c.recovery = Some(crate::lossless_json::from_value(json!({"key":key,"lanes":[{"scope":{
        "provider":"provider","exact_model_id":"model","endpoint_scope":"endpoint","region_scope":"region",
        "credential_scope":"generation","organization_id":"tenant"},"retention_seconds":60.0,"sticky_seconds":30.0}]})).unwrap());
    let mut search = super::search_round::Search::new(
        crate::lossless_json::from_value(fixture["config"]["search"].clone()).unwrap(),
    )
    .unwrap();
    let mut subject: super::search_subject::Subject =
        crate::lossless_json::from_value(fixture["config"]["subject"].clone()).unwrap();
    search.perform(&fixture["steps"][0]["calls"]).unwrap();
    subject
        .append(&search.tools(), search.frames.last().unwrap())
        .unwrap();
    let mut next = key.clone();
    next[3] = subject.recovery_prefix().unwrap().unwrap();
    assert_ne!(next, key);
    let (owner, shared, _) = setup(c);
    owner
        .call(
            "start_attempt",
            json!({"request_id":"request","attempt_ordinal":0}),
        )
        .unwrap();
    owner
        .call(
            "tool_search_round",
            json!({"request_id":"request","route_depth":0,"calls":fixture["steps"][0]["calls"]}),
        )
        .unwrap();
    owner.call("settle", terminal(37, false)).unwrap();
    assert!(shared.recovery.lock().unwrap().inspect(&key).is_some());
    assert!(shared.recovery.lock().unwrap().inspect(&next).is_none());
    owner.call("start_attempt",json!({"request_id":"request","attempt_ordinal":1,"current_depth":0,"tool_search_round":true})).unwrap();
    let mut end = terminal(37, true);
    end["attempt_id"] = json!("attempt-1");
    owner.call("settle", end).unwrap();
    assert!(shared.recovery.lock().unwrap().inspect(&key).is_some());
    assert!(shared.recovery.lock().unwrap().inspect(&next).is_some());
}

#[test]
fn native_search_rebuilds_and_reserves_every_round_without_a_host_callback() {
    let fixture: Value = crate::lossless_json::from_str(
        include_str!("../../tests/generated/search_wires.jsonl")
            .lines()
            .next()
            .unwrap(),
    )
    .unwrap();
    let mut c = config();
    c.features.search = Some(crate::lossless_json::from_value(json!({"search": fixture["configuration"],
        "plans": [fixture["plan"], fixture["plan"]], "counter": fixture["counter"], "subject": fixture["subject"], "checks": []})).unwrap());
    let (owner, _, ledger) = setup(c);
    owner
        .call(
            "start_attempt",
            json!({"request_id": "request", "attempt_ordinal": 0}),
        )
        .unwrap();
    for (i, step) in fixture["steps"].as_array().unwrap().iter().enumerate() {
        let response = owner
            .call(
                "tool_search_round",
                json!({"request_id": "request", "route_depth": 0, "round": i,
            "calls": step["calls"]}),
            )
            .unwrap();
        assert_eq!(response["wire"], step["expected"]);
        let mut end = terminal(37, false);
        end["attempt_id"] = json!(format!("attempt-{i}"));
        owner.call("settle", end).unwrap();
        owner
            .call(
                "start_attempt",
                json!({"request_id": "request", "attempt_ordinal": i + 1, "current_depth": 0,
            "tool_search_round": true}),
            )
            .unwrap();
        let calls = ledger.calls.lock().unwrap();
        let reserve = calls
            .iter()
            .rev()
            .find(|(name, _)| name == "start_attempt")
            .unwrap();
        assert_eq!(
            reserve.1["dispatch"]["reserved_input_tokens"],
            step["reserved_input"]
        );
        assert_eq!(
            reserve.1["dispatch"]["maximum_cost_nano"],
            json!(step["reserved_input"].as_u64().unwrap() * 1000 + 100_000)
        );
    }
}

#[test]
fn search_owner_matches_the_complete_python_dispatch_callback() {
    for (i, line) in include_str!("../../tests/generated/search_dispatches.jsonl")
        .lines()
        .enumerate()
    {
        let fixture: Value = crate::lossless_json::from_str(line).unwrap();
        let mut c = config();
        c.lanes.truncate(1);
        c.rules.truncate(1);
        c.redial_budgets.truncate(1);
        c.features.search =
            Some(crate::lossless_json::from_value(fixture["config"].clone()).unwrap());
        let (owner, _, ledger) = setup(c);
        owner
            .call(
                "start_attempt",
                json!({"request_id": "request", "attempt_ordinal": 0}),
            )
            .unwrap();
        for (j, step) in fixture["steps"].as_array().unwrap().iter().enumerate() {
            let response = owner
                .call(
                    "tool_search_round",
                    json!({"request_id": "request", "route_depth": 0, "calls": step["calls"]}),
                )
                .unwrap();
            assert_eq!(response["wire"], step["expected"], "case {i} round {j}");
            let mut end = terminal(37, false);
            end["attempt_id"] = json!(format!("attempt-{j}"));
            owner.call("settle", end).unwrap();
            owner.call("start_attempt", json!({"request_id": "request", "attempt_ordinal": j + 1, "current_depth": 0, "tool_search_round": true})).unwrap();
            let calls = ledger.calls.lock().unwrap();
            let reserve = &calls
                .iter()
                .rev()
                .find(|(name, _)| name == "start_attempt")
                .unwrap()
                .1["dispatch"];
            assert_eq!(
                reserve["reserved_input_tokens"], step["reserved_input"],
                "case {i} round {j}"
            );
        }
    }
}

#[test]
fn search_input_rejection_preserves_meter_and_waives_the_current_attempt() {
    let fixture: Value = crate::lossless_json::from_str(
        include_str!("../../tests/generated/search_wires.jsonl")
            .lines()
            .next()
            .unwrap(),
    )
    .unwrap();
    let mut c = config();
    c.features.search = Some(crate::lossless_json::from_value(json!({"search": fixture["configuration"],
        "plans": [fixture["plan"], fixture["plan"]], "counter": fixture["counter"], "subject": fixture["subject"],
        "checks": [{"policy": {"protected": true}, "check": {"check_id": "deny-search", "action": "block", "timeout_ms": 1000},
            "adapter": {"kind": "keyword", "needles": ["the matched tools"], "input_limit": 1048576}}]})).unwrap());
    let (owner, _, ledger) = setup(c);
    owner
        .call(
            "start_attempt",
            json!({"request_id": "request", "attempt_ordinal": 0}),
        )
        .unwrap();
    let response = owner
        .call(
            "tool_search_round",
            json!({"request_id": "request", "route_depth": 0,
        "calls": fixture["steps"][0]["calls"]}),
        )
        .unwrap();
    assert_eq!(response["inspection_failure"]["failure_class"], "guardrail");
    assert_eq!(response["rounds"].as_array().unwrap().len(), 1);
    let mut end = terminal(37, true);
    end["outcome"] = json!("failed");
    end["failure"] = response["inspection_failure"].clone();
    owner.call("settle", end).unwrap();
    let calls = ledger.calls.lock().unwrap();
    let end = &calls.iter().find(|(name, _)| name == "settle").unwrap().1["terminal"];
    assert_eq!(end["usage"]["input_tokens"], 37);
    assert_eq!(end["input_guardrail_denied"], true);
}

#[test]
fn native_dispatch_freezes_prices_and_first_meter_without_the_interpreter() {
    let (owner, shared, ledger) = setup(config());
    Python::initialize();
    Python::attach(|_py| {
        let worker = owner.clone();
        let (sender, receiver) = std::sync::mpsc::channel();
        std::thread::spawn(move || {
            let started = worker
                .call(
                    "start_attempt",
                    json!({"request_id": "request", "attempt_ordinal": 0}),
                )
                .unwrap();
            worker.call("price", terminal(37, true)).unwrap();
            worker.call("settle", terminal(999, true)).unwrap();
            sender.send(started).unwrap();
        });
        let started = receiver
            .recv_timeout(Duration::from_secs(5))
            .expect("native callback attached to the interpreter");
        assert_eq!(started["route_depth"], 0);
    });
    assert!(owner.finished());
    let calls = ledger.calls.lock().unwrap();
    assert_eq!(calls[0].1["dispatch"]["maximum_cost_nano"], 200_000);
    assert_eq!(calls[1].1["terminal"], calls[2].1["terminal"]);
    assert_eq!(calls[2].1["terminal"]["usage"]["input_tokens"], 37);
    let key = ("lane-a".into(), "connection".into());
    let p = shared.policies.lock().unwrap();
    assert_eq!(p.loads.inflight(&key, None), 0);
    assert_eq!(
        p.loads.cached_fraction(&key, "tenant", shared.now()),
        20.0 / 37.0
    );
}

#[test]
fn a_local_shed_moves_sideways_and_discloses_the_actual_reservation() {
    let (owner, shared, ledger) = setup(config());
    let key = ("lane-a".into(), "connection".into());
    {
        let mut p = shared.policies.lock().unwrap();
        let r = Reservation {
            organization_id: "tenant".into(),
            bound: Some(2),
            ..Reservation::default()
        };
        p.loads.reserve(&key, &r, shared.now()).unwrap();
        p.loads.reserve(&key, &r, shared.now()).unwrap();
    }
    let started = owner
        .call(
            "start_attempt",
            json!({"request_id": "request", "attempt_ordinal": 0}),
        )
        .unwrap();
    assert_eq!(started["route_depth"], 1);
    owner
        .call("abandon", json!({"request_id": "request"}))
        .unwrap();
    let calls = ledger.calls.lock().unwrap();
    assert_eq!(calls[0].1["dispatch"]["dispatch_reason"], "queue_bound");
    assert_eq!(calls[0].1["dispatch"]["preferred_deployment_id"], "lane-a");
    assert_eq!(
        shared
            .policies
            .lock()
            .unwrap()
            .loads
            .inflight(&("lane-b".into(), "connection".into()), None),
        0
    );
}

#[test]
fn failed_terminal_write_fences_and_replays_without_double_health_or_cache_samples() {
    let (owner, shared, ledger) = setup(config());
    owner
        .call(
            "start_attempt",
            json!({"request_id": "request", "attempt_ordinal": 0}),
        )
        .unwrap();
    ledger.fail_settle.store(true, Ordering::SeqCst);
    assert!(owner.call("settle", terminal(37, true)).is_err());
    assert!(owner.fenced());
    let key = ("lane-a".into(), "connection".into());
    assert_eq!(
        shared.policies.lock().unwrap().loads.inflight(&key, None),
        1
    );
    ledger.fail_settle.store(false, Ordering::SeqCst);
    assert!(owner.recover(Instant::now()).unwrap());
    assert!(!owner.fenced());
    assert_eq!(
        shared.policies.lock().unwrap().loads.inflight(&key, None),
        0
    );
    let calls = ledger.calls.lock().unwrap();
    assert_eq!(calls[1].1, calls[2].1);
}

#[test]
fn unknown_reservation_outcome_is_closed_before_the_error_releases_ownership() {
    let (owner, shared, ledger) = setup(config());
    ledger.fail_reserve.store(true, Ordering::SeqCst);
    assert!(owner
        .call(
            "start_attempt",
            json!({"request_id": "request", "attempt_ordinal": 0})
        )
        .is_err());
    assert!(owner.finished());
    assert_eq!(
        shared
            .policies
            .lock()
            .unwrap()
            .loads
            .inflight(&("lane-a".into(), "connection".into()), None),
        0
    );
    let calls = ledger.calls.lock().unwrap();
    assert_eq!(
        calls.iter().map(|(m, _)| m.as_str()).collect::<Vec<_>>(),
        ["start_attempt", "abandon"]
    );
}

#[test]
fn unowned_meter_cannot_poison_the_request() {
    let (owner, _, _) = setup(config());
    let mut forged = terminal(999, true);
    forged["attempt_id"] = json!("other");
    assert!(owner.call("settle", forged).is_err());
    owner
        .call(
            "start_attempt",
            json!({"request_id": "request", "attempt_ordinal": 0}),
        )
        .unwrap();
    let mut malformed = terminal(37, true);
    malformed["usage_incomplete_due_to_disconnect"] = json!(true);
    assert!(owner.call("settle", malformed).is_err());
    owner.call("settle", terminal(37, true)).unwrap();
    assert!(owner.finished());
}

#[test]
fn late_meter_is_audited_once_without_replaying_cancelled_attempt_effects() {
    use super::ownership::Driver;
    let ledger = Arc::new(Ledger::default());
    let shared = Arc::new(Shared::new(Policy::default(), None, 60.0, 100).unwrap());
    let driver =
        Dispatch::new(config(), shared.clone(), NativeTarget::new(ledger.clone())).unwrap();
    driver
        .call(
            "start_attempt",
            &json!({"request_id":"request","attempt_ordinal":0}),
        )
        .unwrap();
    driver
        .call("abandon", &json!({"request_id":"request"}))
        .unwrap();
    ledger.fail_settle.store(true, Ordering::SeqCst);
    assert!(driver.call("settle", &terminal(123, true)).is_err());
    ledger.fail_settle.store(false, Ordering::SeqCst);
    driver.call("settle", &terminal(456, true)).unwrap();
    driver.call("settle", &terminal(789, true)).unwrap();
    let calls = ledger.calls.lock().unwrap();
    let audits: Vec<_> = calls
        .iter()
        .filter(|(method, _)| method == "record_late_meter")
        .collect();
    assert_eq!(audits.len(), 2);
    assert_eq!(audits[0].1, audits[1].1);
    assert_eq!(audits[1].1["terminal"]["usage"]["input_tokens"], 123);
    assert!(!calls.iter().any(|(method, _)| method == "settle"));
    assert_eq!(
        shared
            .policies
            .lock()
            .unwrap()
            .loads
            .inflight(&("lane-a".into(), "connection".into()), None),
        0
    );
}
