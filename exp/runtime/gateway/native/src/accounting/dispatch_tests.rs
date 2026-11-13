//! Behavioral checks for the complete native accounting callback walk.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use pyo3::Python;
use serde_json::{json, Value};

use crate::errors::{Failure, FailureClass, PublicError};
use crate::native_callbacks::{CallbackTarget, NativeTarget};

use super::dispatch::Dispatch;
use super::health::Policy;
use super::loads::Reservation;
use super::ownership::Owner;
use super::route_config::Registration;
use super::shared::Shared;

#[derive(Default)]
pub(super) struct Ledger {
    pub(super) calls: Mutex<Vec<(String, Value)>>,
    pub(super) fail_settle: AtomicBool,
    pub(super) fail_price: AtomicBool,
    fail_reserve: AtomicBool,
    fail_abandon: AtomicBool,
    incomplete_abandon: AtomicBool,
    pub(super) refuse_request: AtomicBool,
    pub(super) deny_depths: Mutex<Vec<u64>>,
    fail_fleet_release: AtomicBool,
    pub(super) fleet: bool,
    pub(super) funded: bool,
    pub(super) fleet_shed: Option<&'static str>,
    missing_method: Option<&'static str>,
}

impl CallbackTarget for Ledger {
    fn handles(&self, method: &str) -> bool {
        self.missing_method != Some(method)
            && (matches!(
                method,
                "start_attempt" | "settle" | "price" | "abandon" | "record_late_meter"
            ) || (self.fleet && matches!(method, "fleet_acquire" | "fleet_release")))
    }
    fn call(&self, method: &str, argument: &str) -> Result<String, PublicError> {
        let data: Value = crate::lossless_json::from_str(argument).unwrap();
        self.calls
            .lock()
            .unwrap()
            .push((method.into(), data.clone()));
        if (matches!(method, "settle" | "record_late_meter")
            && self.fail_settle.load(Ordering::SeqCst))
            || (method == "price" && self.fail_price.load(Ordering::SeqCst))
            || (method == "start_attempt" && self.fail_reserve.load(Ordering::SeqCst))
            || (method == "abandon" && self.fail_abandon.load(Ordering::SeqCst))
            || (method == "fleet_release" && self.fail_fleet_release.load(Ordering::SeqCst))
        {
            return Err(PublicError::internal());
        }
        Ok(match method {
            "start_attempt" if self.refuse_request.load(Ordering::SeqCst) => json!({"refused": {
                "scope_kind": "request", "scope_id": "tenant", "error": refusal()}}),
            "start_attempt" if self.deny_depths.lock().unwrap().contains(&data["dispatch"]["depth"].as_u64().unwrap()) => json!({"refused":{"scope_kind":"pool","scope_id":"child"}}),
            "start_attempt" => json!({"attempt_id": format!("attempt-{}", data["dispatch"]["ordinal"]), "promo_funded": self.funded}),
            "abandon" => json!({"terminalized": !self.incomplete_abandon.load(Ordering::SeqCst), "known_unbilled": data["certify_no_effects"]}),
            "price" | "settle" | "record_late_meter" => json!({"charge": data["terminal"]["usage"]["input_tokens"]}),
            "fleet_acquire" if self.fleet_shed.is_some() => json!({"shed": self.fleet_shed}),
            "fleet_acquire" => json!({"lease": "lease-1"}),
            "fleet_release" => json!({}),
            _ => panic!("unexpected method"),
        }.to_string())
    }
}

pub(super) fn config() -> Registration {
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
        "recovery_reason": null, "fallback_reason": null, "paid_prework": false, "capacity_message": "Model is at capacity. Please retry.", "caller_narrowing": []
    })).unwrap()
}

pub(super) fn setup(config: Registration) -> (Arc<Owner>, Arc<Shared>, Arc<Ledger>) {
    setup_with(config, Ledger::default())
}

pub(super) fn setup_with(
    config: Registration,
    ledger: Ledger,
) -> (Arc<Owner>, Arc<Shared>, Arc<Ledger>) {
    let ledger = Arc::new(ledger);
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

// The host's public answer when the organization's allocation refuses the request.
fn refusal() -> Value {
    serde_json::to_value(PublicError::new(
        402,
        "monthly_allocation_exhausted",
        "The organization's monthly gateway allocation is exhausted.",
        "insufficient_quota",
    ))
    .unwrap()
}

#[test]
fn a_request_wide_refusal_writes_one_quota_terminal_and_answers_with_the_refusal() {
    for failed_write in [false, true] {
        let (owner, _, ledger) = setup(config());
        ledger.refuse_request.store(true, Ordering::SeqCst);
        ledger.fail_abandon.store(failed_write, Ordering::SeqCst);
        let error = owner
            .call(
                "start_attempt",
                json!({"request_id": "request", "attempt_ordinal": 0}),
            )
            .unwrap_err();
        if failed_write {
            // The refusal stays the request's terminal until a write succeeds.
            assert!(owner.fenced());
            ledger.fail_abandon.store(false, Ordering::SeqCst);
            assert!(owner.recover(Instant::now()).unwrap());
        } else {
            assert_eq!(error.code, "monthly_allocation_exhausted");
            assert_eq!(error.status_code, 402);
        }
        assert!(owner.finished());
        let calls = ledger.calls.lock().unwrap();
        let abandoned: Vec<_> = calls
            .iter()
            .filter(|(method, _)| method == "abandon")
            .collect();
        assert_eq!(abandoned.len(), if failed_write { 2 } else { 1 });
        for (_, input) in abandoned {
            assert_eq!(
                input["terminal"]["failure"]["failure_class"],
                "quota_exceeded"
            );
        }
    }
}

// Observes `failure` on the lead rung, then exhausts a first dial that may not force it.
fn exhaust_first_dial(config: Registration, failure: &Failure) -> (Value, Vec<(String, Value)>) {
    let (owner, shared, ledger) = setup(config);
    let lead = ("revision".into(), "lane-a".into(), "connection".into());
    shared
        .policies
        .lock()
        .unwrap()
        .health
        .failed(&lead, failure, None, shared.now());
    let exhausted = owner
        .call(
            "start_attempt",
            json!({"request_id": "request", "attempt_ordinal": 0}),
        )
        .unwrap();
    assert_eq!(exhausted["exhausted"], true);
    assert!(owner.finished());
    let calls = ledger.calls.lock().unwrap().clone();
    (exhausted, calls)
}

#[test]
fn exhausted_requests_keep_their_terminal_until_cleanup_is_acknowledged() {
    let mut config = config();
    config.selected_route = true;
    config.rules = vec![None, Some(vec!["throttled".into()])];
    let (owner, shared, ledger) = setup(config);
    ledger.incomplete_abandon.store(true, Ordering::SeqCst);
    let mut failure = Failure::new(
        FailureClass::ProviderAuthentication,
        "provider rejected the configured credential",
    );
    failure.customer_owned = true;
    shared.policies.lock().unwrap().health.failed(
        &("revision".into(), "lane-a".into(), "connection".into()),
        &failure,
        None,
        shared.now(),
    );
    let exhausted = owner
        .call(
            "start_attempt",
            json!({"request_id": "request", "attempt_ordinal": 0}),
        )
        .unwrap();
    assert_eq!(exhausted["exhausted"], true);
    assert!(!owner.retired());
    assert!(owner.fenced());
    assert!(owner.needs_recovery(Instant::now()));
    assert!(!owner.recover(Instant::now()).unwrap());
    ledger.incomplete_abandon.store(false, Ordering::SeqCst);
    assert!(owner.recover(Instant::now()).unwrap());
    assert!(owner.retired());
    assert!(!owner.fenced());
    let calls = ledger.calls.lock().unwrap();
    assert_eq!(calls.len(), 3);
    for (method, input) in calls.iter() {
        assert_eq!(method, "abandon");
        assert_eq!(
            input["terminal"]["failure"]["failure_class"],
            "invalid_request"
        );
        assert_eq!(
            input["terminal"]["failure"]["safe_message"],
            failure.safe_message
        );
    }
}

#[test]
fn a_selected_first_dial_answers_with_the_cause_holding_its_own_rung() {
    let selected = || {
        let mut config = config();
        config.selected_route = true;
        // The sibling takes only a throttled failover, never a first dial.
        config.rules = vec![None, Some(vec!["throttled".into()])];
        config
    };
    // The customer's rejected credential holds the rung open: the answer is
    // that account failure, which names their fix, not a gateway outage.
    let mut credential = Failure::new(
        FailureClass::ProviderAuthentication,
        "provider rejected the configured credential",
    );
    credential.customer_owned = true;
    let (exhausted, calls) = exhaust_first_dial(selected(), &credential);
    assert_eq!(
        exhausted["failure"]["failure_class"],
        "provider_authentication"
    );
    assert_eq!(exhausted["failure"]["customer_owned"], true);
    assert_eq!(
        exhausted["failure"]["safe_message"],
        "provider rejected the configured credential"
    );
    // The caller gets their 400 naming the fix; the one ledger write files it client-side.
    let answer: Failure = serde_json::from_value(exhausted["failure"].clone()).unwrap();
    let error = answer.public_error();
    assert_eq!(
        (error.status_code, error.code.as_str()),
        (400, "provider_credential_rejected")
    );
    let written: Vec<_> = calls.iter().map(|(method, _)| method.as_str()).collect();
    assert_eq!(written, ["abandon"]);
    assert_eq!(
        calls[0].1["terminal"]["failure"]["failure_class"],
        "invalid_request"
    );

    // The throttled rung is judged alone: a free sibling the dial may not
    // claim leaves the answer throttled, not unavailable.
    let mut throttle = Failure::new(FailureClass::Throttled, "provider throttled the request");
    throttle.retry_after_seconds = Some(30);
    let (exhausted, _) = exhaust_first_dial(selected(), &throttle);
    assert_eq!(exhausted["failure"]["failure_class"], "throttled");
    assert_eq!(exhausted["failure"]["retry_after_seconds"], 30);
    assert_eq!(
        exhausted["failure"]["safe_message"],
        "all exact-model deployments are inside a provider throttle window; retry in 30s"
    );

    // A caller control that removed a rung is named beside the selector, with the remedy.
    let mut narrowed = selected();
    narrowed.caller_narrowing = vec!["provider.only".into()];
    let (exhausted, _) = exhaust_first_dial(narrowed, &throttle);
    assert_eq!(
        exhausted["failure"]["safe_message"],
        "the route selected by gateway.routing.route_id and provider.only is inside a provider \
         throttle window; retry in 30s, or remove gateway.routing.route_id and provider.only \
         to let the gateway use this model's other routes"
    );
}

#[test]
fn every_required_financial_method_is_checked_before_ownership_transfer() {
    for missing in [
        "start_attempt",
        "settle",
        "abandon",
        "price",
        "record_late_meter",
    ] {
        let ledger = Arc::new(Ledger {
            missing_method: Some(missing),
            ..Ledger::default()
        });
        let shared = Arc::new(Shared::new(Policy::default(), None, 60.0, 100).unwrap());
        assert!(
            Dispatch::new(config(), shared, NativeTarget::new(ledger.clone())).is_err(),
            "{missing}"
        );
        assert!(ledger.calls.lock().unwrap().is_empty());
    }
}

#[test]
fn admitted_search_meters_survive_unmetered_cleanup_and_deadline_recovery() {
    for (active, failed_write, reported) in [false, true].into_iter().flat_map(|active| {
        [false, true].into_iter().flat_map(move |failed_write| {
            [None, Some(0), Some(5)]
                .into_iter()
                .map(move |reported| (active, failed_write, reported))
        })
    }) {
        let mut configuration = config();
        configuration.paid_prework = true;
        configuration.web_search_requests = 3;
        let (owner, _, ledger) = setup(configuration);
        if active {
            owner
                .call(
                    "start_attempt",
                    json!({"request_id":"request","attempt_ordinal":0}),
                )
                .unwrap();
        }
        ledger.fail_abandon.store(failed_write, Ordering::SeqCst);
        let result = owner.call(
            "abandon",
            json!({"request_id": "request", "web_search_requests": reported}),
        );
        assert_eq!(result.is_err(), failed_write);
        if failed_write {
            assert!(owner.fenced());
            ledger.fail_abandon.store(false, Ordering::SeqCst);
            assert!(owner.recover(Instant::now()).unwrap());
        }
        let calls = ledger.calls.lock().unwrap();
        let abandoned: Vec<_> = calls
            .iter()
            .filter(|(method, _)| method == "abandon")
            .collect();
        assert_eq!(abandoned.len(), if failed_write { 2 } else { 1 });
        for (_, input) in abandoned {
            assert_eq!(input["web_search_requests"], reported.unwrap_or(0).max(3));
            assert_eq!(
                input["terminal"]["observations"]["web_search_requests"],
                reported.unwrap_or(0).max(3)
            );
            assert!(input["terminal"]["usage"].is_null());
        }
        assert!(owner.finished());
    }
    for active in [false, true] {
        let mut configuration = config();
        configuration.paid_prework = true;
        configuration.web_search_requests = 3;
        let (owner, _, ledger) = setup(configuration);
        if active {
            owner
                .call(
                    "start_attempt",
                    json!({"request_id":"request","attempt_ordinal":0}),
                )
                .unwrap();
        }
        assert!(owner
            .recover(owner.deadline + Duration::from_secs(5))
            .unwrap());
        let calls = ledger.calls.lock().unwrap();
        let (method, input) = calls.last().unwrap();
        assert_eq!(method, "abandon");
        assert_eq!(input["web_search_requests"], 3);
        assert_eq!(input["terminal"]["observations"]["web_search_requests"], 3);
    }
}

pub(super) fn terminal(input: u64, finalize: bool) -> Value {
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
fn cancellation_during_strict_inspection_prevents_a_waiting_search() {
    let (url, received, release, server) = super::input_tests::server("{\"flagged\":false}", 200);
    let gate =
        super::input::InputGate::start(vec![super::input_tests::check(&url, true)], 5.0).unwrap();
    received.recv_timeout(Duration::from_secs(2)).unwrap();
    let resources = super::resources::Resources::new(Some(gate), None, None, "request").unwrap();
    let fixture: Value = crate::lossless_json::from_str(
        include_str!("../../tests/generated/search_wires.jsonl")
            .lines()
            .next()
            .unwrap(),
    )
    .unwrap();
    let mut c = config();
    c.features.search = Some(
        crate::lossless_json::from_value(json!({"search": fixture["configuration"],
        "plans": [fixture["plan"], fixture["plan"]], "counter": fixture["counter"],
        "subject": fixture["subject"], "checks": []}))
        .unwrap(),
    );
    let ledger = Arc::new(Ledger::default());
    let shared = Arc::new(Shared::new(Policy::default(), None, 60.0, 100).unwrap());
    let driver =
        Dispatch::with_resources(c, shared, NativeTarget::new(ledger.clone()), resources).unwrap();
    let owner = Owner::new(
        "request".into(),
        "tenant".into(),
        Instant::now() + Duration::from_secs(60),
        driver,
    );
    owner
        .call(
            "start_attempt",
            json!({"request_id":"request","attempt_ordinal":0}),
        )
        .unwrap();
    let searching = owner.clone();
    let worker =
        std::thread::spawn(move || {
            searching.call("tool_search_round",
        json!({"request_id":"request","route_depth":0,"calls":fixture["steps"][0]["calls"]}))
        });
    // Give the worker time to enter its classifier wait, whose response is held by a barrier.
    std::thread::sleep(Duration::from_millis(30));
    let cancelling = owner.clone();
    let (sent, done) = std::sync::mpsc::channel();
    let cancellation = std::thread::spawn(move || {
        sent.send(cancelling.call("abandon", json!({"request_id":"request"})))
            .unwrap()
    });
    let timely = done.recv_timeout(Duration::from_millis(200));
    release.send(()).unwrap();
    cancellation.join().unwrap();
    server.join().unwrap();
    let searched = worker.join().unwrap();
    assert!(
        timely.unwrap().is_ok(),
        "cancellation must not wait for inspection"
    );
    assert!(searched.is_err() || searched.unwrap()["inspection_failure"].is_object());
    assert!(owner.retired());
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
fn a_failed_late_meter_write_stays_owned_until_recovery_delivers_it() {
    let (owner, _, ledger) = setup(config());
    owner
        .call(
            "start_attempt",
            json!({"request_id": "request", "attempt_ordinal": 0}),
        )
        .unwrap();
    owner
        .call("abandon", json!({"request_id": "request"}))
        .unwrap();
    assert!(owner.retired());
    ledger.fail_settle.store(true, Ordering::SeqCst);
    assert!(owner.call("settle", terminal(123, true)).is_err());
    // The data plane hands a finalizing write's retry to this owner.
    assert!(!owner.retired());
    assert!(owner.fenced());
    assert!(owner.needs_recovery(Instant::now()));
    assert!(owner.recover(Instant::now()).is_err());
    ledger.fail_settle.store(false, Ordering::SeqCst);
    assert!(owner.recover(Instant::now()).unwrap());
    assert!(!owner.fenced());
    assert!(!owner.needs_recovery(Instant::now()));
    let calls = ledger.calls.lock().unwrap();
    let audits: Vec<_> = calls
        .iter()
        .filter(|(method, _)| method == "record_late_meter")
        .collect();
    // The failed delivery, the failed retry and the delivered retry carry one meter.
    assert_eq!(audits.len(), 3);
    for (_, input) in audits {
        assert_eq!(input["terminal"]["usage"]["input_tokens"], 123);
    }
}

#[test]
fn a_failed_fleet_release_neither_fails_settlement_nor_repeats() {
    let mut config = config();
    for lane in &mut config.lanes {
        lane.policy.fleet = Some(json!({"saturation": "queue"}));
    }
    let (owner, _, ledger) = setup_with(
        config,
        Ledger {
            fleet: true,
            ..Ledger::default()
        },
    );
    owner
        .call(
            "start_attempt",
            json!({"request_id": "request", "attempt_ordinal": 0}),
        )
        .unwrap();
    ledger.fail_fleet_release.store(true, Ordering::SeqCst);
    owner.call("settle", terminal(37, true)).unwrap();
    assert!(owner.retired());
    assert!(!owner.fenced());
    let calls = ledger.calls.lock().unwrap();
    let releases: Vec<_> = calls
        .iter()
        .filter(|(method, _)| method == "fleet_release")
        .collect();
    assert_eq!(releases.len(), 1);
    assert_eq!(releases[0].1["lease"], "lease-1");
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
