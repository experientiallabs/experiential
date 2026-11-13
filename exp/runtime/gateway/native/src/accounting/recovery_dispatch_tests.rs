//! Recovery evidence belongs to the immutable wire scope and native settlement.

use std::sync::atomic::Ordering;
use std::sync::Arc;
use std::time::Duration;
use std::time::Instant;

use serde_json::{json, Value};

use super::dispatch::Dispatch;
use super::dispatch_tests::{config, setup, terminal, Ledger};
use super::loads::Reservation;
use super::ownership::Owner;
use super::session_recovery::{Key, Scope};

#[test]
fn native_funding_receipts_exclude_subsidized_and_estimated_cache_samples() {
    for funded in [false, true] {
        for estimated in [false, true] {
            let mut ledger = Ledger::default();
            ledger.funded = funded;
            let (owner, shared, _) = super::dispatch_tests::setup_with(config(), ledger);
            owner
                .call(
                    "start_attempt",
                    json!({"request_id":"request","attempt_ordinal":0}),
                )
                .unwrap();
            let mut receipt = terminal(100, true);
            if estimated {
                receipt["usage"] = serde_json::Value::Null;
                receipt["output_text"] = json!("generated content");
            }
            owner.call("settle", receipt.clone()).unwrap();
            owner.call("settle", receipt).unwrap();
            let policies = shared.policies.lock().unwrap();
            assert_eq!(
                policies.loads.cached_fraction(
                    &("lane-a".into(), "connection".into()),
                    "tenant",
                    shared.now()
                ),
                if funded || estimated { 0.0 } else { 0.2 }
            );
        }
    }
}
use super::shared::Shared;
use crate::native_callbacks::NativeTarget;

fn key() -> Key {
    [
        "tenant".into(),
        "identity".into(),
        "session".into(),
        "prefix".into(),
    ]
}

fn scope(region: Option<&str>) -> Scope {
    crate::lossless_json::from_value(json!({"provider":"provider", "exact_model_id":"model",
        "endpoint_scope":"endpoint", "region_scope":region, "credential_scope":"K1",
        "organization_id":"tenant"}))
    .unwrap()
}

#[test]
fn native_settlement_never_rebinds_cache_evidence_to_a_rotated_credential() {
    for recovered in [false, true] {
        let mut c = config();
        let original = scope(Some("region"));
        c.recovery = Some(
            crate::lossless_json::from_value(json!({"key":key(), "lanes":[
            {"scope":original,"retention_seconds":60.0,"sticky_seconds":30.0},null]}))
            .unwrap(),
        );
        let (owner, shared, ledger) = setup(c);
        owner
            .call(
                "start_attempt",
                json!({"request_id":"request","attempt_ordinal":0}),
            )
            .unwrap();
        let mut rotated = original.clone();
        rotated.credential_scope = "K2".into();
        let mut receipt = terminal(100, true);
        receipt["usage"]["cached_input_tokens"] = json!(80);
        ledger.fail_settle.store(recovered, Ordering::SeqCst);
        let settled = owner.call("settle", receipt);
        if recovered {
            assert!(settled.is_err());
            assert!(shared.recovery.lock().unwrap().inspect(&key()).is_none());
            ledger.fail_settle.store(false, Ordering::SeqCst);
            assert!(owner.recover(Instant::now()).unwrap());
        } else {
            settled.unwrap();
        }
        let mut recovery = shared.recovery.lock().unwrap();
        for (candidate, expected) in [(original, Some("lane-a")), (rotated, None)] {
            let chosen = recovery
                .choose(
                    &key(),
                    &[("lane-a".into(), candidate)],
                    |_| Ok::<_, ()>(true),
                    None,
                    None::<fn(&str) -> Result<bool, ()>>,
                    super::session_observations::now(),
                )
                .unwrap();
            assert_eq!(chosen.deployment_id.as_deref(), expected);
        }
        assert!(owner.finished());
        assert!(!owner.fenced());
    }
}

#[test]
fn missing_region_cannot_seed_native_settled_cache_evidence() {
    for recovered in [false, true] {
        let mut c = config();
        c.recovery = Some(
            crate::lossless_json::from_value(json!({"key":key(), "lanes":[
            {"scope":scope(None),"retention_seconds":60.0,"sticky_seconds":30.0},null]}))
            .unwrap(),
        );
        let (owner, shared, ledger) = setup(c);
        owner
            .call(
                "start_attempt",
                json!({"request_id":"request","attempt_ordinal":0}),
            )
            .unwrap();
        let mut receipt = terminal(100, true);
        receipt["usage"]["cached_input_tokens"] = json!(80);
        receipt["usage"]["cache_creation_input_tokens"] = json!(10);
        receipt["first_token_at"] = json!("2026-09-18T01:02:03+00:00");
        receipt["upstream_provider"] = json!("upstream");
        ledger.fail_settle.store(recovered, Ordering::SeqCst);
        let settled = owner.call("settle", receipt);
        if recovered {
            assert!(settled.is_err());
            ledger.fail_settle.store(false, Ordering::SeqCst);
            assert!(owner.recover(Instant::now()).unwrap());
        } else {
            settled.unwrap();
        }
        assert!(shared.recovery.lock().unwrap().inspect(&key()).is_none());
        let calls = ledger.calls.lock().unwrap();
        let terminal = &calls
            .iter()
            .rev()
            .find(|(method, _)| method == "settle")
            .unwrap()
            .1["terminal"];
        assert_eq!(terminal["usage"]["cache_creation_input_tokens"], 10);
        assert_eq!(terminal["observations"]["upstream_provider"], "upstream");
        assert!(terminal["first_token_at"].is_string());
    }
}

#[test]
fn scoped_warmth_bypasses_only_the_fresh_threshold_and_expires_without_sticky_bypass() {
    for (verified, seconds, expected_depth) in [(true, 60.0, 0), (true, 0.0, 1), (false, 60.0, 1)] {
        let mut c = config();
        c.recovery_scoped = true;
        c.verified_warm_deployment = verified.then(|| "lane-a".into());
        c.verified_warm_seconds = seconds;
        for lane in &mut c.lanes {
            lane.policy.concurrency_bound = Some(4);
            lane.policy.fresh_session_spill_fraction = Some(0.5);
        }
        let (owner, shared, ledger) = setup(c);
        {
            let mut p = shared.policies.lock().unwrap();
            p.sticky.bind(b"session", "lane-a", 60.0, shared.now());
            for _ in 0..2 {
                p.loads
                    .reserve(
                        &("lane-a".into(), "connection".into()),
                        &Reservation {
                            organization_id: "tenant".into(),
                            bound: Some(4),
                            ..Reservation::default()
                        },
                        shared.now(),
                    )
                    .unwrap();
            }
        }
        let started = owner
            .call(
                "start_attempt",
                json!({"request_id":"request","attempt_ordinal":0}),
            )
            .unwrap();
        assert_eq!(started["route_depth"], expected_depth);
        let calls = ledger.calls.lock().unwrap();
        let dispatch = &calls
            .iter()
            .find(|(method, _)| method == "start_attempt")
            .unwrap()
            .1["dispatch"];
        if expected_depth == 1 {
            assert_ne!(dispatch["dispatch_reason"], Value::Null);
        }
        drop(calls);
        owner
            .call("abandon", json!({"request_id":"request"}))
            .unwrap();
    }
}

fn participant(shared: &Arc<Shared>, name: &str, other_session: bool) -> Arc<Owner> {
    let mut c = config();
    c.request_id = name.into();
    for lane in &mut c.lanes {
        lane.policy.concurrency_bound = Some(1);
    }
    let mut session = key();
    if other_session {
        session[2] = "other-session".into();
    }
    c.recovery = Some(
        crate::lossless_json::from_value(json!({"key":session,"lanes":[
        {"scope":scope(Some("region")),"retention_seconds":60.0,"sticky_seconds":30.0},
        {"scope":scope(Some("region")),"retention_seconds":60.0,"sticky_seconds":30.0}]}))
        .unwrap(),
    );
    let driver = Dispatch::new(
        c,
        shared.clone(),
        NativeTarget::new(Arc::new(Ledger::default())),
    )
    .unwrap();
    Owner::new(
        name.into(),
        "tenant".into(),
        Instant::now() + Duration::from_secs(60),
        driver,
    )
}

fn complete(owner: &Owner, name: &str, started: &Value) {
    let mut receipt = terminal(100, true);
    receipt["request_id"] = json!(name);
    receipt["attempt_id"] = started["attempt_id"].clone();
    receipt["usage"]["cached_input_tokens"] = json!(80);
    owner.call("settle", receipt).unwrap();
}

#[test]
fn capacity_shed_cannot_erase_same_or_other_session_transport_recovery_requirements() {
    for other_session in [false, true] {
        let (_, shared, _) = setup(config());
        let warm = participant(&shared, "warm", false);
        let started = warm
            .call(
                "start_attempt",
                json!({"request_id":"warm","attempt_ordinal":0}),
            )
            .unwrap();
        complete(&warm, "warm", &started);
        let first = participant(&shared, "first", other_session);
        let queued = participant(&shared, "queued", false);
        let started = first
            .call(
                "start_attempt",
                json!({"request_id":"first","attempt_ordinal":0}),
            )
            .unwrap();
        let failed = json!({"failure_class":"transport","safe_message":"transport failed","failover_eligible":true});
        first
            .call(
                "settle",
                json!({"request_id":"first","attempt_id":started["attempt_id"],
            "outcome":"failed","failure":failed,"finalize":other_session}),
            )
            .unwrap();
        if !other_session {
            let retry = first
                .call(
                    "start_attempt",
                    json!({"request_id":"first","attempt_ordinal":1,
                "current_depth":0,"failure":failed}),
                )
                .unwrap();
            assert_eq!(retry["route_depth"], 1);
            complete(&first, "first", &retry);
        }
        let ticket = shared
            .policies
            .lock()
            .unwrap()
            .loads
            .reserve(
                &("lane-a".into(), "connection".into()),
                &Reservation {
                    organization_id: "competitor".into(),
                    bound: Some(1),
                    ..Reservation::default()
                },
                shared.now(),
            )
            .unwrap();
        let shed = queued
            .call(
                "start_attempt",
                json!({"request_id":"queued","attempt_ordinal":0}),
            )
            .unwrap();
        assert_eq!(shed["route_depth"], 1);
        complete(&queued, "queued", &shed);
        shared.policies.lock().unwrap().loads.release(&ticket);
        let mut recovery = shared.recovery.lock().unwrap();
        let chosen = recovery
            .choose(
                &key(),
                &[
                    ("lane-a".into(), scope(Some("region"))),
                    ("lane-b".into(), scope(Some("region"))),
                ],
                |_| Ok::<_, ()>(true),
                None,
                None::<fn(&str) -> Result<bool, ()>>,
                super::session_observations::now() + 6.0,
            )
            .unwrap();
        assert_eq!(chosen.deployment_id.as_deref(), Some("lane-b"));
        assert_eq!(chosen.reason, "retained_warm_fallback");
    }
}
