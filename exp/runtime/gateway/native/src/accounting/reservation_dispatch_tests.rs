//! Real native dispatch contracts for forced failover, hard reserves and fleet leases.

use std::time::{Duration, Instant};

use crate::errors::{Failure, FailureClass};
use serde_json::{json, Value};

use super::dispatch_tests::{config, setup, setup_with, terminal, Ledger};
use super::loads::Reservation;
use super::ownership::Owner;
use super::route_config::Registration;
use super::shared::Shared;

fn refusing() -> Registration {
    let mut c = config();
    c.attempt_policy.maximum_same_deployment_attempts = 1;
    for lane in &mut c.lanes {
        lane.policy.concurrency_bound = Some(4);
        lane.policy.refusing = true;
    }
    c
}

fn fill(shared: &Shared, name: &str, count: usize, forced: bool) -> Vec<String> {
    let mut p = shared.policies.lock().unwrap();
    (0..count)
        .map(|_| {
            p.loads
                .reserve(
                    &(name.into(), "connection".into()),
                    &Reservation {
                        organization_id: "tenant".into(),
                        bound: Some(4),
                        force: forced,
                        failover_cap: forced.then_some(2),
                        ..Reservation::default()
                    },
                    shared.now(),
                )
                .unwrap()
        })
        .collect()
}

fn start(owner: &Owner) -> Value {
    owner
        .call(
            "start_attempt",
            json!({"request_id":"request","attempt_ordinal":0}),
        )
        .unwrap()
}

fn failed(owner: &Owner, started: &Value, class: &str) -> Value {
    let failure = json!({"failure_class":class,"safe_message":"provider failed",
        "retryable_same_deployment":false,"failover_eligible":true});
    owner
        .call(
            "settle",
            json!({"request_id":"request","attempt_id":started["attempt_id"],
        "outcome":"failed","failure":failure,"finalize":false}),
        )
        .unwrap();
    owner
        .call(
            "start_attempt",
            json!({"request_id":"request","attempt_ordinal":1,
        "current_depth":started["route_depth"],"failure":failure}),
        )
        .unwrap()
}

#[test]
fn only_dead_lane_failover_forces_a_refusing_bound_and_every_terminal_releases_it() {
    for class in ["provider_internal", "transport", "throttled", "timeout"] {
        for release in ["settle", "abandon", "sweep"] {
            let (owner, shared, ledger) = setup(refusing());
            fill(&shared, "lane-b", 4, false);
            let first = start(&owner);
            let retry = failed(&owner, &first, class);
            let key = ("lane-b".into(), "connection".into());
            if ["throttled", "timeout"].contains(&class) {
                assert_eq!(retry["exhausted"], true);
                assert_eq!(retry["failure"]["failure_class"], class);
                assert_eq!(
                    shared
                        .policies
                        .lock()
                        .unwrap()
                        .loads
                        .failover_inflight(&key),
                    0
                );
                continue;
            }
            assert_eq!(retry["route_depth"], 1);
            assert_eq!(
                shared
                    .policies
                    .lock()
                    .unwrap()
                    .loads
                    .failover_inflight(&key),
                1
            );
            {
                let calls = ledger.calls.lock().unwrap();
                let dispatch = &calls
                    .iter()
                    .rev()
                    .find(|(m, _)| m == "start_attempt")
                    .unwrap()
                    .1["dispatch"];
                assert_eq!(dispatch["dispatch_reason"], "saturated_overflow");
            }
            match release {
                "settle" => {
                    let mut receipt = terminal(100, true);
                    receipt["attempt_id"] = retry["attempt_id"].clone();
                    owner.call("settle", receipt).unwrap();
                }
                "abandon" => {
                    owner
                        .call("abandon", json!({"request_id":"request"}))
                        .unwrap();
                }
                _ => {
                    assert!(owner
                        .recover(Instant::now() + Duration::from_secs(66))
                        .unwrap());
                }
            }
            assert_eq!(
                shared
                    .policies
                    .lock()
                    .unwrap()
                    .loads
                    .failover_inflight(&key),
                0
            );
            assert!(owner.finished());
        }
    }
}

#[test]
fn initial_dials_and_exhausted_forced_reserves_remain_bounded() {
    for initial in [true, false] {
        let mut c = refusing();
        if initial {
            c.lanes.truncate(1);
            c.rules.truncate(1);
            c.redial_budgets.truncate(1);
        }
        let (owner, shared, _) = setup(c);
        let target = if initial { "lane-a" } else { "lane-b" };
        fill(&shared, target, 4, false);
        if !initial {
            fill(&shared, target, 2, true);
        }
        let first = start(&owner);
        let result = if initial {
            first
        } else {
            failed(&owner, &first, "transport")
        };
        assert_eq!(result["exhausted"], true);
        assert_eq!(result["failure"]["failure_class"], "throttled");
        assert_eq!(result["failure"]["retry_after_seconds"], 5);
    }
}

#[test]
fn a_skipped_full_lane_is_revisited_but_a_free_sibling_wins_before_forcing() {
    for sibling in [false, true] {
        let mut c = refusing();
        if sibling {
            let mut third = c.lanes[0].clone();
            third.key.1 = "lane-c".into();
            c.lanes.push(third);
            c.rules.push(None);
            c.redial_budgets.push(0);
        }
        let (owner, shared, ledger) = setup(c);
        fill(&shared, "lane-a", 4, false);
        let first = start(&owner);
        assert_eq!(first["route_depth"], 1);
        let retry = failed(&owner, &first, "provider_internal");
        assert_eq!(retry["route_depth"], if sibling { 2 } else { 0 });
        let calls = ledger.calls.lock().unwrap();
        let dispatch = &calls
            .iter()
            .rev()
            .find(|(m, _)| m == "start_attempt")
            .unwrap()
            .1["dispatch"];
        assert_eq!(
            dispatch["dispatch_reason"],
            if sibling {
                "queue_bound"
            } else {
                "saturated_overflow"
            }
        );
    }
}

#[test]
fn failover_cannot_borrow_fast_reserve_and_an_empty_ladder_preserves_the_cause() {
    for fast in [false, true] {
        let mut c = refusing();
        if fast {
            let lane = &mut c.lanes[1];
            lane.policy.self_hosted = true;
            lane.policy.fast_reserve_fraction = Some(0.25);
            lane.prices["priority"] = json!({"input_nano_usd_per_million_tokens":1_000_000_000_u64,
                "output_nano_usd_per_million_tokens":2_000_000_000_u64});
        } else {
            c.lanes.truncate(1);
            c.rules.truncate(1);
            c.redial_budgets.truncate(1);
        }
        let (owner, shared, _) = setup(c);
        if fast {
            fill(&shared, "lane-b", 3, false);
        }
        let first = start(&owner);
        let result = failed(&owner, &first, "provider_internal");
        assert_eq!(result["exhausted"], true);
        assert_eq!(
            result["failure"]["failure_class"],
            if fast {
                "throttled"
            } else {
                "provider_internal"
            }
        );
        let p = shared.policies.lock().unwrap();
        assert_eq!(
            p.loads
                .fast_inflight(&("lane-b".into(), "connection".into())),
            0
        );
        assert_eq!(
            p.loads
                .failover_inflight(&("lane-b".into(), "connection".into())),
            0
        );
    }
}

#[test]
fn native_first_dial_discloses_the_ejected_lane_it_bypasses() {
    let (owner, shared, ledger) = setup(config());
    for _ in 0..3 {
        shared.policies.lock().unwrap().health.failed(
            &("revision".into(), "lane-a".into(), "connection".into()),
            &Failure::new(FailureClass::Transport, "refused"),
            None,
            shared.now(),
        );
    }
    assert_eq!(start(&owner)["route_depth"], 1);
    let calls = ledger.calls.lock().unwrap();
    let dispatch = &calls.iter().find(|(m, _)| m == "start_attempt").unwrap().1["dispatch"];
    assert_eq!(dispatch["dispatch_reason"], "lane_ejected");
    assert_eq!(dispatch["preferred_deployment_id"], "lane-a");
}

#[test]
fn ordinary_overflow_can_never_force_a_fleet_denial() {
    let mut c = config();
    c.lanes[0].policy.fleet = Some(json!({"saturation":"queue","requests_per_minute":100}));
    let mut mock = Ledger::default();
    mock.fleet = true;
    mock.fleet_shed = Some("concurrency");
    let (owner, shared, ledger) = setup_with(c, mock);
    {
        let mut p = shared.policies.lock().unwrap();
        for _ in 0..2 {
            p.loads
                .reserve(
                    &("lane-b".into(), "connection".into()),
                    &Reservation {
                        organization_id: "tenant".into(),
                        bound: Some(2),
                        ..Reservation::default()
                    },
                    shared.now(),
                )
                .unwrap();
        }
    }
    assert_eq!(start(&owner)["route_depth"], 1);
    let calls = ledger.calls.lock().unwrap();
    let starts: Vec<_> = calls.iter().filter(|(m, _)| m == "start_attempt").collect();
    assert_eq!(starts.len(), 1);
    assert_eq!(
        starts[0].1["dispatch"]["dispatch_reason"],
        "saturated_overflow"
    );
    assert_eq!(
        calls.iter().filter(|(m, _)| m == "fleet_acquire").count(),
        1
    );
}

#[test]
fn fleet_denials_retract_local_reservations_and_only_spill_when_policy_allows() {
    for reason in ["rate", "concurrency"] {
        for policy in ["queue", "refuse"] {
            for sibling in [false, true] {
                let mut c = config();
                c.lanes[0].policy.fleet =
                    Some(json!({"saturation":policy,"requests_per_minute":100}));
                if !sibling {
                    c.lanes.truncate(1);
                    c.rules.truncate(1);
                    c.redial_budgets.truncate(1);
                }
                let mut mock = Ledger::default();
                mock.fleet = true;
                mock.fleet_shed = Some(reason);
                let (owner, shared, ledger) = setup_with(c, mock);
                let result = start(&owner);
                if sibling && policy == "queue" {
                    assert_eq!(result["route_depth"], 1);
                } else {
                    assert_eq!(result["exhausted"], true);
                    assert_eq!(result["failure"]["failure_class"], "throttled");
                }
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
                let acquired: Vec<_> = calls.iter().filter(|(m, _)| m == "fleet_acquire").collect();
                assert_eq!(acquired.len(), 1);
                assert_eq!(acquired[0].1["reserved_tokens"], 150);
                assert!(!calls.iter().any(|(m, _)| m == "fleet_release"));
            }
        }
    }
}

#[test]
fn native_settlement_releases_one_fleet_lease_with_usage_and_throttle_evidence() {
    for throttled in [false, true] {
        let mut c = config();
        c.lanes[0].policy.fleet = Some(json!({"saturation":"queue","tokens_per_minute":1_000_000}));
        let mut mock = Ledger::default();
        mock.fleet = true;
        let (owner, _, ledger) = setup_with(c, mock);
        start(&owner);
        let mut receipt = terminal(40, true);
        receipt["usage"]["output_tokens"] = json!(5);
        if throttled {
            receipt["outcome"] = json!("failed");
            receipt["failure"] = json!({"failure_class":"throttled","safe_message":"throttled","failover_eligible":true});
        }
        owner.call("settle", receipt.clone()).unwrap();
        owner.call("settle", receipt).unwrap();
        owner
            .call("abandon", json!({"request_id":"request"}))
            .unwrap();
        let calls = ledger.calls.lock().unwrap();
        let releases: Vec<_> = calls.iter().filter(|(m, _)| m == "fleet_release").collect();
        assert_eq!(releases.len(), 1);
        assert_eq!(releases[0].1["lease"], "lease-1");
        assert_eq!(releases[0].1["throttled"], throttled);
        assert_eq!(releases[0].1["used_tokens"], 45);
        assert_eq!(releases[0].1["output_tokens"], 5);
    }
}
