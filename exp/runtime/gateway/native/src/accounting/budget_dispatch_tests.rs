//! Request-local destination refusal handling through the native financial owner.

use std::sync::atomic::Ordering;

use serde_json::{json, Value};

use super::dispatch_tests::{config, setup, terminal};
use super::ownership::Owner;
use super::route_config::Registration;

fn route(suffix_pool: &str, allowed: bool) -> Registration {
    let mut c = config();
    c.attempt_policy.maximum_same_deployment_attempts = 1;
    c.lanes[0].pool_id = "root".into();
    c.lanes[1].pool_id = "child".into();
    c.lanes[1].stage.lead = 1;
    let mut child = c.lanes[1].clone();
    child.key.1 = "child-2".into();
    c.lanes.push(child);
    let mut suffix = c.lanes[0].clone();
    suffix.key.1 = "suffix".into();
    suffix.pool_id = suffix_pool.into();
    suffix.stage.lead = 3;
    c.lanes.push(suffix);
    c.rules = vec![
        None,
        None,
        None,
        Some(vec![if allowed { "transport" } else { "refusal" }.into()]),
    ];
    c.redial_budgets = vec![0; 4];
    c
}

fn start(owner: &Owner, ordinal: u32, depth: Option<usize>) -> Value {
    let mut input = json!({"request_id":"request","attempt_ordinal":ordinal});
    if let Some(depth) = depth {
        input["current_depth"] = json!(depth);
        input["failure"] = json!({"failure_class":"transport","safe_message":"failed",
            "retryable_same_deployment":false,"failover_eligible":true});
    }
    owner.call("start_attempt", input).unwrap()
}

fn failed(owner: &Owner, started: &Value) {
    let mut receipt = terminal(0, false);
    receipt["attempt_id"] = started["attempt_id"].clone();
    receipt["outcome"] = json!("failed");
    receipt["usage"] = Value::Null;
    receipt["failure"] = json!({"failure_class":"transport","safe_message":"failed",
        "failover_eligible":true});
    owner.call("settle", receipt).unwrap();
}

#[test]
fn refused_pool_skips_its_later_members_without_widening_conditional_successors() {
    for allowed in [false, true] {
        let (owner, shared, ledger) = setup(route("root", allowed));
        *ledger.deny_depths.lock().unwrap() = vec![1, 2];
        let first = start(&owner, 0, None);
        failed(&owner, &first);
        let result = start(&owner, 1, Some(0));
        let calls = ledger.calls.lock().unwrap();
        let depths: Vec<_> = calls
            .iter()
            .filter(|(m, _)| m == "start_attempt")
            .map(|(_, v)| v["dispatch"]["depth"].as_u64().unwrap())
            .collect();
        assert_eq!(depths, if allowed { vec![0, 1, 3] } else { vec![0, 1] });
        if allowed {
            assert_eq!(result["route_depth"], 3);
            let dispatch = &calls
                .iter()
                .rev()
                .find(|(m, _)| m == "start_attempt")
                .unwrap()
                .1["dispatch"];
            assert_eq!(dispatch["fallback_reason"], "failover_only_on:transport");
        } else {
            assert_eq!(result["exhausted"], true);
        }
        assert!(!shared.policies.lock().unwrap().health.ejected(
            &("revision".into(), "lane-b".into(), "connection".into()),
            shared.now()
        ));
    }
}

#[test]
fn refusal_after_an_attempt_preserves_forward_pool_boundaries() {
    for suffix in ["root", "child", "other"] {
        let (owner, _, ledger) = setup(route(suffix, true));
        let first = start(&owner, 0, None);
        failed(&owner, &first);
        let child = start(&owner, 1, Some(0));
        assert_eq!(child["route_depth"], 1);
        failed(&owner, &child);
        *ledger.deny_depths.lock().unwrap() = vec![2, 3];
        // A separate destination remains funded; only this child's later members refuse.
        if suffix != "child" {
            *ledger.deny_depths.lock().unwrap() = vec![2];
        }
        let result = start(&owner, 2, Some(1));
        let calls = ledger.calls.lock().unwrap();
        let depths: Vec<_> = calls
            .iter()
            .filter(|(m, _)| m == "start_attempt")
            .map(|(_, v)| v["dispatch"]["depth"].as_u64().unwrap())
            .collect();
        assert_eq!(
            depths,
            if suffix == "child" {
                vec![0, 1, 2]
            } else {
                vec![0, 1, 2, 3]
            }
        );
        if suffix == "child" {
            assert_eq!(result["exhausted"], true);
            assert_eq!(result["failure"]["failure_class"], "quota_exceeded");
        } else {
            assert_eq!(result["route_depth"], 3);
        }
    }
}

#[test]
fn a_request_wide_refusal_at_a_child_cannot_escape_to_a_funded_parent_suffix() {
    let (owner, _, ledger) = setup(route("root", true));
    let first = start(&owner, 0, None);
    failed(&owner, &first);
    ledger.refuse_request.store(true, Ordering::SeqCst);
    let error=owner.call("start_attempt",json!({"request_id":"request","attempt_ordinal":1,
        "current_depth":0,"failure":{"failure_class":"transport","safe_message":"failed","failover_eligible":true}})).unwrap_err();
    assert_eq!(error.status_code, 402);
    assert!(owner.finished());
    let calls = ledger.calls.lock().unwrap();
    assert_eq!(
        calls.iter().filter(|(m, _)| m == "start_attempt").count(),
        2
    );
    assert_eq!(
        calls.last().unwrap().1["terminal"]["failure"]["failure_class"],
        "quota_exceeded"
    );
}
