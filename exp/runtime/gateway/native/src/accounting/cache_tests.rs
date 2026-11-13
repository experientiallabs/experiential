//! Exact parity with Python's cache offers and request callbacks.

use super::*;

fn replace_operation(value: &mut Value, before: &str, after: &str) {
    match value {
        Value::String(s) if s == before => *s = after.into(),
        Value::Array(values) => values
            .iter_mut()
            .for_each(|v| replace_operation(v, before, after)),
        Value::Object(values) => values
            .values_mut()
            .for_each(|v| replace_operation(v, before, after)),
        _ => (),
    }
}

#[test]
fn cache_costs_and_claims_match_python_at_integer_namespace_and_expiry_boundaries() {
    for (index, line) in include_str!("../../tests/generated/cache.jsonl")
        .lines()
        .enumerate()
    {
        let case: Value = crate::lossless_json::from_str(line).unwrap();
        let input = &case["input"];
        match case["kind"].as_str().unwrap() {
            "offer" => {
                let authority: Authority =
                    crate::lossless_json::from_value(case["authority"].clone()).unwrap();
                let result = authority.offer(
                    input["operation_id"].as_str().unwrap().into(),
                    input["request_id"].as_str().unwrap(),
                    input["attempt_id"].as_str().unwrap(),
                    input["account_key_fingerprint"].as_str().unwrap().into(),
                    input["plan_scope_digest"].as_str().unwrap(),
                    input["resource_prefix"].as_str().unwrap(),
                    input["prefix_bytes"].as_u64().unwrap(),
                    input["requested_at"].as_f64().unwrap(),
                );
                assert_eq!(result.is_err(), case["error"], "offer {index}");
                if let Ok(result) = result {
                    assert_eq!(
                        serde_json::to_value(result).unwrap(),
                        case["expected"],
                        "offer {index}"
                    );
                }
            }
            "claim" => {
                let offer: Offer = crate::lossless_json::from_value(case["offer"].clone()).unwrap();
                let claim: Claim = crate::lossless_json::from_value(input.clone()).unwrap();
                assert_eq!(
                    offer.claim(&claim, offer.requested_at, true).is_ok(),
                    case["valid"],
                    "claim {index}"
                );
            }
            "observation" => {
                let offer: Offer = crate::lossless_json::from_value(case["offer"].clone()).unwrap();
                let result = crate::lossless_json::from_value(input.clone()).unwrap();
                assert_eq!(
                    offer.observation(&result).is_ok(),
                    case["valid"],
                    "observation {index}"
                );
            }
            _ => (),
        }
    }
}

#[test]
fn cache_callbacks_match_python_and_never_reissue_creation_after_unknown_results() {
    for (index, line) in include_str!("../../tests/generated/cache.jsonl")
        .lines()
        .enumerate()
    {
        let case: Value = crate::lossless_json::from_str(line).unwrap();
        if case["kind"] != "callbacks" {
            continue;
        }
        let mut cache =
            Cache::new(crate::lossless_json::from_value(case["configuration"].clone()).unwrap());
        cache.clock = || Ok(1_800_000_000.0);
        let active = Active {
            attempt: case["active"]["attempt"].as_str().unwrap().into(),
            depth: 0,
            deployment: case["active"]["deployment"].as_str().unwrap().into(),
        };
        let recorded = Mutex::new(Vec::new());
        let operation = Mutex::new(String::new());
        let reference = "cache-0000000000000000000000000000042f";
        let host = |call: Value| -> Result<Value> {
            match call["action"].as_str().unwrap() {
                "authority" => Ok(case["authority"].clone()),
                "select" => {
                    recorded.lock().unwrap().push(call.clone());
                    Ok(
                        call["candidates"].as_array().unwrap().last().unwrap()["prefix_sha256"]
                            .clone(),
                    )
                }
                "claim" => {
                    let offer = &call["offer"];
                    *operation.lock().unwrap() = offer["operation_id"].as_str().unwrap().into();
                    recorded.lock().unwrap().push(call.clone());
                    match case["host_kind"].as_str().unwrap() {
                        "ready" | "ready_renew" => Ok(
                            json!({"state": "ready", "resource_name": "cachedContents/existing",
                            "expires_at": 1_800_000_100.0, "token_count": 1536,
                            "renewal_operation_id": if case["host_kind"] == "ready_renew" { json!("renew-operation") } else { Value::Null },
                            "renew_until": if case["host_kind"] == "ready_renew" { json!(1_800_000_300.8) } else { Value::Null }}),
                        ),
                        "unavailable" => Ok(json!({"state": "unavailable", "reason": "unknown"})),
                        _ => Ok(
                            json!({"state": "creator", "operation_id": offer["operation_id"], "expires_at": offer["expires_at"]}),
                        ),
                    }
                }
                "record" | "renew" => {
                    let failed =
                        call["action"] == "record" && case["host_kind"] == "record_failure";
                    recorded.lock().unwrap().push(call);
                    if failed {
                        Err(PublicError::internal())
                    } else {
                        Ok(Value::Null)
                    }
                }
                _ => panic!("unexpected cache host operation"),
            }
        };
        for (step, value) in case["steps"].as_array().unwrap().iter().enumerate() {
            let mut input = value["input"].clone();
            replace_operation(&mut input, reference, &operation.lock().unwrap());
            let result = cache.call(
                value["method"].as_str().unwrap(),
                &input,
                input["request_id"].as_str().unwrap(),
                &host,
                &|| Ok(Some(active.clone())),
            );
            assert_eq!(result.is_err(), value["error"], "case {index} step {step}");
            if let Ok(mut result) = result {
                replace_operation(&mut result, &operation.lock().unwrap(), reference);
                assert_eq!(result, value["expected"], "case {index} step {step}");
            }
        }
        let mut calls = json!(*recorded.lock().unwrap());
        replace_operation(&mut calls, &operation.lock().unwrap(), reference);
        assert_eq!(calls, case["calls"], "host calls {index}");
    }
}

fn fixture(kind: &str) -> Value {
    include_str!("../../tests/generated/cache.jsonl")
        .lines()
        .map(|line| crate::lossless_json::from_str::<Value>(line).unwrap())
        .find(|case| case["kind"] == "callbacks" && case["host_kind"] == kind)
        .unwrap()
}

#[test]
fn abandoned_cache_lookup_releases_only_an_unused_renewal_lease() {
    let case = fixture("ready_renew");
    let mut cache =
        Cache::new(crate::lossless_json::from_value(case["configuration"].clone()).unwrap());
    cache.clock = || Ok(1_800_000_000.0);
    let active = Mutex::new(Some(Active {
        attempt: case["active"]["attempt"].as_str().unwrap().into(),
        depth: 0,
        deployment: case["active"]["deployment"].as_str().unwrap().into(),
    }));
    let calls = Mutex::new(Vec::new());
    let input = &case["steps"][0]["input"];
    let result = cache.call("prepare_explicit_cache", input, input["request_id"].as_str().unwrap(), &|call| {
        calls.lock().unwrap().push(call.clone());
        match call["action"].as_str().unwrap() {
            "authority" => Ok(case["authority"].clone()),
            "claim" => {
                *active.lock().unwrap() = None;
                Ok(json!({"state": "ready", "resource_name": "cachedContents/ready", "expires_at": 1_800_000_100.0,
                    "token_count": 1536, "renewal_operation_id": "unused", "renew_until": 1_800_000_300.0}))
            }
            "renew" => Ok(Value::Null),
            _ => panic!("unexpected host operation"),
        }
    }, &|| Ok(active.lock().unwrap().clone())).unwrap();
    assert_eq!(result, unavailable());
    let calls = calls.lock().unwrap();
    assert_eq!(calls.len(), 3);
    assert_eq!(
        calls[2],
        json!({"action": "renew", "renewal": {"operation_id": "unused",
        "outcome": "failed", "observed_at": 1_800_000_000.0, "expire_time": null}})
    );
    assert!(cache.state().unwrap().renewals.is_empty());
    assert!(cache.state().unwrap().operations.is_empty());
}

#[test]
fn expiry_during_a_durable_claim_never_authorizes_http_or_refunds_uncertainty() {
    use std::sync::atomic::{AtomicU64, Ordering};
    static CLOCK: AtomicU64 = AtomicU64::new(1_800_000_000);
    let case = fixture("creator");
    let mut cache =
        Cache::new(crate::lossless_json::from_value(case["configuration"].clone()).unwrap());
    cache.clock = || Ok(CLOCK.load(Ordering::SeqCst) as f64);
    let active = Active {
        attempt: case["active"]["attempt"].as_str().unwrap().into(),
        depth: 0,
        deployment: case["active"]["deployment"].as_str().unwrap().into(),
    };
    let input = &case["steps"][0]["input"];
    let result = cache.call("prepare_explicit_cache", input, input["request_id"].as_str().unwrap(), &|call| {
        match call["action"].as_str().unwrap() {
            "authority" => Ok(case["authority"].clone()),
            "claim" => {
                CLOCK.store(1_800_000_295, Ordering::SeqCst);
                Ok(json!({"state": "creator", "operation_id": call["offer"]["operation_id"], "expires_at": call["offer"]["expires_at"]}))
            }
            _ => panic!("an expired claim cannot refund or retry"),
        }
    }, &|| Ok(Some(active.clone()))).unwrap();
    assert_eq!(result, unavailable());
    assert!(cache.state().unwrap().operations.is_empty());
}

#[test]
fn concurrent_cache_prepares_issue_only_one_durable_claim() {
    use std::sync::{
        atomic::{AtomicUsize, Ordering},
        Barrier,
    };
    let case = fixture("creator");
    let mut cache =
        Cache::new(crate::lossless_json::from_value(case["configuration"].clone()).unwrap());
    cache.clock = || Ok(1_800_000_000.0);
    let active = Active {
        attempt: case["active"]["attempt"].as_str().unwrap().into(),
        depth: 0,
        deployment: case["active"]["deployment"].as_str().unwrap().into(),
    };
    let input = &case["steps"][0]["input"];
    let entered = Barrier::new(2);
    let release = Barrier::new(2);
    let claims = AtomicUsize::new(0);
    let host = |call: Value| match call["action"].as_str().unwrap() {
        "authority" => Ok(case["authority"].clone()),
        "claim" => {
            claims.fetch_add(1, Ordering::SeqCst);
            entered.wait();
            release.wait();
            Ok(
                json!({"state": "creator", "operation_id": call["offer"]["operation_id"], "expires_at": call["offer"]["expires_at"]}),
            )
        }
        _ => panic!("unexpected cache mutation"),
    };
    let prepare = || {
        cache.call(
            "prepare_explicit_cache",
            input,
            input["request_id"].as_str().unwrap(),
            &host,
            &|| Ok(Some(active.clone())),
        )
    };
    std::thread::scope(|scope| {
        let first = scope.spawn(prepare);
        entered.wait();
        assert_eq!(prepare().unwrap(), unavailable());
        release.wait();
        assert_eq!(first.join().unwrap().unwrap()["state"], "create");
    });
    assert_eq!(claims.load(Ordering::SeqCst), 1);
}

#[test]
fn concurrent_receipts_and_lost_acknowledgements_preserve_the_first_observation() {
    use std::sync::{
        atomic::{AtomicUsize, Ordering},
        Barrier,
    };
    let case = fixture("creator");
    let mut cache =
        Cache::new(crate::lossless_json::from_value(case["configuration"].clone()).unwrap());
    cache.clock = || Ok(1_800_000_000.0);
    let active = Active {
        attempt: case["active"]["attempt"].as_str().unwrap().into(),
        depth: 0,
        deployment: case["active"]["deployment"].as_str().unwrap().into(),
    };
    let prepare = &case["steps"][0]["input"];
    let created = cache.call("prepare_explicit_cache", prepare, prepare["request_id"].as_str().unwrap(), &|call| {
        match call["action"].as_str().unwrap() {
            "authority" => Ok(case["authority"].clone()),
            "claim" => Ok(json!({"state": "creator", "operation_id": call["offer"]["operation_id"], "expires_at": call["offer"]["expires_at"]})),
            _ => panic!("unexpected cache mutation"),
        }
    }, &|| Ok(Some(active.clone()))).unwrap();
    let mut input = case["steps"][1]["input"].clone();
    input["operation_id"] = created["operation_id"].clone();
    let entered = Barrier::new(2);
    let release = Barrier::new(2);
    let calls = AtomicUsize::new(0);
    let receipts = Mutex::new(Vec::new());
    let host = |call: Value| {
        assert_eq!(call["action"], "record");
        receipts.lock().unwrap().push(call);
        if calls.fetch_add(1, Ordering::SeqCst) == 0 {
            entered.wait();
            release.wait();
            Err(PublicError::internal())
        } else {
            Ok(Value::Null)
        }
    };
    let finish = || {
        cache.call(
            "finish_explicit_cache",
            &input,
            input["request_id"].as_str().unwrap(),
            &host,
            &|| Ok(Some(active.clone())),
        )
    };
    std::thread::scope(|scope| {
        let first = scope.spawn(finish);
        entered.wait();
        let _ = finish();
        assert_eq!(calls.load(Ordering::SeqCst), 1);
        release.wait();
        let _ = first.join().unwrap();
    });
    assert_eq!(finish().unwrap()["state"], "ready");
    assert_eq!(finish().unwrap()["state"], "ready");
    assert_eq!(calls.load(Ordering::SeqCst), 2);
    let receipts = receipts.lock().unwrap();
    assert_eq!(receipts[0], receipts[1]);
}
