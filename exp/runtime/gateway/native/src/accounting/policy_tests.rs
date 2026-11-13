use std::collections::HashMap;

use serde_json::{json, Value};

use super::health::{Claim, Health, Policy};
use super::loads::{Loads, Reservation};

#[test]
fn circuit_throttles_and_cross_revision_ejection_match_python() {
    replay_health(include_str!("../../tests/generated/health.jsonl"));
}

#[test]
fn account_circuits_and_newer_throttle_windows_match_python() {
    replay_health(include_str!("../../tests/generated/circuits.jsonl"));
}

fn replay_health(trace: &str) {
    let mut health = Health::new(Policy::default()).unwrap();
    let keys = [
        ("revision-a", "lane-a", "connection-a"),
        ("revision-b", "lane-a", "connection-a"),
        ("revision-a", "lane-b", "connection-b"),
    ]
    .map(|(r, l, c)| (r.into(), l.into(), c.into()));
    for (index, line) in trace.lines().enumerate() {
        let case: Value = crate::lossless_json::from_str(line).unwrap();
        let now = case["now"].as_f64().unwrap();
        let key = &keys[case["key"].as_u64().unwrap() as usize];
        let result = match case["op"].as_str().unwrap() {
            "claim" => json!(health.claim(key, Claim::Normal, now)),
            "claim_last_resort" => json!(health.claim(key, Claim::LastResort, now)),
            "claim_forced" => json!(health.claim(key, Claim::Forced, now)),
            "claim_throttle_redial" => json!(health.claim(key, Claim::ThrottleRedial, now)),
            "succeeded" | "dispatch_opened" => {
                health.succeeded(key, None, now);
                Value::Null
            }
            "failed" => {
                health.failed(
                    key,
                    &crate::lossless_json::from_value(case["argument"].clone()).unwrap(),
                    None,
                    now,
                );
                Value::Null
            }
            "exhausted" => {
                health.exhausted(key, case["argument"].as_f64().unwrap(), now);
                Value::Null
            }
            "release_probe" => {
                health.release(key, None);
                Value::Null
            }
            op => panic!("unknown operation {op}"),
        };
        assert_eq!(result, case["result"], "result {index}");
        assert_eq!(
            json!(keys
                .iter()
                .map(|k| health.ejected(k, now))
                .collect::<Vec<_>>()),
            case["ejected"],
            "ejected {index}"
        );
        assert_eq!(
            json!(keys
                .iter()
                .map(|k| health.suppressed(k, now))
                .collect::<Vec<_>>()),
            case["suppressed"],
            "suppressed {index}"
        );
        assert_eq!(
            json!(health.order(&keys, &[0, 1, 2], now)),
            case["order"],
            "order {index}"
        );
        close(
            &json!(health.throttled_remaining(&keys, now)),
            &case["remaining"],
            index,
        );
        let judged: Vec<Vec<_>> = JUDGED_DEPTHS
            .iter()
            .map(|depths| depths.iter().map(|d| keys[*d].clone()).collect())
            .collect();
        assert_eq!(
            json!(judged
                .iter()
                .map(|keys| health.circuit_failure(keys, now).map(|f| json!({
                    "failure_class": f.failure_class.as_str(),
                    "safe_message": f.safe_message,
                    "customer_owned": f.customer_owned,
                })))
                .collect::<Vec<_>>()),
            case["circuit"],
            "circuit {index}"
        );
        close(
            &json!(judged
                .iter()
                .map(|keys| health.throttled_remaining(keys, now))
                .collect::<Vec<_>>()),
            &case["windows"],
            index,
        );
    }
}

/// Depth sets a reservation may claim, mirrored from the fixture builder.
const JUDGED_DEPTHS: [&[usize]; 6] = [&[], &[0], &[1], &[2], &[0, 2], &[0, 1, 2]];

fn close(actual: &Value, expected: &Value, index: usize) {
    match (actual, expected) {
        (Value::Number(a), Value::Number(b)) => {
            let a = a.as_f64().unwrap();
            let b = b.as_f64().unwrap();
            assert!(
                (a - b).abs() < 1e-12 * b.abs().max(1.0),
                "case {index}: {a} != {b}"
            );
        }
        (Value::Array(a), Value::Array(b)) => {
            assert_eq!(a.len(), b.len());
            for (a, b) in a.iter().zip(b) {
                close(a, b, index);
            }
        }
        (Value::Object(a), Value::Object(b)) => {
            assert_eq!(a.len(), b.len());
            for (k, v) in a {
                close(v, &b[k], index);
            }
        }
        _ => assert_eq!(actual, expected, "case {index}"),
    }
}

#[test]
fn local_rate_fairness_fast_reserves_and_cache_signals_match_python() {
    let mut loads = Loads::new(None, 10.0).unwrap();
    let keys =
        [("lane-a", "connection-a"), ("lane-b", "connection-b")].map(|(l, c)| (l.into(), c.into()));
    let mut tickets = HashMap::new();
    for (index, line) in include_str!("../../tests/generated/loads.jsonl")
        .lines()
        .enumerate()
    {
        let case: Value = crate::lossless_json::from_str(line).unwrap();
        let now = case["now"].as_f64().unwrap();
        let key = &keys[case["key"].as_u64().unwrap() as usize];
        let args = &case["arguments"];
        let ticket = args["ticket"].as_i64().unwrap_or(-1);
        let held = tickets.get(&ticket).map(String::as_str).unwrap_or("absent");
        let result = match case["op"].as_str().unwrap() {
            "reserve" => {
                let request: Reservation = crate::lossless_json::from_value(args.clone()).unwrap();
                assert!(request.valid());
                match loads.reserve(key, &request, now) {
                    Ok(ticket) => {
                        tickets.insert(index as i64, ticket);
                        json!({"ticket": index})
                    }
                    Err(shed) => json!({"shed": shed}),
                }
            }
            "can_admit" => {
                json!(loads.can_admit(
                    key,
                    &crate::lossless_json::from_value(args.clone()).unwrap(),
                    now
                ))
            }
            "throttle" => {
                loads.record_throttle(key, now);
                Value::Null
            }
            "sample" => {
                loads.record_settle(
                    key,
                    args["organization_id"].as_str().unwrap(),
                    args["cached_tokens"].as_u64().unwrap(),
                    args["input_tokens"].as_u64().unwrap(),
                    now,
                );
                Value::Null
            }
            "release" => {
                loads.release(held);
                Value::Null
            }
            "retract" => {
                loads.retract(held);
                Value::Null
            }
            "restamp" => {
                loads.restamp(held, now);
                Value::Null
            }
            "bind" => {
                loads.bind(held, &format!("attempt-{ticket}"));
                Value::Null
            }
            "release_attempt" => {
                loads.release_attempt(&format!("attempt-{ticket}"));
                Value::Null
            }
            op => panic!("unknown operation {op}"),
        };
        close(&result, &case["result"], index);
        let inflight: Vec<_> = keys
            .iter()
            .map(|k| {
                let mut counts = vec![loads.inflight(k, None)];
                counts.extend((0..5).map(|n| loads.inflight(k, Some(&format!("org-{n}")))));
                counts.extend([loads.fast_inflight(k), loads.failover_inflight(k)]);
                counts
            })
            .collect();
        close(&json!(inflight), &case["inflight"], index);
        let cached: Vec<Vec<_>> = keys
            .iter()
            .map(|k| {
                (0..5)
                    .map(|n| loads.cached_fraction(k, &format!("org-{n}"), now))
                    .collect()
            })
            .collect();
        close(&json!(cached), &case["cached"], index);
        close(&json!(loads.learned_ceilings()), &case["ceilings"], index);
    }
}
