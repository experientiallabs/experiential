use serde::Deserialize;
use serde_json::{json, Value};

use crate::errors::Failure;

use super::health::{Health, Policy};
use super::selection::{self, AttemptPolicy, Retry, Rules, Stage};
use super::sticky::Sticky;

#[derive(Deserialize)]
struct Case {
    prior: Vec<Failure>,
    failure: Failure,
    current: usize,
    physical_counts: Vec<u32>,
    ordinary_counts: Vec<u32>,
    total: u32,
    policy: AttemptPolicy,
    stage: Stage,
    rules: Rules,
    #[serde(deserialize_with = "crate::lossless_json::deserialize")]
    cached_fraction: f64,
    refusal_failover: bool,
    throttle_backoff: bool,
    redial_budget: u32,
    result: Option<usize>,
    disposition: Option<String>,
}

#[test]
fn retries_and_conditional_failover_match_python() {
    let keys = (0..4)
        .map(|n| ("revision".into(), format!("lane-{n}"), "connection".into()))
        .collect::<Vec<_>>();
    for (index, line) in include_str!("../../tests/generated/selection.jsonl")
        .lines()
        .enumerate()
    {
        let case: Case = crate::lossless_json::from_str(line).unwrap();
        assert!(case.policy.valid());
        let mut health = Health::new(Policy::default()).unwrap();
        for (key, failure) in keys.iter().zip(&case.prior) {
            health.failed(key, failure, None, 100.0);
        }
        let retry = Retry {
            failure: &case.failure,
            current: case.current,
            ordinary_counts: &case.ordinary_counts,
            physical_counts: &case.physical_counts,
            total: case.total,
            policy: &case.policy,
            refusal_failover: case.refusal_failover,
            stage: &case.stage,
            cached_fraction: case.cached_fraction,
            throttle_backoff: case.throttle_backoff,
            redial_budget: case.redial_budget,
            rules: &case.rules,
        };
        let (depth, disposition) = selection::next(&mut health, &keys, &retry, 100.0);
        assert_eq!(depth, case.result, "selection {index}");
        assert_eq!(
            disposition.map(|d| d.reason()),
            case.disposition.as_deref(),
            "disposition {index}"
        );
    }
}

#[test]
fn sticky_affinity_expiry_and_lru_match_python() {
    let mut sticky = Sticky::new(4).unwrap();
    for (index, line) in include_str!("../../tests/generated/sticky.jsonl")
        .lines()
        .enumerate()
    {
        let case: Value = crate::lossless_json::from_str(line).unwrap();
        let now = case["now"].as_f64().unwrap();
        let fingerprint = case["fingerprint"].as_str().unwrap().as_bytes();
        if case["op"] == "bind" {
            sticky.bind(
                fingerprint,
                case["deployment"].as_str().unwrap(),
                case["ttl"].as_f64().unwrap(),
                now,
            );
        } else {
            sticky.clear(fingerprint);
        }
        let bindings: Vec<_> = (0..7)
            .map(|n| {
                sticky
                    .bound(n.to_string().as_bytes(), now)
                    .map(str::to_string)
            })
            .collect();
        assert_eq!(json!(bindings), case["bindings"], "bindings {index}");
        assert_eq!(json!(sticky.len()), case["size"], "size {index}");
    }
}
