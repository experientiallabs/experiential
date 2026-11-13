//! Native dispatch preserves paced recovery from provider-unsized throttle windows.

use serde_json::json;

use super::dispatch_tests::{config, setup, terminal};
use super::health::{Claim, Health, HealthKey, Policy};
use super::selection;
use crate::errors::{Failure, FailureClass};

fn key(revision: &str, name: &str) -> HealthKey {
    (revision.into(), name.into(), "connection".into())
}

fn throttled() -> Failure {
    Failure::new(FailureClass::Throttled, "provider throttled the request")
}

#[test]
fn hintless_probes_are_paced_across_revisions_and_only_the_bound_attempt_lifts() {
    let mut h = Health::new(Policy::default()).unwrap();
    let keys = [key("old", "lane"), key("new", "lane")];
    for k in &keys {
        h.failed(k, &throttled(), None, 100.0);
    }
    assert!(h.claim(&keys[0], Claim::ThrottleProbe, 100.0));
    assert!(!h.claim(&keys[1], Claim::ThrottleProbe, 100.5));
    h.succeeded(&keys[0], Some("older"), 100.5);
    assert!(h.suppressed(&keys[0], 100.5));
    h.release(&keys[0], Some("unrelated"));
    h.bind_throttle_probe(&keys[0], "probe", true, 100.5);
    h.succeeded(&keys[0], Some("probe"), 100.6);
    assert!(!h.suppressed(&keys[0], 100.6));
    assert!(h.claim(&keys[1], Claim::ThrottleProbe, 101.0));
}

#[test]
fn explicit_retry_after_and_plan_windows_are_not_probed() {
    for stated in [false, true] {
        let mut h = Health::new(Policy::default()).unwrap();
        let k = key("revision", "lane");
        let mut failure = throttled();
        if stated {
            failure.retry_after_seconds = Some(5);
        }
        h.failed(&k, &failure, None, 100.0);
        if !stated {
            h.exhausted(&k, 5.0, 100.0);
        }
        assert!(!h.claim(&k, Claim::ThrottleProbe, 100.0));
        if !stated {
            // The provider-sized part expired; only the engine's longer guess remains.
            assert!(h.claim(&k, Claim::ThrottleProbe, 105.0));
        }
    }
}

#[test]
fn late_success_released_probe_and_redials_cannot_clear_a_new_window() {
    for action in ["newer_429", "reset", "released", "redial", "outside"] {
        let mut h = Health::new(Policy::default()).unwrap();
        let k = key("revision", "lane");
        if action == "outside" {
            h.bind_throttle_probe(&k, "probe", true, 99.0);
        }
        h.failed(&k, &throttled(), None, 100.0);
        assert!(h.claim(&k, Claim::ThrottleProbe, 100.0));
        if action != "outside" {
            h.bind_throttle_probe(&k, "probe", action != "redial", 100.0);
        }
        match action {
            "newer_429" => h.failed(&k, &throttled(), Some("another"), 100.1),
            "reset" => h.exhausted(&k, 60.0, 100.1),
            "released" => h.release(&k, Some("probe")),
            _ => (),
        }
        h.succeeded(&k, Some("probe"), 100.2);
        assert!(h.suppressed(&k, 100.2), "{action}");
        assert!(!h.claim(&k, Claim::ThrottleProbe, 100.2), "{action}");
    }
}

#[test]
fn no_failure_walk_prefers_unthrottled_siblings_and_failover_never_probes() {
    let mut h = Health::new(Policy::default()).unwrap();
    let keys = [key("revision", "a"), key("revision", "b")];
    h.failed(&keys[0], &throttled(), None, 100.0);
    assert_eq!(
        selection::claim(&mut h, &keys, 0, &[0, 1], true, 100.0),
        Some(1)
    );
    h.failed(&keys[1], &throttled(), None, 100.0);
    assert_eq!(
        selection::claim(&mut h, &keys, 0, &[0, 1], false, 100.0),
        None
    );
    assert_eq!(
        selection::claim(&mut h, &keys, 0, &[0, 1], true, 100.0),
        Some(0)
    );
    h.release(&keys[0], None);
    assert_eq!(
        selection::claim(&mut h, &keys, 1, &[0, 1], true, 100.0),
        Some(1)
    );
}

#[test]
fn native_reservation_binds_probe_and_terminal_success_reopens_lone_lane() {
    for (selected, exhausted) in [(false, false), (true, false), (false, true)] {
        let mut c = config();
        c.lanes.truncate(1);
        c.rules.truncate(1);
        c.redial_budgets.truncate(1);
        c.selected_route = selected;
        let k = c.lanes[0].key.clone();
        let (owner, shared, _) = setup(c);
        shared
            .policies
            .lock()
            .unwrap()
            .health
            .failed(&k, &throttled(), None, shared.now());
        let first = owner
            .call(
                "start_attempt",
                json!({"request_id":"request", "attempt_ordinal":0}),
            )
            .unwrap();
        if selected {
            assert_eq!(first["exhausted"], true);
        } else {
            assert_eq!(first["route_depth"], 0);
            let mut result = terminal(100, true);
            if exhausted {
                result["rate_limit_headers"] = json!({
                    "x-codex-primary-used-percent": "100",
                    "x-codex-primary-reset-after-seconds": "60"
                });
            }
            owner.call("settle", result).unwrap();
            assert_eq!(
                shared
                    .policies
                    .lock()
                    .unwrap()
                    .health
                    .suppressed(&k, shared.now()),
                exhausted
            );
        }
    }
}
