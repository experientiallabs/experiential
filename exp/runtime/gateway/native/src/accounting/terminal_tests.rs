use std::time::{Duration, UNIX_EPOCH};

use serde_json::{json, Value};

use super::{observations, prices, terminal};

#[test]
fn each_physical_call_freezes_its_own_schedule_across_the_peak_boundary() {
    let card = json!({"input_nano_usd_per_million_tokens":300_000_000,
        "cached_input_nano_usd_per_million_tokens":6_000_000,
        "output_nano_usd_per_million_tokens":1_200_000_000,
        "off_peak":{"prices":{"input_nano_usd_per_million_tokens":150_000_000,
            "cached_input_nano_usd_per_million_tokens":3_000_000,
            "output_nano_usd_per_million_tokens":600_000_000},"off_peak_dates":[],
            "peak_windows":[{"days":[1,2,3,4,5],"start_minute":360,"end_minute":600}]}});
    let ceiling = prices::Ceiling::Tokens {
        input: 10_000,
        output: 2_000,
        cached: true,
        writes: false,
        hour: false,
        twelve: false,
        reasoning: false,
    };
    let pick = |at: &str| {
        prices::select(
            &card,
            Some("priority"),
            true,
            false,
            &ceiling,
            chrono::DateTime::parse_from_rfc3339(at).unwrap().into(),
        )
        .unwrap()
    };
    let (first, tier, reserved) = pick("2026-10-19T09:59:59Z");
    let (guard_band, _, guarded) = pick("2026-10-19T10:00:58Z");
    let (retry, retry_tier, next) = pick("2026-10-19T10:01:01Z");
    assert!(tier.is_null() && retry_tier.is_null());
    assert_eq!(first, guard_band);
    assert_eq!(reserved, Some(5_400_000));
    assert_eq!(guarded, reserved);
    assert_eq!(next, Some(2_700_000));
    assert_eq!(ceiling.amount(&first).unwrap(), reserved);
    assert_eq!(ceiling.amount(&retry).unwrap(), next);
    assert!(first["off_peak"].is_null() && retry["off_peak"].is_null());
    assert_eq!(first["input_nano_usd_per_million_tokens"], 300_000_000);
    assert_eq!(retry["input_nano_usd_per_million_tokens"], 150_000_000);
}

#[test]
fn reservation_prices_match_python_reference() {
    for (index, line) in include_str!("../../tests/generated/prices.jsonl")
        .lines()
        .enumerate()
    {
        let case: Value = crate::lossless_json::from_str(line).unwrap();
        let ceiling = crate::lossless_json::from_value(case["ceiling"].clone()).unwrap();
        let now = chrono::DateTime::parse_from_rfc3339(case["at"].as_str().unwrap()).unwrap();
        let (effective, admission, amount) = prices::select(
            &case["prices"],
            case["requested"].as_str(),
            case["forwards"].as_bool().unwrap(),
            case["customer"].as_bool().unwrap(),
            &ceiling,
            now.into(),
        )
        .unwrap();
        assert_eq!(effective, case["effective"], "effective case {index}");
        assert_eq!(admission, case["admission"], "admission case {index}");
        assert_eq!(json!(amount), case["amount"], "ceiling case {index}");
    }
}

#[test]
fn terminal_meters_match_python_reference() {
    for (index, line) in include_str!("../../tests/generated/terminal.jsonl")
        .lines()
        .enumerate()
    {
        let case: Value = crate::lossless_json::from_str(line).unwrap();
        let terms = crate::lossless_json::from_value(case["terms"].clone()).unwrap();
        let result = terminal::normalize(
            &case["input"],
            &terms,
            case["cached"].as_f64().unwrap(),
            UNIX_EPOCH,
        );
        if case["invalid"] == true {
            assert!(result.is_err(), "invalid case {index}");
        } else {
            let result = result.unwrap_or_else(|_| panic!("valid case {index}"));
            for (key, expected) in case["expected"].as_object().unwrap() {
                assert_eq!(&result[key], expected, "case {index} field {key}");
            }
        }
    }
}

#[test]
fn header_observations_match_python_reference() {
    for (index, line) in include_str!("../../tests/generated/observations.jsonl")
        .lines()
        .enumerate()
    {
        let case: Value = crate::lossless_json::from_str(line).unwrap();
        let parsed = observations::parse(
            &case["input"],
            UNIX_EPOCH + Duration::from_secs(1_760_000_000),
        );
        assert_eq!(
            serde_json::to_value(parsed).unwrap(),
            case["expected"],
            "case {index}"
        );
    }
}

#[test]
fn disconnect_estimates_do_not_replace_the_original_provider_meter() {
    let terms: terminal::Terms = crate::lossless_json::from_value(json!({
        "surface":"chat_completions", "counted_input_tokens":11,
        "output_ceiling":100, "output_budget":100,
        "service_tier":null, "self_served":false, "paid_prework":false,
    }))
    .unwrap();
    for observed in [
        Value::Null,
        json!({"input_tokens":19}),
        json!({"input_tokens":19,"output_tokens":7}),
    ] {
        let result = terminal::normalize(&json!({
            "outcome":"failed", "failure":{"failure_class":"cancelled", "safe_message":"Caller left"},
            "opened":true, "dispatched":true, "usage_incomplete_due_to_disconnect":true,
            "usage":observed, "streamed_output":{"text":"A partial answer", "reasoning":""},
        }), &terms, 0.0, UNIX_EPOCH).unwrap();
        assert_eq!(result["usage_estimated"], true);
        assert!(result["usage"]["input_tokens"].as_u64().is_some());
        assert!(result["usage"]["output_tokens"].as_u64().is_some());
        for field in ["input_tokens", "output_tokens"] {
            assert_eq!(result["provider_usage"][field], observed[field]);
        }
    }
}

#[test]
fn paid_search_counts_survive_unmetered_terminal_quotes_and_write_recovery() {
    use std::sync::atomic::Ordering;
    use std::time::Instant;

    use super::dispatch_tests::{config, setup};

    for reported in [None, Some(0), Some(5)] {
        let mut registration = config();
        registration.paid_prework = true;
        registration.web_search_requests = 3;
        let (owner, _, ledger) = setup(registration);
        owner
            .call(
                "start_attempt",
                json!({"request_id":"request", "attempt_ordinal":0}),
            )
            .unwrap();
        let evidence = json!({
            "request_id":"request", "attempt_id":"attempt-0", "outcome":"failed",
            "failure":{"failure_class":"transport", "safe_message":"Provider failed"},
            "finalize":true, "opened":false, "dispatched":true, "usage":null,
            "web_search_requests":reported, "tool_search_requests":2,
        });
        owner.call("price", evidence.clone()).unwrap();
        ledger.fail_settle.store(true, Ordering::SeqCst);
        assert!(owner.call("settle", evidence.clone()).is_err());
        assert!(owner.fenced());
        ledger.fail_settle.store(false, Ordering::SeqCst);
        assert!(owner.recover(Instant::now()).unwrap());
        // A replay uses the durable result and never sends a new paid receipt.
        owner.call("settle", evidence).unwrap();
        let calls = ledger.calls.lock().unwrap();
        let terminals: Vec<_> = calls
            .iter()
            .filter(|(method, _)| method != "start_attempt")
            .collect();
        assert_eq!(terminals.len(), 3);
        assert_eq!(terminals[0].0, "price");
        for (_, input) in terminals {
            assert_eq!(
                input["terminal"]["observations"]["web_search_requests"],
                reported.unwrap_or(0).max(3)
            );
            assert_eq!(input["terminal"]["tool_search_requests"], 2);
            assert!(input["terminal"]["usage"].is_null());
            assert!(input["terminal"]["provider_usage"].is_null());
        }
    }
}

#[test]
fn admitted_search_floor_belongs_only_to_the_final_attempt() {
    use super::dispatch_tests::{config, setup};

    let mut registration = config();
    registration.paid_prework = true;
    registration.web_search_requests = 3;
    let (owner, _, ledger) = setup(registration);
    for ordinal in 0..2 {
        owner
            .call(
                "start_attempt",
                json!({"request_id":"request", "attempt_ordinal":ordinal}),
            )
            .unwrap();
        owner.call("settle", json!({
            "request_id":"request", "attempt_id":format!("attempt-{ordinal}"), "outcome":"failed",
            "failure":{"failure_class":"transport", "safe_message":"Provider failed"},
            "finalize":ordinal == 1, "opened":false, "dispatched":true, "usage":null,
        })).unwrap();
    }
    let calls = ledger.calls.lock().unwrap();
    let settlements: Vec<_> = calls
        .iter()
        .filter(|(method, _)| method == "settle")
        .collect();
    assert_eq!(settlements.len(), 2);
    assert_eq!(
        settlements[0].1["terminal"]["observations"]["web_search_requests"],
        0
    );
    assert_eq!(
        settlements[1].1["terminal"]["observations"]["web_search_requests"],
        3
    );
}
