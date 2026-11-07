use std::time::{Duration, UNIX_EPOCH};

use serde_json::{json, Value};

use super::{observations, prices, terminal};

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
