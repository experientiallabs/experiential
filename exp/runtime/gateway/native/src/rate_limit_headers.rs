//! Allowlisted provider rate-limit response headers for settlement.
//!
//! Providers describe their rate limiting in a small set of response headers
//! (`retry-after` plus the OpenAI `x-ratelimit-*` and Anthropic
//! `anthropic-ratelimit-*` families). Settlement forwards exactly that
//! allowlist to the control plane as the optional `rate_limit_headers` map,
//! on successes and failures alike, where the python side normalizes the raw
//! strings into typed integers for the ledger and sizes throttle windows from
//! the provider's own stated wait. The allowlist is deliberately closed:
//! arbitrary provider headers are never forwarded, because exotic providers
//! put account identifiers and other non-content-free material in theirs.

use reqwest::header::HeaderMap;
use serde_json::{Map, Value};

/// The closed set of forwarded rate-limit headers, lowercased. The `x-codex-*`
/// entries are the ChatGPT plan backend's rolling usage windows (percent used,
/// seconds to reset, window length); numbers only, never the opaque
/// `x-codex-turn-state` or any account label.
const ALLOWLISTED_HEADERS: [&str; 15] = [
    "retry-after",
    "x-ratelimit-limit-requests",
    "x-ratelimit-remaining-requests",
    "x-ratelimit-limit-tokens",
    "x-ratelimit-remaining-tokens",
    "anthropic-ratelimit-requests-limit",
    "anthropic-ratelimit-requests-remaining",
    "anthropic-ratelimit-tokens-limit",
    "anthropic-ratelimit-tokens-remaining",
    "x-codex-primary-used-percent",
    "x-codex-primary-reset-after-seconds",
    "x-codex-primary-window-minutes",
    "x-codex-secondary-used-percent",
    "x-codex-secondary-reset-after-seconds",
    "x-codex-secondary-window-minutes",
];

/// Collect the allowlisted rate-limit headers from one provider response.
///
/// Returns `None` when the response carries none of them, so settlement can
/// omit the field entirely; header values that are not visible ASCII are
/// skipped rather than lossily decoded.
pub fn harvest_rate_limit_headers(headers: &HeaderMap) -> Option<Map<String, Value>> {
    let mut harvested = Map::new();
    for name in ALLOWLISTED_HEADERS {
        if let Some(value) = headers.get(name).and_then(|value| value.to_str().ok()) {
            harvested.insert(name.to_string(), Value::String(value.to_string()));
        }
    }
    if harvested.is_empty() {
        None
    } else {
        Some(harvested)
    }
}

/// Parse one `Retry-After` header as whole seconds.
///
/// Only the integer-seconds form parses here; the HTTP-date form still rides
/// the harvested map verbatim, where the control plane's fuller parser
/// handles it. Anything unparseable yields `None` rather than a guess.
pub fn retry_after_seconds(headers: &HeaderMap) -> Option<u32> {
    headers
        .get("retry-after")
        .and_then(|value| value.to_str().ok())
        .and_then(|value| value.trim().parse::<u32>().ok())
        .filter(|seconds| *seconds > 0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use reqwest::header::{HeaderName, HeaderValue};

    fn headers(pairs: &[(&str, &str)]) -> HeaderMap {
        let mut map = HeaderMap::new();
        for (name, value) in pairs {
            map.insert(
                name.parse::<HeaderName>().unwrap(),
                HeaderValue::from_str(value).unwrap(),
            );
        }
        map
    }

    #[test]
    fn harvest_keeps_only_the_allowlist() {
        let harvested = harvest_rate_limit_headers(&headers(&[
            ("retry-after", "30"),
            ("x-ratelimit-remaining-requests", "9999"),
            ("anthropic-ratelimit-tokens-limit", "12000000"),
            ("content-type", "application/json"),
            ("x-request-id", "req_secretive"),
        ]))
        .unwrap();
        assert_eq!(harvested.len(), 3);
        assert_eq!(harvested["retry-after"], "30");
        assert_eq!(harvested["x-ratelimit-remaining-requests"], "9999");
        assert_eq!(harvested["anthropic-ratelimit-tokens-limit"], "12000000");
    }

    #[test]
    fn harvest_keeps_plan_window_numbers_but_never_the_turn_state() {
        let harvested = harvest_rate_limit_headers(&headers(&[
            ("x-codex-primary-used-percent", "22"),
            ("x-codex-primary-reset-after-seconds", "11511"),
            ("x-codex-secondary-window-minutes", "10080"),
            ("x-codex-turn-state", "gAAAAABqn4w3"),
            ("x-codex-plan-type", "team"),
        ]))
        .unwrap();
        assert_eq!(harvested.len(), 3);
        assert_eq!(harvested["x-codex-primary-used-percent"], "22");
        assert!(!harvested.contains_key("x-codex-turn-state"));
        assert!(!harvested.contains_key("x-codex-plan-type"));
    }

    #[test]
    fn harvest_is_none_without_rate_limit_headers() {
        assert!(
            harvest_rate_limit_headers(&headers(&[("content-type", "text/event-stream")]))
                .is_none()
        );
    }

    #[test]
    fn retry_after_parses_only_positive_integer_seconds() {
        assert_eq!(
            retry_after_seconds(&headers(&[("retry-after", "3600")])),
            Some(3600)
        );
        assert_eq!(
            retry_after_seconds(&headers(&[("retry-after", " 7 ")])),
            Some(7)
        );
        assert_eq!(retry_after_seconds(&headers(&[("retry-after", "0")])), None);
        assert_eq!(
            retry_after_seconds(&headers(&[(
                "retry-after",
                "Mon, 07 Sep 2026 12:00:00 GMT"
            )])),
            None
        );
        assert_eq!(retry_after_seconds(&HeaderMap::new()), None);
    }
}
