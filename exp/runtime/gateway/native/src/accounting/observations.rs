//! Bounded provider rate observations used by receipts and lane health.

use std::collections::HashMap;
use std::time::{SystemTime, UNIX_EPOCH};

use serde::Serialize;
use serde_json::Value;

const MAX_WAIT: u64 = 7 * 24 * 3600;
const MAX_COUNT: u64 = 1_000_000_000_000_000;

#[derive(Default, Serialize)]
pub(super) struct Observation {
    pub retry_after_seconds: Option<u64>,
    pub limit_requests: Option<u64>,
    pub remaining_requests: Option<u64>,
    pub limit_tokens: Option<u64>,
    pub remaining_tokens: Option<u64>,
    pub subscription_windows: Vec<Window>,
}

#[derive(Serialize)]
pub(super) struct Window {
    pub window: &'static str,
    pub used_percent: u64,
    pub reset_after_seconds: Option<u64>,
    pub window_minutes: Option<u64>,
}

impl Observation {
    pub fn exhausted_wait(&self) -> Option<u64> {
        self.subscription_windows
            .iter()
            .filter(|window| window.used_percent >= 100)
            .filter_map(|window| window.reset_after_seconds)
            .max()
            .map(|wait| wait.clamp(1, MAX_WAIT))
    }
}

/// Invalid headers never cause settlement to fail or manufacture zero usage.
pub(super) fn parse(value: &Value, now: SystemTime) -> Observation {
    let Some(values) = value.as_object() else {
        return Observation::default();
    };
    let headers: HashMap<_, _> = values
        .iter()
        .filter_map(|(key, value)| value.as_str().map(|value| (key.to_lowercase(), value)))
        .collect();
    let get = |key: &str| headers.get(key).copied();
    let count_pair = |open: &str, anthropic: &str| {
        get(open)
            .and_then(count)
            .or_else(|| get(anthropic).and_then(count))
    };
    let reference = now
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs_f64();
    let mut windows = Vec::new();
    for (name, tag, minutes) in [("primary", "5h", 300), ("secondary", "7d", 10080)] {
        let prefix = format!("anthropic-ratelimit-unified-{tag}-");
        let utilization = get(&format!("{prefix}utilization")).and_then(|text| percent(text, true));
        let rejected = get(&format!("{prefix}status"))
            .is_some_and(|text| text.trim().eq_ignore_ascii_case("rejected"));
        if utilization.is_some() || rejected {
            windows.push(Window {
                window: name,
                used_percent: if rejected { 100 } else { utilization.unwrap() },
                reset_after_seconds: get(&format!("{prefix}reset"))
                    .and_then(|text| until(text, reference)),
                window_minutes: Some(minutes),
            });
        }
    }
    if windows.is_empty() {
        for name in ["primary", "secondary"] {
            let prefix = format!("x-codex-{name}-");
            let Some(used) =
                get(&format!("{prefix}used-percent")).and_then(|text| percent(text, false))
            else {
                continue;
            };
            let reset = match get(&format!("{prefix}reset-after-seconds")) {
                Some(raw) => count(raw),
                None => get(&format!("{prefix}reset-at")).and_then(|text| until(text, reference)),
            };
            windows.push(Window {
                window: name,
                used_percent: used,
                reset_after_seconds: reset,
                window_minutes: get(&format!("{prefix}window-minutes")).and_then(count),
            });
        }
    }
    Observation {
        retry_after_seconds: get("retry-after").and_then(|text| retry_after(text, now)),
        limit_requests: count_pair(
            "x-ratelimit-limit-requests",
            "anthropic-ratelimit-requests-limit",
        ),
        remaining_requests: count_pair(
            "x-ratelimit-remaining-requests",
            "anthropic-ratelimit-requests-remaining",
        ),
        limit_tokens: count_pair(
            "x-ratelimit-limit-tokens",
            "anthropic-ratelimit-tokens-limit",
        ),
        remaining_tokens: count_pair(
            "x-ratelimit-remaining-tokens",
            "anthropic-ratelimit-tokens-remaining",
        ),
        subscription_windows: windows,
    }
}

fn integer(text: &str) -> Option<u128> {
    let text = text.trim();
    let negative = text.starts_with('-');
    let text = text.strip_prefix(['+', '-']).unwrap_or(text);
    if text.is_empty()
        || text.starts_with('_')
        || text.ends_with('_')
        || text.contains("__")
        || !text.bytes().all(|c| c.is_ascii_digit() || c == b'_')
    {
        return None;
    }
    let digits: String = text.chars().filter(|c| *c != '_').collect();
    let value = digits.parse().unwrap_or(u128::MAX);
    (!negative || value == 0).then_some(value)
}

fn count(text: &str) -> Option<u64> {
    integer(text)
        .filter(|count| *count <= MAX_COUNT.into())
        .map(|count| count as u64)
}

fn retry_after(text: &str, now: SystemTime) -> Option<u64> {
    if let Some(seconds) = integer(text) {
        return Some(seconds.clamp(1, MAX_WAIT.into()) as u64);
    }
    let instant = httpdate::parse_http_date(text.trim()).ok()?;
    let remaining = instant.duration_since(now).unwrap_or_default();
    Some((remaining.as_secs() + u64::from(remaining.subsec_nanos() > 0)).clamp(1, MAX_WAIT))
}

fn percent(text: &str, fraction: bool) -> Option<u64> {
    let text = text.trim();
    if text.as_bytes().iter().enumerate().any(|(i, c)| {
        *c == b'_'
            && (i == 0
                || !text.as_bytes()[i - 1].is_ascii_digit()
                || !text.as_bytes().get(i + 1).is_some_and(u8::is_ascii_digit))
    }) {
        return None;
    }
    let mut number: f64 = text.replace('_', "").parse().ok()?;
    if fraction && text.contains(['.', 'e', 'E']) {
        number *= 100.0;
    }
    if !number.is_finite() || number < 0.0 {
        return None;
    }
    // Python rounds six fractional places before flooring float noise.
    Some(
        ((number * 1_000_000.0).round_ties_even() / 1_000_000.0)
            .floor()
            .min(100.0) as u64,
    )
}

fn until(text: &str, reference: f64) -> Option<u64> {
    let count = count(text)?;
    let moment = if count > 100_000_000_000 {
        count as f64 / 1000.0
    } else {
        count as f64
    };
    Some((moment - reference).ceil().clamp(0.0, MAX_WAIT as f64) as u64)
}
