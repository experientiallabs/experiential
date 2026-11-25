//! Google's structured verdicts on the ACCOUNT behind a Gemini-wire rung,
//! read from the `google.rpc.ErrorInfo` reason of an error body.
//!
//! The Cloud Code Assist backend an Antigravity plan dispatches to answers
//! three account states the HTTP status alone cannot tell apart:
//!
//! * `429 RESOURCE_EXHAUSTED` with reason `QUOTA_EXHAUSTED` or
//!   `INSUFFICIENT_G1_CREDITS_BALANCE`: the plan's quota window (or its
//!   credit balance) is spent. That is the account's funding state, filed as
//!   `ProviderQuota` with the stated reset as its wait, never a throttle.
//! * `429 RESOURCE_EXHAUSTED` with reason `RATE_LIMIT_EXCEEDED`: a throttle;
//!   its `RetryInfo.retryDelay` is the wait when no `Retry-After` header came.
//! * `403 PERMISSION_DENIED` with reason `TOS_VIOLATION`: Google suspended
//!   the account. It is filed as a credential failure (never redialed, the
//!   ladder may try another account) marked `provider_account_suspended`,
//!   with the appeal URL Google returned, so the host can pull the account and
//!   tell its owner where to appeal.
//!
//! Only the structured reason and Google's own metadata fields are read; the
//! sentence is never parsed for routing.

use serde_json::Value;

use crate::dialects::Dialect;
use crate::error_envelope::parse_error_document;
use crate::errors::{Failure, FailureClass};

/// ErrorInfo reasons naming an exhausted plan quota or credit balance.
const QUOTA_REASONS: &[&str] = &["QUOTA_EXHAUSTED", "INSUFFICIENT_G1_CREDITS_BALANCE"];

/// The ErrorInfo reason for an account Google suspended.
const SUSPENSION_REASON: &str = "TOS_VIOLATION";

/// Longest appeal URL carried to settlement; with its prefix it stays within
/// the 240-character ledger detail.
pub const MAXIMUM_APPEAL_URL_CHARS: usize = 200;

/// Separates the appeal URL from the reason in a suspension's ledger detail.
const APPEAL_MARKER: &str = "; appeal_url=";

/// Upper bound on a body-stated wait (one week), so one absurd value cannot
/// suppress an account for longer than any plan window lasts.
const MAXIMUM_STATED_WAIT_SECONDS: f64 = 7.0 * 24.0 * 3_600.0;

/// The caller-facing message of a suspended account.
pub const SUSPENDED_MESSAGE: &str = "the provider suspended the account behind this connection \
     for a terms-of-service violation; the account owner must appeal to the provider, and the \
     gateway operator should disconnect the account";

/// The caller-facing message of an exhausted plan account.
const QUOTA_MESSAGE: &str = "the provider account's plan quota or credit balance is exhausted; \
     it resets on the provider's schedule, or the gateway operator can fund the account";

/// One account verdict read from a Google error body.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GoogleAccountVerdict {
    /// The account's quota window or credit balance is spent.
    QuotaExhausted {
        reason: String,
        reset_seconds: Option<u32>,
    },
    /// Google suspended the account.
    Suspended { appeal_url: Option<String> },
}

impl GoogleAccountVerdict {
    /// The failure this verdict files, with `status_detail` (`http 429`)
    /// heading its ledger detail.
    pub fn failure(&self, status_detail: &str) -> Failure {
        match self {
            GoogleAccountVerdict::QuotaExhausted {
                reason,
                reset_seconds,
            } => {
                let mut failure = Failure::new(FailureClass::ProviderQuota, QUOTA_MESSAGE)
                    .with_retry(false, true)
                    .with_provider_detail(Some(format!("{status_detail}: {reason}")));
                failure.retry_after_seconds = *reset_seconds;
                failure
            }
            GoogleAccountVerdict::Suspended { appeal_url } => {
                let appeal = appeal_url
                    .as_deref()
                    .map(|url| format!("{APPEAL_MARKER}{url}"))
                    .unwrap_or_default();
                let mut failure =
                    Failure::new(FailureClass::ProviderAuthentication, SUSPENDED_MESSAGE)
                        .with_retry(false, true)
                        .with_provider_detail(Some(format!(
                            "{status_detail}: {SUSPENSION_REASON}{appeal}"
                        )));
                failure.account_suspended = true;
                failure
            }
        }
    }
}

/// The appeal URL of a suspended account's failure, read back from the
/// ledger detail the verdict wrote it into.
pub fn appeal_url_of(failure: &Failure) -> Option<&str> {
    if !failure.account_suspended {
        return None;
    }
    let detail = failure.provider_detail.as_deref()?;
    let start = detail.find(APPEAL_MARKER)? + APPEAL_MARKER.len();
    Some(&detail[start..])
}

/// Read Google's account verdict off one Gemini-wire error body, when it
/// states one under the status it is documented with.
pub fn google_account_verdict(
    dialect: Dialect,
    status: u16,
    body: Option<&str>,
) -> Option<GoogleAccountVerdict> {
    if dialect != Dialect::GeminiGenerateContent {
        return None;
    }
    let value = parse_error_document(body?)?;
    let error = google_error(&value)?;
    let details = error.get("details").and_then(Value::as_array)?;
    let info = details
        .iter()
        .find(|detail| is_detail(detail, "google.rpc.ErrorInfo"))?;
    let reason = info.get("reason").and_then(Value::as_str)?.trim();
    match status {
        429 if QUOTA_REASONS.contains(&reason) => Some(GoogleAccountVerdict::QuotaExhausted {
            reason: reason.to_string(),
            reset_seconds: google_retry_delay_seconds(&value),
        }),
        403 if reason == SUSPENSION_REASON => Some(GoogleAccountVerdict::Suspended {
            appeal_url: info
                .get("metadata")
                .and_then(|metadata| metadata.get("appeal_url"))
                .and_then(Value::as_str)
                .and_then(appeal_url),
        }),
        _ => None,
    }
}

/// The `google.rpc.ErrorInfo` reason of a Gemini-wire error body, when it is a
/// bounded identifier (`RATE_LIMIT_EXCEEDED`): the token the ledger detail
/// records so the host can tell Google's account and window states apart.
pub fn google_error_info_reason(dialect: Dialect, body: Option<&str>) -> Option<String> {
    if dialect != Dialect::GeminiGenerateContent {
        return None;
    }
    let value = parse_error_document(body?)?;
    let details = google_error(&value)?.get("details")?.as_array()?;
    let reason = details
        .iter()
        .find(|detail| is_detail(detail, "google.rpc.ErrorInfo"))?
        .get("reason")?
        .as_str()?
        .trim();
    let identifier = !reason.is_empty()
        && reason.len() <= 64
        && reason
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || matches!(c, '_' | '.' | '-'));
    identifier.then(|| reason.to_string())
}

/// The wait a Google error body states: `RetryInfo.retryDelay`, else
/// `ErrorInfo.metadata.quotaResetDelay`, rounded up to whole seconds.
pub fn google_body_retry_seconds(dialect: Dialect, body: Option<&str>) -> Option<u32> {
    if dialect != Dialect::GeminiGenerateContent {
        return None;
    }
    google_retry_delay_seconds(&parse_error_document(body?)?)
}

/// The `error` object of a Google envelope; a streaming endpoint may wrap
/// the envelope in a one-element array.
fn google_error(value: &Value) -> Option<&Value> {
    match value {
        Value::Array(items) => items.first()?.get("error"),
        other => other.get("error"),
    }
}

fn is_detail(detail: &Value, kind: &str) -> bool {
    detail
        .get("@type")
        .and_then(Value::as_str)
        .is_some_and(|value| value.ends_with(kind))
}

fn google_retry_delay_seconds(value: &Value) -> Option<u32> {
    let details = google_error(value)?.get("details")?.as_array()?;
    let stated = details
        .iter()
        .filter(|detail| is_detail(detail, "google.rpc.RetryInfo"))
        .find_map(|detail| detail.get("retryDelay").and_then(Value::as_str))
        .or_else(|| {
            details
                .iter()
                .filter(|detail| is_detail(detail, "google.rpc.ErrorInfo"))
                .find_map(|detail| {
                    detail
                        .get("metadata")
                        .and_then(|metadata| metadata.get("quotaResetDelay"))
                        .and_then(Value::as_str)
                })
        })?;
    let seconds = duration_seconds(stated)?;
    Some(seconds.min(MAXIMUM_STATED_WAIT_SECONDS).ceil().max(1.0) as u32)
}

/// Parse a Go/proto duration such as `3.5s`, `373.8ms` or `1h2m3s`.
fn duration_seconds(text: &str) -> Option<f64> {
    let mut rest = text.trim();
    if rest.is_empty() {
        return None;
    }
    let mut total = 0.0;
    while !rest.is_empty() {
        let number_end = rest
            .find(|c: char| !(c.is_ascii_digit() || c == '.'))
            .unwrap_or(rest.len());
        if number_end == 0 {
            return None;
        }
        let number: f64 = rest[..number_end].parse().ok()?;
        rest = &rest[number_end..];
        let (scale, unit_len) = if rest.starts_with("ms") {
            (0.001, 2)
        } else if rest.starts_with('h') {
            (3_600.0, 1)
        } else if rest.starts_with('m') {
            (60.0, 1)
        } else if rest.starts_with('s') {
            (1.0, 1)
        } else {
            return None;
        };
        total += number * scale;
        rest = &rest[unit_len..];
    }
    total.is_finite().then_some(total)
}

/// Keep an appeal URL only when it is one bounded https URL. It is kept whole
/// (its query can name the appeal form), because its reader is the account's
/// owner.
fn appeal_url(raw: &str) -> Option<String> {
    let url = raw.trim();
    let usable = url.starts_with("https://")
        && url.len() <= MAXIMUM_APPEAL_URL_CHARS
        && !url.chars().any(|c| c.is_whitespace() || c.is_control());
    usable.then(|| url.to_string())
}

#[cfg(test)]
#[path = "google_account_verdicts_tests.rs"]
mod tests;
