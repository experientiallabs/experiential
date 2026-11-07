//! Convert trusted native phase evidence into one frozen accounting terminal.

use std::time::SystemTime;

use chrono::{DateTime, Utc};
use serde::Deserialize;
use serde_json::{json, Value};

use crate::errors::{Failure, FailureClass, PublicError};

use super::{estimate, observations};

#[derive(Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct Terms {
    pub surface: String,
    pub counted_input_tokens: u64,
    pub output_ceiling: Option<u64>,
    pub output_budget: Option<u64>,
    #[serde(deserialize_with = "crate::lossless_json::deserialize")]
    pub service_tier: Value,
    pub self_served: bool,
    pub paid_prework: bool,
}

#[derive(Deserialize)]
#[serde(default)]
struct Provenance {
    graceful_transport_retirement: bool,
    opened: Option<bool>,
    dispatched: bool,
    finalize: bool,
    usage_incomplete_due_to_disconnect: bool,
    usage_unreported_truncation: bool,
}

impl Default for Provenance {
    fn default() -> Self {
        Self {
            graceful_transport_retirement: false,
            opened: None,
            dispatched: false,
            finalize: true,
            usage_incomplete_due_to_disconnect: false,
            usage_unreported_truncation: false,
        }
    }
}

pub(super) fn normalize(
    data: &Value,
    terms: &Terms,
    cached: f64,
    now: SystemTime,
) -> Result<Value, PublicError> {
    normalize_inner(data, terms, cached, now, true)
}

/// Check owned identities and phase evidence before they can win retained state.
pub(super) fn validate(data: &Value, terms: &Terms) -> Result<(), PublicError> {
    normalize_inner(data, terms, 0.0, SystemTime::now(), false).map(|_| ())
}

fn normalize_inner(
    data: &Value,
    terms: &Terms,
    cached: f64,
    now: SystemTime,
    estimate_usage: bool,
) -> Result<Value, PublicError> {
    let provenance: Provenance =
        crate::lossless_json::from_value(data.clone()).map_err(|_| PublicError::internal())?;
    let outcome = data["outcome"]
        .as_str()
        .filter(|s| matches!(*s, "completed" | "incomplete" | "failed"))
        .ok_or_else(PublicError::internal)?;
    let mut failure = if data["failure"].is_object() {
        Some(
            crate::lossless_json::from_value::<Failure>(data["failure"].clone())
                .map_err(|_| PublicError::internal())?,
        )
    } else {
        None
    };
    let observed = observations::parse(&data["rate_limit_headers"], now);
    if let Some(failure) = &mut failure {
        if failure.failure_class == FailureClass::Throttled && failure.retry_after_seconds.is_none()
        {
            failure.retry_after_seconds = observed
                .retry_after_seconds
                .or_else(|| observed.exhausted_wait())
                .map(|v| v as u32);
        }
        if failure.customer_owned
            && matches!(
                failure.failure_class,
                FailureClass::ProviderAuthentication | FailureClass::ProviderQuota
            )
        {
            failure.failure_class = FailureClass::InvalidRequest;
        }
    }
    let class = failure.as_ref().map(|failure| failure.failure_class);
    if (provenance.usage_incomplete_due_to_disconnect
        && (!provenance.dispatched
            || !provenance.finalize
            || outcome != "failed"
            || class != Some(FailureClass::Cancelled)))
        || (provenance.usage_unreported_truncation
            && (!provenance.dispatched || failure.is_some() || outcome != "incomplete"))
        || (provenance.graceful_transport_retirement
            && (outcome != "failed"
                || provenance.opened != Some(false)
                || !data["usage"].is_null()
                || class != Some(FailureClass::Transport)))
    {
        return Err(PublicError::internal());
    }
    let mut usage = usage(data)?;
    if outcome != "failed"
        && usage.as_ref().is_some_and(|usage| {
            usage["billed_units"].is_null()
                && usage["input_tokens"] == 0
                && usage["output_tokens"] == 0
        })
    {
        usage = None;
    }
    let estimate = (provenance.usage_incomplete_due_to_disconnect
        || provenance.usage_unreported_truncation)
        && provenance.opened == Some(true)
        && matches!(
            terms.surface.as_str(),
            "chat_completions" | "responses" | "messages"
        );
    let mut estimated = false;
    if estimate && estimate_usage {
        if let Some(value) = estimate::usage(
            data,
            usage.as_ref(),
            terms.counted_input_tokens,
            cached,
            if provenance.usage_unreported_truncation {
                terms.output_budget.unwrap_or(0)
            } else {
                0
            },
        )? {
            usage = Some(value);
            estimated = true;
        }
    }
    if let (Some(usage), Some(ceiling)) = (&mut usage, terms.output_ceiling) {
        if usage["output_tokens"].as_u64().is_some_and(|n| n > ceiling) {
            usage["output_tokens"] = json!(ceiling);
            for subset in ["reasoning_tokens", "image_output_tokens"] {
                if let Some(value) = usage[subset].as_u64() {
                    usage[subset] = json!(value.min(ceiling));
                }
            }
        }
    }
    let first = data["first_token_at"]
        .as_str()
        .and_then(|text| DateTime::parse_from_rfc3339(text).ok())
        .map(|at| at.with_timezone(&Utc));
    let provider = data["upstream_provider"]
        .as_str()
        .map(str::trim)
        .filter(|text| !text.is_empty() && text.chars().count() <= 128);
    let tier = tier(
        &terms.service_tier,
        &data["service_tier"],
        terms.self_served,
    )?;
    let state = match class {
        Some(FailureClass::Cancelled) => "cancelled",
        Some(_) => "failed",
        None => outcome,
    };
    let searches = usage
        .as_ref()
        .and_then(|v| v["web_search_requests"].as_u64())
        .unwrap_or(0);
    let tool_searches = usage
        .as_ref()
        .and_then(|v| v["tool_search_requests"].as_u64())
        .unwrap_or(0);
    let failure = failure.as_ref().map(|f| {
        json!({"failure_class": f.failure_class.as_str(), "safe_message": f.safe_message,
        "provider_detail": f.provider_detail.as_ref().filter(|value| !value.is_empty()),
        "customer_owned": f.customer_owned, "retry_after_seconds": f.retry_after_seconds})
    });
    Ok(
        json!({"state": state, "failure": failure, "usage": usage, "first_token_at": first,
            "usage_estimated": estimated, "usage_incomplete_due_to_disconnect": provenance.usage_incomplete_due_to_disconnect,
            "decision_provider_rejected": terms.surface == "decisions" && outcome == "failed" && usage.is_none()
                && provenance.opened == Some(false) && data["decision_provider_rejected"] == true,
            "finalize": provenance.finalize,
            "observations": {"retry_after_seconds": observed.retry_after_seconds,
                "ratelimit_limit_requests": observed.limit_requests, "ratelimit_remaining_requests": observed.remaining_requests,
                "ratelimit_limit_tokens": observed.limit_tokens, "ratelimit_remaining_tokens": observed.remaining_tokens,
                "upstream_provider": provider, "web_search_requests": searches},
            "service_tier": tier, "incomplete_reason": data["incomplete_reason"].as_str(), "tool_search_requests": tool_searches,
            "stream_opened": provenance.opened, "paid_prework": provenance.opened.map(|_| terms.paid_prework)
        }),
    )
}

pub(super) const TOKEN_FIELDS: [&str; 8] = [
    "input_tokens",
    "output_tokens",
    "cached_input_tokens",
    "cache_creation_input_tokens",
    "cache_creation_1h_input_tokens",
    "cache_creation_12h_input_tokens",
    "reasoning_tokens",
    "image_output_tokens",
];

pub(super) fn usage(data: &Value) -> Result<Option<Value>, PublicError> {
    let payload = &data["usage"];
    let names: Vec<_> = data["tool_names"]
        .as_array()
        .map(|items| items.iter().filter_map(Value::as_str).collect())
        .unwrap_or_default();
    let billed = &payload["billed_units"];
    if !billed.is_null()
        && (!billed.is_object()
            || !matches!(
                billed["kind"].as_str(),
                Some("character" | "audio_second" | "video_second" | "image")
            )
            || billed["quantity_milli"].as_u64().is_none()
            || billed.get("variant").is_some_and(|v| !v.is_string()))
    {
        return Err(PublicError::internal());
    }
    let tokened = !payload["input_tokens"].is_null() || !payload["output_tokens"].is_null();
    if !tokened && names.is_empty() && billed.is_null() {
        return Ok(None);
    }
    let mut result = json!({"tool_names": names, "billed_units": if billed.is_null() {Value::Null} else {
        json!({"kind": billed["kind"], "quantity_milli": billed["quantity_milli"], "variant": billed.get("variant").cloned().unwrap_or(json!(""))})}});
    for field in TOKEN_FIELDS {
        if tokened && payload[field].as_i64().is_some_and(|n| n < 0) {
            return Err(PublicError::internal());
        }
        result[field] = json!(tokened.then(|| payload[field].as_u64()).flatten());
    }
    if result["input_tokens"].is_null()
        && result["output_tokens"].is_null()
        && result["tool_names"].as_array().unwrap().is_empty()
        && billed.is_null()
    {
        return Err(PublicError::internal());
    }
    for (subset, total) in [
        (
            "cache_creation_1h_input_tokens",
            "cache_creation_input_tokens",
        ),
        (
            "cache_creation_12h_input_tokens",
            "cache_creation_input_tokens",
        ),
        ("image_output_tokens", "output_tokens"),
    ] {
        if let Some(value) = result[subset].as_u64() {
            let covering = result[total].as_u64().ok_or_else(PublicError::internal)?;
            let additional = if subset == "cache_creation_12h_input_tokens" {
                result["cache_creation_1h_input_tokens"]
                    .as_u64()
                    .unwrap_or(0)
            } else {
                0
            };
            if u128::from(value) + u128::from(additional) > u128::from(covering) {
                return Err(PublicError::internal());
            }
        }
    }
    search_counts(&mut result, data);
    Ok(Some(result))
}

pub(super) fn search_counts(usage: &mut Value, data: &Value) {
    let searches = data["web_search_requests"].as_u64().unwrap_or(0);
    usage["web_search_requests"] = json!(searches);
    usage["tool_search_requests"] = json!(data["tool_search_requests"].as_u64().unwrap_or(0));
    for field in ["web_search_results", "web_search_cost_nano_usd"] {
        usage[field] = json!((searches > 0).then(|| data[field].as_u64()).flatten());
    }
}

fn tier(admission: &Value, observed: &Value, self_served: bool) -> Result<Value, PublicError> {
    if admission.is_null() {
        return Ok(Value::Null);
    }
    if !observed.is_null()
        && (!observed.is_object()
            || observed.as_object().is_some_and(|fields| {
                fields
                    .keys()
                    .any(|key| !matches!(key.as_str(), "resolution" | "served"))
            })
            || observed
                .get("resolution")
                .is_some_and(|value| !value.is_string())
            || observed
                .get("served")
                .is_some_and(|value| !value.is_null() && !value.is_string()))
    {
        return Err(PublicError::internal());
    }
    let requested = admission["requested"]
        .as_str()
        .filter(|tier| matches!(*tier, "priority" | "flex"))
        .ok_or_else(PublicError::internal)?;
    let mut resolution = observed["resolution"].as_str().unwrap_or("missing");
    let mut served = observed["served"].as_str();
    if !matches!(
        resolution,
        "confirmed" | "missing" | "unknown" | "conflicting"
    ) || served.is_some_and(|tier| !matches!(tier, "default" | "priority" | "flex"))
    {
        return Err(PublicError::internal());
    }
    if self_served && resolution == "missing" {
        resolution = "confirmed";
        served = Some(requested);
    }
    let prices = if resolution == "confirmed" {
        if served == Some("default") {
            admission["standard_prices"].clone()
        } else if served == Some(requested) {
            admission["requested_prices"].clone()
        } else {
            resolution = "unknown";
            Value::Null
        }
    } else {
        Value::Null
    };
    Ok(
        json!({"requested": requested, "served": if prices.is_object() {served} else {None},
        "resolution": resolution, "prices": prices}),
    )
}
