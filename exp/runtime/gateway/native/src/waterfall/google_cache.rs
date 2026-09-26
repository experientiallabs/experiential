//! One host-reserved Google cache create, with no retries or shared route mutation.

use super::{DeploymentWire, WaterfallContext};
use crate::errors::{Failure, FailureClass, PublicError};
use reqwest::Url;
use serde::Deserialize;
use serde_json::{json, Value};
use std::future::Future;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

const MAXIMUM_RESPONSE_BYTES: usize = 64 * 1024;
const MAXIMUM_LIFETIME_SECONDS: f64 = 300.0;

#[derive(Deserialize)]
#[serde(tag = "state", rename_all = "snake_case")]
enum Preparation {
    Disabled,
    Unavailable,
    Ready {
        resource_name: String,
        resource_prefix: String,
        payload: Value,
        expires_at: f64,
    },
    Create {
        operation_id: String,
        resource_prefix: String,
        url: String,
        payload: Value,
        expires_at: f64,
    },
}

#[derive(Deserialize)]
#[serde(tag = "state", rename_all = "snake_case")]
enum Completion {
    Ready { payload: Value, expires_at: f64 },
    Unavailable,
}

#[derive(Clone, Copy)]
enum EndpointPolicy {
    Official,
    #[cfg(test)]
    Loopback,
}

struct CacheEndpoint {
    url: Url,
    resource_prefix: String,
}

struct Created {
    name: String,
    expire_time: String,
    create_time: Option<String>,
    total_tokens: u64,
}

/// Resolve a private wire overlay only after the host has recorded its accounting.
pub(super) async fn prepare(
    ctx: &WaterfallContext<'_>,
    wire: &DeploymentWire,
    repaired: bool,
) -> Result<Option<DeploymentWire>, Failure> {
    execute(
        ctx.http,
        wire,
        ctx.request_id,
        ctx.deadline,
        repaired || ctx.tool_search.is_some(),
        EndpointPolicy::Official,
        |method, argument| ctx.bridge.call(method, argument),
    )
    .await
}

/// Keep the bridge seam injectable while the real pooled HTTP transport is exercised.
async fn execute<F, Fut>(
    http: &crate::upstream::UpstreamClient,
    wire: &DeploymentWire,
    request_id: &str,
    deadline: Instant,
    repaired_or_search: bool,
    policy: EndpointPolicy,
    mut call: F,
) -> Result<Option<DeploymentWire>, Failure>
where
    F: FnMut(&'static str, String) -> Fut,
    Fut: Future<Output = Result<String, PublicError>>,
{
    if !wire.explicit_cache
        || wire.dialect != "gemini_generate_content"
        || wire.upstream_body.is_some()
        || repaired_or_search
    {
        return Ok(None);
    }
    let Some(endpoint) = cache_endpoint(&wire.url, policy) else {
        return Ok(None);
    };
    ensure_remaining(deadline)?;
    let scope = json!({"request_id": request_id, "deployment_id": wire.deployment_id});
    let reply = callback(deadline, call("prepare_explicit_cache", scope.to_string())).await?;
    let preparation: Preparation = serde_json::from_str(&reply).map_err(|_| internal())?;
    match preparation {
        Preparation::Disabled | Preparation::Unavailable => {
            ensure_remaining(deadline)?;
            Ok(None)
        }
        Preparation::Ready {
            resource_name,
            resource_prefix,
            payload,
            expires_at,
        } => {
            ensure_remaining(deadline)?;
            let endpoint = bind_resource_prefix(endpoint, &resource_prefix)?;
            overlay(wire, payload, &resource_name, &endpoint, expires_at)
        }
        Preparation::Create {
            operation_id,
            resource_prefix,
            url,
            payload,
            expires_at,
        } => {
            let endpoint = bind_resource_prefix(endpoint, &resource_prefix)?;
            // The host persists an unknown claim before returning create. Dropping this
            // future anywhere below therefore quarantines it without a Drop callback.
            if operation_id.is_empty() || operation_id.len() > 256 {
                return Err(internal());
            }
            let mut status = None;
            let created = if valid_create(&endpoint, &url, &payload, expires_at)
                && wire.timeout_seconds.is_finite()
                && wire.timeout_seconds > 0.0
            {
                let bound = deadline
                    .saturating_duration_since(Instant::now())
                    .min(Duration::from_secs_f64(wire.timeout_seconds.min(300.0)));
                if bound.is_zero() {
                    None
                } else {
                    tokio::time::timeout(
                        bound,
                        create(http, wire, &endpoint, &payload, expires_at, &mut status),
                    )
                    .await
                    .ok()
                    .flatten()
                }
            } else {
                None
            };
            let mut finish = json!({
                "request_id": request_id,
                "deployment_id": wire.deployment_id,
                "operation_id": operation_id,
                "outcome": if created.is_some() { "ready" } else { "unknown" },
            });
            if let Some(status) = status {
                finish["http_status"] = json!(status);
            }
            if let Some(created) = &created {
                finish["name"] = json!(created.name);
                finish["expire_time"] = json!(created.expire_time);
                if let Some(create_time) = &created.create_time {
                    finish["create_time"] = json!(create_time);
                }
                finish["total_tokens"] = json!(created.total_tokens);
            }
            // No generation is permitted if this accounting acknowledgement fails.
            let reply =
                callback(deadline, call("finish_explicit_cache", finish.to_string())).await?;
            let completion: Completion = serde_json::from_str(&reply).map_err(|_| internal())?;
            ensure_remaining(deadline)?;
            match completion {
                Completion::Unavailable => Ok(None),
                Completion::Ready {
                    payload,
                    expires_at,
                } => {
                    let created = created.ok_or_else(internal)?;
                    overlay(wire, payload, &created.name, &endpoint, expires_at)
                }
            }
        }
    }
}

async fn callback<F>(deadline: Instant, future: F) -> Result<String, Failure>
where
    F: Future<Output = Result<String, PublicError>>,
{
    tokio::time::timeout(deadline.saturating_duration_since(Instant::now()), future)
        .await
        .map_err(|_| internal())?
        .map_err(|_| internal())
}

fn internal() -> Failure {
    Failure::new(
        FailureClass::Internal,
        "explicit cache accounting could not be confirmed; contact the gateway operator",
    )
}

fn ensure_remaining(deadline: Instant) -> Result<(), Failure> {
    if Instant::now() >= deadline {
        Err(Failure::new(
            FailureClass::Timeout,
            "gateway execution deadline exceeded",
        ))
    } else {
        Ok(())
    }
}

fn overlay(
    wire: &DeploymentWire,
    payload: Value,
    name: &str,
    endpoint: &CacheEndpoint,
    expires_at: f64,
) -> Result<Option<DeploymentWire>, Failure> {
    if !valid_resource(endpoint, name)
        || !payload.is_object()
        || payload.get("cachedContent").and_then(Value::as_str) != Some(name)
        || !expires_at.is_finite()
    {
        return Err(internal());
    }
    let mut result = wire.clone();
    result.upstream_payload = payload;
    // Recheck after the host callback and payload clone. Staleness alone needs
    // neither another create nor another callback; generation retains its input.
    if expires_at <= epoch_now() + 5.0 {
        return Ok(None);
    }
    Ok(Some(result))
}

/// Derive the create operation from an official generation path, never a host-supplied origin.
fn cache_endpoint(generation_url: &str, policy: EndpointPolicy) -> Option<CacheEndpoint> {
    let mut url = Url::parse(generation_url).ok()?;
    if !url.username().is_empty()
        || url.password().is_some()
        || url.fragment().is_some()
        || generation_url.chars().any(char::is_whitespace)
    {
        return None;
    }
    let host = url.host_str()?;
    let official = url.scheme() == "https" && url.port_or_known_default() == Some(443);
    let studio = official && host == "generativelanguage.googleapis.com";
    let vertex = official
        && (host == "aiplatform.googleapis.com"
            || host
                .strip_suffix("-aiplatform.googleapis.com")
                .is_some_and(|region| {
                    !region.is_empty()
                        && region.bytes().all(|byte| {
                            byte.is_ascii_lowercase() || byte.is_ascii_digit() || byte == b'-'
                        })
                }));
    let loopback = match policy {
        EndpointPolicy::Official => false,
        #[cfg(test)]
        EndpointPolicy::Loopback => url.scheme() == "http" && host == "127.0.0.1",
    };
    if !studio && !vertex && !loopback {
        return None;
    }
    let parts: Vec<&str> = url.path().strip_prefix('/')?.split('/').collect();
    let operation = parts.last()?.strip_suffix(":streamGenerateContent")?;
    if !identifier(operation) {
        return None;
    }
    let (path, prefix) = if (studio || loopback)
        && parts.len() == 3
        && matches!(parts[0], "v1" | "v1beta")
        && parts[1] == "models"
    {
        (
            format!("/{}/cachedContents", parts[0]),
            "cachedContents/".to_string(),
        )
    } else if (vertex || loopback)
        && parts.len() == 9
        && matches!(parts[0], "v1" | "v1beta1")
        && parts[1] == "projects"
        && identifier(parts[2])
        && parts[3] == "locations"
        && identifier(parts[4])
        && parts[5..8] == ["publishers", "google", "models"]
    {
        let prefix = format!(
            "projects/{}/locations/{}/cachedContents/",
            parts[2], parts[4]
        );
        (
            format!("/{}/{}", parts[0], prefix.trim_end_matches('/')),
            prefix,
        )
    } else {
        return None;
    };
    // Only the admitted Studio key can accompany create; alt=sse is generation-only.
    let mut auth_key = None;
    let mut seen_alt = false;
    for (name, value) in url.query_pairs() {
        match name.as_ref() {
            "key" if prefix == "cachedContents/" && auth_key.is_none() && !value.is_empty() => {
                auth_key = Some(value.into_owned());
            }
            "alt" if !seen_alt && value == "sse" => seen_alt = true,
            _ => return None,
        }
    }
    url.set_path(&path);
    url.set_query(None);
    if let Some(key) = auth_key {
        url.query_pairs_mut().append_pair("key", &key);
    }
    Some(CacheEndpoint {
        url,
        resource_prefix: prefix,
    })
}

/// Bind only the host-verified numeric namespace, preserving the admitted create URL.
fn bind_resource_prefix(
    mut endpoint: CacheEndpoint,
    authorized: &str,
) -> Result<CacheEndpoint, Failure> {
    if endpoint.resource_prefix == "cachedContents/" {
        return if authorized == endpoint.resource_prefix {
            Ok(endpoint)
        } else {
            Err(internal())
        };
    }
    let original: Vec<&str> = endpoint.resource_prefix.split('/').collect();
    let canonical: Vec<&str> = authorized.split('/').collect();
    if original.len() != 6
        || canonical.len() != 6
        || canonical[0] != "projects"
        || canonical[2] != "locations"
        || canonical[3] != original[3]
        || canonical[4] != "cachedContents"
        || !canonical[5].is_empty()
        || canonical[1].is_empty()
        || canonical[1].len() > 20
        || canonical[1].starts_with('0')
        || !canonical[1].bytes().all(|byte| byte.is_ascii_digit())
        || (original[1].bytes().all(|byte| byte.is_ascii_digit()) && original[1] != canonical[1])
    {
        return Err(internal());
    }
    // Only the trusted preparation callback supplies this mapping. Provider names
    // must match it exactly; they cannot change the project, location or origin.
    endpoint.resource_prefix = authorized.to_string();
    Ok(endpoint)
}

fn identifier(value: &str) -> bool {
    !value.is_empty()
        && value.len() <= 256
        && value != "."
        && value != ".."
        && value
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'_' | b'-' | b'.'))
}

fn valid_resource(endpoint: &CacheEndpoint, name: &str) -> bool {
    name.strip_prefix(&endpoint.resource_prefix)
        .is_some_and(identifier)
}

fn epoch_now() -> f64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(f64::INFINITY, |time| time.as_secs_f64())
}

fn valid_create(endpoint: &CacheEndpoint, url: &str, payload: &Value, expires_at: f64) -> bool {
    let now = epoch_now();
    expires_at.is_finite()
        && expires_at > now + 5.0
        && expires_at <= now + MAXIMUM_LIFETIME_SECONDS
        && matching_create_url(&endpoint.url, url)
        && payload.is_object()
        && payload.get("ttl").is_none()
        && payload
            .get("expireTime")
            .and_then(Value::as_str)
            .and_then(expiry_epoch)
            .is_some_and(|absolute| absolute == expires_at)
}

/// Compare the complete operation and decoded auth query without accepting added parameters.
fn matching_create_url(expected: &Url, candidate: &str) -> bool {
    if candidate.chars().any(char::is_whitespace) {
        return false;
    }
    let Ok(mut candidate) = Url::parse(candidate) else {
        return false;
    };
    if !candidate.query_pairs().eq(expected.query_pairs()) {
        return false;
    }
    candidate.set_query(expected.query());
    candidate == *expected
}

/// Read only bounded JSON evidence. Error bodies and headers never cross the bridge.
async fn create(
    http: &crate::upstream::UpstreamClient,
    wire: &DeploymentWire,
    endpoint: &CacheEndpoint,
    payload: &Value,
    expires_at: f64,
    status: &mut Option<u16>,
) -> Option<Created> {
    let mut request = http.post(endpoint.url.as_str()).ok()?;
    for (name, value) in &wire.headers {
        if !name.eq_ignore_ascii_case("idempotency-key")
            && !name.eq_ignore_ascii_case("content-length")
            && !name.eq_ignore_ascii_case("content-type")
        {
            request = request.header(name, value);
        }
    }
    // ctx.http is built by upstream::build_client with redirect::none and retry::never.
    let mut response = request.json(payload).send().await.ok()?;
    *status = Some(response.status().as_u16());
    if !response.status().is_success()
        || response
            .content_length()
            .is_some_and(|length| length > MAXIMUM_RESPONSE_BYTES as u64)
    {
        return None;
    }
    let mut body = Vec::new();
    while let Some(chunk) = response.chunk().await.ok()? {
        if body.len().saturating_add(chunk.len()) > MAXIMUM_RESPONSE_BYTES {
            return None;
        }
        body.extend_from_slice(&chunk);
    }
    let body: Value = serde_json::from_slice(&body).ok()?;
    let name = body.get("name")?.as_str()?;
    let expire_time = body.get("expireTime")?.as_str()?;
    let actual_expiry = expiry_epoch(expire_time)?;
    let total_tokens = body
        .get("usageMetadata")?
        .get("totalTokenCount")?
        .as_u64()?;
    if !valid_resource(endpoint, name)
        || total_tokens == 0
        || actual_expiry <= 0.0
        || actual_expiry > expires_at
    {
        return None;
    }
    // Optional creation evidence does not decide resource usability. Never infer
    // a missing or malformed timestamp from the offer, response arrival or TTL.
    let observed_at = epoch_now();
    let create_time = body
        .get("createTime")
        .and_then(Value::as_str)
        .filter(|value| {
            expiry_epoch(value).is_some_and(|created| {
                observed_at.is_finite()
                    && created > 0.0
                    && created <= actual_expiry
                    && created <= observed_at
            })
        });
    // Expired resources still have known billable facts. The host records those
    // facts first and separately decides whether the resource can serve a generation.
    Some(Created {
        name: name.to_string(),
        expire_time: expire_time.to_string(),
        create_time: create_time.map(str::to_owned),
        total_tokens,
    })
}

/// Parse bounded UTC RFC3339 timestamps used by Google's resource interval.
fn expiry_epoch(value: &str) -> Option<f64> {
    if !value.is_ascii() || value.len() < 20 || value.len() > 35 {
        return None;
    }
    let value = value
        .strip_suffix('Z')
        .or_else(|| value.strip_suffix("+00:00"))?;
    if value.len() < 19
        || &value[4..5] != "-"
        || &value[7..8] != "-"
        || &value[10..11] != "T"
        || &value[13..14] != ":"
        || &value[16..17] != ":"
    {
        return None;
    }
    let number = |start, end| value.get(start..end)?.parse::<u32>().ok();
    let year = number(0, 4)?;
    let month = number(5, 7)?;
    let day = number(8, 10)?;
    let hour = number(11, 13)?;
    let minute = number(14, 16)?;
    let second = number(17, 19)?;
    if !(1970..=9999).contains(&year)
        || !(1..=12).contains(&month)
        || hour > 23
        || minute > 59
        || second > 59
    {
        return None;
    }
    let leap = |year: u32| {
        year.is_multiple_of(4) && (!year.is_multiple_of(100) || year.is_multiple_of(400))
    };
    let days = [
        31,
        if leap(year) { 29 } else { 28 },
        31,
        30,
        31,
        30,
        31,
        31,
        30,
        31,
        30,
        31,
    ];
    if day == 0 || day > days[month as usize - 1] {
        return None;
    }
    let fraction = if value.len() == 19 {
        0.0
    } else {
        let digits = value[19..].strip_prefix('.')?;
        if digits.is_empty()
            || digits.len() > 9
            || !digits.bytes().all(|byte| byte.is_ascii_digit())
        {
            return None;
        }
        value[19..].parse::<f64>().ok()?
    };
    let prior = year - 1;
    let elapsed_year_days = 365 * (year - 1970) + (prior / 4 - prior / 100 + prior / 400)
        - (1969 / 4 - 1969 / 100 + 1969 / 400);
    let elapsed_days = elapsed_year_days + days[..month as usize - 1].iter().sum::<u32>() + day - 1;
    Some(
        f64::from(elapsed_days) * 86400.0
            + f64::from(hour * 3600 + minute * 60 + second)
            + fraction,
    )
}

#[cfg(test)]
#[path = "google_cache_tests.rs"]
mod tests;

#[cfg(test)]
#[path = "google_cache_bridge_tests.rs"]
mod bridge_tests;

#[cfg(test)]
#[path = "google_cache_scope_tests.rs"]
mod scope_tests;
