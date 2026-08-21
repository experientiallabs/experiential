//! Upstream provider HTTP transport over one shared pooled client.

use std::collections::HashMap;
use std::time::Duration;

use serde_json::Value;

use crate::errors::{Failure, FailureClass};

/// Per-token completion allowance mirroring
/// `providers.base.COMPLETION_SECONDS_PER_OUTPUT_TOKEN`.
const COMPLETION_SECONDS_PER_OUTPUT_TOKEN: f64 = 0.03;
const MAXIMUM_COMPLETION_TIMEOUT_SECONDS: f64 = 600.0;

/// Derive one bounded completion timeout, mirroring
/// `providers.base.completion_timeout_seconds`.
pub fn completion_timeout_seconds(
    configured_timeout_seconds: f64,
    maximum_output_tokens: Option<u64>,
) -> f64 {
    match maximum_output_tokens {
        None => configured_timeout_seconds,
        Some(tokens) => {
            let scaled = tokens as f64 * COMPLETION_SECONDS_PER_OUTPUT_TOKEN;
            configured_timeout_seconds.max(scaled.min(MAXIMUM_COMPLETION_TIMEOUT_SECONDS))
        }
    }
}

/// Build the shared pooled upstream client, mirroring the pooling constants in
/// `providers.async_transport` (256 connections, 64 keep-alive).
pub fn build_client() -> Result<reqwest::Client, String> {
    reqwest::Client::builder()
        .pool_max_idle_per_host(64)
        .connect_timeout(Duration::from_secs(10))
        .use_rustls_tls()
        .build()
        .map_err(|error| format!("upstream client construction failed: {error}"))
}

fn transport_failure(status: Option<u16>) -> Failure {
    let (class, message) = match status {
        Some(401) | Some(403) => (
            FailureClass::ProviderAuthentication,
            "provider rejected the configured credential",
        ),
        Some(404) => (
            FailureClass::ProviderNotFound,
            "provider endpoint or model was not found",
        ),
        Some(408) => (FailureClass::Timeout, "provider request timed out"),
        Some(429) => (FailureClass::Throttled, "provider throttled the request"),
        Some(code) if code >= 500 => (
            FailureClass::ProviderInternal,
            "provider returned an internal error",
        ),
        Some(_) => (FailureClass::InvalidRequest, "provider rejected the request"),
        None => (FailureClass::Transport, "provider transport failed"),
    };
    Failure::new(class, message)
}

/// Open one streaming POST and return the response on HTTP success.
pub async fn open_stream(
    client: &reqwest::Client,
    url: &str,
    headers: &HashMap<String, String>,
    idempotency_key: &str,
    payload: &Value,
    phase_timeout: Duration,
) -> Result<reqwest::Response, Failure> {
    let mut request = client.post(url).timeout(phase_timeout);
    for (name, value) in headers {
        if name.eq_ignore_ascii_case("idempotency-key") {
            continue;
        }
        request = request.header(name, value);
    }
    request = request.header("Idempotency-Key", idempotency_key);
    let response = match request.json(payload).send().await {
        Ok(response) => response,
        Err(error) => {
            if error.is_timeout() {
                return Err(Failure::new(
                    FailureClass::Timeout,
                    "provider request timed out",
                ));
            }
            return Err(transport_failure(None));
        }
    };
    let status = response.status().as_u16();
    if !(200..300).contains(&status) {
        return Err(transport_failure(Some(status)));
    }
    Ok(response)
}
