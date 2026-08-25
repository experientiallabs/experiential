//! Upstream provider HTTP transport over one shared pooled client.

use std::collections::HashMap;
use std::time::Duration;

use serde_json::Value;

use crate::errors::{Failure, FailureClass};

/// Build the shared pooled upstream client, mirroring the pooling constants in
/// `providers.async_transport` (64 keep-alive) and its no-redirect policy so a
/// provider 3xx can never re-send credentials to an attacker-chosen location.
pub fn build_client() -> Result<reqwest::Client, String> {
    reqwest::Client::builder()
        .pool_max_idle_per_host(64)
        .connect_timeout(Duration::from_secs(10))
        .redirect(reqwest::redirect::Policy::none())
        .use_rustls_tls()
        .build()
        .map_err(|error| format!("upstream client construction failed: {error}"))
}

/// Classify one sanitized HTTP or connection failure by status only,
/// mirroring `providers.errors._transport_failure`: classes, wording, the
/// same-deployment retry policy, and failover eligibility across the
/// certified deployment ladder.
pub fn transport_failure(status: Option<u16>) -> Failure {
    let (class, message, retryable, failover) = match status {
        Some(401) | Some(403) => (
            FailureClass::ProviderAuthentication,
            "provider authentication failed; ask the gateway operator to verify \
             the provider connection credential",
            false,
            true,
        ),
        Some(404) => (
            FailureClass::ProviderNotFound,
            "provider deployment was not found; ask the gateway operator to verify \
             the deployment model ID in the catalog",
            false,
            true,
        ),
        Some(429) => (
            FailureClass::Throttled,
            "provider throttled the request; retry after the delay in the Retry-After header",
            false,
            true,
        ),
        Some(408) => (
            FailureClass::Timeout,
            "provider request timed out; retry the request",
            true,
            true,
        ),
        Some(code) if code >= 500 => (
            FailureClass::ProviderInternal,
            "provider service failed; retry after a short delay",
            true,
            true,
        ),
        Some(409) | Some(425) => (
            FailureClass::ProviderInternal,
            "provider reported a transient conflict; retry the request",
            true,
            true,
        ),
        Some(code) if (400..500).contains(&code) => (
            FailureClass::InvalidRequest,
            "provider rejected the request; verify the request fields against \
             the model alias capabilities",
            false,
            false,
        ),
        // Redirects are disabled, so a 3xx (or any other status) is an
        // unexpected provider response, never followed.
        Some(_) => (
            FailureClass::ProviderInternal,
            "provider returned an unexpected status; retry the request",
            false,
            true,
        ),
        None => (
            FailureClass::Transport,
            "provider transport failed; retry the request",
            true,
            true,
        ),
    };
    Failure::new(class, message).with_retry(retryable, failover)
}

/// The classified open-phase timeout, matching `_transport_failure(408)`.
fn open_timeout_failure() -> Failure {
    Failure::new(
        FailureClass::Timeout,
        "provider request timed out; retry the request",
    )
    .with_retry(true, true)
}

/// Open one streaming POST and return the response on HTTP success. The
/// timeout bounds only the request/response-header phase; body-read pacing is
/// bounded per chunk by the caller, mirroring the python transport split.
///
/// `raw_body` carries the exact pre-serialized body for body-signing dialects
/// (Bedrock SigV4): its signature covers those exact bytes, so it is sent
/// verbatim with the signed headers instead of re-serializing `payload`.
pub async fn open_stream(
    client: &reqwest::Client,
    url: &str,
    headers: &HashMap<String, String>,
    idempotency_key: &str,
    payload: &Value,
    raw_body: Option<&str>,
    phase_timeout: Duration,
) -> Result<reqwest::Response, Failure> {
    let mut request = client.post(url);
    for (name, value) in headers {
        if name.eq_ignore_ascii_case("idempotency-key") {
            continue;
        }
        request = request.header(name, value);
    }
    request = request.header("Idempotency-Key", idempotency_key);
    let send = match raw_body {
        Some(body) => request.body(body.to_string()).send(),
        None => request.json(payload).send(),
    };
    let response = match tokio::time::timeout(phase_timeout, send).await {
        Ok(Ok(response)) => response,
        Ok(Err(error)) => {
            if error.is_timeout() {
                return Err(open_timeout_failure());
            }
            return Err(transport_failure(None));
        }
        Err(_) => return Err(open_timeout_failure()),
    };
    let status = response.status().as_u16();
    if !(200..300).contains(&status) {
        return Err(transport_failure(Some(status)));
    }
    Ok(response)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn transport_failure_flags_mirror_the_python_taxonomy() {
        // (status, retryable_same_deployment, failover_eligible)
        let table = [
            (Some(401), false, true),
            (Some(403), false, true),
            (Some(404), false, true),
            (Some(429), false, true),
            (Some(408), true, true),
            (Some(500), true, true),
            (Some(503), true, true),
            (Some(409), true, true),
            (Some(425), true, true),
            (Some(400), false, false),
            (Some(422), false, false),
            (Some(301), false, true),
            (None, true, true),
        ];
        for (status, retryable, failover) in table {
            let failure = transport_failure(status);
            assert_eq!(
                failure.retryable_same_deployment, retryable,
                "retryable for {status:?}"
            );
            assert_eq!(
                failure.failover_eligible, failover,
                "failover for {status:?}"
            );
        }
    }
}
