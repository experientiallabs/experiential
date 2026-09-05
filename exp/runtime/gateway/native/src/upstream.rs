//! Upstream provider HTTP transport over one shared pooled client.

use std::collections::HashMap;
use std::time::Duration;

use serde_json::Value;

use crate::dialects::Dialect;
use crate::errors::{Failure, FailureClass};
use crate::param_attribution::{rejected_detail, rejected_model_not_found, rejected_parameter};

/// Build the shared pooled upstream client, mirroring the pooling constants in
/// `providers.async_transport` (64 keep-alive) and its no-redirect policy so a
/// provider 3xx can never re-send credentials to an attacker-chosen location.
///
/// `connect_timeout` bounds only the TCP+TLS connect phase; a dead lane whose
/// host never accepts the connection fails over after this window instead of
/// hanging on the per-deployment request timeout.
pub fn build_client(connect_timeout: Duration) -> Result<reqwest::Client, String> {
    reqwest::Client::builder()
        .pool_max_idle_per_host(64)
        .connect_timeout(connect_timeout)
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
        // 402 is the provider ACCOUNT's billing state (trial quota exhausted,
        // postpaid billing disabled), never the caller's request fields: it is
        // operator-actionable deadness, so it fails over in every failover mode
        // instead of surfacing a corrective 400 to the caller.
        Some(402) => (
            FailureClass::ProviderQuota,
            "provider account quota or billing is exhausted; ask the gateway operator \
             to fund or enable the provider account",
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

/// Classify a lead that connected but never completed the request/response-header
/// phase within `phase_timeout`. A deployment that accepted the connection but
/// stalled awaiting response headers is the same dead-lane signal as a stalled
/// first byte, so it mirrors `relay::first_byte_timeout_failure`: failover-eligible
/// (advance to the next certified rung) but deliberately *not* same-deployment
/// retryable. Redialing the same stalled deployment would only burn another full
/// header-timeout window before failing over; skipping straight to the next rung
/// keeps a stalled lead's cost near one fail-fast window. It stays a
/// `FailureClass::Timeout`, so it feeds the health circuit like other timeouts.
fn open_timeout_failure() -> Failure {
    Failure::new(
        FailureClass::Timeout,
        "provider did not send response headers in time",
    )
    .with_retry(false, true)
}

/// Open one streaming POST and return the response on HTTP success. The
/// timeout bounds only the request/response-header phase; body-read pacing is
/// bounded per chunk by the caller, mirroring the python transport split.
///
/// `raw_body` carries the exact pre-serialized body for body-signing dialects
/// (Bedrock SigV4): its signature covers those exact bytes, so it is sent
/// verbatim with the signed headers instead of re-serializing `payload`.
#[allow(clippy::too_many_arguments)]
pub async fn open_stream(
    client: &reqwest::Client,
    url: &str,
    headers: &HashMap<String, String>,
    idempotency_key: &str,
    payload: &Value,
    raw_body: Option<&str>,
    phase_timeout: Duration,
    dialect: Dialect,
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
        let failure = transport_failure(Some(status));
        // Only the generic client-error class may carry attribution: the body
        // is read bounded, and the relayable facts are a validated parameter
        // path plus the provider's own bounded explanation of what the caller
        // got wrong; every other class stays content-free.
        if failure.failure_class != FailureClass::InvalidRequest {
            return Err(failure);
        }
        let body = match tokio::time::timeout(ERROR_BODY_READ_TIMEOUT, bounded_error_body(response))
            .await
        {
            Ok(Some(body)) => Some(body),
            _ => None,
        };
        // A client-error status whose body names a missing model is the
        // catalog's fault, not the caller's: it takes the 404 policy so the
        // ladder advances instead of surfacing one dead rung as a 400.
        if body
            .as_deref()
            .is_some_and(|body| rejected_model_not_found(dialect, body))
        {
            return Err(transport_failure(Some(404)));
        }
        let parameter = body
            .as_deref()
            .and_then(|body| rejected_parameter(dialect, body));
        // The payload's own model id is a caller-known word: a provider
        // sentence naming it unquoted (Anthropic's client-version gate does)
        // must not be redacted as infrastructure.
        let request_words: Vec<&str> = payload
            .get("model")
            .and_then(Value::as_str)
            .into_iter()
            .collect();
        let detail = body
            .as_deref()
            .and_then(|body| rejected_detail(dialect, body, &request_words));
        return Err(failure
            .with_rejected_parameter(parameter)
            .with_provider_detail(detail));
    }
    Ok(response)
}

/// Longest provider error body read for parameter attribution.
const ERROR_BODY_READ_LIMIT: usize = 16 * 1024;

/// Bound on the whole attribution body read; a stalling error stream is
/// abandoned and the failure stays content-free.
const ERROR_BODY_READ_TIMEOUT: Duration = Duration::from_secs(2);

/// Read at most `ERROR_BODY_READ_LIMIT` bytes of one error response body.
async fn bounded_error_body(mut response: reqwest::Response) -> Option<String> {
    let mut collected: Vec<u8> = Vec::new();
    while let Ok(Some(chunk)) = response.chunk().await {
        if collected.len() + chunk.len() > ERROR_BODY_READ_LIMIT {
            return None;
        }
        collected.extend_from_slice(&chunk);
    }
    String::from_utf8(collected).ok()
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
            (Some(402), false, true),
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

    #[tokio::test]
    async fn a_402_with_the_literal_tokenhub_body_classes_provider_quota() {
        // TokenHub (the Tencent relay) answers HTTP 402 with provider code
        // 401008 on EVERY request shape once the account's free trial is
        // exhausted and postpaid billing is off. Classing that invalid_request
        // blamed 652 callers' request fields for the provider's billing state
        // (2026-09 incident): the class must be the provider-side quota family,
        // the rung must fail over, and the body must never be relayed.
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
            .await
            .expect("bind");
        let addr = listener.local_addr().expect("addr");
        tokio::spawn(async move {
            use tokio::io::{AsyncReadExt, AsyncWriteExt};
            let (mut socket, _) = listener.accept().await.expect("accept");
            let mut buffer = [0u8; 8192];
            let _ = socket.read(&mut buffer).await;
            let body = "{\"error\":{\"code\":401008,\"message\":\"free trial quota exhausted \
                        and postpaid billing is not enabled - enable in Console > Online \
                        Inference Service\",\"type\":\"payment_required\"}}";
            let response = format!(
                "HTTP/1.1 402 Payment Required\r\ncontent-type: application/json\r\n\
                 content-length: {}\r\nconnection: close\r\n\r\n{}",
                body.len(),
                body,
            );
            socket.write_all(response.as_bytes()).await.expect("write");
        });
        let client = build_client(Duration::from_secs(2)).expect("client");
        let failure = open_stream(
            &client,
            &format!("http://{addr}/v1/chat/completions"),
            &HashMap::new(),
            "idem-402",
            &serde_json::json!({"model": "m", "messages": []}),
            None,
            Duration::from_secs(5),
            Dialect::OpenAiCompatible,
        )
        .await
        .expect_err("a 402 must classify as a failure");
        assert_eq!(failure.failure_class, FailureClass::ProviderQuota);
        assert!(failure.failover_eligible, "an unfunded rung must fail over");
        assert!(!failure.retryable_same_deployment);
        assert!(
            failure.provider_detail.is_none() && failure.rejected_parameter.is_none(),
            "a billing failure must stay content-free"
        );
        assert!(
            !failure.safe_message.contains("request fields"),
            "the caller must never be told to fix their fields for a provider billing state"
        );
    }

    #[test]
    fn header_phase_timeout_fails_over_without_a_same_deployment_redial() {
        // A lead that connects but never completes the response-header phase must
        // skip straight to the next rung (failover-eligible) instead of redialing
        // the same stalled deployment for another full header-timeout window.
        let failure = open_timeout_failure();
        assert_eq!(failure.failure_class, FailureClass::Timeout);
        assert!(
            !failure.retryable_same_deployment,
            "a header-phase stall must not redial the same deployment"
        );
        assert!(
            failure.failover_eligible,
            "a header-phase stall must fail over to the next certified rung"
        );
    }
}
