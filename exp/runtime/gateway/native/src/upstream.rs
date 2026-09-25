//! Upstream provider HTTP transport over one shared pooled client.

use std::collections::HashMap;
use std::time::{Duration, Instant};

use serde_json::Value;

use crate::dialects::Dialect;
use crate::errors::{Failure, FailureClass};
use crate::param_attribution::{
    bounded_masked_line, content_filtered_completion, generic_error_code,
    rejected_by_account_quota, rejected_by_lane_limitation, rejected_by_routing_gate,
    rejected_caller_reference_not_found, rejected_code, rejected_detail,
    rejected_encrypted_reasoning, rejected_model_not_found, rejected_parameter,
    rejected_via_decode_failure, sanitized_detail,
};
use crate::rate_limit_headers::{harvest_rate_limit_headers, retry_after_seconds};

/// Build the shared pooled upstream client, mirroring the pooling constants in
/// `providers.async_transport` (64 keep-alive) and its no-redirect policy so a
/// provider 3xx can never re-send credentials to an attacker-chosen location.
///
/// `connect_timeout` bounds only the TCP+TLS connect phase; a dead lane whose
/// host never accepts the connection fails over after this window instead of
/// hanging on the per-deployment request timeout.
pub fn build_client(connect_timeout: Duration) -> Result<reqwest::Client, String> {
    client_builder(connect_timeout)
        .build()
        .map_err(|error| format!("upstream client construction failed: {error}"))
}

/// One transport configuration shared by production and protocol-level tests.
fn client_builder(connect_timeout: Duration) -> reqwest::ClientBuilder {
    reqwest::Client::builder()
        .pool_max_idle_per_host(64)
        .connect_timeout(connect_timeout)
        .redirect(reqwest::redirect::Policy::none())
        // The waterfall owns every physical retry and its reservation, including H2 nacks.
        .retry(reqwest::retry::never())
        .use_rustls_tls()
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

/// Pin TypeSafe rejection evidence to the received status, never an inferred class.
pub(crate) fn decision_http_failure(mut failure: Failure, status: u16) -> Failure {
    failure.decision_provider_rejected = matches!(status, 400 | 401 | 403 | 404 | 422);
    // Zero-work accounting evidence does not itself authorize another call.
    // Only a rejected credential may try an independently credentialed rung;
    // every other status ends this request without automatic dispatch.
    failure.retryable_same_deployment = false;
    failure.failover_eligible = status == 401;
    failure
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
    // SystemOne defines no idempotency contract, so neither configured nor
    // synthesized replay keys may imply protection against duplicate billing.
    if dialect != Dialect::TypesafeSystemone {
        request = request.header("Idempotency-Key", idempotency_key);
    }
    let send = match raw_body {
        Some(body) => request.body(body.to_string()).send(),
        None => request.json(payload).send(),
    };
    let phase_started = Instant::now();
    let response = match tokio::time::timeout(phase_timeout, send).await {
        Ok(Ok(response)) => response,
        Ok(Err(error)) => {
            if error.is_timeout() {
                return Err(open_timeout_failure());
            }
            // The failure stays a content-free transport class on the wire;
            // the engine's own account of WHICH transport fault (never
            // provider text) rides to the ledger so the row is diagnosable.
            return Err(transport_failure(None)
                .with_provider_detail(Some(transport_error_detail("open", &error))));
        }
        Err(_) => return Err(open_timeout_failure()),
    };
    let status = response.status().as_u16();
    if !(200..300).contains(&status) {
        // Rate-limit facts are read off the headers before anything consumes
        // the response: a 429's `retry-after` and remaining-quota counts ride
        // the failure into settlement (never to the caller), where the
        // control plane sizes throttle windows and persists them per attempt.
        let rate_limit = harvest_rate_limit_headers(response.headers());
        let retry_after = retry_after_seconds(response.headers());
        let failure =
            transport_failure(Some(status)).with_rate_limit_facts(rate_limit.clone(), retry_after);
        // Only the generic client-error class may carry attribution: the body
        // is read bounded, and the relayable facts are a validated parameter
        // path plus the provider's own bounded explanation of what the caller
        // got wrong; every other class stays content-free. A 403 is read too,
        // only to tell an aggregator routing gate from a credential verdict,
        // a 404 to tell a caller's dangling reference from a missing model,
        // and a 429 to tell an exhausted ACCOUNT from a throttle and to file
        // the provider's code token (never its sentence) in the ledger.
        // Every status-only classification carries the status itself as its
        // ledger detail (`http 503`): the class alone could not tell a 502
        // relay from a 500 model fault, and none of these classes relays
        // detail to the caller.
        let failure = if failure.failure_class == FailureClass::InvalidRequest {
            failure
        } else {
            failure.with_provider_detail(Some(status_detail(status)))
        };
        if dialect == Dialect::TypesafeSystemone {
            // This non-conversational wire has no documented chat error
            // envelope. Its actual HTTP status, not provider prose or a
            // guessed error class, establishes pre-execution rejection.
            return Err(decision_http_failure(failure, status));
        }
        if failure.failure_class != FailureClass::InvalidRequest
            && status != 403
            && status != 404
            && status != 429
        {
            return Err(failure);
        }
        // The attribution read never outlives the rung's own header-phase
        // budget: a provider that answers its status and then stalls the body
        // costs at most what was left of that window, never a further two
        // seconds past the caller's deadline. A throttle is the hot path
        // under load and its failover must stay near-immediate, so its read
        // gets only the short budget: the small envelope normally arrives
        // with the headers, and a provider that stalls after a 429 simply
        // fails over content-free as before.
        let read_timeout = if status == 429 {
            THROTTLE_BODY_READ_TIMEOUT
        } else {
            ERROR_BODY_READ_TIMEOUT
        };
        let body_budget = read_timeout.min(phase_timeout.saturating_sub(phase_started.elapsed()));
        let body = match tokio::time::timeout(body_budget, bounded_error_body(response)).await {
            Ok(Some(body)) => Some(body),
            _ => None,
        };
        if status == 429 {
            // OpenAI answers an out-of-quota account with 429
            // `insufficient_quota`, the same status as a throttle. A status-
            // only read filed both as `throttled`, so the house exhaustion
            // sweep (which reads `provider_quota`) never saw the account die.
            // Any other code token rides the failure into the ledger only
            // (a throttle's public error never relays detail): Novita's
            // `RATE_LIMIT_EXCEEDED` versus `TOKEN_LIMIT_EXCEEDED` names which
            // window closed, which its headers do not (2026-09-14: 205
            // Novita 429s with 992-999 of 1000 requests remaining and no
            // Retry-After).
            let code = body
                .as_deref()
                .and_then(|body| rejected_code(dialect, body));
            let detail = code
                .as_deref()
                .filter(|token| !generic_error_code(token))
                .map(|token| format!("{}: {token}", status_detail(status)));
            if crate::stream_errors::is_quota_code(code.as_deref()) {
                return Err(transport_failure(Some(402))
                    .with_provider_detail(detail)
                    .with_rate_limit_facts(rate_limit.clone(), retry_after));
            }
            return Err(match detail {
                Some(detail) => failure.with_provider_detail(Some(detail)),
                None => failure,
            });
        }
        if status == 404 {
            // OpenAI answers 404 for an `item_reference`, `conversation`, or
            // similar handle the caller sent but the provider does not hold
            // (store=false items are never persisted). The catalog is fine and
            // every rung would answer the same, so it is the caller's 400 with
            // the provider's sentence, never a lane 404 that walks the ladder.
            if body
                .as_deref()
                .is_some_and(|body| rejected_caller_reference_not_found(dialect, body))
            {
                let request_words: Vec<&str> = payload
                    .get("model")
                    .and_then(Value::as_str)
                    .into_iter()
                    .collect();
                let detail = body
                    .as_deref()
                    .and_then(|body| rejected_detail(dialect, body, &request_words));
                let parameter = body
                    .as_deref()
                    .and_then(|body| rejected_parameter(dialect, body));
                return Err(Failure::new(
                    FailureClass::InvalidRequest,
                    "the request references a provider-side item, response, or conversation \
                     the provider does not hold; resend that content inline",
                )
                .with_retry(false, false)
                .with_rejected_parameter(parameter)
                .with_provider_detail(detail)
                .with_rate_limit_facts(rate_limit.clone(), retry_after));
            }
            return Err(failure);
        }
        if status == 403 {
            if body
                .as_deref()
                .is_some_and(|body| rejected_by_routing_gate(dialect, body))
            {
                return Err(Failure::new(
                    FailureClass::ProviderNotFound,
                    "provider does not route this model for the gateway's account; ask \
                     the gateway operator to change or disable the lane",
                )
                .with_retry(false, true)
                .with_rate_limit_facts(rate_limit.clone(), retry_after));
            }
            // A reseller's balance verdict under a 403 (Novita answers an
            // unfunded account `403 NOT_ENOUGH_BALANCE`) is the ACCOUNT's
            // funding state, not a credential one: it takes the quota class
            // the house exhaustion sweep and pool rotation read, with the
            // status and token as its ledger detail like every other
            // operator-facing class.
            let code = body
                .as_deref()
                .and_then(|body| rejected_code(dialect, body));
            if crate::stream_errors::is_quota_code(code.as_deref()) {
                let token = code.unwrap_or_default();
                return Err(transport_failure(Some(402))
                    .with_provider_detail(Some(format!("{}: {token}", status_detail(status))))
                    .with_rate_limit_facts(rate_limit.clone(), retry_after));
            }
            // A compatible relay can put Gemini's content verdict under 403.
            // Read only the raw error envelope, never sanitized or echoed text.
            let refusal = body.as_deref().and_then(|body| {
                if !matches!(
                    dialect,
                    Dialect::OpenAiCompatible | Dialect::OpenAiResponses
                ) {
                    return None;
                }
                let value = crate::error_envelope::parse_error_document(body)?;
                let envelope = crate::error_envelope::openai_family_envelope(&value)?;
                let reason = crate::stream_errors::relayed_gemini_refusal(
                    envelope.code.as_deref(),
                    envelope.message,
                )?;
                let detail = envelope
                    .message
                    .and_then(|message| sanitized_detail(message, &[]));
                Some((reason, detail))
            });
            if let Some((reason, detail)) = refusal {
                return Err(Failure::refusal(reason)
                    .with_provider_detail(detail)
                    .with_rate_limit_facts(rate_limit.clone(), retry_after));
            }
            return Err(failure);
        }
        // A client-error status whose body names a missing model is the
        // catalog's fault, not the caller's: it takes the 404 policy so the
        // ladder advances instead of surfacing one dead rung as a 400.
        if body
            .as_deref()
            .is_some_and(|body| rejected_model_not_found(dialect, body))
        {
            return Err(transport_failure(Some(404))
                .with_provider_detail(Some(format!("{}: model_not_found", status_detail(status))))
                .with_rate_limit_facts(rate_limit.clone(), retry_after));
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
        // A 4xx whose body is a COMPLETION finished `content_filter` (Azure
        // Foundry's DeepSeek lanes answer their output filter this way, with
        // no error envelope at all) is the model's verdict on the content:
        // file and answer it as a refusal, the blocked label kept ledger-only,
        // never as a request-shape error and never a failover (the next rung
        // refuses the same content or, worse, serves it).
        if let Some(filtered) = body
            .as_deref()
            .and_then(|body| content_filtered_completion(dialect, body))
        {
            let reason = crate::stream_errors::refusal_reason(
                Some(&filtered.code),
                filtered.message.as_deref(),
            );
            let detail = filtered
                .message
                .as_deref()
                .and_then(|message| sanitized_detail(message, &request_words))
                .or(Some(filtered.code));
            return Err(Failure::refusal(reason)
                .with_provider_detail(detail)
                .with_rate_limit_facts(rate_limit.clone(), retry_after));
        }
        let code = body
            .as_deref()
            .and_then(|body| rejected_code(dialect, body));
        let detail = body
            .as_deref()
            .and_then(|body| rejected_detail(dialect, body, &request_words))
            // A sentence the identifier screen dropped still leaves the
            // provider's own code token: "invalid_value" beats "verify the
            // request fields" for the caller and the ledger alike. A generic
            // family type or bare status adds nothing and is not relayed.
            .or_else(|| code.clone().filter(|token| !generic_error_code(token)));
        // A relay that could not decode the UPSTREAM error it received (Novita's
        // Go relay on a numeric `error.code`) answers its own 400 whose
        // sentence embeds that upstream document: the class and the sentence
        // the caller needs are the upstream's, so a relayed throttle or quota
        // fails over as such and a relayed caller error keeps its real sentence
        // instead of the decoder's noise.
        let (code, detail) = match body
            .as_deref()
            .and_then(|body| rejected_via_decode_failure(dialect, body))
        {
            Some((upstream_code, upstream_sentence)) => {
                let kind = crate::stream_errors::classify_stream_error(
                    upstream_code.as_deref(),
                    Some(&upstream_sentence),
                );
                let sanitized = sanitized_detail(&upstream_sentence, &request_words);
                // The relay answered a client-error status, so an upstream
                // sentence the classifier cannot place (its default is the
                // provider-fault class) keeps this status's caller class with
                // the upstream sentence; only a positively classified throttle,
                // quota, credential, not-found or refusal verdict overrides it.
                if !matches!(
                    kind,
                    crate::stream_errors::StreamErrorKind::InvalidRequest
                        | crate::stream_errors::StreamErrorKind::ProviderInternal
                ) {
                    let ledger_detail =
                        sanitized.map(|text| format!("{}: {text}", status_detail(status)));
                    return Err(crate::stream_errors::stream_failure(kind, ledger_detail)
                        .with_rate_limit_facts(rate_limit.clone(), retry_after));
                }
                (upstream_code, sanitized)
            }
            None => (code, detail),
        };
        // A client-error status whose CODE or SENTENCE says the provider
        // ACCOUNT cannot pay (Novita `400 "Insufficient quota available for
        // instant inference"` on a drained prepaid balance) is the house
        // account's funding state, never the caller's request: it takes the
        // quota class the exhaustion sweep and pool rotation read, fails over,
        // and keeps the sentence ledger-only like every operator-facing class.
        if crate::stream_errors::is_quota_code(code.as_deref())
            || body
                .as_deref()
                .is_some_and(|body| rejected_by_account_quota(dialect, body))
        {
            let ledger_detail = detail.as_deref().map_or_else(
                || status_detail(status),
                |text| format!("{}: {text}", status_detail(status)),
            );
            return Err(transport_failure(Some(402))
                .with_provider_detail(Some(ledger_detail))
                .with_rate_limit_facts(rate_limit.clone(), retry_after));
        }
        // A content-filter CODE under a 4xx is the model's verdict on the
        // content (Azure and Gemini answer 400 for it), not a request-shape
        // error: file and answer it as a refusal naming its bounded category,
        // detail kept ledger-only. Only the authoritative code decides here; a
        // sentence saying "blocked by" could be about a firewall or a limit.
        if crate::stream_errors::is_refusal_code(code.as_deref()) {
            let reason = crate::stream_errors::refusal_reason(code.as_deref(), None);
            return Err(Failure::refusal(reason)
                .with_provider_detail(detail)
                .with_rate_limit_facts(rate_limit.clone(), retry_after));
        }
        // A sentence naming a limitation of THIS lane's serving stack (a chat
        // template that rejects a mid-conversation system turn the OpenAI
        // contract allows) keeps the caller's class and detail but fails over:
        // another rung serves the same request, and only a route with no other
        // rung surfaces the 400.
        let lane_limitation = body
            .as_deref()
            .is_some_and(|body| rejected_by_lane_limitation(dialect, body));
        // A refused replayed reasoning payload keeps the caller's class too,
        // but the waterfall re-dials this rung once without those items
        // before the class is allowed to surface.
        let encrypted_reasoning = body
            .as_deref()
            .is_some_and(|body| rejected_encrypted_reasoning(dialect, body));
        return Err(failure
            .with_retry(false, lane_limitation)
            .with_rejected_parameter(parameter)
            .with_provider_detail(detail)
            .with_encrypted_reasoning_rejected(encrypted_reasoning));
    }
    Ok(response)
}

/// The ledger detail of one status-only classification.
fn status_detail(status: u16) -> String {
    format!("http {status}")
}

/// Longest transport cause text kept in a ledger detail.
const TRANSPORT_CAUSE_LIMIT: usize = 120;

/// The engine's own account of one connection-level failure, for the ledger.
///
/// A bare `transport` class hid what actually broke (2026-09-15: ~150
/// transport settlements a day with no cause recorded), so the failure names
/// the phase (`open` before headers, `stream` mid-body), reqwest's fault kind,
/// and the innermost cause's own sentence (an OS error such as "Connection
/// reset by peer (os error 54)", a TLS alert, hyper's "connection closed
/// before message completed"). The reqwest layer's own text is skipped
/// because it names the request URL; the cause text is then held to the same
/// identifier mask as a provider sentence and bounded, so nothing shaped like
/// a host or handle crosses. This is engine vocabulary about the engine's own
/// socket, never provider body content, and no class it rides on relays
/// detail to callers.
pub(crate) fn transport_error_detail(phase: &str, error: &reqwest::Error) -> String {
    let kind = if error.is_timeout() {
        "timed out"
    } else if error.is_connect() {
        "connect failed"
    } else if error.is_body() {
        "body read failed"
    } else if error.is_decode() {
        "decode failed"
    } else if error.is_request() {
        "request failed"
    } else if error.is_redirect() {
        "redirect refused"
    } else {
        "failed"
    };
    let mut cause: Option<String> = None;
    let mut source = std::error::Error::source(error);
    while let Some(inner) = source {
        cause = Some(inner.to_string());
        source = inner.source();
    }
    let mut detail = format!("{phase} {kind}");
    if let Some(cause) = cause {
        let collapsed = cause.split_whitespace().collect::<Vec<_>>().join(" ");
        let bounded: String = bounded_masked_line(&collapsed, &[])
            .chars()
            .take(TRANSPORT_CAUSE_LIMIT)
            .collect();
        if !bounded.is_empty() {
            detail.push_str(": ");
            detail.push_str(&bounded);
        }
    }
    detail
}

/// Longest provider error body read for parameter attribution.
const ERROR_BODY_READ_LIMIT: usize = 16 * 1024;

/// Bound on the whole attribution body read; a stalling error stream is
/// abandoned and the failure stays content-free.
const ERROR_BODY_READ_TIMEOUT: Duration = Duration::from_secs(2);

/// Bound on a 429 body read: only the code token is wanted, the envelope
/// normally rides in the same segment as the headers, and a throttle's
/// failover must not wait on a provider that dribbles its error body.
const THROTTLE_BODY_READ_TIMEOUT: Duration = Duration::from_millis(250);

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
mod tests;

#[cfg(test)]
#[path = "upstream_reseller_tests.rs"]
mod reseller_tests;
