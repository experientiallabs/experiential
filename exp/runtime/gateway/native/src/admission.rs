//! The admitted-request contract and per-request orchestration shared by the
//! chat, Responses, and Messages surfaces: the admission wire shape with its
//! frozen route policy, the commit-independent and commit-dependent response
//! headers, request-guard construction, wire-drift abandonment, the bounded
//! active-dispatch permit, and the optional output guardrail.

use std::time::{Duration, Instant};

use axum::response::Response;
use serde::Deserialize;
use serde_json::Value;

use crate::encode_responses::ResponsesEnvelope;
use crate::errors::empty_completion_headers;
use crate::errors::{Failure, FailureClass, PublicError};
use crate::events::Event;
use crate::guardrails;
use crate::guardrails::plan::OutputPlan;
use crate::metrics::METRICS;
use crate::replay_repair::replay_repair_headers;
use crate::respond::error_response;
use crate::server::AppState;
use crate::settlement::AttemptGuard;
use crate::throttle_backoff::ThrottleRedial;
use crate::tool_search::{ToolSearchAdmission, ToolSearchRound};
use crate::waterfall::{DeploymentWire, RoutePolicy, Served};
use crate::web_search::WebSearchAdmission;

/// The wire configuration returned by one successful admission: the full
/// ordered certified route (one wire configuration per deployment, each with
/// its payload fully built by the shared python dialect builders) plus the
/// frozen retry-policy facts. No attempt is started at admission; each
/// physical dispatch is reserved through the `start_attempt` callback.
#[derive(Debug, Clone, Deserialize)]
pub(crate) struct Admission {
    pub request_id: String,
    pub alias: String,
    pub alias_revision_id: String,
    pub stream: bool,
    pub include_usage: bool,
    pub exact_model_id: String,
    pub route_reason: String,
    pub route: Vec<DeploymentWire>,
    #[serde(default)]
    pub ignored_parameters: Vec<String>,
    pub maximum_total_attempts: u32,
    pub maximum_same_deployment_attempts: u32,
    #[serde(default)]
    pub physical_route_cap: Option<u32>,
    #[serde(default)]
    pub backoff: Option<crate::request_policy::Backoff>,
    #[serde(default)]
    pub refusal_failover: bool,
    /// The pool's backoff-and-redial schedule for throttled rungs; absent on
    /// pools that keep throttles failover-only, so their waterfall is
    /// unchanged.
    #[serde(default)]
    pub throttle_redial: Option<ThrottleRedial>,
    /// Responses-only request-reflecting envelope fields; chat admissions
    /// omit it.
    #[serde(default)]
    pub envelope: Option<ResponsesEnvelope>,
    /// How the identity's output chain must be enforced for this request.
    /// Unguarded admissions omit the field and never call a guardrail
    /// callback. See [`OutputGuardrailMode`].
    #[serde(default)]
    pub output_guardrail: OutputGuardrailMode,
    /// The resolved output chain when every check binds a deterministic
    /// detector. The data plane enforces it in place, so the request pays no
    /// python callback. A chain with any non-deterministic adapter omits the
    /// plan and sets `output_guardrail` instead.
    #[serde(default)]
    pub guardrail_output_plan: Option<OutputPlan>,
    /// The control plane's pre-dispatch count of the prompt (the reservation
    /// estimator without its headroom). Messages admissions carry it so the
    /// caller's `message_start` shows a real input figure when the upstream
    /// reports nothing before its final chunk; display-only, never settled.
    #[serde(default)]
    pub input_token_estimate: Option<u64>,
    /// The caller's own output cap (`max_tokens`, `max_completion_tokens`
    /// or `max_output_tokens`, normalized by the control plane), absent when
    /// the request is uncapped. The waterfall reads it to classify a `stop`
    /// that carried no output and no usage report: on a capped request that
    /// is a budget the provider's hidden reasoning exhausted (an honest
    /// `length`), never a completed empty answer.
    #[serde(default)]
    pub maximum_output_tokens: Option<u64>,
    /// The caller's stable identity (organization and identity ids), the
    /// scope of the data plane's per-caller replay-repair memory. The
    /// request's own bearer cannot serve: in a hosted worker it is the
    /// front's ephemeral exchanged token, different on every request.
    /// Absent from an older control plane, which disables that memory.
    #[serde(default)]
    pub caller_scope: Option<String>,
    /// The ONE web search the control plane executed before dispatch for a
    /// request that asked for it on a route unable to serve it natively:
    /// the query, the billed request count, and the ranked results it
    /// injected into the prompt. Absent when no search ran, which leaves
    /// every response byte exactly as before.
    #[serde(default)]
    pub web_search: Option<WebSearchAdmission>,
    /// The gateway-run tool search for this request: the function tool the
    /// model was given in place of the deferred catalog, and the round
    /// budget. Absent when the gateway runs no search (no deferred tools, or
    /// a route that searches natively), which leaves every byte as before.
    #[serde(default)]
    pub tool_search: Option<ToolSearchAdmission>,
    /// The search rounds the waterfall completed for this request, moved
    /// here from the winning attempt (`tool_search::adopt_outcome`) so every
    /// response surface renders them ahead of the answer. Never on the wire.
    #[serde(skip)]
    pub tool_search_rounds: Vec<ToolSearchRound>,
}

/// How one admission's output chain is enforced on the data plane.
///
/// `Buffer` collects the whole winning completion and calls `enforce_output`
/// once before any caller byte or replay retention: the only safe shape for a
/// check that can block or for a detector that needs the full text. `Stream`
/// releases the completion incrementally through `enforce_output_segment`,
/// holding back only the bounded trailing window the detector cannot yet
/// decide about. The control plane picks the mode at admit time.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Deserialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum OutputGuardrailMode {
    #[default]
    Off,
    Buffer,
    Stream,
}

impl OutputGuardrailMode {
    /// Whether this admission runs any output guardrail work at all.
    pub(crate) fn enforces(self) -> bool {
        !matches!(self, Self::Off)
    }
}

impl Admission {
    /// Reject a control-plane mismatch before reserving or dispatching an attempt.
    pub(crate) fn preserves_chat_probabilities(&self) -> bool {
        self.route.iter().all(|wire| {
            wire.upstream_payload.get("logprobs") != Some(&Value::Bool(true))
                || (wire.dialect == "openai_compatible"
                    && !self.output_guardrail.enforces()
                    && wire.stop_sequences.is_empty())
        })
    }

    pub(crate) fn policy(&self) -> RoutePolicy {
        RoutePolicy {
            maximum_total_attempts: self.maximum_total_attempts,
            maximum_same_deployment_attempts: self.maximum_same_deployment_attempts,
            refusal_failover: self.refusal_failover,
            throttle_redial: self.throttle_redial,
            physical_route_cap: self.physical_route_cap,
            backoff: self.backoff,
        }
    }

    /// How many gateway-executed web searches this request bills; `0` when
    /// the control plane ran none.
    pub(crate) fn web_search_requests(&self) -> u32 {
        self.web_search
            .as_ref()
            .map_or(0, |web_search| web_search.requests)
    }

    /// Whether the winning completion must be buffered for an output chain,
    /// natively or across the python boundary.
    pub(crate) fn buffers_output(&self) -> bool {
        self.output_guardrail.enforces() || self.guardrail_output_plan.is_some()
    }

    /// Whether the rung at `depth` returns plaintext reasoning to the caller.
    pub(crate) fn reasoning_exposed_at(&self, depth: usize) -> bool {
        self.route
            .get(depth)
            .is_some_and(|wire| wire.reasoning_output_exposed)
    }

    /// Whether the attempt at `depth` may enforce its output chain as bytes
    /// stream, instead of buffering the whole completion first.
    ///
    /// Admission already proved the chain is one deterministic modify-only
    /// check over a streamed request that offers no tools and asks for no
    /// reasoning text. The remaining fact belongs to the winning rung: a
    /// deployment that returns plaintext reasoning to the caller keeps the
    /// buffered path, where a rewrite still drops that channel wholesale.
    pub(crate) fn stream_incremental(&self, depth: usize) -> bool {
        self.output_guardrail == OutputGuardrailMode::Stream
            && self.stream
            && !self.reasoning_exposed_at(depth)
    }

    /// The per-chunk transport bound of the deployment serving `depth`.
    pub(crate) fn phase_timeout(&self, depth: usize) -> Duration {
        let seconds = self
            .route
            .get(depth)
            .map(|wire| wire.timeout_seconds)
            .unwrap_or(60.0);
        Duration::from_secs_f64(seconds.max(0.001))
    }
}

/// Commit-independent headers, mirroring `commit_independent_headers`,
/// including the caller's echoed request identity when one was supplied.
pub(crate) fn commit_independent(
    admission: &Admission,
    client_request_id: Option<&str>,
) -> Vec<(String, String)> {
    let mut headers = vec![
        ("x-request-id".to_string(), admission.request_id.clone()),
        ("x-gateway-alias".to_string(), admission.alias.clone()),
        (
            "x-gateway-alias-revision".to_string(),
            admission.alias_revision_id.clone(),
        ),
    ];
    if let Some(value) = client_request_id {
        headers.push(("x-client-request-id".to_string(), value.to_string()));
    }
    headers
}

/// Commit-dependent headers, mirroring `commit_dependent_headers`: the
/// deployment identity and route depth that actually served the request.
pub(crate) fn commit_dependent(admission: &Admission, depth: usize) -> Vec<(String, String)> {
    let served = admission.route.get(depth);
    let (provider, deployment_id) = served
        .map(|wire| (wire.provider.clone(), wire.deployment_id.clone()))
        .unwrap_or_default();
    let mut headers = vec![
        (
            "x-gateway-canonical-model".to_string(),
            admission
                .route
                .get(depth)
                .filter(|wire| !wire.exact_model_id.is_empty())
                .map(|wire| wire.exact_model_id.clone())
                .unwrap_or_else(|| admission.exact_model_id.clone()),
        ),
        ("x-gateway-provider".to_string(), provider),
        ("x-gateway-deployment".to_string(), deployment_id),
        ("x-gateway-route-depth".to_string(), depth.to_string()),
        (
            "x-gateway-route-reason".to_string(),
            admission.route_reason.clone(),
        ),
    ];
    // A rung dispatched under OpenRouter's zero-data-retention constraint
    // says so, so the host can attest the answer as ZDR-served; absent on
    // every ordinary rung rather than a literal `false`.
    if served.is_some_and(|wire| wire.zdr_constrained) {
        headers.push(("x-gateway-zdr-constrained".to_string(), "true".to_string()));
    }
    headers
}

/// Every header one served attempt carries: the request identity, the rung
/// that served, and any data-plane repair of the replayed input.
pub(crate) fn served_headers(
    admission: &Admission,
    client_request_id: Option<&str>,
    served: Served,
) -> Vec<(String, String)> {
    let mut headers = commit_independent(admission, client_request_id);
    headers.extend(commit_dependent(admission, served.depth));
    headers.extend(replay_repair_headers(served.encrypted_reasoning_stripped));
    headers.extend(empty_completion_headers(served.empty_completion));
    headers
}

/// Build one request guard bound to this server's settlement bookkeeping.
pub(crate) fn new_guard(state: &AppState, request_id: String, started: Instant) -> AttemptGuard {
    AttemptGuard::new(
        state.bridge.clone(),
        state.pending_settlements.clone(),
        request_id,
        started,
    )
}

/// Abandon one accepted request whose admission body failed to deserialize.
pub(crate) async fn wire_drift_response(
    state: &AppState,
    admission_value: &Value,
    started: Instant,
) -> Response {
    let request_id = admission_value
        .get("request_id")
        .and_then(Value::as_str)
        .unwrap_or_default()
        .to_string();
    if !request_id.is_empty() {
        let mut guard = new_guard(state, request_id, started);
        guard
            .abandon(&Failure::new(
                FailureClass::Internal,
                "gateway admission wire contract failed",
            ))
            .await;
    }
    error_response(&PublicError::internal())
}

/// Wait for one bounded active-dispatch permit after admission, like the
/// python executor: protocol and authority errors answer immediately even at
/// capacity, and a queue-deadline expiry terminalizes the accepted request.
pub(crate) async fn acquire_permit(
    state: &AppState,
    guard: &mut AttemptGuard,
    deadline: Instant,
) -> Result<tokio::sync::OwnedSemaphorePermit, Box<Response>> {
    let permit_wait_started = Instant::now();
    match tokio::time::timeout_at(deadline.into(), state.permits.clone().acquire_owned()).await {
        Ok(Ok(permit)) => {
            METRICS.permit_wait_ms.record(permit_wait_started.elapsed());
            Ok(permit)
        }
        Ok(Err(_)) => {
            guard
                .abandon(&Failure::new(
                    FailureClass::Cancelled,
                    "gateway is draining and is not accepting new requests",
                ))
                .await;
            Err(Box::new(error_response(&PublicError::draining())))
        }
        Err(_) => {
            let failure = Failure::new(
                FailureClass::Timeout,
                "gateway execution queue deadline exceeded",
            );
            let error = failure.public_error();
            guard.abandon(&failure).await;
            Err(Box::new(error_response(&error)))
        }
    }
}

/// Enforce the winning completion's output chain before any caller byte.
///
/// A deterministic chain is enforced natively against the compiled detectors
/// this server was started with. Every other guarded admission crosses the
/// python boundary exactly as before, and an unguarded admission does
/// neither.
pub(crate) async fn apply_output_guardrail(
    state: &AppState,
    admission: &Admission,
    events: Vec<Event>,
    deadline: Instant,
) -> Result<Vec<Event>, Failure> {
    if let Some(plan) = admission.guardrail_output_plan.as_ref() {
        return guardrails::plan::enforce(plan, &state.guardrail_detectors, events, deadline);
    }
    if !admission.output_guardrail.enforces() {
        return Ok(events);
    }
    guardrails::enforce_collected_output(&state.bridge, &admission.request_id, events).await
}

#[cfg(test)]
mod tests {
    use super::*;

    fn admission(zdr_constrained: bool) -> Admission {
        serde_json::from_value(serde_json::json!({
            "request_id": "req",
            "alias": "public-model",
            "alias_revision_id": "rev",
            "stream": true,
            "include_usage": true,
            "exact_model_id": "exact",
            "route_reason": "direct",
            "route": [{
                "provider": "openrouter",
                "deployment_id": "or-rung",
                "dialect": "openai_compatible",
                "url": "https://openrouter.ai/api/v1/chat/completions",
                "headers": {},
                "model_id": "anthropic/claude-opus-5",
                "billing_customer_managed": false,
                "timeout_seconds": 60.0,
                "upstream_payload": {},
                "upstream_body": null,
                "idempotency_key": "op",
                "zdr_constrained": zdr_constrained,
            }],
            "maximum_total_attempts": 1,
            "maximum_same_deployment_attempts": 1,
        }))
        .expect("admission decodes")
    }

    #[test]
    fn a_zdr_constrained_rung_names_the_constraint_on_its_served_headers() {
        let headers = commit_dependent(&admission(true), 0);
        assert!(headers.contains(&("x-gateway-zdr-constrained".to_string(), "true".to_string())));
        assert!(headers.contains(&("x-gateway-deployment".to_string(), "or-rung".to_string())));
    }

    #[test]
    fn an_ordinary_rung_carries_no_constraint_header_at_all() {
        let headers = commit_dependent(&admission(false), 0);
        assert!(headers
            .iter()
            .all(|(name, _)| name != "x-gateway-zdr-constrained"));
    }
}
