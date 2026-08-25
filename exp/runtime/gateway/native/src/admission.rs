//! The admitted-request contract and per-request orchestration shared by the
//! chat, Responses, and Messages surfaces: the admission wire shape with its
//! frozen route policy, the commit-independent and commit-dependent response
//! headers, request-guard construction, wire-drift abandonment, the bounded
//! active-dispatch permit, and the optional output guardrail.

use std::time::{Duration, Instant};

use axum::response::Response;
use serde::Deserialize;
use serde_json::Value;

use crate::bridge::Bridge;
use crate::encode_responses::ResponsesEnvelope;
use crate::errors::{Failure, FailureClass, PublicError};
use crate::events::Event;
use crate::guardrails;
use crate::metrics::METRICS;
use crate::respond::error_response;
use crate::server::AppState;
use crate::settlement::AttemptGuard;
use crate::waterfall::{DeploymentWire, RoutePolicy};

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
    pub refusal_failover: bool,
    /// Responses-only request-reflecting envelope fields; chat admissions
    /// omit it.
    #[serde(default)]
    pub envelope: Option<ResponsesEnvelope>,
    /// When true, buffer the winning completion and call `enforce_output` once
    /// before any caller byte or replay retention. Unguarded admissions omit
    /// the flag (default false) and never invoke that callback.
    #[serde(default)]
    pub output_guardrail: bool,
}

impl Admission {
    pub(crate) fn policy(&self) -> RoutePolicy {
        RoutePolicy {
            maximum_total_attempts: self.maximum_total_attempts.max(1),
            maximum_same_deployment_attempts: self.maximum_same_deployment_attempts.max(1),
            refusal_failover: self.refusal_failover,
        }
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
    let (provider, deployment_id) = admission
        .route
        .get(depth)
        .map(|wire| (wire.provider.clone(), wire.deployment_id.clone()))
        .unwrap_or_default();
    vec![
        (
            "x-gateway-canonical-model".to_string(),
            admission.exact_model_id.clone(),
        ),
        ("x-gateway-provider".to_string(), provider),
        ("x-gateway-deployment".to_string(), deployment_id),
        ("x-gateway-route-depth".to_string(), depth.to_string()),
        (
            "x-gateway-route-reason".to_string(),
            admission.route_reason.clone(),
        ),
    ]
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

pub(crate) async fn apply_output_guardrail(
    admission: &Admission,
    bridge: &Bridge,
    events: Vec<Event>,
) -> Result<Vec<Event>, Failure> {
    if !admission.output_guardrail {
        return Ok(events);
    }
    guardrails::enforce_collected_output(bridge, &admission.request_id, events).await
}
