//! The admitted route's wire facts: one deployment's dispatch configuration,
//! the frozen retry policy, and everything one waterfall run carries beside
//! its request guard.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};

use serde::Deserialize;
use serde_json::Value;

use crate::bridge::Bridge;
use crate::throttle_backoff::ThrottleRedial;
use crate::tool_search::ToolSearchAdmission;

/// One deployment's wire configuration inside the admitted ordered route.
/// Payloads are built python-side per deployment, since model identities and
/// dialects may differ across one certified pool.
#[derive(Debug, Clone, Deserialize)]
pub struct DeploymentWire {
    pub provider: String,
    pub deployment_id: String,
    /// Canonical identity of this actual stage, not the requested root model.
    #[serde(default)]
    pub exact_model_id: String,
    pub dialect: String,
    pub url: String,
    pub headers: HashMap<String, String>,
    /// Exact provider model identifier the payload carries; a caller-known
    /// word the stream-error detail screen must not redact.
    #[serde(default)]
    pub model_id: String,
    /// Whether this rung dispatches on the customer's own (BYOK) credential.
    /// A rejected credential or exhausted account on such a rung is the
    /// customer's configuration, surfaced as their 400, never operator
    /// deadness that fails over.
    #[serde(default)]
    pub billing_customer_managed: bool,
    /// Host-authorized Google explicit-cache execution for this attempt.
    /// The host still reserves and settles each cache operation separately.
    #[serde(default)]
    pub explicit_cache: bool,
    pub timeout_seconds: f64,
    /// Structured payload the data plane serializes itself; null for
    /// body-signing dialects, whose route entry carries `upstream_body`.
    #[serde(default)]
    pub upstream_payload: Value,
    /// Exact pre-serialized body for body-signing dialects (Bedrock SigV4).
    /// When present it is sent verbatim: the signature covers these exact
    /// bytes, so re-serializing a structured payload here could invalidate
    /// it.
    #[serde(default)]
    pub upstream_body: Option<String>,
    #[serde(default)]
    pub fireworks_reasoning_route_sha256: Option<String>,
    /// Tencent Hunyuan preserved-thinking route identity; like the Fireworks
    /// field it turns on provider reasoning-content capture and stamps each
    /// delta with this exact route, but seals under the Hunyuan carrier scheme.
    #[serde(default)]
    pub hunyuan_reasoning_route_sha256: Option<String>,
    /// When true, this rung returns the model's plaintext `reasoning_content`
    /// to the caller for display (Tencent/DeepSeek think mode). The sealed
    /// round-trip carrier is emitted independently; elsewhere reasoning stays
    /// stripped. Defaults false so every other provider is unchanged.
    #[serde(default)]
    pub reasoning_output_exposed: bool,
    /// Caller stop sequences the data plane enforces on this rung's stream
    /// because the provider wire has no stop field (OpenAI Responses). The
    /// relay cuts visible text at the first match and terminates with
    /// `Event::StoppedAtSequence`. Empty when the payload carries `stop`.
    #[serde(default)]
    pub stop_sequences: Vec<String>,
    /// The caller sent `parallel_tool_calls: false` and this rung's wire has
    /// no such control: the relay serializes the turn to one tool call.
    #[serde(default)]
    pub serialize_tool_calls: bool,
    /// Codex native-tool inversion map for this request (provider-facing
    /// mangled name -> origin name, namespace, is-custom). Empty unless the
    /// request carried translated Codex native tools; see
    /// `codex_native_inversion`.
    #[serde(default)]
    pub native_tool_translation: std::collections::HashMap<String, (String, Option<String>, bool)>,
    /// The served model emits images (`emits_images` on the lane -- never the
    /// Images-API claim `supports_image_generation`, whose reuse admitted
    /// image generations onto OpenRouter chat lanes on 2026-09-15) that the
    /// chat normalizers carry no event for, so a turn with no
    /// renderable output is the EXPECTED shape of an image generation, not a
    /// transient empty answer: the waterfall neither redials nor advances the
    /// ladder on it (each attempt bills the house a whole image -- $0.24 list
    /// for one doubled gpt-5.4-image-2 request, 2026-09-15) and the caller
    /// receives the typed empty answer at once. Absent on older admissions.
    #[serde(default)]
    pub image_output: bool,
    pub idempotency_key: String,
    /// Deployment override for the flat first-byte allowance; the serving
    /// configuration's default applies when absent.
    #[serde(default)]
    pub time_to_first_byte_base_seconds: Option<f64>,
    /// Deployment override for the input-scaled first-byte allowance in
    /// seconds per million approximate input tokens; the serving
    /// configuration's default applies when absent.
    #[serde(default)]
    pub time_to_first_byte_seconds_per_million_input_tokens: Option<f64>,
    /// Deployment override for the flat first-TOKEN allowance (the wait from
    /// the dial to the first semantic event); the serving configuration's
    /// default applies when absent. Absent or null on older admissions.
    #[serde(default)]
    pub time_to_first_token_base_seconds: Option<f64>,
    /// How many post-backoff redials a throttle on this rung is worth on
    /// this request: the pool's `throttle_redial` schedule scaled by the
    /// requesting organization's cache at stake here (the full schedule at
    /// or above any authored threshold, a proportional share below it). The
    /// waterfall backs off and re-dials this rung that many times before the
    /// ladder advances; zero keeps the rung's throttle failover-only.
    #[serde(default)]
    pub throttle_redial_budget: u32,
    /// Stage-local schedule; its budget is zero when this stage disables redial.
    #[serde(default)]
    pub throttle_redial: Option<ThrottleRedial>,
    /// The rung's payload was tightened to OpenRouter's zero-data-retention
    /// routing constraint at admission; an answer it serves carries
    /// `x-gateway-zdr-constrained: true` so the host can attest it.
    #[serde(default)]
    pub zdr_constrained: bool,
    /// Failure tokens this rung serves as a failover for (see
    /// `fallback_rules`): a rung carrying a set is never the first dial and
    /// is dialed as a successor only when the failure being failed over from
    /// spells one of them. Absent or null on an unrestricted rung, which
    /// behaves exactly as before.
    #[serde(default)]
    pub failover_only_on: Option<Vec<String>>,
}

/// The frozen retry-policy facts returned by admission.
#[derive(Debug, Clone, Copy)]
pub struct RoutePolicy {
    pub maximum_total_attempts: u32,
    pub maximum_same_deployment_attempts: u32,
    pub refusal_failover: bool,
    /// The pool's backoff-and-redial schedule for throttled rungs, when
    /// authored.
    pub throttle_redial: Option<ThrottleRedial>,
    pub physical_route_cap: Option<u32>,
    pub backoff: Option<crate::request_policy::Backoff>,
}

impl RoutePolicy {
    /// Every dispatch must fit the server caps and any explicit caller cap.
    pub fn permits(self, total: u32, physical_at_route: u32) -> bool {
        total < self.maximum_total_attempts.min(8)
            && self
                .physical_route_cap
                .is_none_or(|cap| physical_at_route < cap.min(4))
    }

    /// Reject a malformed internal admission rather than silently widening it.
    pub fn valid(self) -> bool {
        (1..=8).contains(&self.maximum_total_attempts)
            && (1..=4).contains(&self.maximum_same_deployment_attempts)
            && self
                .physical_route_cap
                .is_none_or(|cap| (1..=4).contains(&cap))
            && self.backoff.is_none_or(|backoff| backoff.valid())
    }
}

/// Everything one waterfall run needs besides its request guard.
pub struct WaterfallContext<'a> {
    pub bridge: &'a Arc<Bridge>,
    pub http: &'a reqwest::Client,
    pub request_id: &'a str,
    /// The presented virtual key, forwarded so hosted budget-error policy
    /// can shape a rejected reservation for the caller.
    pub raw_key: &'a str,
    /// The caller's stable identity from admission, scoping the per-caller
    /// replay-repair memory; `None` disables that memory for the request.
    pub caller_scope: Option<&'a str>,
    pub route: &'a [DeploymentWire],
    pub policy: RoutePolicy,
    pub deadline: Instant,
    /// Fail-fast flat bound on the wait for each physical attempt's first
    /// provider byte. Applied per attempt (each redial and each failover
    /// advance gets a fresh window); it bounds the connect/header/first-byte
    /// phase only and never caps generation once the provider has started
    /// answering. Deployments may override it per wire entry.
    pub time_to_first_byte: Duration,
    /// Default input-scaled first-byte allowance in seconds per million
    /// approximate input tokens, so a very large prompt whose prefill
    /// legitimately takes longer than the flat bound is not misread as a
    /// dead lane. Deployments may override it per wire entry.
    pub time_to_first_byte_slope_seconds_per_million_input_tokens: f64,
    /// Fail-fast flat bound on the wait for each physical attempt's first
    /// TOKEN (the first semantic event), absolute from the dial and sharing
    /// the slope above; keepalive comments and role-only frames do not
    /// satisfy it. Deployments may override it per wire entry.
    pub time_to_first_token: Duration,
    /// Approximate input tokens for this request: the raw body's bytes
    /// divided by four. An allowance heuristic only, never a billing
    /// quantity.
    pub approximate_input_tokens: f64,
    pub chat_logprobs: bool,
    /// The bridge `remember` argument retaining an output-less turn: a
    /// successful terminal reached before any semantic output still answers
    /// the caller with a response id, and a response id the caller received
    /// must stay continuable (api.openai.com persists `incomplete` responses
    /// too). Only the Responses route carries one; it runs ahead of the
    /// attempt's settlement so the control plane can still resolve the
    /// request's continuation context.
    pub output_less_retention: Option<String>,
    /// The caller's output cap when the request carries one. A `stop` with
    /// no semantic output and NO usage report on a capped request is read
    /// as the budget exhausted before the first visible token (the provider
    /// mislabelled a truncation; Meta's muse-spark lanes do this whenever
    /// hidden reasoning eats `max_tokens`, 2026-09-15) and answers
    /// `Incomplete`; the same shape on an uncapped request is the provider
    /// delivering nothing at all and takes the ladder.
    pub output_token_cap: Option<u64>,
    /// The gateway-run tool search admitted for this request: the relay
    /// withholds every call to its tool and the waterfall runs at most
    /// `max_rounds` search rounds. `None` withholds nothing.
    pub tool_search: Option<&'a ToolSearchAdmission>,
}

/// The bound on one dial's open (request/response-header) phase: the
/// remaining request deadline or the remaining first-byte allowance,
/// whichever is nearer. The deployment's per-chunk `timeout_seconds` is
/// deliberately NOT a term: it paces body reads after the first byte, and
/// letting it cap the open phase made every authored first-byte allowance
/// above it a silent no-op.
pub(crate) fn open_phase_bound(
    deadline_remaining: Duration,
    first_byte_remaining: Duration,
) -> Duration {
    deadline_remaining.min(first_byte_remaining)
}

/// The effective first-byte allowance for one attempt: the deployment's (or
/// serving default's) flat base plus its input-scaled allowance.
pub(crate) fn first_byte_allowance(
    wire: &DeploymentWire,
    default_base: Duration,
    default_slope_seconds_per_million: f64,
    approximate_input_tokens: f64,
) -> Duration {
    let base = wire
        .time_to_first_byte_base_seconds
        .unwrap_or(default_base.as_secs_f64());
    let slope = wire
        .time_to_first_byte_seconds_per_million_input_tokens
        .unwrap_or(default_slope_seconds_per_million);
    let scaled = slope * (approximate_input_tokens.max(0.0) / 1_000_000.0);
    Duration::from_secs_f64((base + scaled).max(0.001))
}

/// The effective first-TOKEN allowance for one attempt: the deployment's (or
/// serving default's) flat first-token base plus the same input-scaled
/// allowance the header bound uses (prefill delays both).
pub(crate) fn first_token_allowance(
    wire: &DeploymentWire,
    default_base: Duration,
    default_slope_seconds_per_million: f64,
    approximate_input_tokens: f64,
) -> Duration {
    let base = wire
        .time_to_first_token_base_seconds
        .unwrap_or(default_base.as_secs_f64());
    let slope = wire
        .time_to_first_byte_seconds_per_million_input_tokens
        .unwrap_or(default_slope_seconds_per_million);
    let scaled = slope * (approximate_input_tokens.max(0.0) / 1_000_000.0);
    Duration::from_secs_f64((base + scaled).max(0.001))
}
