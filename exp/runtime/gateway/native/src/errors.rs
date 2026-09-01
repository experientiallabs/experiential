//! OpenAI-shaped public errors mirroring `exp.runtime.openai_protocol.errors`.

use serde::{Deserialize, Serialize};
use serde_json::json;

/// One sanitized public protocol error carrying its HTTP representation.
///
/// Field names match the JSON payload attached to `NativeBridgeError` on the
/// Python side so a bridge failure deserializes directly into this struct.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PublicError {
    pub status_code: u16,
    pub code: String,
    pub message: String,
    #[serde(default = "default_error_type")]
    pub error_type: String,
    #[serde(default)]
    pub param: Option<String>,
    #[serde(default)]
    pub retry_after_seconds: Option<u32>,
}

fn default_error_type() -> String {
    "invalid_request_error".to_string()
}

impl PublicError {
    pub fn new(status_code: u16, code: &str, message: &str, error_type: &str) -> Self {
        Self {
            status_code,
            code: code.to_string(),
            message: message.to_string(),
            error_type: error_type.to_string(),
            param: None,
            retry_after_seconds: None,
        }
    }

    /// The OpenAI error envelope body, matching `OpenAIProtocolError.json_body()`.
    pub fn json_body(&self) -> serde_json::Value {
        json!({
            "error": {
                "message": self.message,
                "type": self.error_type,
                "param": self.param,
                "code": self.code,
            }
        })
    }

    pub fn invalid_key() -> Self {
        Self::new(
            401,
            "invalid_key",
            "A valid gateway Bearer key is required. Send the virtual key as \
             'Authorization: Bearer <key>'.",
            "authentication_error",
        )
    }

    pub fn invalid_json() -> Self {
        Self::new(
            400,
            "invalid_json",
            "Request body must contain valid JSON. Re-encode the payload and resend.",
            "invalid_request_error",
        )
    }

    pub fn internal() -> Self {
        Self::new(
            500,
            "internal_error",
            "The gateway request failed. Retry the request; if this persists, \
             ask the gateway operator to inspect the server logs.",
            "api_error",
        )
    }

    pub fn draining() -> Self {
        let mut error = Self::new(
            503,
            "gateway_draining",
            "The gateway is draining and is not accepting new requests. \
             Retry after the delay in the Retry-After header.",
            "api_error",
        );
        error.retry_after_seconds = Some(10);
        error
    }

    pub fn request_too_large() -> Self {
        Self::new(
            413,
            "request_too_large",
            "Request body exceeds the gateway limit. Reduce the request size and resend.",
            "invalid_request_error",
        )
    }

    pub fn provider_output_too_large() -> Self {
        Self::new(
            502,
            "provider_output_too_large",
            "Provider output exceeded the gateway response limit. \
             Request less output, for example with a lower max_tokens value.",
            "api_error",
        )
    }
}

/// Stable failure classes shared with `GatewayFailureClass`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FailureClass {
    InvalidRequest,
    UnsupportedCapability,
    Authentication,
    Authorization,
    QuotaExceeded,
    Throttled,
    Transport,
    Timeout,
    ProviderAuthentication,
    ProviderNotFound,
    Refusal,
    MalformedResponse,
    ProviderInternal,
    Cancelled,
    Guardrail,
    Internal,
    Unavailable,
}

impl FailureClass {
    pub fn as_str(&self) -> &'static str {
        match self {
            FailureClass::InvalidRequest => "invalid_request",
            FailureClass::UnsupportedCapability => "unsupported_capability",
            FailureClass::Authentication => "authentication",
            FailureClass::Authorization => "authorization",
            FailureClass::QuotaExceeded => "quota_exceeded",
            FailureClass::Throttled => "throttled",
            FailureClass::Transport => "transport",
            FailureClass::Timeout => "timeout",
            FailureClass::ProviderAuthentication => "provider_authentication",
            FailureClass::ProviderNotFound => "provider_not_found",
            FailureClass::Refusal => "refusal",
            FailureClass::MalformedResponse => "malformed_response",
            FailureClass::ProviderInternal => "provider_internal",
            FailureClass::Cancelled => "cancelled",
            FailureClass::Guardrail => "guardrail",
            FailureClass::Internal => "internal",
            FailureClass::Unavailable => "unavailable",
        }
    }
}

/// One sanitized provider failure, the Rust mirror of `GatewayFailure`,
/// including the executor's per-failure retry classification: whether the
/// same deployment may be redialed and whether a later certified deployment
/// may serve the request instead.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Failure {
    pub failure_class: FailureClass,
    pub safe_message: String,
    #[serde(default)]
    pub retryable_same_deployment: bool,
    #[serde(default)]
    pub failover_eligible: bool,
    /// Validated provider-named parameter path; one of the two facts a
    /// sanitized client-error may relay.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub rejected_parameter: Option<String>,
    /// The provider's own bounded single-line explanation of a client error,
    /// relayed only for that class so the caller sees what was actually
    /// refused. Every other class stays content-free.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub provider_detail: Option<String>,
}

impl Failure {
    pub fn new(failure_class: FailureClass, safe_message: &str) -> Self {
        Self {
            failure_class,
            safe_message: safe_message.to_string(),
            retryable_same_deployment: false,
            failover_eligible: false,
            rejected_parameter: None,
            provider_detail: None,
        }
    }

    /// Attach one already-validated provider parameter path.
    pub fn with_rejected_parameter(mut self, parameter: Option<String>) -> Self {
        self.rejected_parameter = parameter;
        self
    }

    /// Attach one already-sanitized provider explanation.
    pub fn with_provider_detail(mut self, detail: Option<String>) -> Self {
        self.provider_detail = detail;
        self
    }

    /// Attach the python taxonomy's retry classification to this failure.
    pub fn with_retry(mut self, retryable_same_deployment: bool, failover_eligible: bool) -> Self {
        self.retryable_same_deployment = retryable_same_deployment;
        self.failover_eligible = failover_eligible;
        self
    }

    /// Coerce this failure to its boundary form: the python engine replaces
    /// malformed-response detail with one generic safe message before both
    /// accounting and the public error, so the native plane does the same.
    pub fn boundary(self) -> Self {
        match self.failure_class {
            FailureClass::MalformedResponse
                if self.safe_message
                    != "provider returned a malformed response; retry the request" =>
            {
                // The generic boundary message is about to erase the specific
                // parse-reject reason, so emit it once as a structured,
                // content-free operator line (the crate's stderr idiom) before
                // it is lost. The reason is always a static parser label, never
                // provider payload, so it is safe to log.
                let line = json!({
                    "event": "malformed_response_boundary",
                    "reason": self.safe_message,
                });
                eprintln!("exp-gateway-native: {line}");
                Failure::new(
                    FailureClass::MalformedResponse,
                    "provider returned a malformed response; retry the request",
                )
                .with_retry(self.retryable_same_deployment, self.failover_eligible)
            }
            _ => self,
        }
    }

    /// Map one failure to its public error, mirroring `public_failure_error`.
    ///
    /// Quota exhaustion omits the Python engine's month-boundary suffix because
    /// the reset boundary is computed control-plane side; the PoC returns the
    /// plain safe message with a one-hour retry hint instead.
    pub fn public_error(&self) -> PublicError {
        let (status, code, error_type) = match self.failure_class {
            FailureClass::InvalidRequest => (400, "invalid_request", "invalid_request_error"),
            FailureClass::UnsupportedCapability => {
                (400, "unsupported_capability", "invalid_request_error")
            }
            FailureClass::Authentication => (401, "invalid_key", "authentication_error"),
            FailureClass::Authorization => (403, "model_not_granted", "permission_error"),
            FailureClass::QuotaExceeded => (429, "insufficient_quota", "insufficient_quota"),
            FailureClass::Throttled => (429, "unavailable_route", "api_error"),
            FailureClass::Timeout => (504, "deadline_exceeded", "api_error"),
            FailureClass::Cancelled => (499, "request_cancelled", "api_error"),
            FailureClass::Guardrail => (400, "content_filter", "invalid_request_error"),
            FailureClass::Unavailable => (503, "gateway_unavailable", "api_error"),
            _ => (502, "all_routes_failed", "api_error"),
        };
        let mut error = PublicError::new(status, code, &self.safe_message, error_type);
        if self.failure_class == FailureClass::InvalidRequest {
            error.param = self.rejected_parameter.clone();
            if let Some(detail) = self.provider_detail.as_deref() {
                // The provider's own sentence replaces the generic "verify the
                // request fields" advice: it says which field and why, which is
                // the whole point of relaying it.
                let head = self
                    .safe_message
                    .split(';')
                    .next()
                    .unwrap_or_default()
                    .trim();
                error.message = format!("{head}: {detail}");
            }
        }
        error.retry_after_seconds = match self.failure_class {
            FailureClass::Throttled => Some(5),
            FailureClass::QuotaExceeded => Some(3600),
            FailureClass::Unavailable => Some(2),
            _ => None,
        };
        error
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejected_parameter_reaches_the_public_error_only_for_invalid_requests() {
        let attributed = Failure::new(FailureClass::InvalidRequest, "provider rejected")
            .with_rejected_parameter(Some("input[1].status".to_string()));
        assert_eq!(
            attributed.public_error().param.as_deref(),
            Some("input[1].status")
        );
        // Any other class stays param-free even if a parameter leaked in.
        let internal = Failure::new(FailureClass::ProviderInternal, "provider failed")
            .with_rejected_parameter(Some("input[1].status".to_string()));
        assert_eq!(internal.public_error().param, None);
        // Serde omits the field when absent, so boundary payloads are unchanged.
        let bare = serde_json::to_value(Failure::new(FailureClass::InvalidRequest, "x"))
            .expect("serializable");
        assert!(bare.get("rejected_parameter").is_none());
        let carried = serde_json::to_value(attributed).expect("serializable");
        assert_eq!(
            carried["rejected_parameter"].as_str(),
            Some("input[1].status")
        );
    }

    #[test]
    fn provider_detail_replaces_the_generic_advice_only_for_invalid_requests() {
        let explained = Failure::new(
            FailureClass::InvalidRequest,
            "provider rejected the request; verify the request fields against \
             the model alias capabilities",
        )
        .with_provider_detail(Some("`top_p` is deprecated for this model.".to_string()));
        assert_eq!(
            explained.public_error().message,
            "provider rejected the request: `top_p` is deprecated for this model."
        );
        // Any other class keeps its own message even if a detail leaked in.
        let internal = Failure::new(FailureClass::ProviderInternal, "provider failed")
            .with_provider_detail(Some("account 4711 is over its map".to_string()));
        assert_eq!(internal.public_error().message, "provider failed");
        let bare = serde_json::to_value(Failure::new(FailureClass::InvalidRequest, "x"))
            .expect("serializable");
        assert!(bare.get("provider_detail").is_none());
        let carried = serde_json::to_value(explained).expect("serializable");
        assert_eq!(
            carried["provider_detail"].as_str(),
            Some("`top_p` is deprecated for this model.")
        );
    }

    #[test]
    fn boundary_replaces_malformed_detail_with_the_generic_message() {
        let coerced = Failure::new(FailureClass::MalformedResponse, "specific detail").boundary();
        assert_eq!(
            coerced.safe_message,
            "provider returned a malformed response; retry the request"
        );
        let transport = Failure::new(FailureClass::Transport, "kept").boundary();
        assert_eq!(transport.safe_message, "kept");
    }

    #[test]
    fn failure_classes_round_trip_through_their_wire_names() {
        for class in [
            FailureClass::InvalidRequest,
            FailureClass::QuotaExceeded,
            FailureClass::MalformedResponse,
            FailureClass::Cancelled,
            FailureClass::Unavailable,
        ] {
            let wire = serde_json::to_value(class).expect("serializable");
            let back: FailureClass = serde_json::from_value(wire).expect("round trip");
            assert_eq!(back.as_str(), class.as_str());
        }
    }

    #[test]
    fn unavailable_maps_to_a_retryable_503() {
        // Parity with the python control plane's UNAVAILABLE mapping: a
        // transient roll condition is a retryable 503, not a closed 500/502.
        let failure = Failure::new(FailureClass::Unavailable, "the gateway is updating");
        let error = failure.public_error();
        assert_eq!(error.status_code, 503);
        assert_eq!(error.code, "gateway_unavailable");
        assert_eq!(error.error_type, "api_error");
        assert_eq!(error.retry_after_seconds, Some(2));
        // The wire name matches the python GatewayFailureClass member so a
        // failure serialized on either side deserializes on the other.
        assert_eq!(FailureClass::Unavailable.as_str(), "unavailable");
    }
}
