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
    Internal,
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
            FailureClass::Internal => "internal",
        }
    }
}

/// One sanitized provider failure, the Rust mirror of `GatewayFailure`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Failure {
    pub failure_class: FailureClass,
    pub safe_message: String,
}

impl Failure {
    pub fn new(failure_class: FailureClass, safe_message: &str) -> Self {
        Self {
            failure_class,
            safe_message: safe_message.to_string(),
        }
    }

    /// Coerce this failure to its boundary form: the python engine replaces
    /// malformed-response detail with one generic safe message before both
    /// accounting and the public error, so the native plane does the same.
    pub fn boundary(self) -> Self {
        match self.failure_class {
            FailureClass::MalformedResponse
                if self.safe_message != "provider returned a malformed response; retry the request" =>
            {
                Failure::new(
                    FailureClass::MalformedResponse,
                    "provider returned a malformed response; retry the request",
                )
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
            _ => (502, "all_routes_failed", "api_error"),
        };
        let mut error = PublicError::new(status, code, &self.safe_message, error_type);
        error.retry_after_seconds = match self.failure_class {
            FailureClass::Throttled => Some(5),
            FailureClass::QuotaExceeded => Some(3600),
            _ => None,
        };
        error
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
        ] {
            let wire = serde_json::to_value(class).expect("serializable");
            let back: FailureClass = serde_json::from_value(wire).expect("round trip");
            assert_eq!(back.as_str(), class.as_str());
        }
    }
}
