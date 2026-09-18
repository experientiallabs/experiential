//! The Anthropic error envelope and the surface's sanitized refusal failure,
//! split from `encode_messages` so the implementation stays within the
//! repository line budget.

use serde_json::{json, Value};

use crate::errors::{Failure, FailureClass, PublicError};

const REFUSAL_MESSAGE: &str = "provider refused the request";

/// The sanitized failure for provider refusals on this surface, mirroring
/// `refusal_failure` in the python encoder.
pub fn refusal_failure() -> Failure {
    Failure::new(FailureClass::Refusal, REFUSAL_MESSAGE)
}

/// Render one sanitized public error as the Anthropic error envelope,
/// mirroring `anthropic_error_body`: status decides the Anthropic type
/// first, then the OpenAI envelope type, and a present `param` pointer is
/// folded into the message text.
pub fn anthropic_error_body(error: &PublicError) -> Value {
    let error_type = match error.status_code {
        401 => "authentication_error",
        403 => "permission_error",
        404 => "not_found_error",
        413 => "request_too_large",
        429 => "rate_limit_error",
        503 => "overloaded_error",
        _ if error.error_type == "invalid_request_error" => "invalid_request_error",
        _ => "api_error",
    };
    let message = match &error.param {
        Some(param) if !param.is_empty() => format!("{} (param: {param})", error.message),
        _ => error.message.clone(),
    };
    let mut body = json!({
        "type": "error",
        "error": {"type": error_type, "message": message},
    });
    // A refusal carries its bounded category on the Anthropic envelope too, so
    // a Messages caller reads the same machine-readable reason as a Chat one.
    if let Some(reason) = error.refusal_reason {
        body["error"]["refusal_reason"] = json!(reason.as_str());
    }
    body
}
