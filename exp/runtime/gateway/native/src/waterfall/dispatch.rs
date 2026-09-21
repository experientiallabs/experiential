//! Per-dial dispatch facts: body signing immediately before the provider
//! POST, and the customer ownership of a BYOK rung's credential failures.

use std::collections::HashMap;

use serde_json::{json, Value};

use super::DeploymentWire;
use crate::bridge::Bridge;
use crate::encode::compact_json;
use crate::errors::{Failure, PublicError};

/// Resolve the dispatch headers for one physical open attempt. Body-signing
/// dialects (Bedrock SigV4) are signed here, immediately before the provider
/// POST, so neither queue time nor a spent earlier attempt can age the
/// signature toward AWS's short clock window. Other dialects use the route
/// entry's headers unchanged.
pub(super) async fn dispatch_headers(
    bridge: &Bridge,
    request_id: &str,
    wire: &DeploymentWire,
) -> Result<HashMap<String, String>, PublicError> {
    let mut headers = wire.headers.clone();
    let Some(body) = wire.upstream_body.as_deref() else {
        return Ok(headers);
    };
    let argument = compact_json(&json!({
        "request_id": request_id,
        "url": wire.url,
        "body": body,
    }));
    let text = bridge.call("sign_dispatch", argument).await?;
    let signed: HashMap<String, String> = serde_json::from_str::<Value>(&text)
        .ok()
        .and_then(|value| {
            serde_json::from_value(value.get("headers").cloned().unwrap_or(Value::Null)).ok()
        })
        .ok_or_else(PublicError::internal)?;
    headers.extend(signed);
    Ok(headers)
}

/// Open and read one physical attempt up to commitment or its terminal.
/// On a customer-managed rung, a rejected credential or exhausted provider
/// account at stream OPEN is the customer's to fix (see
/// `stream_errors::customer_credential_failure`); failures declared on the
/// open stream take the same path inside the relay. House rungs are unchanged.
pub(super) fn customer_owned(failure: Failure, wire: &DeploymentWire) -> Failure {
    if wire.billing_customer_managed {
        crate::stream_errors::customer_credential_failure(failure, &wire.provider)
    } else {
        failure
    }
}
