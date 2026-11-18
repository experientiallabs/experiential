//! The decisions admission bridge argument and its clone-free answer parse.

use axum::http::HeaderMap;
use serde::{Deserialize, Serialize};
use serde_json::{json, Map, Value};

use super::{DecisionWire, DecisionsAdmission};
use crate::respond::{client_ip, with_app_identity};

/// An admission bridge answer that is not a dispatchable decision route.
pub(super) enum Parsed {
    Escalate(String),
    Drift(Value),
    Malformed,
}

#[derive(Deserialize)]
struct EscalationProbe {
    #[serde(default)]
    escalate: Option<Value>,
}

/// Decode the admission straight from its text: the route carries every
/// rung's provider payload, so no intermediate `Value` copy is kept.
pub(super) fn parse_admission(
    text: &str,
    expected_wires: &[DecisionWire],
) -> Result<DecisionsAdmission, Parsed> {
    let Ok(probe) = serde_json::from_str::<EscalationProbe>(text) else {
        return Err(Parsed::Malformed);
    };
    if let Some(reason) = probe.escalate {
        return Err(Parsed::Escalate(
            reason.as_str().unwrap_or_default().to_string(),
        ));
    }
    match serde_json::from_str::<DecisionsAdmission>(text) {
        Ok(admission) if expected_wires.contains(&admission.wire) => Ok(admission),
        _ => Err(Parsed::Drift(
            serde_json::from_str(text).unwrap_or(Value::Null),
        )),
    }
}

#[derive(Serialize)]
struct AdmissionArgument<'a> {
    body: &'a str,
    /// UTF-8 length of `body`, so admission bounds it without re-encoding.
    body_bytes: usize,
    #[serde(flatten)]
    fields: Map<String, Value>,
}

/// Forward authority, app identity, and trusted proxy IP without caller idempotency.
pub(super) fn admission_argument(raw_key: &str, body: &str, headers: &HeaderMap) -> String {
    let mut fields = json!({
        "raw_key": raw_key,
        "client_ip": client_ip(headers),
        "capture_session_id": crate::capture::session_id(headers),
    });
    with_app_identity(&mut fields, headers);
    let Value::Object(fields) = fields else {
        unreachable!("the admission fields are built as an object")
    };
    serde_json::to_string(&AdmissionArgument {
        body,
        body_bytes: body.len(),
        fields,
    })
    .unwrap_or_else(|_| "null".to_string())
}
