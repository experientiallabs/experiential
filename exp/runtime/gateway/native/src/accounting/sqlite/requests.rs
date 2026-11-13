//! Durable request acceptance, idempotency and no-effects certificates.

use rusqlite::{params, Connection};
use serde_json::{json, Value};

use super::{
    authority::text,
    db::{self, query, Result},
};
use crate::errors::PublicError;

pub(super) fn accept(
    db: &Connection,
    authorization: &Value,
    now: &str,
    deadline: &str,
) -> Result<()> {
    let org = text(authorization, "organization_id")?;
    let identity = text(authorization, "identity_id")?;
    let revision = text(authorization, "alias_revision_id")?;
    if !authorization["caller_operation_sha256"].is_null() {
        let prior = query(db, "SELECT canonical_request_sha256, terminal_state, failed_without_effects,
            NOT EXISTS (SELECT 1 FROM gateway_attempts a WHERE a.request_id = gateway_requests.request_id) AS no_dispatch
            FROM gateway_requests WHERE organization_id=? AND identity_id=? AND alias_revision_id=?
            AND api_surface=? AND caller_operation_sha256=? ORDER BY rowid DESC LIMIT 1",
            params![org, identity, revision, text(authorization, "surface")?, text(authorization, "caller_operation_sha256")?])?;
        if let Some(prior) = prior.first() {
            if prior["canonical_request_sha256"] != authorization["canonical_request_sha256"] {
                return Err(PublicError::new(
                    409,
                    "idempotency_conflict",
                    "Caller operation key was reused with different request content.",
                    "invalid_request_error",
                ));
            }
            if !matches!(
                prior["terminal_state"].as_str(),
                Some("expired_before_dispatch" | "unknown_after_crash")
            ) && !(prior["terminal_state"] == "failed"
                && prior["failed_without_effects"] == 1
                && prior["no_dispatch"] == 1)
            {
                return Err(PublicError::new(
                    409,
                    "idempotency_replay_unavailable",
                    "Matching keyed request exists but durable content replay is unavailable.",
                    "invalid_request_error",
                ));
            }
        }
    }
    let row = query(
        db,
        "SELECT alias_id FROM alias_revisions WHERE organization_id=? AND revision_id=?",
        params![org, revision],
    )?
    .pop()
    .ok_or_else(PublicError::internal)?;
    let mut fields = vec![
        ("alias_id", row["alias_id"].clone()),
        ("api_surface", authorization["surface"].clone()),
        ("key_id", authorization["virtual_key_id"].clone()),
        ("accepted_at", json!(now)),
        ("deadline_at", json!(deadline)),
    ];
    for field in [
        "request_id",
        "organization_id",
        "identity_id",
        "alias_revision_id",
        "canonical_request_sha256",
        "caller_operation_sha256",
        "app_referer",
        "app_title",
        "client_app",
        "user_agent",
    ] {
        fields.push((field, authorization[field].clone()));
    }
    db::insert(db, "gateway_requests", &fields)
}

pub(super) fn accepted(db: &Connection, authorization: &Value) -> Result<Value> {
    let row = query(
        db,
        "SELECT * FROM gateway_requests WHERE request_id=?",
        params![text(authorization, "request_id")?],
    )?
    .pop()
    .ok_or_else(PublicError::internal)?;
    for field in ["organization_id", "identity_id", "alias_revision_id"] {
        if row[field] != authorization[field] {
            return Err(PublicError::internal());
        }
    }
    Ok(row)
}

pub(super) fn finish(
    db: &Connection,
    authorization: &Value,
    terminal: &Value,
    certify: bool,
    searches: u64,
    now: &str,
) -> Result<bool> {
    let row = accepted(db, authorization)?;
    let request = text(authorization, "request_id")?;
    let attempts = query(
        db,
        "SELECT attempt_id FROM gateway_attempts WHERE request_id=? LIMIT 1",
        params![request],
    )?;
    if searches > 0 && !attempts.is_empty() {
        return Err(PublicError::internal());
    }
    let state = text(terminal, "state")?;
    if !row["terminal_state"].is_null() {
        if row["terminal_state"] != state
            || (searches > 0 && row["web_search_requests"] != searches)
        {
            return Err(PublicError::internal());
        }
        return Ok(row["failed_without_effects"] == 1 && attempts.is_empty());
    }
    let certified = certify
        && searches == 0
        && attempts.is_empty()
        && state == "failed"
        && authorization["target"]["kind"] == "direct";
    db::update(
        db,
        "gateway_requests",
        "request_id",
        request,
        &[
            ("terminal_state", json!(state)),
            ("terminal_at", json!(now)),
            ("failed_without_effects", json!(certified)),
            ("web_search_requests", json!(searches)),
        ],
    )?;
    Ok(certified)
}
