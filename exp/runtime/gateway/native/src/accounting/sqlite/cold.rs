//! Administrative entry points into the same native financial implementation.

use super::db::{query, Result};
use super::{attempts, authority, budgets, db, requests, Store};
use crate::errors::PublicError;
use chrono::{DateTime, Duration};
use rusqlite::{params, Connection};
use serde_json::{json, Value};

pub(super) fn apply(
    db: &Connection,
    store: &Store,
    operation: &str,
    input: &Value,
) -> Result<Value> {
    // These calls join an explicit administrative transaction, never autocommit.
    if db.is_autocommit() {
        return Err(PublicError::internal());
    }
    let now = authority::text(input, "now")?;
    let stamp = DateTime::parse_from_rfc3339(now).map_err(|_| PublicError::internal())?;
    let auth = &input["authorization"];
    let proof = if matches!(operation, "accept" | "start") {
        let seconds = input["remaining_seconds"]
            .as_f64()
            .filter(|s| s.is_finite() && *s >= 0.0)
            .ok_or_else(PublicError::internal)?
            .min(86400.0);
        Some(store.proof(
            db,
            auth,
            std::time::Instant::now() + std::time::Duration::from_secs_f64(seconds),
        )?)
    } else {
        None
    };
    if let Some(proof) = &proof {
        proof.validate(db, auth)?;
    }
    let result = match operation {
        "backfill_budget" => {
            budgets::backfill(db, authority::text(input, "budget_id")?)?;
            Value::Null
        }
        "accept" => {
            requests::accept(db, auth, now, authority::text(input, "deadline_at")?)?;
            Value::Null
        }
        "start" => attempts::start(
            db,
            store,
            input,
            &input["dispatch"],
            now,
            &stamp.format("%Y-%m-01T00:00:00+00:00").to_string(),
        )?,
        "settle" => {
            let id = authority::text(input, "attempt_id")?;
            let row = query(
                db,
                "SELECT request_id, organization_id FROM gateway_attempts WHERE attempt_id=?",
                params![id],
            )?
            .pop()
            .ok_or_else(PublicError::internal)?;
            attempts::settle(db, &row, id, input["terminal"].clone(), now)?
        }
        "finish" => json!(requests::finish(
            db,
            auth,
            &input["terminal"],
            input["certify_no_effects"] == true,
            input["web_search_requests"]
                .as_u64()
                .ok_or_else(PublicError::internal)?,
            now
        )?),
        "reconcile_liability" => {
            let id = authority::text(input, "attempt_id")?;
            let cost = input["assigned_cost_nano_usd"]
                .as_i64()
                .filter(|n| *n >= 0)
                .ok_or_else(PublicError::internal)?;
            let row = query(db, "SELECT a.*, r.api_surface FROM gateway_attempts a JOIN gateway_requests r ON r.request_id=a.request_id WHERE a.attempt_id=?", params![id])?.pop().ok_or_else(PublicError::internal)?;
            if row["api_surface"] != "decisions"
                || row["state"] == "dispatched"
                || row["usage_source"] != "unknown"
                || row["budget_reserved_nano_usd"].is_null()
            {
                return Err(db::invariant(
                    "Attempt has no terminal decision liability to reconcile.",
                ));
            }
            if !row["budget_settled_nano_usd"].is_null() && row["budget_settled_nano_usd"] != cost {
                return Err(db::invariant(
                    "Decision liability already resolved at another cost.",
                ));
            }
            budgets::settle(db, id, Some(cost))?;
            db::update(
                db,
                "gateway_attempts",
                "attempt_id",
                id,
                &[("budget_settled_nano_usd", json!(cost))],
            )?;
            Value::Null
        }
        "recover" => {
            let grace = input["cleanup_grace_seconds"]
                .as_i64()
                .filter(|n| *n >= 0)
                .ok_or_else(PublicError::internal)?;
            let mut expired = 0;
            let mut unknown = 0;
            for row in query(db, "SELECT r.*, EXISTS(SELECT 1 FROM gateway_attempts a WHERE a.request_id=r.request_id) AS has_attempt FROM gateway_requests r WHERE terminal_state IS NULL", [])? {
                let deadline=DateTime::parse_from_rfc3339(authority::text(&row,"deadline_at")?).map_err(|_| PublicError::internal())?;
                if row["has_attempt"]==0 && deadline<=stamp {
                    db::update(db,"gateway_requests","request_id",authority::text(&row,"request_id")?,&[("terminal_state",json!("expired_before_dispatch")),("terminal_at",json!(now))])?;
                    expired+=1;
                }
            }
            for row in query(db,"SELECT a.attempt_id,a.request_id,r.deadline_at FROM gateway_attempts a JOIN gateway_requests r ON r.request_id=a.request_id WHERE a.state='dispatched'",[])? {
                let deadline=DateTime::parse_from_rfc3339(authority::text(&row,"deadline_at")?).map_err(|_| PublicError::internal())?;
                if deadline.checked_add_signed(Duration::seconds(grace)).ok_or_else(PublicError::internal)?>stamp { continue; }
                db::update(db,"gateway_attempts","attempt_id",authority::text(&row,"attempt_id")?,&[("state",json!("unknown_after_crash")),("terminal_at",json!(now)),("usage_source",json!("unknown"))])?;
                db.execute("UPDATE gateway_requests SET terminal_state='unknown_after_crash',terminal_at=? WHERE request_id=? AND terminal_state IS NULL",params![now,authority::text(&row,"request_id")?]).map_err(db::error)?;
                unknown+=1;
            }
            json!([expired, unknown])
        }
        _ => return Err(PublicError::internal()),
    };
    if let Some(proof) = &proof {
        proof.validate(db, auth)?;
    }
    Ok(result)
}
