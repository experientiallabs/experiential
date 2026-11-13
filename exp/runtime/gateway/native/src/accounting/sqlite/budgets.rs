//! Atomic local budget reservations and settlement under the ledger write fence.

use rusqlite::{params, Connection};
use serde_json::{json, Value};

use crate::errors::PublicError;

use super::db::{query, Result};

pub(super) struct Reservation<'a> {
    pub organization: &'a str,
    pub identity: &'a str,
    pub alias: &'a str,
    pub root_pool: &'a str,
    pub pool: &'a str,
    pub deployment: &'a str,
    pub attempt: &'a str,
    pub period: &'a str,
    pub maximum: Option<i64>,
}

pub(super) fn reserve(db: &Connection, r: &Reservation<'_>) -> Result<Option<Value>> {
    let budgets = query(db, "SELECT * FROM gateway_monthly_budgets
        WHERE organization_id = ? AND period_start = ? AND (
            scope_kind = 'team' OR (scope_kind = 'identity' AND identity_id = ?)
            OR (scope_kind = 'pool' AND alias_id = ? AND pool_id IN (?, ?))
            OR (scope_kind = 'deployment' AND alias_id = ? AND pool_id = ? AND deployment_id = ?))
        ORDER BY CASE WHEN scope_kind IN ('team', 'identity') OR (scope_kind = 'pool' AND pool_id = ?)
            THEN 0 ELSE 1 END, scope_kind, scope_key",
        params![r.organization, r.period, r.identity, r.alias, r.pool, r.root_pool, r.alias, r.pool, r.deployment, r.root_pool])?;
    // Validate every scope before touching any counter. A refusal is an ordinary
    // return value, so the caller need not accidentally commit a partial hold.
    for b in &budgets {
        let limit = integer(b, "limit_nano_usd")?;
        let charged = integer(b, "reserved_nano_usd")?
            .checked_add(integer(b, "settled_nano_usd")?)
            .ok_or_else(PublicError::internal)?;
        let strict = integer(b, "strict_unknown_cost")? != 0;
        if (strict && (r.maximum.is_none() || integer(b, "unknown_cost_attempts")? > 0))
            || r.maximum
                .is_some_and(|amount| i128::from(amount) > i128::from(limit) - i128::from(charged))
        {
            let scoped = matches!(b["scope_kind"].as_str(), Some("pool" | "deployment"))
                && !(b["scope_kind"] == "pool" && b["pool_id"] == r.root_pool);
            return Ok(Some(json!({
                "scope_kind": if scoped { b["scope_kind"].clone() } else { json!("request") },
                "scope_id": if b["scope_kind"] == "deployment" { &b["deployment_id"] } else { &b["pool_id"] },
                "budget_scope": {"kind":b["scope_kind"],"identity_id":b["identity_id"],"alias_id":b["alias_id"],"pool_id":b["pool_id"],"deployment_id":b["deployment_id"]},
                "application": if scoped {"destination"} else if b["scope_kind"]=="pool" && r.pool != r.root_pool {"root"} else {"shared"},
                "error": PublicError::new(429, "insufficient_quota", if strict && r.maximum.is_none() {
                    "This allocation requires a known maximum attempt cost."
                } else if strict && b["unknown_cost_attempts"].as_i64().unwrap_or(0) > 0 {
                    "This allocation has prior attempts with unknown cost."
                } else { "Monthly gateway allocation is exhausted." }, "insufficient_quota")
            })));
        }
    }
    for b in budgets {
        let id = b["budget_id"].as_str().ok_or_else(PublicError::internal)?;
        if let Some(maximum) = r.maximum {
            let held = integer(&b, "reserved_nano_usd")?
                .checked_add(maximum)
                .ok_or_else(PublicError::internal)?;
            db.execute(
                "UPDATE gateway_monthly_budgets SET reserved_nano_usd = ? WHERE budget_id = ?",
                params![held, id],
            )
            .map_err(super::db::error)?;
        } else {
            let unknown = integer(&b, "unknown_cost_attempts")?
                .checked_add(1)
                .ok_or_else(PublicError::internal)?;
            db.execute(
                "UPDATE gateway_monthly_budgets SET unknown_cost_attempts = ? WHERE budget_id = ?",
                params![unknown, id],
            )
            .map_err(super::db::error)?;
        }
        db.execute("INSERT INTO gateway_attempt_budget_charges (attempt_id, budget_id, reserved_nano_usd) VALUES (?, ?, ?)", params![r.attempt, id, r.maximum]).map_err(super::db::error)?;
    }
    Ok(None)
}

pub(super) fn settle(db: &Connection, attempt: &str, settled: Option<i64>) -> Result<()> {
    let Some(settled) = settled else {
        return Ok(());
    };
    let charges = query(
        db,
        "SELECT c.*, b.settled_nano_usd AS budget_settled_nano_usd
        FROM gateway_attempt_budget_charges AS c
        JOIN gateway_monthly_budgets AS b ON b.budget_id = c.budget_id WHERE c.attempt_id = ?",
        params![attempt],
    )?;
    for charge in charges {
        if !charge["settled_nano_usd"].is_null() {
            continue;
        }
        let id = charge["budget_id"]
            .as_str()
            .ok_or_else(PublicError::internal)?;
        let total = integer(&charge, "budget_settled_nano_usd")?
            .checked_add(settled)
            .ok_or_else(PublicError::internal)?;
        let changed = if let Some(reserved) = charge["reserved_nano_usd"].as_i64() {
            db.execute("UPDATE gateway_monthly_budgets SET reserved_nano_usd = reserved_nano_usd - ?, settled_nano_usd = ? WHERE budget_id = ? AND reserved_nano_usd >= ?", params![reserved, total, id, reserved])
        } else {
            db.execute("UPDATE gateway_monthly_budgets SET unknown_cost_attempts = unknown_cost_attempts - 1, settled_nano_usd = ? WHERE budget_id = ? AND unknown_cost_attempts > 0", params![total, id])
        }.map_err(super::db::error)?;
        if changed != 1 || db.execute("UPDATE gateway_attempt_budget_charges SET settled_nano_usd = ? WHERE budget_id = ? AND attempt_id = ? AND settled_nano_usd IS NULL", params![settled, id, attempt]).map_err(super::db::error)? != 1 {
            return Err(PublicError::internal());
        }
    }
    Ok(())
}

fn integer(row: &Value, field: &str) -> Result<i64> {
    row[field]
        .as_i64()
        .filter(|n| *n >= 0)
        .ok_or_else(PublicError::internal)
}

pub(super) fn replace(
    db: &Connection,
    attempt: &str,
    old: Option<i64>,
    new: Option<i64>,
) -> Result<()> {
    let charges = query(db, "SELECT c.*, b.settled_nano_usd AS budget_settled, b.reserved_nano_usd AS budget_reserved, b.unknown_cost_attempts FROM gateway_attempt_budget_charges c JOIN gateway_monthly_budgets b ON b.budget_id=c.budget_id WHERE c.attempt_id=?", params![attempt])?;
    for row in charges {
        let prior = row["settled_nano_usd"].as_i64();
        if prior.is_none() || old.is_none() || prior != old || prior == new {
            continue;
        }
        let prior = prior.unwrap();
        let total = integer(&row, "budget_settled")?
            .checked_sub(prior)
            .filter(|n| *n >= 0)
            .and_then(|n| n.checked_add(new.unwrap_or(0)))
            .ok_or_else(|| {
                super::db::invariant(
                    "reconciled monthly gateway cost exceeds SQLite integer capacity",
                )
            })?;
        let mut reserved = integer(&row, "budget_reserved")?;
        let mut unknown = integer(&row, "unknown_cost_attempts")?;
        if new.is_none() {
            if let Some(bound) = row["reserved_nano_usd"].as_i64() {
                reserved = reserved.checked_add(bound).ok_or_else(|| {
                    super::db::invariant(
                        "reconciled monthly gateway cost exceeds SQLite integer capacity",
                    )
                })?;
            } else {
                unknown = unknown.checked_add(1).ok_or_else(PublicError::internal)?;
            }
        }
        let id = row["budget_id"]
            .as_str()
            .ok_or_else(PublicError::internal)?;
        let changed = db.execute("UPDATE gateway_monthly_budgets SET settled_nano_usd=?,reserved_nano_usd=?,unknown_cost_attempts=? WHERE budget_id=? AND settled_nano_usd>=?",params![total,reserved,unknown,id,prior]).map_err(super::db::error)?;
        if changed != 1 {
            return Err(super::db::invariant(
                "monthly budget counters are inconsistent",
            ));
        }
        db.execute("UPDATE gateway_attempt_budget_charges SET settled_nano_usd=? WHERE budget_id=? AND attempt_id=?",params![new,id,attempt]).map_err(super::db::error)?;
    }
    settle(db, attempt, new)
}

/// Attach a new limit to prior attempts without losing holds or explicit adjustments.
pub(super) fn backfill(db: &Connection, budget_id: &str) -> Result<()> {
    let b = query(
        db,
        "SELECT * FROM gateway_monthly_budgets WHERE budget_id=?",
        params![budget_id],
    )?
    .pop()
    .ok_or_else(PublicError::internal)?;
    let rows = query(db, "SELECT a.attempt_id, a.budget_reserved_nano_usd, a.budget_settled_nano_usd, a.estimated_cost_nano_usd
        FROM gateway_attempts a JOIN gateway_requests r ON r.request_id=a.request_id
        WHERE a.organization_id=?1 AND a.budget_period_start=?2 AND (
            ?3='team' OR (?3='identity' AND r.identity_id=?4) OR
            (?3='pool' AND r.alias_id=?5 AND (a.pool_id=?6 OR EXISTS (
                SELECT 1 FROM alias_revisions v WHERE v.organization_id=r.organization_id
                AND v.alias_id=r.alias_id AND v.revision_id=r.alias_revision_id AND v.pool_id=?6))) OR
            (?3='deployment' AND r.alias_id=?5 AND a.pool_id=?6 AND a.deployment_id=?7))
        ORDER BY a.attempt_id", params![b["organization_id"].as_str(), b["period_start"].as_str(), b["scope_kind"].as_str(),
            b["identity_id"].as_str(), b["alias_id"].as_str(), b["pool_id"].as_str(), b["deployment_id"].as_str()])?;
    let (mut held, mut charged, mut unknown) = (0_i64, 0_i64, 0_i64);
    for row in rows {
        let settled = row["budget_settled_nano_usd"].as_i64().or_else(|| {
            row["budget_reserved_nano_usd"]
                .is_null()
                .then(|| row["estimated_cost_nano_usd"].as_i64())
                .flatten()
        });
        let reserved = if settled.is_none() {
            row["budget_reserved_nano_usd"].as_i64()
        } else {
            None
        };
        let overflow = || {
            super::db::invariant("historical monthly gateway cost exceeds SQLite integer capacity")
        };
        held = held
            .checked_add(reserved.unwrap_or(0))
            .ok_or_else(overflow)?;
        charged = charged
            .checked_add(settled.unwrap_or(0))
            .ok_or_else(overflow)?;
        unknown = unknown
            .checked_add(i64::from(settled.is_none() && reserved.is_none()))
            .ok_or_else(overflow)?;
        db.execute("INSERT INTO gateway_attempt_budget_charges (budget_id,attempt_id,reserved_nano_usd,settled_nano_usd) VALUES (?,?,?,?)",
            params![budget_id, row["attempt_id"].as_str().ok_or_else(PublicError::internal)?, reserved, settled]).map_err(super::db::error)?;
    }
    db.execute("UPDATE gateway_monthly_budgets SET reserved_nano_usd=?,settled_nano_usd=?,unknown_cost_attempts=? WHERE budget_id=?",
        params![held,charged,unknown,budget_id]).map_err(super::db::error)?;
    Ok(())
}
