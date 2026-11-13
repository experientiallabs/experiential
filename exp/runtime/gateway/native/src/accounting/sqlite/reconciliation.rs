//! Late meters may correct provisional attribution, never immutable delivery or explicit waivers.

use super::{
    attempts::{decode, rates, TOKENS},
    budgets,
    db::{self, Result},
};
use crate::errors::PublicError;
use rusqlite::Connection;
use serde_json::{json, Value};

const CONFLICT: &str =
    "attempt settlement replay differs from durable accounting evidence (durable usage or cost)";

fn recovered(row: &Value) -> bool {
    row["accounting_recovery"] == 1 || row["state"] == "unknown_after_crash"
}
fn complete(terminal: &Value) -> bool {
    let usage = &terminal["usage"];
    terminal["usage_estimated"] != true
        && terminal["usage_incomplete_due_to_disconnect"] != true
        && ((usage["input_tokens"].is_u64() && usage["output_tokens"].is_u64())
            || (usage["billed_units"].is_object() && TOKENS.iter().all(|k| usage[*k].is_null())))
}
fn same_meter(row: &Value, terminal: &Value) -> bool {
    let usage = &terminal["usage"];
    TOKENS.iter().all(|key| row[*key] == usage[*key])
        && [
            ("billed_unit_kind", "kind"),
            ("billed_unit_variant", "variant"),
            ("billed_quantity_milli", "quantity_milli"),
        ]
        .iter()
        .all(|(column, field)| row[*column] == usage["billed_units"][*field])
}
fn retain_tier(db: &Connection, row: &Value, id: &str, terminal: &Value) -> Result<()> {
    let prior = decode(row, "service_tier_settlement")?;
    let tier = &terminal["service_tier"];
    if prior["resolution"] != "confirmed" && tier["resolution"] == "confirmed" {
        let mut fields = vec![("service_tier_settlement", tier.clone())];
        rates(&mut fields, &tier["prices"])?;
        // The private terminal retains the meter independently of an unmetered receipt.
        let mut stored = decode(row, "native_accounting_terminal")?;
        if stored.is_object() {
            stored["service_tier"] = tier.clone();
            fields.push(("native_accounting_terminal", stored));
        }
        db::update(db, "gateway_attempts", "attempt_id", id, &fields)?;
    }
    Ok(())
}

/// Return true only when the caller owes a complete replacement meter and budget delta.
pub(super) fn upgrade(
    db: &Connection,
    row: &Value,
    id: &str,
    terminal: &Value,
    supplied_tier: bool,
    cost: Option<i64>,
) -> Result<bool> {
    let usage = &terminal["usage"];
    let provisional = recovered(row) || row["state"] == "cancelled";
    if terminal["accounting_release_reservation"] == true {
        if !provisional {
            let observed = row["usage_source"] == "observed"
                && (row["billed_quantity_milli"].is_i64()
                    || (row["input_tokens"].is_i64() && row["output_tokens"].is_i64()));
            return Err(PublicError::new(
                500,
                if observed {
                    "accounting_recovery_superseded"
                } else {
                    "accounting_recovery_unresolved"
                },
                if observed {
                    "accounting recovery was superseded by an immutable provider terminal"
                } else {
                    "customer waiver requires durable accounting recovery"
                },
                "api_error",
            ));
        }
        budgets::replace(db, id, row["budget_settled_nano_usd"].as_i64(), Some(0))?;
        db::update(
            db,
            "gateway_attempts",
            "attempt_id",
            id,
            &[("budget_settled_nano_usd", json!(0))],
        )?;
        return Ok(false);
    }
    if complete(terminal) {
        if row["usage_source"] == "estimated" || (row["usage_source"] == "unknown" && provisional) {
            return Ok(true);
        }
        if row["usage_source"] == "unknown" {
            return Err(db::invariant(CONFLICT));
        }
        let prior = decode(row, "service_tier_settlement")?;
        if !same_meter(row, terminal)
            || ((!prior.is_object() || prior["resolution"] == "confirmed")
                && row["estimated_cost_nano_usd"]
                    .as_i64()
                    .is_some_and(|n| Some(n) != cost))
        {
            return Err(db::invariant(CONFLICT));
        }
        if prior.is_object()
            && ((prior["resolution"] != "confirmed" && supplied_tier)
                || (prior["resolution"] == "confirmed"
                    && recovered(row)
                    && row["estimated_cost_nano_usd"].is_null()))
        {
            return Ok(true);
        }
        return Ok(false);
    }
    if usage.is_null()
        && !terminal["failure"].is_null()
        && matches!(terminal["state"].as_str(), Some("failed" | "cancelled"))
        && recovered(row)
    {
        if supplied_tier {
            retain_tier(db, row, id, terminal)?;
        }
        return Ok(false);
    }
    if row["state"] != terminal["state"]
        && !(row["state"] == "unknown_after_crash" && supplied_tier)
    {
        return Err(db::invariant(
            "attempt is already settled with another terminal state",
        ));
    }
    if !supplied_tier || terminal["service_tier"]["resolution"] != "confirmed" {
        return Ok(false);
    }
    if row["state"] == "unknown_after_crash" && row["usage_source"] == "unknown" {
        return Err(db::invariant(
            "crash tier recovery requires complete observed usage",
        ));
    }
    if !same_meter(row, terminal) {
        return Err(db::invariant(
            "service-tier receipt replay differs from the durable usage",
        ));
    }
    let source = if usage.is_null() {
        "unknown"
    } else if terminal["usage_estimated"] == true {
        "estimated"
    } else {
        "observed"
    };
    if row["usage_source"] != source {
        return Err(db::invariant(
            "service-tier receipt replay differs from the usage evidence",
        ));
    }
    let prior = decode(row, "service_tier_settlement")?;
    if prior == terminal["service_tier"]
        && cost.is_none()
        && row["budget_settled_nano_usd"].is_null()
    {
        return Ok(false);
    }
    if let Some(charged) = row["budget_settled_nano_usd"].as_i64() {
        if Some(charged) != cost {
            return Err(db::invariant(
                "service-tier replay differs from the settled amount",
            ));
        }
        return Ok(false);
    }
    if cost.is_none() {
        return Err(db::invariant(
            "confirmed service-tier receipt still lacks a complete price or meter",
        ));
    }
    Ok(true)
}
