//! Local attempt records with immutable pricing, recoverable terminals and scoped holds.

use rusqlite::{params, Connection};
use serde_json::{json, Value};

use super::{
    authority::text,
    budgets,
    db::{self, query, Result},
    reconciliation, requests, valuation,
};
use crate::errors::PublicError;

const RATES: [(&str, &str); 6] = [
    ("input_rate", "input"),
    ("cached_input_rate", "cached_input"),
    ("cache_creation_input_rate", "cache_creation_input"),
    ("cache_creation_1h_input_rate", "cache_creation_1h_input"),
    ("output_rate", "output"),
    ("reasoning_rate", "reasoning"),
];
const LONG: [&str; 6] = [
    "long_context_input_rate",
    "long_context_cached_input_rate",
    "long_context_cache_creation_input_rate",
    "long_context_cache_creation_1h_input_rate",
    "long_context_output_rate",
    "long_context_reasoning_rate",
];
const PREFERRED: [&str; 6] = [
    "preferred_input_rate",
    "preferred_cached_input_rate",
    "preferred_cache_creation_input_rate",
    "preferred_cache_creation_1h_input_rate",
    "preferred_output_rate",
    "preferred_reasoning_rate",
];
pub(super) const TOKENS: [&str; 6] = [
    "input_tokens",
    "output_tokens",
    "cached_input_tokens",
    "cache_creation_input_tokens",
    "cache_creation_1h_input_tokens",
    "reasoning_tokens",
];

pub(super) fn rates(fields: &mut Vec<(&str, Value)>, prices: &Value) -> Result<()> {
    for (column, field) in RATES {
        fields.push((column, json!(valuation::rate(prices, field)?)));
    }
    fields.push((
        "long_context_threshold_tokens",
        prices["long_context"]["input_threshold_tokens"].clone(),
    ));
    for (column, (_, field)) in LONG.into_iter().zip(RATES) {
        fields.push((
            column,
            json!(valuation::rate(&prices["long_context"], field)?),
        ));
    }
    Ok(())
}

pub(super) fn start(
    db: &Connection,
    store: &super::Store,
    registration: &Value,
    dispatch: &Value,
    now: &str,
    period: &str,
) -> Result<Value> {
    let auth = &registration["authorization"];
    let snapshot = &registration["snapshot"];
    let depth = dispatch["depth"]
        .as_u64()
        .and_then(|d| usize::try_from(d).ok())
        .ok_or_else(PublicError::internal)?;
    let stage = store.stage(snapshot, depth)?;
    let deployment = registration["deployments"]
        .as_array()
        .and_then(|lanes| lanes.get(depth))
        .ok_or_else(PublicError::internal)?;
    let deployment_id = text(deployment, "deployment_id")?;
    if snapshot["deployment_ids"][depth] != deployment_id
        || deployment["exact_model_id"] != stage["exact_model_id"]
    {
        return Err(PublicError::internal());
    }
    for field in ["route_reason", "fallback_reason", "dispatch_reason"] {
        if !dispatch[field].is_null()
            && dispatch[field]
                .as_str()
                .is_none_or(|s| s.chars().count() > 512 || s.chars().any(|c| c < ' '))
        {
            return Err(db::invariant("Dispatch reasons must be display-safe."));
        }
    }
    let maximum = if dispatch["maximum_cost_nano"].is_null() {
        None
    } else {
        Some(
            dispatch["maximum_cost_nano"]
                .as_i64()
                .filter(|n| *n >= 0)
                .ok_or_else(PublicError::internal)?,
        )
    };
    let row = requests::accepted(db, auth)?;
    let preferred = dispatch["preferred_deployment_id"]
        .as_str()
        .map(|id| {
            if id == deployment_id {
                return Err(db::invariant("Preferred deployment must be divergent."));
            }
            registration["deployments"]
                .as_array()
                .and_then(|lanes| lanes.iter().find(|lane| lane["deployment_id"] == id))
                .ok_or_else(PublicError::internal)
        })
        .transpose()?;
    let prices = &dispatch["prices"];
    if !prices.is_object() {
        return Err(PublicError::internal());
    }
    let effective_at = deployment["gateway"]["pricing_effective_at"]
        .as_str()
        .map(|value| {
            chrono::DateTime::parse_from_rfc3339(value)
                .map(|time| super::timestamp(time.with_timezone(&chrono::Utc)))
                .map_err(|_| PublicError::internal())
        })
        .transpose()?;
    let id = format!("attempt-{}", uuid::Uuid::new_v4().simple());
    let mut fields = vec![
        ("attempt_id", json!(id)),
        ("request_id", auth["request_id"].clone()),
        ("organization_id", auth["organization_id"].clone()),
        ("attempt_ordinal", dispatch["ordinal"].clone()),
        ("route_depth", json!(depth)),
        ("deployment_id", json!(deployment_id)),
        ("provider", deployment["provider"].clone()),
        ("exact_model_id", stage["exact_model_id"].clone()),
        ("pool_id", stage["pool_id"].clone()),
        ("catalog_sha256", auth["catalog_sha256"].clone()),
        ("billing_source", deployment["billing_source"].clone()),
        (
            "pricing_source",
            deployment["gateway"]["pricing_source"].clone(),
        ),
        ("pricing_effective_at", json!(effective_at)),
        (
            "preferred_deployment_id",
            dispatch["preferred_deployment_id"].clone(),
        ),
        ("unit_prices", prices["units"].clone()),
        ("state", json!("dispatched")),
        ("started_at", json!(now)),
        ("budget_period_start", json!(period)),
        ("budget_reserved_nano_usd", json!(maximum)),
        ("service_tier_admission", dispatch["service_tier"].clone()),
        (
            "native_accounting_terms",
            json!({"prices": prices, "preferred_prices": preferred.map(|d| &d["gateway"]["prices"]), "dispatch": dispatch}),
        ),
    ];
    for field in ["route_reason", "fallback_reason", "dispatch_reason"] {
        fields.push((field, dispatch[field].clone()));
    }
    rates(&mut fields, prices)?;
    for (column, (_, field)) in PREFERRED.into_iter().zip(RATES) {
        fields.push((
            column,
            json!(preferred
                .map(|d| valuation::rate(&d["gateway"]["prices"], field))
                .transpose()?
                .flatten()),
        ));
    }
    let existing = db::query(
        db,
        "SELECT * FROM gateway_attempts WHERE request_id=? AND attempt_ordinal=?",
        params![text(auth, "request_id")?, dispatch["ordinal"].as_i64()],
    )?;
    if let Some(existing) = existing.first() {
        for name in [
            "organization_id",
            "route_depth",
            "deployment_id",
            "provider",
            "exact_model_id",
            "pool_id",
            "catalog_sha256",
            "billing_source",
            "pricing_source",
            "pricing_effective_at",
        ] {
            if fields
                .iter()
                .find(|(key, _)| *key == name)
                .map(|(_, value)| value)
                != Some(&existing[name])
            {
                return Err(db::invariant(
                    "attempt reservation replay differs from durable accounting input",
                ));
            }
        }
        if fields
            .iter()
            .find(|(key, _)| *key == "native_accounting_terms")
            .map(|(_, value)| value)
            != Some(&decode(existing, "native_accounting_terms")?)
        {
            return Err(db::invariant(
                "attempt reservation replay differs from durable accounting input",
            ));
        }
        return Ok(json!({"attempt_id": existing["attempt_id"], "funded": false}));
    }
    if !row["terminal_state"].is_null() {
        return Err(db::invariant("Request is already terminal."));
    }
    db::insert(db, "gateway_attempts", &fields)?;
    let reservation = budgets::Reservation {
        organization: text(auth, "organization_id")?,
        identity: text(auth, "identity_id")?,
        alias: text(&row, "alias_id")?,
        root_pool: text(snapshot, "pool_id")?,
        pool: text(stage, "pool_id")?,
        deployment: deployment_id,
        attempt: &id,
        period,
        maximum,
    };
    if let Some(refused) = budgets::reserve(db, &reservation)? {
        db.execute(
            "DELETE FROM gateway_attempts WHERE attempt_id=?",
            params![id],
        )
        .map_err(db::error)?;
        return Ok(json!({"refused": refused}));
    }
    Ok(json!({"attempt_id": id, "funded": false}))
}

pub(super) fn decode(row: &Value, column: &str) -> Result<Value> {
    match row[column].as_str() {
        Some(text) => crate::lossless_json::from_str(text).map_err(|_| PublicError::internal()),
        None => Ok(Value::Null),
    }
}

/// Rows created before native ownership still carry their immutable price columns.
fn persisted_terms(row: &Value) -> Result<Value> {
    let stored = decode(row, "native_accounting_terms")?;
    if !stored.is_null() {
        return Ok(stored);
    }
    let mut prices = json!({"units": decode(row, "unit_prices")?});
    let mut preferred = json!({});
    for ((column, field), preferred_column) in RATES.into_iter().zip(PREFERRED) {
        prices[format!("{field}_nano_usd_per_million_tokens")] = row[column].clone();
        preferred[format!("{field}_nano_usd_per_million_tokens")] = row[preferred_column].clone();
    }
    if !row["long_context_threshold_tokens"].is_null() {
        let mut long = json!({"input_threshold_tokens":row["long_context_threshold_tokens"]});
        for (column, (_, field)) in LONG.into_iter().zip(RATES) {
            long[format!("{field}_nano_usd_per_million_tokens")] = row[column].clone();
        }
        prices["long_context"] = long;
    }
    Ok(
        json!({"prices":prices,"preferred_prices":if row["preferred_deployment_id"].is_null(){Value::Null}else{preferred}}),
    )
}

fn tier(row: &Value, terminal: &mut Value) -> Result<()> {
    let admission = decode(row, "service_tier_admission")?;
    let receipt = &terminal["service_tier"];
    if admission.is_null() {
        if !receipt.is_null() {
            return Err(db::invariant(
                "service-tier receipt differs from the admitted schedules",
            ));
        }
        return Ok(());
    }
    if receipt.is_null() {
        terminal["service_tier"] = json!({"requested": admission["requested"], "served": null, "prices": null, "resolution": "missing"});
        return Ok(());
    }
    let receipt = &terminal["service_tier"];
    let expected = if receipt["resolution"] == "confirmed" {
        if receipt["served"] == "default" {
            &admission["standard_prices"]
        } else if receipt["served"] == admission["requested"] {
            &admission["requested_prices"]
        } else {
            return Err(db::invariant(
                "service-tier receipt differs from the admitted schedules",
            ));
        }
    } else {
        if !matches!(
            receipt["resolution"].as_str(),
            Some("missing" | "unknown" | "conflicting")
        ) || !receipt["served"].is_null()
        {
            return Err(db::invariant(
                "service-tier receipt differs from the admitted schedules",
            ));
        }
        &Value::Null
    };
    if receipt["requested"] != admission["requested"] || receipt["prices"] != *expected {
        return Err(db::invariant(
            "service-tier receipt differs from the admitted schedules",
        ));
    }
    Ok(())
}

struct Quote {
    row: Value,
    terms: Value,
    terminal: Value,
    supplied_tier: bool,
    cost: Option<i64>,
}

fn quote(db: &Connection, auth: &Value, id: &str, mut terminal: Value) -> Result<Quote> {
    let row = query(db, "SELECT a.*, r.api_surface FROM gateway_attempts a JOIN gateway_requests r ON r.request_id=a.request_id WHERE a.attempt_id=? AND a.request_id=? AND a.organization_id=?",
        params![id, text(auth, "request_id")?, text(auth, "organization_id")?])?.pop().ok_or_else(PublicError::internal)?;
    let terms = persisted_terms(&row)?;
    let replay = row["state"] != "dispatched";
    let supplied_tier = !terminal["service_tier"].is_null();
    let prior_tier = decode(&row, "service_tier_settlement")?;
    if replay && prior_tier["resolution"] == "confirmed" {
        let weaker_recovery = terminal["usage"].is_null()
            && !terminal["failure"].is_null()
            && (row["accounting_recovery"] == 1 || row["state"] == "unknown_after_crash")
            && terminal["service_tier"]["resolution"] != "confirmed";
        if supplied_tier && terminal["service_tier"] != prior_tier && !weaker_recovery {
            return Err(db::invariant(
                "service-tier receipt differs from the settled tier",
            ));
        }
        terminal["service_tier"] = prior_tier;
    }
    tier(&row, &mut terminal)?;
    let cost = valuation::terminal_cost(&terms["prices"], &terminal)?;
    Ok(Quote {
        row,
        terms,
        terminal,
        supplied_tier,
        cost,
    })
}

/// A pre-delivery quote validates persisted terms without taking a SQLite write lock.
pub(super) fn price(db: &Connection, auth: &Value, id: &str, terminal: Value) -> Result<Value> {
    quote(db, auth, id, terminal)?;
    Ok(json!({}))
}

pub(super) fn settle(
    db: &Connection,
    auth: &Value,
    id: &str,
    terminal: Value,
    now: &str,
) -> Result<Value> {
    let Quote {
        row,
        terms,
        terminal,
        supplied_tier,
        cost,
    } = quote(db, auth, id, terminal)?;
    let replay = row["state"] != "dispatched";
    let usage = &terminal["usage"];
    let source = if usage.is_null() {
        "unknown"
    } else if terminal["usage_estimated"] == true {
        "estimated"
    } else {
        "observed"
    };
    let waived = terminal["input_guardrail_denied"] == true
        || terminal["accounting_release_reservation"] == true;
    let charge = if waived {
        Some(0)
    } else if terminal["service_tier"].is_object() {
        cost
    } else if row["api_surface"] == "decisions" && cost.is_none() {
        (terminal["state"] == "failed"
            && terminal["decision_provider_rejected"] == true
            && usage.is_null())
        .then_some(0)
    } else {
        cost.or(row["budget_reserved_nano_usd"].as_i64())
    };
    let mut charge = charge;
    if replay {
        if !reconciliation::upgrade(db, &row, id, &terminal, supplied_tier, cost)? {
            return Ok(json!({}));
        }
        let prior_charge = row["budget_settled_nano_usd"].as_i64();
        let provisional_cost = row["estimated_cost_nano_usd"]
            .as_i64()
            .or(row["budget_reserved_nano_usd"].as_i64());
        if prior_charge.is_some()
            && (prior_charge == Some(0)
                || row["api_surface"] == "decisions"
                || prior_charge != provisional_cost)
        {
            charge = prior_charge;
        }
        budgets::replace(db, id, prior_charge, charge)?;
    } else {
        if terminal["accounting_recovery"] == true
            && (!usage.is_null() || terminal["failure"].is_null())
        {
            return Err(db::invariant(
                "accounting recovery requires an unmetered failure",
            ));
        }
        budgets::settle(db, id, charge)?;
    }
    let counterfactual = if terms["preferred_prices"].is_null() || !usage["billed_units"].is_null()
    {
        None
    } else {
        valuation::usage_cost(&terms["preferred_prices"], usage)?
    };
    let mut fields = vec![
        ("usage_source", json!(source)),
        ("estimated_cost_nano_usd", json!(cost)),
        ("budget_settled_nano_usd", json!(charge)),
        ("counterfactual_cost_nano_usd", json!(counterfactual)),
        ("service_tier_settlement", terminal["service_tier"].clone()),
        ("native_accounting_terminal", terminal.clone()),
    ];
    for field in TOKENS {
        fields.push((field, usage[field].clone()));
    }
    let units = &usage["billed_units"];
    fields.extend([
        ("billed_unit_kind", units["kind"].clone()),
        ("billed_unit_variant", units["variant"].clone()),
        ("billed_quantity_milli", units["quantity_milli"].clone()),
        (
            "billed_unit_rate",
            json!(valuation::unit_rate(&terms["prices"], units)?),
        ),
    ]);
    if !replay {
        fields.extend([
            ("state", terminal["state"].clone()),
            ("terminal_at", json!(now)),
            ("first_token_at", terminal["first_token_at"].clone()),
            (
                "failure_class",
                terminal["failure"]["failure_class"].clone(),
            ),
            (
                "failure_message",
                terminal["failure"]["provider_detail"].clone(),
            ),
            (
                "accounting_recovery",
                json!(terminal["accounting_recovery"] == true),
            ),
        ]);
        for field in [
            "retry_after_seconds",
            "ratelimit_limit_requests",
            "ratelimit_remaining_requests",
            "ratelimit_limit_tokens",
            "ratelimit_remaining_tokens",
            "upstream_provider",
        ] {
            fields.push((field, terminal["observations"][field].clone()));
        }
    }
    if terminal["service_tier"]["prices"].is_object() {
        rates(&mut fields, &terminal["service_tier"]["prices"])?;
    }
    db::update(db, "gateway_attempts", "attempt_id", id, &fields)?;
    if terminal["finalize"] == true {
        db.execute("UPDATE gateway_requests SET terminal_state=?, terminal_at=? WHERE request_id=? AND terminal_state IS NULL",
            params![text(&terminal, "state")?, now, text(auth, "request_id")?]).map_err(db::error)?;
    }
    Ok(json!({}))
}
