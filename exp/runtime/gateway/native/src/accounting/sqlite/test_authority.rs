//! Test-only durable chain authority. Excluded from every production wheel.
//!
//! Financial writes use the production SQLite owner. These tables model a host's
//! independently issued authority without a Python callback after admission.

use chrono::DateTime;
use rusqlite::{params, Connection};
use serde_json::Value;

use super::authority::text;
use super::db::{self, query, Result};

const GENERATION: &str = "17fca02b-7c90-4bca-89ca-2538fdd71c0a";

pub(super) fn selected(db: &Connection, auth: &Value) -> Result<bool> {
    if query(
        db,
        "SELECT 1 FROM sqlite_schema WHERE name='test_chain_floors'",
        [],
    )?
    .is_empty()
    {
        return Ok(false);
    }
    Ok(!query(
        db,
        "SELECT 1 FROM alias_revisions r JOIN test_chain_floors f ON
        f.organization_id=r.organization_id AND (f.alias_id=r.alias_id OR f.digest=r.catalog_sha256)
        WHERE r.organization_id=? AND r.revision_id=?",
        params![
            text(auth, "organization_id")?,
            text(auth, "alias_revision_id")?
        ],
    )?
    .is_empty())
}

fn future(value: &Value) -> bool {
    value
        .as_str()
        .and_then(|s| DateTime::parse_from_rfc3339(s).ok())
        .is_some_and(|date| date > super::now())
}

pub(super) fn validate(db: &Connection, auth: &Value) -> Result<()> {
    let org = text(auth, "organization_id")?;
    let revision = text(auth, "alias_revision_id")?;
    let identity = text(auth, "identity_id")?;
    let request = text(auth, "request_id")?;
    let accepted = query(
        db,
        "SELECT organization_id, identity_id, alias_revision_id FROM
        gateway_requests WHERE request_id=?",
        params![request],
    )?;
    if accepted.first().is_some_and(|row| {
        ["organization_id", "identity_id", "alias_revision_id"]
            .iter()
            .any(|field| row[field] != auth[field])
    }) {
        return Err(db::invariant("test host accepted authority differs"));
    }
    let grant = query(db, "SELECT k.expires_at FROM virtual_keys k JOIN identities i
        ON i.organization_id=k.organization_id AND i.identity_id=k.identity_id
        JOIN organizations o ON o.organization_id=i.organization_id
        JOIN identity_alias_grants g ON g.organization_id=i.organization_id AND g.identity_id=i.identity_id
        JOIN gateway_aliases a ON a.organization_id=g.organization_id AND a.alias_id=g.alias_id
        JOIN alias_revisions r ON r.organization_id=a.organization_id AND r.alias_id=a.alias_id
        WHERE k.organization_id=? AND k.identity_id=? AND k.key_id=? AND r.revision_id=?
        AND k.revoked_at IS NULL AND i.active=1 AND o.active=1 AND a.active=1",
        params![org, identity, text(auth,"virtual_key_id")?, revision])?;
    let grant = grant
        .first()
        .ok_or_else(|| db::invariant("test host live grant is invalid"))?;
    if !grant["expires_at"].is_null() && !future(&grant["expires_at"]) {
        return Err(db::invariant("test host key has expired"));
    }
    let floors = query(
        db,
        "SELECT f.alias_id,r.revision_id,r.catalog_sha256,f.epoch FROM
        test_chain_floors f JOIN alias_revisions r ON r.alias_id=f.alias_id
        AND r.organization_id=f.organization_id WHERE r.organization_id=? AND r.revision_id=?",
        params![org, revision],
    )?;
    let Some(floor) = floors.first() else {
        let plain = query(db, "SELECT r.pool_id,r.catalog_sha256,c.catalog FROM alias_revisions r
            JOIN test_chain_floors c ON c.organization_id=r.organization_id AND c.digest=r.catalog_sha256
            WHERE r.organization_id=? AND r.revision_id=?", params![org, revision])?;
        let plain = plain
            .first()
            .ok_or_else(|| db::invariant("test host plain alias is missing"))?;
        let catalog: Value = crate::lossless_json::from_str(text(plain, "catalog")?)
            .map_err(|_| db::invariant("test host catalog is invalid"))?;
        let pool = catalog["pools"]
            .as_array()
            .and_then(|pools| {
                pools
                    .iter()
                    .find(|pool| pool["pool_id"] == plain["pool_id"])
            })
            .ok_or_else(|| db::invariant("test host selected pool is missing"))?;
        if !auth["model_chain_authority"].is_null()
            || auth["catalog_sha256"] != plain["catalog_sha256"]
            || auth["target"]["pool_id"] != plain["pool_id"]
            || catalog["model_chains"].as_array().is_some_and(|chains| {
                chains
                    .iter()
                    .any(|chain| chain["model_id"] == pool["exact_model_id"])
            })
        {
            return Err(db::invariant("test host plain alias authority mismatch"));
        }
        return Ok(());
    };
    let receipt = &auth["model_chain_authority"];
    for field in [
        "organization_id",
        "identity_id",
        "virtual_key_id",
        "alias_revision_id",
        "catalog_sha256",
        "request_id",
    ] {
        if receipt[field] != auth[field] {
            return Err(db::invariant(
                "test host receipt is not bound to the request",
            ));
        }
    }
    if receipt["contract_version"] != 1
        || receipt["mode"] != "dispatch"
        || receipt["process_generation"] != GENERATION
        || !future(&receipt["expires_at"])
    {
        return Err(db::invariant("test host receipt is not live"));
    }
    let issued = query(
        db,
        "SELECT binding FROM test_chain_receipts WHERE receipt_id=?",
        params![text(receipt, "receipt_id")?],
    )?;
    let issued = issued
        .first()
        .ok_or_else(|| db::invariant("test host receipt was not issued"))?;
    let issued: Value = crate::lossless_json::from_str(text(issued, "binding")?)
        .map_err(|_| db::invariant("test host receipt is invalid"))?;
    if issued != *receipt {
        return Err(db::invariant("test host receipt was altered"));
    }
    for (left, right) in [
        ("stable_alias_id", "alias_id"),
        ("alias_revision_id", "revision_id"),
        ("catalog_sha256", "catalog_sha256"),
        ("feature_epoch", "epoch"),
    ] {
        if receipt[left] != floor[right] {
            return Err(db::invariant("test host receipt floor is stale"));
        }
    }
    Ok(())
}

pub(super) fn stage(snapshot: &Value, depth: usize) -> Result<&Value> {
    let stages = snapshot["model_stages"]
        .as_array()
        .ok_or_else(|| db::invariant("test host model stages are missing"))?;
    let mut offset = 0;
    for stage in stages {
        let count = stage["deployment_ids"]
            .as_array()
            .ok_or_else(|| db::invariant("test host stage deployments are missing"))?
            .len();
        if (offset..offset + count).contains(&depth) {
            return Ok(stage);
        }
        offset += count;
    }
    Err(db::invariant("test host route depth has no stage"))
}
