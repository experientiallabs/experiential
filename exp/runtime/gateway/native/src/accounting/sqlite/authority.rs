//! Revalidate local alias authority and exact snapshot generations before each write.

use std::collections::BTreeSet;
use std::path::{Path, PathBuf};
use std::time::Instant;

use rusqlite::{params, Connection};
use serde_json::Value;

use super::db::{query, Result};
use super::snapshot::Snapshot;
use crate::errors::PublicError;

pub(super) fn text<'a>(value: &'a Value, field: &str) -> Result<&'a str> {
    value[field]
        .as_str()
        .filter(|s| !s.is_empty())
        .ok_or_else(PublicError::internal)
}

fn rows(db: &Connection, authorization: &Value) -> Result<Vec<Value>> {
    let rows = query(
        db,
        "SELECT r.organization_id, r.alias_id, r.revision_id, r.catalog_sha256,
        r.snapshot_ref, a.active_revision_id, active.catalog_sha256 AS active_catalog_sha256,
        active.snapshot_ref AS active_snapshot_ref FROM alias_revisions r JOIN gateway_aliases a
        ON a.organization_id=r.organization_id AND a.alias_id=r.alias_id
        LEFT JOIN alias_revisions active ON active.organization_id=a.organization_id
        AND active.revision_id=a.active_revision_id WHERE r.organization_id=? AND r.revision_id=?",
        params![
            text(authorization, "organization_id")?,
            text(authorization, "alias_revision_id")?
        ],
    )?;
    if rows.is_empty() {
        return Err(PublicError::internal());
    }
    Ok(rows)
}

pub(super) struct Proof {
    rows: Vec<Value>,
    files: Vec<Snapshot>,
    deadline: Instant,
    #[cfg(feature = "test-support")]
    fixture: bool,
}

impl Proof {
    pub(super) fn prepare(
        db: &Connection,
        root: &Path,
        authorization: &Value,
        maximum: u64,
        deadline: Instant,
    ) -> Result<Self> {
        if !authorization["model_chain_authority"].is_null() || Instant::now() >= deadline {
            return Err(PublicError::internal());
        }
        let rows = rows(db, authorization)?;
        let references: BTreeSet<_> = rows
            .iter()
            .flat_map(|row| {
                [
                    row["snapshot_ref"].as_str(),
                    row["active_snapshot_ref"].as_str(),
                ]
            })
            .flatten()
            .collect();
        let mut files = Vec::new();
        for reference in references {
            let path = PathBuf::from(reference);
            for (source, field) in [
                (path.clone(), "model_chains"),
                (path.with_extension("models.json"), "gateway_model_chains"),
            ] {
                let mut snapshot = Snapshot::open(root, &source)?;
                snapshot.classify(maximum, field)?;
                files.push(snapshot);
            }
        }
        if Instant::now() >= deadline {
            return Err(PublicError::internal());
        }
        Ok(Self {
            rows,
            files,
            deadline,
            #[cfg(feature = "test-support")]
            fixture: false,
        })
    }

    pub(super) fn validate(&self, db: &Connection, authorization: &Value) -> Result<()> {
        #[cfg(feature = "test-support")]
        if self.fixture {
            if db.is_autocommit() || Instant::now() >= self.deadline {
                return Err(PublicError::internal());
            }
            return super::test_authority::validate(db, authorization);
        }
        if db.is_autocommit()
            || Instant::now() >= self.deadline
            || self.rows != rows(db, authorization)?
        {
            return Err(PublicError::internal());
        }
        for file in &self.files {
            file.validate()?;
        }
        Ok(())
    }

    #[cfg(feature = "test-support")]
    pub(super) fn fixture(
        db: &Connection,
        authorization: &Value,
        deadline: Instant,
    ) -> Result<Self> {
        super::test_authority::validate(db, authorization)?;
        Ok(Self {
            rows: Vec::new(),
            files: Vec::new(),
            deadline,
            fixture: true,
        })
    }
}
