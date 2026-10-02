//! Local SQLite destination. Collection and delivery remain in the shared collector.
use serde::Deserialize;
use serde_json::json;
use sha2::{Digest, Sha256};

use super::delivery::Sink;
use super::record::Record;
use super::{local_store, projection::CapturedResponse};

#[derive(Debug, Clone, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub(crate) struct Scope {
    pub user_id: String,
    pub application_id: String,
}

#[derive(Debug, Clone, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub(crate) struct Policy {
    pub scope: Scope,
    pub enabled: bool,
    pub maximum_experiences: usize,
    pub maximum_storage_bytes: usize,
    pub maximum_experience_bytes: usize,
    pub retention_seconds: u64,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct Binding {
    pub alias: String,
    pub policy: Policy,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct CaptureConfiguration {
    pub database_path: String,
    pub bindings: Vec<Binding>,
    pub queue_capacity: usize,
}

pub(crate) struct SqliteSink {
    connection: rusqlite::Connection,
    policies: Vec<Policy>,
    maintenance_failed: u64,
}

impl SqliteSink {
    pub(crate) fn open(config: CaptureConfiguration) -> Result<Option<Self>, String> {
        local_store::validate(&config)?;
        let policies: Vec<_> = config
            .bindings
            .into_iter()
            .filter(|binding| binding.policy.enabled)
            .map(|binding| binding.policy)
            .collect();
        if policies.is_empty() {
            return Ok(None);
        }
        let connection = local_store::open_database(std::path::Path::new(&config.database_path))?;
        let mut sink = Self {
            connection,
            policies,
            maintenance_failed: 0,
        };
        sink.maintain()
            .map_err(|_| "cannot prune local capture database".to_owned())?;
        Ok(Some(sink))
    }
}

impl Sink for SqliteSink {
    type Prepared = Option<local_store::Pending>;

    fn preparation_bytes(maximum_record_bytes: usize) -> usize {
        // Encoded payload plus its response-id copy, two bounded scope strings,
        // the fixed-size experience id, and the owning struct. SQLite's own
        // transaction workspace is separate from the delivery queue budget.
        2 * maximum_record_bytes + 1024 + 256 + std::mem::size_of::<local_store::Pending>()
    }

    fn prepare(&self, record: &Record, maximum_bytes: usize) -> Result<Self::Prepared, ()> {
        let scope = &record.request.scope;
        let policy = self
            .policies
            .iter()
            .find(|policy| {
                policy.scope.user_id == scope.identity_id
                    && policy.scope.application_id == scope.application_id
            })
            .ok_or(())?;
        let Some(response) = CapturedResponse::completed_record(record) else {
            return Ok(None);
        };
        let response_id = response
            .get("id")
            .and_then(serde_json::Value::as_str)
            .ok_or(())?;
        let protocol = serde_json::to_value(record.request.protocol).map_err(|_| ())?;
        let experience_id = format!(
            "experience-{:x}",
            Sha256::digest(
                serde_json::to_vec(&json!([
                    scope.identity_id,
                    scope.application_id,
                    record.request.request_id,
                    protocol
                ]))
                .map_err(|_| ())?
            )
        );
        let payload = super::local_payload::encode(
            record,
            &response,
            &experience_id,
            maximum_bytes.min(policy.maximum_experience_bytes),
        )
        .ok_or(())?;
        Ok(Some(local_store::Pending {
            policy: policy.clone(),
            payload,
            experience_id,
            response_id: response_id.to_owned(),
            captured_at: record.captured_at as u64,
        }))
    }

    fn write(&mut self, prepared: &Self::Prepared) -> Result<(), ()> {
        let Some(prepared) = prepared else {
            return Ok(());
        };
        let persisted = local_store::persist(&mut self.connection, prepared).map_err(|_| ())?;
        self.maintenance_failed += u64::from(persisted.maintenance_failed);
        Ok(())
    }

    fn take_maintenance_failures(&mut self) -> u64 {
        std::mem::take(&mut self.maintenance_failed)
    }

    fn maintain(&mut self) -> Result<(), ()> {
        for policy in &self.policies {
            local_store::prune(&self.connection, policy, local_store::now()).map_err(|_| ())?;
        }
        Ok(())
    }
}

#[cfg(test)]
#[path = "local_test.rs"]
mod tests;
