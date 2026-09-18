//! Local SQLite destination. Collection and delivery remain in the shared collector.
use serde::Deserialize;
use serde_json::json;
use sha2::{Digest, Sha256};

use super::delivery::Sink;
use super::record::Record;
use super::{local_store, projection};

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
        };
        sink.maintain()
            .map_err(|_| "cannot prune local capture database".to_owned())?;
        Ok(Some(sink))
    }
}

impl Sink for SqliteSink {
    fn write(&mut self, encoded: &str) -> Result<(), ()> {
        let record: Record = serde_json::from_str(encoded).map_err(|_| ())?;
        let scope = &record.request.scope;
        let policy = self
            .policies
            .iter()
            .find(|policy| {
                policy.scope.user_id == scope.identity_id
                    && policy.scope.application_id == scope.application_id
            })
            .ok_or(())?;
        let Some(response) = projection::completed_response(&record) else {
            return Ok(());
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
        let request = json!({
            "exp_context": record.request.context,
            "previous_response_id": record.request.context["request"]["previous_response_id"]
        });
        let experience = json!({
            "schema_version": 1, "experience_id": experience_id, "response_id": response_id,
            "episode_id": null, "parent_response_id": request["previous_response_id"],
            "scope": {"user_id": scope.identity_id, "application_id": scope.application_id},
            "protocol": protocol, "captured_at": record.captured_at, "request": request,
            "response": response,
            "provenance": {"source_kind": "traffic", "source_id": record.request.request_id,
                "model_id": record.request.model_id, "model_revision": null,
                "deployment_id": record.deployment_id, "policy_revision": null,
                "source_experience_ids": []}, "exact_tokens": null
        });
        let payload = serde_json::to_string(&experience).map_err(|_| ())?;
        if payload.len() > policy.maximum_experience_bytes {
            return Err(());
        }
        local_store::persist(
            &mut self.connection,
            local_store::Pending {
                policy: policy.clone(),
                payload,
                experience_id,
                response_id: response_id.to_owned(),
                captured_at: record.captured_at as u64,
            },
        )
        .map_err(|_| ())
    }

    fn maintain(&mut self) -> Result<(), ()> {
        for policy in &self.policies {
            local_store::prune(&self.connection, policy, local_store::now()).map_err(|_| ())?;
        }
        Ok(())
    }
}
