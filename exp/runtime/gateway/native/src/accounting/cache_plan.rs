//! Admission-frozen Google cache bodies and their verified resource namespace.

use serde::Deserialize;
use serde_json::{json, Value};

use super::cache_policy::{self as policy, Project, Result};
use crate::errors::PublicError;

#[derive(Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct Plan {
    pub create_url: String,
    pub model: String,
    pub resource_prefix: String,
    pub prefix_sha256: String,
    pub conservative_input_bound: u64,
    #[serde(deserialize_with = "crate::lossless_json::deserialize")]
    pub create_payload: Value,
    #[serde(deserialize_with = "crate::lossless_json::deserialize")]
    pub generation_payload: Value,
}

impl Plan {
    pub fn bind(&self, project: Option<&Project>) -> Result<Option<Self>> {
        policy::digest(&self.prefix_sha256)?;
        if self.resource_prefix == "cachedContents/" {
            policy::require(project.is_none())?;
            return Ok(Some(self.clone()));
        }
        let url = reqwest::Url::parse(&self.create_url).map_err(|_| PublicError::internal())?;
        let parts: Vec<_> = url.path().split('/').collect();
        policy::require(
            parts.len() == 7
                && parts[0].is_empty()
                && parts[1] == "v1"
                && parts[2] == "projects"
                && parts[4] == "locations"
                && parts[6] == "cachedContents",
        )?;
        let endpoint = parts[3];
        let numeric = |s: &str| {
            !s.is_empty()
                && s.len() <= 20
                && !s.starts_with('0')
                && s.bytes().all(|b| b.is_ascii_digit())
        };
        let number = if let Some(project) = project {
            let id = &project.endpoint_project;
            policy::require(
                !id.is_empty()
                    && id.len() <= 256
                    && id.as_bytes()[0].is_ascii_alphanumeric()
                    && id
                        .bytes()
                        .all(|b| b.is_ascii_alphanumeric() || b == b'_' || b == b'-')
                    && numeric(&project.project_number)
                    && id == endpoint
                    && (!id.bytes().all(|b| b.is_ascii_digit()) || id == &project.project_number),
            )?;
            project.project_number.as_str()
        } else if numeric(endpoint) {
            endpoint
        } else {
            return Ok(None);
        };
        let prefix = format!("projects/{number}/locations/{}/cachedContents/", parts[5]);
        policy::resource(&prefix, &format!("{prefix}a"))?;
        let mut bound = self.clone();
        if prefix != self.resource_prefix {
            bound.prefix_sha256 = policy::hash(
                &json!({"plan": self.prefix_sha256, "resource_prefix": prefix}),
                false,
            )?;
            bound.resource_prefix = prefix;
        }
        Ok(Some(bound))
    }

    pub fn account(&self, connection: &str, credential_scope: &str) -> Result<String> {
        let mut url = reqwest::Url::parse(&self.create_url).map_err(|_| PublicError::internal())?;
        url.set_query(None);
        url.set_fragment(None);
        policy::hash(
            &json!({"endpoint": url.as_str(), "model": self.model,
            "connection": connection, "credential_scope": credential_scope}),
            false,
        )
    }

    pub fn apply(&self, name: &str) -> Result<Value> {
        policy::resource(&self.resource_prefix, name)?;
        let id = name
            .strip_prefix(&self.resource_prefix)
            .ok_or_else(PublicError::internal)?;
        policy::require(id.as_bytes()[0].is_ascii_alphanumeric())?;
        let mut payload = self.generation_payload.clone();
        payload
            .as_object_mut()
            .ok_or_else(PublicError::internal)?
            .insert("cachedContent".into(), json!(name));
        Ok(payload)
    }
}

#[derive(Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct Binding {
    pub deployment_id: String,
    pub connection: String,
    pub plan: Plan,
    pub automatic_plans: Vec<Plan>,
}

#[derive(Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct Configuration {
    pub bindings: Vec<Option<Binding>>,
    pub automatic: bool,
}
