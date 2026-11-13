//! Frozen recovery scope and once-per-attempt effects, without interpreter calls.

use serde::Deserialize;
use serde_json::Value;
use std::time::{SystemTime, UNIX_EPOCH};

use super::{
    health::{self, Cause},
    session_recovery::{Key, Scope, Success},
    shared::Shared,
};
use crate::errors::{Failure, PublicError};

#[derive(Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct Lane {
    pub scope: Scope,
    #[serde(default, deserialize_with = "crate::lossless_json::deserialize")]
    pub retention_seconds: Option<f64>,
    #[serde(default, deserialize_with = "crate::lossless_json::deserialize")]
    pub sticky_seconds: Option<f64>,
}
#[derive(Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct Context {
    pub key: Key,
    pub lanes: Vec<Option<Lane>>,
}
pub(super) fn now() -> f64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs_f64()
}
impl Context {
    pub fn validate(&self, organization: &str, count: usize) -> Result<(), PublicError> {
        if self.key[0] != organization
            || self.key.iter().any(|part| part.len() > 1024)
            || self.lanes.len() != count
            || self.lanes.iter().flatten().any(|l| {
                l.scope.organization_id != organization
                    || [
                        &l.scope.operational.provider,
                        &l.scope.operational.exact_model_id,
                        &l.scope.operational.endpoint_scope,
                        &l.scope.credential_scope,
                    ]
                    .iter()
                    .any(|part| part.len() > 4096)
                    || l.scope
                        .operational
                        .region_scope
                        .as_ref()
                        .is_some_and(|part| part.len() > 1024)
                    || l.retention_seconds.is_some_and(|v| !v.is_finite())
                    || l.sticky_seconds.is_some_and(|v| !v.is_finite())
            })
        {
            return Err(PublicError::internal());
        }
        Ok(())
    }
    #[allow(clippy::too_many_arguments)]
    pub fn departed(
        &self,
        key: &Key,
        shared: &Shared,
        depth: usize,
        deployment: &str,
        cause: &str,
        retry: f64,
        observed: f64,
    ) {
        let Some(lane) = self.lanes.get(depth).and_then(Option::as_ref) else {
            return;
        };
        if lane.scope.operational.region_scope.is_none()
            || lane.scope.operational.endpoint_scope.is_empty()
            || lane.scope.credential_scope.is_empty()
        {
            return;
        }
        if let Ok(mut registry) = shared.recovery.lock() {
            registry.depart(key, deployment, lane.scope.clone(), cause, retry, observed);
        }
    }
    #[allow(clippy::too_many_arguments)]
    pub fn settled(
        &self,
        key: &Key,
        shared: &Shared,
        depth: usize,
        deployment: &str,
        terminal: &Value,
        failure: Option<&Failure>,
        observed: f64,
    ) {
        let Some(lane) = self.lanes.get(depth).and_then(Option::as_ref) else {
            return;
        };
        if lane.scope.operational.region_scope.is_none()
            || lane.scope.operational.endpoint_scope.is_empty()
            || lane.scope.credential_scope.is_empty()
        {
            return;
        }
        if let Some(failure) = failure {
            let cause = match health::cause(failure.failure_class) {
                Some(Cause::Transport) => "transport",
                Some(Cause::Credential) => "credential",
                Some(Cause::Throttle) => "throttle",
                None => return,
            };
            self.departed(
                key,
                shared,
                depth,
                deployment,
                cause,
                failure.retry_after_seconds.unwrap_or(0) as f64,
                observed,
            );
        } else if terminal["usage"].is_object() {
            if let Ok(mut registry) = shared.recovery.lock() {
                registry.success(
                    key,
                    deployment,
                    lane.scope.clone(),
                    Success {
                        cached_tokens: terminal["usage"]["cached_input_tokens"]
                            .as_i64()
                            .unwrap_or(0),
                        cache_write_tokens: terminal["usage"]["cache_creation_input_tokens"]
                            .as_i64()
                            .unwrap_or(0),
                        retention_seconds: lane.retention_seconds,
                        sticky_seconds: lane.sticky_seconds,
                    },
                    observed,
                    now(),
                );
            }
        }
    }
}
