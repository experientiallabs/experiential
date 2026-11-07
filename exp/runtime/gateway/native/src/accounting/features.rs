//! Request-owned response features with no interpreter callbacks.

use serde::Deserialize;
use serde_json::Value;

use crate::errors::PublicError;

use super::{carriers, continuations, signing};

#[derive(Clone, Default, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub(super) struct Features {
    pub signing: Vec<Option<signing::Binding>>,
    pub reasoning: Vec<Option<carriers::Authority>>,
    pub continuation: Option<continuations::Context>,
    pub output_checks: Option<Vec<super::inspection::Check>>,
    pub search: Option<super::search_dispatch::Configuration>,
    pub cache: Option<super::cache_plan::Configuration>,
}

impl Features {
    pub fn validate(&self, lanes: usize) -> Result<(), PublicError> {
        if [self.signing.len(), self.reasoning.len()]
            .iter()
            .any(|len| *len != 0 && *len != lanes)
        {
            return Err(PublicError::internal());
        }
        if self.search.as_ref().is_some_and(|s| s.plans.len() != lanes) {
            return Err(PublicError::internal());
        }
        if self
            .cache
            .as_ref()
            .is_some_and(|c| c.bindings.len() != lanes)
        {
            return Err(PublicError::internal());
        }
        Ok(())
    }
    pub fn call(
        &self,
        method: &str,
        request: &str,
        depth: Option<usize>,
        input: &Value,
    ) -> Option<Result<Value, PublicError>> {
        match method {
            "sign_dispatch" => Some(
                depth
                    .and_then(|d| self.signing.get(d))
                    .and_then(Option::as_ref)
                    .ok_or_else(PublicError::internal)
                    .and_then(|binding| {
                        binding.headers(input, std::time::SystemTime::now().into())
                    }),
            ),
            "seal_reasoning_content" => Some(
                depth
                    .and_then(|d| self.reasoning.get(d))
                    .and_then(Option::as_ref)
                    .ok_or_else(PublicError::internal)
                    .and_then(|authority| authority.seal(request, depth.unwrap(), input)),
            ),
            _ => None,
        }
    }
}
