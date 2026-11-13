//! Frozen admission facts for the native dispatch walk.

use serde::Deserialize;
use serde_json::Value;

use crate::errors::PublicError;

use super::health::{HealthKey, LaneKey};
use super::prices::Ceiling;
use super::selection::{AttemptPolicy, Rules, Stage};

#[derive(Clone, Default, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub(super) struct DispatchPolicy {
    pub concurrency_bound: Option<u64>,
    pub requests_per_minute: Option<u64>,
    pub tokens_per_minute: Option<u64>,
    #[serde(default, deserialize_with = "crate::lossless_json::deserialize")]
    pub cache_priority_alpha: Option<f64>,
    #[serde(default, deserialize_with = "crate::lossless_json::deserialize")]
    pub fresh_session_spill_fraction: Option<f64>,
    pub refusing: bool,
    #[serde(default, deserialize_with = "crate::lossless_json::deserialize")]
    pub priority_overflow_paying: Option<f64>,
    #[serde(default, deserialize_with = "crate::lossless_json::deserialize")]
    pub priority_overflow_pro: Option<f64>,
    pub self_hosted: bool,
    pub priority_tier_attested: bool,
    #[serde(default, deserialize_with = "crate::lossless_json::deserialize")]
    pub fast_reserve_fraction: Option<f64>,
    #[serde(default, deserialize_with = "crate::lossless_json::deserialize")]
    pub sticky_spill_seconds: Option<f64>,
    #[serde(default, deserialize_with = "crate::lossless_json::deserialize")]
    pub fleet: Option<Value>,
}

#[derive(Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct Lane {
    pub key: HealthKey,
    pub pool_id: String,
    pub stage: Stage,
    pub policy: DispatchPolicy,
    #[serde(deserialize_with = "crate::lossless_json::deserialize")]
    pub prices: Value,
    pub ceiling: Ceiling,
    pub output_tokens: u64,
    pub customer_managed: bool,
    pub forwards_tier: bool,
    pub route_reason: Option<String>,
    pub keeps_pin: bool,
    pub cache_placed: bool,
    pub output_ceiling: Option<u64>,
}

impl Lane {
    pub fn load_key(&self) -> LaneKey {
        (self.key.1.clone(), self.key.2.clone())
    }
    pub fn priority_card(&self) -> bool {
        self.prices["priority"].is_object()
    }
    pub fn fast(&self, config: &Registration) -> bool {
        config.fast && self.policy.self_hosted && self.priority_card()
    }
    pub fn fast_reserve(&self) -> Option<u64> {
        let bound = self.policy.concurrency_bound?;
        let fraction = self.policy.fast_reserve_fraction?;
        if !self.policy.self_hosted || !self.priority_card() {
            return None;
        }
        let regular = (((1.0 - fraction) * bound as f64 + 0.5).floor() as u64).max(1);
        (regular < bound).then_some(bound - regular)
    }
}

#[derive(Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct Registration {
    #[serde(default)]
    pub recovery: Option<super::session_observations::Context>,
    #[serde(default)]
    pub features: super::features::Features,
    pub request_id: String,
    pub organization_id: String,
    #[serde(deserialize_with = "crate::lossless_json::deserialize")]
    pub remaining_seconds: f64,
    pub surface: String,
    pub lanes: Vec<Lane>,
    pub rules: Rules,
    pub attempt_policy: AttemptPolicy,
    pub refusal_failover: bool,
    pub selected_route: bool,
    pub priority: u8,
    pub weight: u64,
    pub fast: bool,
    pub class: Option<i64>,
    pub class_cap: Option<u64>,
    #[serde(default, deserialize_with = "crate::lossless_json::deserialize")]
    pub pace: Option<f64>,
    #[serde(default, deserialize_with = "crate::lossless_json::deserialize")]
    pub pace_saturation: Option<f64>,
    pub input_tokens: u64,
    pub counted_input_tokens: u64,
    pub output_ceiling: Option<u64>,
    pub service_tier: Option<String>,
    pub redial_budgets: Vec<u32>,
    pub fingerprint: Option<String>,
    pub bind_sticky: bool,
    pub sticky_preferred: bool,
    pub caching_off: bool,
    pub recovery_scoped: bool,
    pub verified_warm_deployment: Option<String>,
    #[serde(deserialize_with = "crate::lossless_json::deserialize")]
    pub verified_warm_seconds: f64,
    pub recovery_reason: Option<String>,
    pub fallback_reason: Option<String>,
    pub paid_prework: bool,
    /// Paid search completed before this owner was registered.
    #[serde(default)]
    pub web_search_requests: u64,
    /// Caller-facing capacity copy supplied by the admitting authority.
    pub capacity_message: String,
    /// Public spellings of the caller's routing controls that removed at least
    /// one rung; a throttled refusal names only these.
    pub caller_narrowing: Vec<String>,
}

impl Registration {
    pub fn validate(&self) -> Result<(), PublicError> {
        self.features.validate(self.lanes.len())?;
        if let Some(recovery) = &self.recovery {
            recovery.validate(&self.organization_id, self.lanes.len())?;
        }
        let len = self.lanes.len();
        if self.request_id.is_empty()
            || self.request_id.len() > 512
            || self.organization_id.is_empty()
            || self.organization_id.len() > 512
            || self.fingerprint.as_ref().is_some_and(|v| v.len() > 512)
            || !(1..=256).contains(&len)
            || self.rules.len() != len
            || self.redial_budgets.len() != len
            || !self.attempt_policy.valid()
            || self.priority > 2
            || self.weight == 0
            || !self.remaining_seconds.is_finite()
            || !(0.0..=86400.0).contains(&self.remaining_seconds)
            || !self.verified_warm_seconds.is_finite()
            || self.verified_warm_seconds < 0.0
            || self.class_cap == Some(0)
            || self.caller_narrowing.len() > 8
            || self
                .caller_narrowing
                .iter()
                .any(|control| control.is_empty() || control.len() > 128)
            || self.pace.is_some_and(|v| !v.is_finite() || v <= 0.0)
            || self
                .pace_saturation
                .is_some_and(|v| !v.is_finite() || v < 0.0)
            || self
                .lanes
                .windows(2)
                .any(|pair| pair[0].stage.lead > pair[1].stage.lead)
            || self.lanes.iter().enumerate().any(|(depth, lane)| {
                let p = &lane.policy;
                lane.stage.lead > depth
                    || self.lanes[lane.stage.lead].stage.lead != lane.stage.lead
                    || [&lane.key.0, &lane.key.1, &lane.key.2]
                        .iter()
                        .any(|key| key.len() > 4096)
                    || lane.pool_id.len() > 512
                    || lane.key.1.is_empty()
                    || lane.pool_id.is_empty()
                    || p.concurrency_bound == Some(0)
                    || [
                        p.cache_priority_alpha,
                        p.fresh_session_spill_fraction,
                        p.priority_overflow_paying,
                        p.priority_overflow_pro,
                        p.fast_reserve_fraction,
                        p.sticky_spill_seconds,
                        lane.stage.throttle_cache_threshold,
                    ]
                    .into_iter()
                    .flatten()
                    .any(|v| !v.is_finite() || v < 0.0)
                    || p.fast_reserve_fraction.is_some_and(|v| v >= 1.0)
            })
        {
            return Err(PublicError::internal());
        }
        for lane in &self.lanes {
            lane.ceiling.validate()?;
        }
        Ok(())
    }
}
