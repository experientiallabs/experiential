//! The initial admission view reads the same registries mutated by native execution.

use serde::de::DeserializeOwned;
use serde_json::{json, Value};

use crate::errors::{Failure, PublicError};

use super::health::{Claim, HealthKey, LaneKey};
use super::loads::Reservation;
use super::shared::Shared;

fn field<T: DeserializeOwned + 'static>(data: &Value, name: &str) -> Result<T, PublicError> {
    crate::lossless_json::from_value(data[name].clone()).map_err(|_| PublicError::internal())
}

pub(super) fn call(
    shared: &Shared,
    operation: &str,
    argument: &str,
) -> Result<String, PublicError> {
    let data: Value =
        crate::lossless_json::from_str(argument).map_err(|_| PublicError::internal())?;
    let mut p = shared
        .policies
        .lock()
        .map_err(|_| PublicError::internal())?;
    let now = shared.now();
    let result = match operation {
        "health_suppressed" => json!(p.health.suppressed(&field::<HealthKey>(&data, "key")?, now)),
        "health_ejected" => json!(p.health.ejected(&field::<HealthKey>(&data, "key")?, now)),
        "health_order" => {
            let keys: Vec<HealthKey> = field(&data, "keys")?;
            let depths: Vec<usize> = field(&data, "depths")?;
            if depths.iter().any(|d| *d >= keys.len()) {
                return Err(PublicError::internal());
            }
            json!(p.health.order(&keys, &depths, now))
        }
        "health_claim" => {
            let mode = match data["mode"].as_str() {
                Some("normal") => Claim::Normal,
                Some("last_resort") => Claim::LastResort,
                Some("forced") => Claim::Forced,
                Some("throttle_redial") => Claim::ThrottleRedial,
                _ => return Err(PublicError::internal()),
            };
            json!(p
                .health
                .claim(&field::<HealthKey>(&data, "key")?, mode, now))
        }
        "health_release" => {
            p.health.release(&field::<HealthKey>(&data, "key")?);
            Value::Null
        }
        "health_failed" => {
            p.health.failed(
                &field::<HealthKey>(&data, "key")?,
                &field::<Failure>(&data, "failure")?,
                now,
            );
            Value::Null
        }
        "health_succeeded" => {
            p.health.succeeded(&field::<HealthKey>(&data, "key")?);
            Value::Null
        }
        "health_exhausted" => {
            p.health.exhausted(
                &field::<HealthKey>(&data, "key")?,
                field(&data, "reset_seconds")?,
                now,
            );
            Value::Null
        }
        "health_throttled_remaining" => json!(p
            .health
            .throttled_remaining(&field::<Vec<HealthKey>>(&data, "keys")?, now)),
        "health_circuit_failure" => json!(p
            .health
            .circuit_failure(&field::<Vec<HealthKey>>(&data, "keys")?, now)),
        "loads_cached_fraction" => json!(p.loads.cached_fraction(
            &field::<LaneKey>(&data, "key")?,
            &field::<String>(&data, "organization_id")?,
            now
        )),
        "loads_inflight" => json!(p.loads.inflight(
            &field::<LaneKey>(&data, "key")?,
            data["organization_id"].as_str()
        )),
        "loads_fast_inflight" => json!(p.loads.fast_inflight(&field::<LaneKey>(&data, "key")?)),
        "loads_failover_inflight" => {
            json!(p.loads.failover_inflight(&field::<LaneKey>(&data, "key")?))
        }
        "loads_learned_ceilings" => json!(p.loads.learned_ceilings()),
        "loads_default_bound" => json!(p.loads.default_bound),
        "loads_can_admit" => {
            let request: Reservation = field(&data, "reservation")?;
            if !request.valid() {
                return Err(PublicError::internal());
            }
            json!(p
                .loads
                .can_admit(&field::<LaneKey>(&data, "key")?, &request, now))
        }
        "loads_reserve" => {
            let request: Reservation = field(&data, "reservation")?;
            if !request.valid() {
                return Err(PublicError::internal());
            }
            match p
                .loads
                .reserve(&field::<LaneKey>(&data, "key")?, &request, now)
            {
                Ok(ticket) => json!({"ticket": ticket}),
                Err(shed) => json!({"shed": shed}),
            }
        }
        "loads_bind" => {
            p.loads.bind(
                &field::<String>(&data, "ticket")?,
                &field::<String>(&data, "attempt_id")?,
            );
            Value::Null
        }
        "loads_release_ticket" => {
            p.loads.release(&field::<String>(&data, "ticket")?);
            Value::Null
        }
        "loads_retract" => {
            p.loads.retract(&field::<String>(&data, "ticket")?);
            Value::Null
        }
        "loads_restamp" => {
            p.loads.restamp(&field::<String>(&data, "ticket")?, now);
            Value::Null
        }
        "loads_release_attempt" => {
            p.loads
                .release_attempt(&field::<String>(&data, "attempt_id")?);
            Value::Null
        }
        "loads_record_throttle" => {
            p.loads
                .record_throttle(&field::<LaneKey>(&data, "key")?, now);
            Value::Null
        }
        "loads_record_settle" => {
            p.loads.record_settle(
                &field::<LaneKey>(&data, "key")?,
                &field::<String>(&data, "organization_id")?,
                field(&data, "cached_tokens")?,
                field(&data, "input_tokens")?,
                now,
            );
            Value::Null
        }
        "sticky_bound" => json!(p
            .sticky
            .bound(field::<String>(&data, "fingerprint")?.as_bytes(), now)),
        "sticky_bind" => {
            let ttl: f64 = field(&data, "ttl_seconds")?;
            if !ttl.is_finite() || ttl < 0.0 {
                return Err(PublicError::internal());
            }
            p.sticky.bind(
                field::<String>(&data, "fingerprint")?.as_bytes(),
                &field::<String>(&data, "deployment_id")?,
                ttl,
                now,
            );
            Value::Null
        }
        "sticky_clear" => {
            p.sticky
                .clear(field::<String>(&data, "fingerprint")?.as_bytes());
            Value::Null
        }
        "sticky_len" => json!(p.sticky.len()),
        _ => return Err(PublicError::internal()),
    };
    Ok(result.to_string())
}
