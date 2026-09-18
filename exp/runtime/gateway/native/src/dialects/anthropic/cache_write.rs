//! Preserve an observed cache-write TTL breakdown without inferring it from a request.

use serde_json::{Map, Value};

use crate::dialects::malformed;
use crate::errors::Failure;
use crate::events::{bounded_ledger_sum, count_if_present};

/// Return the one-hour subset only when both TTL counters cover the reported total.
/// Missing or partial breakdowns remain unknown; malformed or contradictory
/// evidence fails the provider stream rather than silently inventing a price.
pub(super) fn hour_subset(usage: &Map<String, Value>, total: u64) -> Result<Option<u64>, Failure> {
    let details = match usage.get("cache_creation") {
        None | Some(Value::Null) => return Ok(None),
        Some(value) => value
            .as_object()
            .ok_or_else(|| malformed("Anthropic cache_creation must be an object"))?,
    };
    let five = count_if_present(
        details,
        "ephemeral_5m_input_tokens",
        "Anthropic cache_creation",
    )
    .map_err(|message| malformed(&message))?;
    let hour = count_if_present(
        details,
        "ephemeral_1h_input_tokens",
        "Anthropic cache_creation",
    )
    .map_err(|message| malformed(&message))?;
    if five.is_some_and(|count| count > total) || hour.is_some_and(|count| count > total) {
        return Err(malformed(
            "Anthropic cache_creation TTL count exceeds total",
        ));
    }
    match (five, hour) {
        (Some(five), Some(hour)) => {
            let sum = bounded_ledger_sum(&[five, hour], "Anthropic cache_creation")
                .map_err(|message| malformed(&message))?;
            if sum != total {
                return Err(malformed(
                    "Anthropic cache_creation TTL counts differ from total",
                ));
            }
            Ok(Some(hour))
        }
        _ => Ok(None),
    }
}

#[cfg(test)]
mod tests {
    use super::hour_subset;
    use serde_json::json;

    #[test]
    fn complete_and_partial_ttl_evidence_are_distinct() {
        for (details, expected) in [
            (json!(null), None),
            (json!({"ephemeral_5m_input_tokens": 100}), None),
            (json!({"ephemeral_1h_input_tokens": 200}), None),
            (
                json!({"ephemeral_5m_input_tokens": 100, "ephemeral_1h_input_tokens": 200}),
                Some(200),
            ),
            (
                json!({"ephemeral_5m_input_tokens": 300, "ephemeral_1h_input_tokens": 0}),
                Some(0),
            ),
        ] {
            let usage = json!({"cache_creation": details});
            assert_eq!(
                hour_subset(usage.as_object().unwrap(), 300).unwrap(),
                expected
            );
        }
    }

    #[test]
    fn malformed_or_contradictory_ttl_counts_fail_closed() {
        for details in [
            json!(7),
            json!({"ephemeral_5m_input_tokens": -1}),
            json!({"ephemeral_1h_input_tokens": true}),
            json!({"ephemeral_1h_input_tokens": 301}),
            json!({"ephemeral_5m_input_tokens": 100, "ephemeral_1h_input_tokens": 100}),
            json!({"ephemeral_5m_input_tokens": 250, "ephemeral_1h_input_tokens": 100}),
        ] {
            let usage = json!({"cache_creation": details});
            assert!(hour_subset(usage.as_object().unwrap(), 300).is_err());
        }
    }
}
