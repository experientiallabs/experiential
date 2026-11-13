//! Exact integer attribution for the local content-free ledger.

use serde_json::Value;

use crate::errors::PublicError;

pub(super) type Result<T> = std::result::Result<T, PublicError>;

fn count(value: &Value) -> Result<Option<u64>> {
    if value.is_null() {
        return Ok(None);
    }
    value
        .as_u64()
        .filter(|n| *n <= i64::MAX as u64)
        .map(Some)
        .ok_or_else(PublicError::internal)
}

pub(super) fn rate(prices: &Value, field: &str) -> Result<Option<u64>> {
    count(&prices[format!("{field}_nano_usd_per_million_tokens")])
}

fn rounded(numerator: u128, divisor: u128) -> Result<i64> {
    let rounded = numerator
        .checked_add(divisor / 2)
        .ok_or_else(PublicError::internal)?
        / divisor;
    i64::try_from(rounded).map_err(|_| PublicError::internal())
}

pub(super) fn unit_rate(prices: &Value, units: &Value) -> Result<Option<u64>> {
    let card = &prices["units"];
    if units.is_null() || card.is_null() || card["kind"] != units["kind"] {
        return Ok(None);
    }
    let variant = units["variant"].as_str().unwrap_or("");
    let selected = card["rates"].get(variant).unwrap_or(&card["rates"][""]);
    count(selected)
}

/// Preserve absent evidence and charge each observed subset only once.
pub(super) fn usage_cost(prices: &Value, usage: &Value) -> Result<Option<i64>> {
    if usage.is_null() {
        return Ok(None);
    }
    let units = &usage["billed_units"];
    let units_cost = if !units.is_null() {
        match unit_rate(prices, units)?.zip(count(&units["quantity_milli"])?) {
            Some((rate, quantity)) => {
                rounded(u128::from(rate) * u128::from(quantity), 1000).map(Some)
            }
            None => Ok(None),
        }?
    } else {
        Some(0)
    };
    if !units.is_null() && usage["input_tokens"].is_null() && usage["output_tokens"].is_null() {
        return Ok(units_cost);
    }
    let Some(units_cost) = units_cost else {
        return Ok(None);
    };
    let (Some(input), Some(output)) = (
        count(&usage["input_tokens"])?,
        count(&usage["output_tokens"])?,
    ) else {
        return Ok(None);
    };
    let long = &prices["long_context"];
    let prices = if long["input_threshold_tokens"]
        .as_u64()
        .is_some_and(|n| input >= n)
    {
        long
    } else {
        prices
    };
    let input_rate = rate(prices, "input")?;
    let output_rate = rate(prices, "output")?;
    let write_rate = rate(prices, "cache_creation_input")?;
    let hour_rate = rate(prices, "cache_creation_1h_input")?;
    let image_rate = rate(prices, "image_output")?;
    let read = count(&usage["cached_input_tokens"])?
        .unwrap_or(0)
        .min(input);
    let written = count(&usage["cache_creation_input_tokens"])?
        .unwrap_or(0)
        .min(input - read);
    let hour = count(&usage["cache_creation_1h_input_tokens"])?;
    if written > 0 && hour.is_none() && (write_rate.is_none() || write_rate != hour_rate) {
        return Ok(None);
    }
    let hour = hour.unwrap_or(0).min(written);
    let twelve = count(&usage["cache_creation_12h_input_tokens"])?
        .unwrap_or(0)
        .min(written - hour);
    let image = count(&usage["image_output_tokens"])?;
    if output > 0 && image.is_none() && image_rate.is_some() && image_rate != output_rate {
        return Ok(None);
    }
    let image = if image_rate.is_some() {
        image.unwrap_or(0).min(output)
    } else {
        0
    };
    let reasoning = count(&usage["reasoning_tokens"])?
        .unwrap_or(0)
        .min(output - image);
    let twelve_rate = rate(prices, "cache_creation_12h_input")?
        .map(u128::from)
        .or_else(|| input_rate.map(|n| (u128::from(n) * 5).div_ceil(2)));
    let dimensions = [
        (input - read - written, input_rate.map(u128::from)),
        (read, rate(prices, "cached_input")?.map(u128::from)),
        (written - hour - twelve, write_rate.map(u128::from)),
        (hour, hour_rate.map(u128::from)),
        (twelve, twelve_rate),
        (output - image - reasoning, output_rate.map(u128::from)),
        (reasoning, rate(prices, "reasoning")?.map(u128::from)),
        (image, image_rate.map(u128::from)),
    ];
    let mut numerator = 0_u128;
    for (tokens, rate) in dimensions {
        if tokens > 0 && rate.is_none() {
            return Ok(None);
        }
        numerator = numerator
            .checked_add(u128::from(tokens) * rate.unwrap_or(0))
            .ok_or_else(PublicError::internal)?;
    }
    rounded(numerator, 1_000_000)?
        .checked_add(units_cost)
        .map(Some)
        .ok_or_else(PublicError::internal)
}

pub(super) fn terminal_cost(prices: &Value, terminal: &Value) -> Result<Option<i64>> {
    if terminal["usage_incomplete_due_to_disconnect"] == true && terminal["usage_estimated"] != true
    {
        return Ok(None);
    }
    let tier = &terminal["service_tier"];
    if tier.is_object() {
        if tier["resolution"] != "confirmed" || !tier["prices"].is_object() {
            return Ok(None);
        }
        return usage_cost(&tier["prices"], &terminal["usage"]);
    }
    usage_cost(prices, &terminal["usage"])
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn long_context_threshold_and_cache_write_dimensions() {
        let prices = json!({"input_nano_usd_per_million_tokens":1000000,
            "cached_input_nano_usd_per_million_tokens":100000,
            "cache_creation_input_nano_usd_per_million_tokens":2000000,
            "cache_creation_1h_input_nano_usd_per_million_tokens":3000000,
            "output_nano_usd_per_million_tokens":4000000,
            "long_context":{"input_threshold_tokens":100,
                "input_nano_usd_per_million_tokens":2000000,
                "cached_input_nano_usd_per_million_tokens":200000,
                "cache_creation_input_nano_usd_per_million_tokens":4000000,
                "cache_creation_1h_input_nano_usd_per_million_tokens":6000000,
                "output_nano_usd_per_million_tokens":8000000}});
        let mut usage = json!({"input_tokens":99,"output_tokens":10,
            "cached_input_tokens":10,"cache_creation_input_tokens":20,"cache_creation_1h_input_tokens":5});
        assert_eq!(usage_cost(&prices, &usage).unwrap(), Some(155));
        usage["input_tokens"] = json!(100);
        assert_eq!(usage_cost(&prices, &usage).unwrap(), Some(312));
        usage["cache_creation_1h_input_tokens"] = Value::Null;
        assert_eq!(usage_cost(&prices, &usage).unwrap(), None);
    }

    #[test]
    fn missing_ttl_requires_uniform_prices_in_the_selected_schedule() {
        for (hour, expected) in [
            (json!(5000000), Some(408)),
            (json!(6000000), None),
            (Value::Null, None),
        ] {
            let prices = json!({"input_nano_usd_per_million_tokens":2000000,
                "output_nano_usd_per_million_tokens":4000000,
                "cache_creation_input_nano_usd_per_million_tokens":2500000,
                "cache_creation_1h_input_nano_usd_per_million_tokens":2500000,
                "long_context":{"input_threshold_tokens":100,
                    "input_nano_usd_per_million_tokens":4000000,
                    "cached_input_nano_usd_per_million_tokens":400000,
                    "cache_creation_input_nano_usd_per_million_tokens":5000000,
                    "cache_creation_1h_input_nano_usd_per_million_tokens":hour,
                    "output_nano_usd_per_million_tokens":2000000}});
            let usage = json!({"input_tokens":100,"output_tokens":10,"cached_input_tokens":20,"cache_creation_input_tokens":60});
            assert_eq!(usage_cost(&prices, &usage).unwrap(), expected);
        }
    }
}
