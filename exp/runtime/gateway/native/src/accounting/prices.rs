//! Resolve physical-attempt schedules and conservative integer reservation ceilings.

use std::time::SystemTime;

use chrono::{DateTime, Datelike, Duration, Timelike, Utc};
use serde::Deserialize;
use serde_json::{json, Value};

use crate::errors::PublicError;

const RATE_FIELDS: [&str; 8] = [
    "input_nano_usd_per_million_tokens",
    "cached_input_nano_usd_per_million_tokens",
    "cache_creation_input_nano_usd_per_million_tokens",
    "cache_creation_1h_input_nano_usd_per_million_tokens",
    "cache_creation_12h_input_nano_usd_per_million_tokens",
    "output_nano_usd_per_million_tokens",
    "reasoning_nano_usd_per_million_tokens",
    "image_output_nano_usd_per_million_tokens",
];

#[derive(Clone, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub(super) enum Ceiling {
    Tokens {
        input: u64,
        output: u64,
        cached: bool,
        writes: bool,
        hour: bool,
        twelve: bool,
        reasoning: bool,
    },
    Embeddings {
        input: u64,
    },
    Images {
        input: u64,
        count: u64,
        variant: String,
    },
    Audio {
        input: u64,
        output: Option<u64>,
        unit: String,
        quantity: u64,
    },
    Units {
        unit: String,
        variant: String,
        quantity: u64,
    },
}

impl Ceiling {
    pub fn search_input(&self, input_tokens: u64) -> Result<Self, PublicError> {
        let mut ceiling = self.clone();
        let Self::Tokens { input, .. } = &mut ceiling else {
            return Err(PublicError::internal());
        };
        *input = input_tokens;
        ceiling.validate()?;
        Ok(ceiling)
    }

    pub fn validate(&self) -> Result<(), PublicError> {
        let counts = match self {
            Self::Tokens { input, output, .. } => vec![*input, *output],
            Self::Embeddings { input } => vec![*input],
            Self::Images { input, count, .. } => vec![*input, *count],
            Self::Audio {
                input,
                output,
                quantity,
                ..
            } => vec![*input, output.unwrap_or(0), *quantity],
            Self::Units { quantity, .. } => vec![*quantity],
        };
        if counts.into_iter().any(|n| n > i64::MAX as u64) {
            return Err(PublicError::internal());
        }
        Ok(())
    }

    pub fn amount(&self, prices: &Value) -> Result<Option<i64>, PublicError> {
        let value = match self {
            Self::Tokens {
                input,
                output,
                cached,
                writes,
                hour,
                twelve,
                reasoning,
            } => {
                let mut schedules = vec![prices];
                let tier = &prices["long_context"];
                if tier["input_threshold_tokens"]
                    .as_u64()
                    .is_some_and(|n| u128::from(*input) * 100 >= u128::from(n) * 80)
                {
                    schedules.push(tier);
                }
                let mut max_input = 0;
                let mut max_output = 0;
                for schedule in schedules {
                    let r = rates(schedule)?;
                    if r[0].is_none()
                        || r[5].is_none()
                        || (*cached && r[1].is_none())
                        || (*writes && (r[2].is_none() || (*hour && r[3].is_none())))
                        || (*reasoning && r[6].is_none())
                    {
                        return Ok(None);
                    }
                    let twelve_rate = if *twelve {
                        r[4].or_else(|| r[0].map(|n| (n * 5).div_ceil(2)))
                    } else {
                        None
                    };
                    max_input = max_input.max(
                        [
                            r[0],
                            r[1],
                            r[2].filter(|_| *writes),
                            r[3].filter(|_| *writes && *hour),
                            twelve_rate,
                        ]
                        .into_iter()
                        .flatten()
                        .max()
                        .unwrap_or(0),
                    );
                    max_output =
                        max_output.max([r[5], r[6], r[7]].into_iter().flatten().max().unwrap_or(0));
                }
                Some(token_amount(*input, max_input, *output, max_output))
            }
            Self::Embeddings { input } => {
                rates(prices)?[0].map(|rate| token_amount(*input, rate, 0, 0))
            }
            Self::Images {
                input,
                count,
                variant,
            } => {
                if let Some(rate) = unit_rate(prices, "image", variant)? {
                    Some(u128::from(*count) * u128::from(rate))
                } else {
                    let r = rates(prices)?;
                    r[0].zip(r[5]).map(|(i, o)| {
                        (u128::from(*input) * u128::from(i)
                            + u128::from(*count) * 6240 * u128::from(o))
                        .div_ceil(1_000_000)
                    })
                }
            }
            Self::Audio {
                input,
                output,
                unit,
                quantity,
            } => {
                if let Some(rate) = unit_rate(prices, unit, "")? {
                    Some(u128::from(*quantity) * u128::from(rate))
                } else {
                    let Some(output) = output else {
                        return Ok(None);
                    };
                    let mut r = rates(prices)?;
                    if prices["long_context"].is_object() {
                        let tier = rates(&prices["long_context"])?;
                        for index in [0, 5] {
                            r[index] = r[index].zip(tier[index]).map(|(a, b)| a.max(b));
                        }
                    }
                    r[0].zip(r[5])
                        .map(|(i, o)| token_amount(*input, i, *output, o))
                }
            }
            Self::Units {
                unit,
                variant,
                quantity,
            } => unit_rate(prices, unit, variant)?
                .map(|rate| u128::from(*quantity) * u128::from(rate)),
        };
        value
            .map(|n| i64::try_from(n).map_err(|_| PublicError::internal()))
            .transpose()
    }
}

fn token_amount(input: u64, input_rate: u64, output: u64, output_rate: u64) -> u128 {
    (u128::from(input) * u128::from(input_rate) + u128::from(output) * u128::from(output_rate))
        .div_ceil(1_000_000)
}

fn rate(value: &Value) -> Result<Option<u64>, PublicError> {
    if value.is_null() {
        return Ok(None);
    }
    value
        .as_u64()
        .filter(|n| *n <= 1_000_000_000_000)
        .map(Some)
        .ok_or_else(PublicError::internal)
}

fn rates(prices: &Value) -> Result<[Option<u64>; 8], PublicError> {
    if !prices.is_object() {
        return Err(PublicError::internal());
    }
    let mut values = [None; 8];
    for (index, field) in RATE_FIELDS.iter().enumerate() {
        values[index] = rate(&prices[field])?;
    }
    Ok(values)
}

fn unit_rate(prices: &Value, kind: &str, variant: &str) -> Result<Option<u64>, PublicError> {
    let units = &prices["units"];
    if units["kind"] != kind {
        return Ok(None);
    }
    rate(units["rates"].get(variant).unwrap_or(&units["rates"][""]))
}

/// Off-peak replacement precedes processing-tier selection, once per physical call.
pub(super) fn select(
    prices: &Value,
    requested: Option<&str>,
    forwards: bool,
    customer: bool,
    ceiling: &Ceiling,
    now: SystemTime,
) -> Result<(Value, Value, Option<i64>), PublicError> {
    let mut standard = prices.clone();
    if prices["off_peak"].is_object() {
        if off_peak(&prices["off_peak"], now)? {
            standard = card(&prices["off_peak"]["prices"], Value::Null);
        } else {
            standard["off_peak"] = Value::Null;
        }
    }
    let tier = requested
        .filter(|tier| forwards && matches!(*tier, "flex" | "priority"))
        .filter(|tier| standard[*tier].is_object());
    let effective = tier.map_or_else(
        || standard.clone(),
        |tier| card(&standard[tier], standard["units"].clone()),
    );
    let mut admission = Value::Null;
    let mut amount = ceiling.amount(&effective)?;
    if let Some(tier) = tier.filter(|_| !customer) {
        let mut flat = standard;
        flat["flex"] = Value::Null;
        flat["priority"] = Value::Null;
        amount = amount.zip(ceiling.amount(&flat)?).map(|(a, b)| a.max(b));
        admission =
            json!({"requested": tier, "standard_prices": flat, "requested_prices": effective});
    }
    Ok((effective, admission, amount))
}

fn card(value: &Value, units: Value) -> Value {
    let mut result = json!({"units": units, "flex": null, "priority": null, "off_peak": null,
        "long_context": value["long_context"]});
    for field in RATE_FIELDS {
        result[field] = value[field].clone();
    }
    result
}

fn off_peak(value: &Value, now: SystemTime) -> Result<bool, PublicError> {
    #[derive(Deserialize)]
    struct Window {
        days: Vec<u32>,
        start_minute: u32,
        end_minute: u32,
    }
    #[derive(Deserialize)]
    struct Schedule {
        peak_windows: Vec<Window>,
        off_peak_dates: Vec<String>,
    }
    let schedule: Schedule =
        crate::lossless_json::from_value(value.clone()).map_err(|_| PublicError::internal())?;
    if schedule.peak_windows.is_empty()
        || schedule.peak_windows.len() > 32
        || schedule.off_peak_dates.len() > 128
        || schedule.peak_windows.iter().any(|w| {
            w.days.is_empty()
                || w.days.iter().any(|d| !(1..=7).contains(d))
                || w.start_minute >= w.end_minute
                || w.end_minute > 1440
        })
    {
        return Err(PublicError::internal());
    }
    let now = DateTime::<Utc>::from(now);
    for offset in -1..=15 {
        let at = now + Duration::minutes(offset);
        if schedule
            .off_peak_dates
            .contains(&at.date_naive().to_string())
        {
            continue;
        }
        let minute = at.hour() * 60 + at.minute();
        if schedule.peak_windows.iter().any(|w| {
            w.days.contains(&at.weekday().number_from_monday())
                && minute >= w.start_minute
                && minute < w.end_minute
        }) {
            return Ok(false);
        }
    }
    Ok(true)
}
