//! Native o200k estimation of generated output whose provider meter never arrived.

use std::sync::OnceLock;

use serde::Deserialize;
use serde_json::{json, Value};

use crate::errors::PublicError;

use super::terminal::TOKEN_FIELDS;

static ENCODER: OnceLock<Result<tiktoken_rs::CoreBPE, ()>> = OnceLock::new();

pub(super) fn warm() -> Result<(), PublicError> {
    encoder().map(|_| ())
}

fn encoder() -> Result<&'static tiktoken_rs::CoreBPE, PublicError> {
    ENCODER
        .get_or_init(|| tiktoken_rs::o200k_base().map_err(|_| ()))
        .as_ref()
        .map_err(|_| PublicError::internal())
}

#[derive(Default, Deserialize)]
#[serde(default, deny_unknown_fields)]
struct Streamed {
    text: String,
    reasoning: String,
    text_overflow_chars: u64,
    reasoning_overflow_chars: u64,
    images: u64,
}

pub(super) fn usage(
    data: &Value,
    observed: Option<&Value>,
    input: u64,
    cached: f64,
    output_floor: u64,
) -> Result<Option<Value>, PublicError> {
    let Ok(streamed) =
        crate::lossless_json::from_value::<Streamed>(data["streamed_output"].clone())
    else {
        return Ok(None);
    };
    if streamed.images > 0 {
        return Ok(None);
    }
    let reasoning = text_tokens(&streamed.reasoning, streamed.reasoning_overflow_chars)?;
    let visible = text_tokens(&streamed.text, streamed.text_overflow_chars)?;
    let observed = observed.unwrap_or(&Value::Null);
    let mut result = json!({"tool_names": observed.get("tool_names").cloned().unwrap_or(json!([])), "billed_units": null});
    for field in TOKEN_FIELDS {
        result[field] = observed[field].clone();
    }
    let reasoning_total = observed["reasoning_tokens"]
        .as_u64()
        .unwrap_or(0)
        .max(reasoning);
    let output = visible
        .checked_add(reasoning)
        .ok_or_else(PublicError::internal)?
        .max(observed["output_tokens"].as_u64().unwrap_or(0))
        .max(reasoning_total)
        .max(output_floor);
    let reads = observed["cached_input_tokens"].as_u64();
    let writes = observed["cache_creation_input_tokens"]
        .as_u64()
        .unwrap_or(0);
    let input = observed["input_tokens"].as_u64().unwrap_or(
        input.max(
            reads
                .unwrap_or(0)
                .checked_add(writes)
                .ok_or_else(PublicError::internal)?,
        ),
    );
    result["input_tokens"] = json!(input);
    result["output_tokens"] = json!(output);
    result["reasoning_tokens"] = json!(reasoning_total);
    if reads.is_none() && cached > 0.0 && input > writes {
        result["cached_input_tokens"] =
            json!((input - writes).min((input as f64 * cached.min(1.0)) as u64));
    }
    result["image_output_tokens"] = Value::Null;
    for field in ["web_search_requests", "tool_search_requests"] {
        result[field] = observed.get(field).cloned().unwrap_or(json!(0));
    }
    for field in ["web_search_results", "web_search_cost_nano_usd"] {
        result[field] = observed[field].clone();
    }
    Ok(Some(result))
}

fn text_tokens(text: &str, overflow: u64) -> Result<u64, PublicError> {
    let count = if text.is_empty() {
        0
    } else {
        encoder()?.count_ordinary(text) as u64
    };
    if overflow == 0 {
        return Ok(count);
    }
    let characters = text.chars().count() as u64;
    let extra = if characters == 0 || count == 0 {
        u128::from(overflow).div_ceil(4)
    } else {
        (u128::from(overflow) * u128::from(count)).div_ceil(u128::from(characters))
    };
    u64::try_from(u128::from(count) + extra).map_err(|_| PublicError::internal())
}
