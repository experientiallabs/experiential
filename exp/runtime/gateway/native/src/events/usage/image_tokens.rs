//! Generated-image output tokens: the count a provider prices at its image
//! rate (Gemini `candidatesTokensDetails` IMAGE entries, OpenRouter
//! `completion_tokens_details.image_tokens`). It is a detail of the output,
//! disjoint from `reasoning_tokens`, and is carried exactly as reported: a
//! Gemini 3 thinking image model's final meter can name more IMAGE tokens
//! than its `candidatesTokenCount`, so consumers bound it by the output total.

use serde_json::{Map, Value};

use super::{bounded_ledger_sum, count_or_zero, optional_usage_detail};

/// Sum Gemini's IMAGE-modality candidate entries; `None` when none was reported.
pub(super) fn gemini_image_output_tokens(
    object: &Map<String, Value>,
) -> Result<Option<u64>, String> {
    let details = match object.get("candidatesTokensDetails") {
        None | Some(Value::Null) => return Ok(None),
        Some(value) => value
            .as_array()
            .ok_or("Gemini candidatesTokensDetails must be an array")?,
    };
    let mut image: Option<u64> = None;
    for entry in details {
        let entry = entry
            .as_object()
            .ok_or("Gemini candidatesTokensDetails entries must be objects")?;
        if entry.get("modality").and_then(Value::as_str) != Some("IMAGE") {
            continue;
        }
        let count = count_or_zero(entry, "tokenCount", "Gemini image tokenCount")?;
        image = Some(bounded_ledger_sum(
            &[image.unwrap_or(0), count],
            "Gemini image output",
        )?);
    }
    Ok(image)
}

/// Read the OpenAI-shaped `image_tokens` output detail (OpenRouter's image models).
pub(super) fn openai_image_output_tokens(
    object: &Map<String, Value>,
    output_details: &str,
) -> Result<Option<u64>, String> {
    optional_usage_detail(
        object,
        output_details,
        "image_tokens",
        "OpenAI image_tokens",
    )
}
