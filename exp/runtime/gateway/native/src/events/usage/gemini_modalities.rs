//! Gemini's per-modality output breakdown, read for the image-output meter.

use serde_json::{Map, Value};

use super::{bounded_ledger_sum, count_or_zero};

/// The IMAGE entries of Gemini's `candidatesTokensDetails` (per-modality
/// output counts). A reported breakdown without an IMAGE entry is a real 0;
/// only an absent breakdown leaves the subset unknown. A malformed entry
/// fails the meter.
pub(super) fn gemini_image_candidates(object: &Map<String, Value>) -> Result<Option<u64>, String> {
    let details = match object.get("candidatesTokensDetails") {
        None | Some(Value::Null) => return Ok(None),
        Some(Value::Array(details)) => details,
        Some(_) => return Err("Gemini candidatesTokensDetails must be an array".to_string()),
    };
    let mut image = 0u64;
    for detail in details {
        let detail = detail
            .as_object()
            .ok_or("Gemini candidatesTokensDetails entries must be objects")?;
        if detail.get("modality").and_then(Value::as_str) == Some("IMAGE") {
            let count = count_or_zero(detail, "tokenCount", "Gemini image tokenCount")?;
            image = bounded_ledger_sum(&[image, count], "Gemini image output")?;
        }
    }
    Ok(Some(image))
}

/// The output total with the image subset inside it.
///
/// Gemini usually counts the image inside `candidatesTokenCount`
/// (gemini-3.1-flash-image: candidates 1278 with IMAGE 1120). Gemini 3 Pro
/// Image can report IMAGE 1120 beside a candidates count of 55 to 71 and
/// totals that leave it out (live 2026-10-19). Such an image is added to the
/// output, so the subset stays inclusive and the generated image is never
/// unbilled; an output that already covers it is unchanged.
pub fn gemini_output_covering_image(output: u64, image: Option<u64>) -> Result<u64, String> {
    match image {
        Some(image) if image > output => bounded_ledger_sum(&[output, image], "Gemini output"),
        _ => Ok(output),
    }
}
