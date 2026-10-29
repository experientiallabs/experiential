//! Validation and wire shapes for complete generated images inside chat turns.

mod webp;

use base64::engine::general_purpose::STANDARD;
use base64::Engine;
use image::ImageDecoder;
use serde_json::{json, Value};

use crate::errors::{Failure, FailureClass};

const MAXIMUM_ENCODED_IMAGE_BYTES: usize = 16 * 1024 * 1024;
const MAXIMUM_COMPRESSED_IMAGE_BYTES: usize = 12 * 1024 * 1024;
const MAXIMUM_RASTER_BYTES: u64 = 64 * 1024 * 1024;
const MAXIMUM_IMAGE_DIMENSION: u32 = 16_384;

fn image_limit_failure() -> Failure {
    Failure::new(
        FailureClass::MalformedResponse,
        "provider generated image exceeds validation limits",
    )
    .with_retry(false, false)
}

/// Reject invalid image payloads without exposing provider content in diagnostics.
fn invalid_image() -> Failure {
    Failure::new(
        FailureClass::MalformedResponse,
        "provider returned an invalid generated image",
    )
    .with_retry(false, false)
}

/// Validate an inline raster image and preserve its encoded bytes exactly.
pub fn inline_image(media_type: &str, data: &str) -> Result<String, Failure> {
    let format = match media_type {
        "image/png" => image::ImageFormat::Png,
        "image/jpeg" => image::ImageFormat::Jpeg,
        "image/webp" => image::ImageFormat::WebP,
        "image/gif" => image::ImageFormat::Gif,
        _ => return Err(invalid_image()),
    };
    // Bound the allocation performed by base64 decoding, not just the later raster.
    if data.len() > MAXIMUM_ENCODED_IMAGE_BYTES
        || data
            .len()
            .div_ceil(4)
            .checked_mul(3)
            .is_none_or(|bytes| bytes > MAXIMUM_COMPRESSED_IMAGE_BYTES)
    {
        return Err(image_limit_failure());
    }
    let bytes = STANDARD.decode(data).map_err(|_| invalid_image())?;
    if format == image::ImageFormat::WebP {
        webp::validate_frame_headers(&bytes)?;
    }
    let mut reader = image::ImageReader::with_format(std::io::Cursor::new(&bytes), format);
    let mut limits = image::Limits::default();
    limits.max_image_width = Some(MAXIMUM_IMAGE_DIMENSION);
    limits.max_image_height = Some(MAXIMUM_IMAGE_DIMENSION);
    // This is a codec working-memory hint, not a strict heap ceiling across formats.
    limits.max_alloc = Some(MAXIMUM_RASTER_BYTES);
    reader.limits(limits);
    let decoder = reader.into_decoder().map_err(|_| invalid_image())?;
    let (width, height) = decoder.dimensions();
    let raster_bytes = u64::from(width)
        .checked_mul(u64::from(height))
        .and_then(|pixels| pixels.checked_mul(u64::from(decoder.color_type().bytes_per_pixel())));
    if width > MAXIMUM_IMAGE_DIMENSION
        || height > MAXIMUM_IMAGE_DIMENSION
        || raster_bytes.is_none_or(|bytes| bytes > MAXIMUM_RASTER_BYTES)
        || decoder.total_bytes() > MAXIMUM_RASTER_BYTES
    {
        return Err(image_limit_failure());
    }
    // Header construction precedes this check; full raster allocation does not.
    image::DynamicImage::from_decoder(decoder).map_err(|_| invalid_image())?;
    drop(bytes);
    Ok(format!("data:{media_type};base64,{data}"))
}

/// Decode the OpenRouter-style image URL envelope without fetching external URLs.
pub fn chat_image(value: &Value) -> Result<String, Failure> {
    let url = value
        .pointer("/image_url/url")
        .and_then(Value::as_str)
        .ok_or_else(invalid_image)?;
    let (media_type, data) = url
        .strip_prefix("data:")
        .and_then(|value| value.split_once(";base64,"))
        .ok_or_else(invalid_image)?;
    inline_image(media_type, data)
}

/// Encode the chat images extension for one completed image.
pub fn chat_image_value(url: &str) -> Value {
    json!({"type": "image_url", "image_url": {"url": url}})
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dialects::{Dialect, Normalizer};
    use crate::events::Event;
    use crate::sse::SseEvent;

    const PNG: &str = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAIAAACQd1PeAAAADElEQVR4nGP4z8AAAAMBAQDJ/pLvAAAAAElFTkSuQmCC";

    #[test]
    fn inline_raster_is_preserved_and_remote_or_active_content_is_refused() {
        let url = inline_image("image/png", PNG).unwrap();
        assert_eq!(chat_image(&chat_image_value(&url)).unwrap(), url);
        for image in [
            json!({}),
            json!({"image_url":{"url":"https://example.test/a.png"}}),
            json!({"image_url":{"url":"data:image/svg+xml;base64,PHN2Zz4="}}),
        ] {
            let failure = chat_image(&image).unwrap_err();
            assert!(!failure.retryable_same_deployment);
        }
        assert!(inline_image("image/png", "iVBORw0KGgo=").is_err());
        assert!(inline_image("image/png", "garbage").is_err());
        assert!(inline_image("image/png", "aGVsbG8=").is_err());
    }

    #[test]
    fn oversized_encoded_image_is_rejected_before_base64_decoding() {
        // A real PNG with trailing bytes is still decodable, but exceeds the input budget.
        let mut bytes = STANDARD.decode(PNG).unwrap();
        bytes.resize(MAXIMUM_COMPRESSED_IMAGE_BYTES + 3, 0);
        let encoded = STANDARD.encode(bytes);
        let failure = inline_image("image/png", &encoded).unwrap_err();
        assert_eq!(failure.safe_message, image_limit_failure().safe_message);
        assert!(!failure.retryable_same_deployment && !failure.failover_eligible);
    }

    #[test]
    fn inflated_gif_canvas_is_bounded_before_raster_allocation() {
        // A valid one-pixel GIF can declare a much larger canvas in its logical header.
        let gif = STANDARD
            .decode("R0lGODlhAQABAIAAAAAAAP///ywAAAAAAQABAAACAUwAOw==")
            .unwrap();
        assert!(inline_image("image/gif", &STANDARD.encode(&gif)).is_ok());
        for (width, height, accepted) in [(4096_u16, 4096_u16, true), (4097, 4096, false)] {
            let mut bytes = gif.clone();
            bytes[6..8].copy_from_slice(&width.to_le_bytes());
            bytes[8..10].copy_from_slice(&height.to_le_bytes());
            let decoder = image::ImageReader::with_format(
                std::io::Cursor::new(&bytes),
                image::ImageFormat::Gif,
            )
            .into_decoder()
            .unwrap();
            assert_eq!(decoder.dimensions(), (u32::from(width), u32::from(height)));
            assert_eq!(decoder.total_bytes() <= MAXIMUM_RASTER_BYTES, accepted);
            let result = inline_image("image/gif", &STANDARD.encode(&bytes));
            if accepted {
                assert!(result.is_ok());
            } else {
                let failure = result.unwrap_err();
                assert_eq!(failure.safe_message, image_limit_failure().safe_message);
                assert!(!failure.retryable_same_deployment && !failure.failover_eligible);
            }
        }
    }

    #[test]
    fn both_chat_dialects_preserve_visible_images_without_thought_images() {
        let url = inline_image("image/png", PNG).unwrap();
        let cases = [
            (
                Dialect::OpenAiCompatible,
                json!({"choices":[{"index":0,"delta":{"content":"A cat", "images":[chat_image_value(&url)]},"finish_reason":"stop"}]}),
            ),
            (
                Dialect::GeminiGenerateContent,
                json!({"candidates":[{"content":{"parts":[{"thought":true,"inlineData":{"mimeType":"image/png","data":PNG}},{"text":"A cat"},{"inlineData":{"mimeType":"image/png","data":PNG}}]},"finishReason":"STOP"}]}),
            ),
        ];
        for (dialect, payload) in cases {
            let mut normalizer = Normalizer::new(dialect);
            let events = normalizer
                .feed(&SseEvent {
                    event: None,
                    data: payload.to_string(),
                })
                .unwrap();
            assert_eq!(
                events
                    .iter()
                    .filter(|e| matches!(e, Event::Image(_)))
                    .count(),
                1
            );
            assert!(events
                .iter()
                .any(|e| matches!(e, Event::Image(value) if value == &url)));
            assert!(events
                .iter()
                .any(|e| matches!(e, Event::TextDelta(value) if value == "A cat")));
        }
    }
}
