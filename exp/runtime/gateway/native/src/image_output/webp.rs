//! Bound every WebP frame before the codec can allocate from embedded dimensions.

use super::{image_limit_failure, invalid_image, MAXIMUM_IMAGE_DIMENSION, MAXIMUM_RASTER_BYTES};
use crate::errors::Failure;

fn little_u24(bytes: &[u8]) -> u32 {
    u32::from_le_bytes([bytes[0], bytes[1], bytes[2], 0])
}

fn dimensions(width: u32, height: u32, expected: Option<(u32, u32)>) -> Result<(), Failure> {
    if width > MAXIMUM_IMAGE_DIMENSION
        || height > MAXIMUM_IMAGE_DIMENSION
        || u64::from(width) * u64::from(height) * 4 > MAXIMUM_RASTER_BYTES
    {
        return Err(image_limit_failure());
    }
    if width == 0 || height == 0 || expected.is_some_and(|size| size != (width, height)) {
        return Err(invalid_image());
    }
    Ok(())
}

/// Inspect RIFF chunks without allocation, including bitstreams inside animation frames.
pub(super) fn validate_frame_headers(bytes: &[u8]) -> Result<(), Failure> {
    let header = bytes.get(..12).ok_or_else(invalid_image)?;
    if &header[..4] != b"RIFF" || &header[8..12] != b"WEBP" {
        return Err(invalid_image());
    }
    let size = u32::from_le_bytes(header[4..8].try_into().unwrap()) as usize;
    let end = size.checked_add(8).ok_or_else(invalid_image)?;
    if end != bytes.len() {
        return Err(invalid_image());
    }
    chunks(bytes.get(12..end).ok_or_else(invalid_image)?, None, true)
}

fn chunks(
    mut bytes: &[u8],
    mut expected: Option<(u32, u32)>,
    allow_animation: bool,
) -> Result<(), Failure> {
    while !bytes.is_empty() {
        let header = bytes.get(..8).ok_or_else(invalid_image)?;
        let size = u32::from_le_bytes(header[4..8].try_into().unwrap()) as usize;
        let end = size.checked_add(8).ok_or_else(invalid_image)?;
        let data = bytes.get(8..end).ok_or_else(invalid_image)?;
        match &header[..4] {
            b"VP8X" if allow_animation && expected.is_none() => {
                let frame = data.get(..10).ok_or_else(invalid_image)?;
                let size = (little_u24(&frame[4..7]) + 1, little_u24(&frame[7..10]) + 1);
                dimensions(size.0, size.1, None)?;
                expected = Some(size);
            }
            b"VP8 " => {
                let frame = data.get(..10).ok_or_else(invalid_image)?;
                if frame[0] & 1 != 0 || frame[3..6] != [0x9d, 0x01, 0x2a] {
                    return Err(invalid_image());
                }
                let width = u16::from_le_bytes([frame[6], frame[7]]) & 0x3fff;
                let height = u16::from_le_bytes([frame[8], frame[9]]) & 0x3fff;
                dimensions(u32::from(width), u32::from(height), expected)?;
            }
            b"VP8L" => {
                let frame = data.get(..5).ok_or_else(invalid_image)?;
                let bits = u32::from_le_bytes(frame[1..5].try_into().unwrap());
                if frame[0] != 0x2f || bits >> 29 != 0 {
                    return Err(invalid_image());
                }
                dimensions((bits & 0x3fff) + 1, ((bits >> 14) & 0x3fff) + 1, expected)?;
            }
            b"ANMF" if allow_animation => {
                let frame = data.get(..16).ok_or_else(invalid_image)?;
                let canvas = expected.ok_or_else(invalid_image)?;
                let size = (little_u24(&frame[6..9]) + 1, little_u24(&frame[9..12]) + 1);
                dimensions(size.0, size.1, None)?;
                if little_u24(&frame[..3]) * 2 + size.0 > canvas.0
                    || little_u24(&frame[3..6]) * 2 + size.1 > canvas.1
                {
                    return Err(invalid_image());
                }
                chunks(&data[16..], Some(size), false)?;
            }
            b"ANMF" | b"VP8X" => return Err(invalid_image()),
            _ => {}
        }
        bytes = bytes.get(end + size % 2..).ok_or_else(invalid_image)?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use base64::engine::general_purpose::STANDARD;
    use base64::Engine;
    use image::{ImageDecoder, ImageEncoder};

    fn chunk(kind: &[u8; 4], data: &[u8]) -> Vec<u8> {
        let mut bytes = kind.to_vec();
        bytes.extend_from_slice(&(data.len() as u32).to_le_bytes());
        bytes.extend_from_slice(data);
        if !data.len().is_multiple_of(2) {
            bytes.push(0);
        }
        bytes
    }

    fn mismatched_image(lossless: bool, animated: bool, width: u16) -> Vec<u8> {
        let bitstream = if lossless {
            let bits = u32::from(width - 1) | (u32::from(width - 1) << 14);
            let mut data = vec![0x2f];
            data.extend_from_slice(&bits.to_le_bytes());
            chunk(b"VP8L", &data)
        } else {
            let mut data = vec![0x10, 0, 0, 0x9d, 0x01, 0x2a];
            data.extend_from_slice(&width.to_le_bytes());
            data.extend_from_slice(&width.to_le_bytes());
            chunk(b"VP8 ", &data)
        };
        let mut canvas = [0; 10];
        canvas[0] = if animated { 2 } else { 0 };
        let mut body = b"WEBP".to_vec();
        body.extend(chunk(b"VP8X", &canvas));
        if animated {
            body.extend(chunk(b"ANIM", &[0; 6]));
            let mut frame = vec![0; 16];
            frame.extend(bitstream);
            body.extend(chunk(b"ANMF", &frame));
        } else {
            body.extend(bitstream);
        }
        chunk(b"RIFF", &body)
    }

    #[test]
    fn extended_and_animated_bitstream_dimensions_are_checked_before_decode() {
        for lossless in [false, true] {
            for animated in [false, true] {
                for width in [2, 16_383] {
                    let bytes = mismatched_image(lossless, animated, width);
                    // The library header sees only the 1x1 outer canvas. Decoding its
                    // lossy frame would allocate from the untrusted embedded size first.
                    let decoder = image::ImageReader::with_format(
                        std::io::Cursor::new(&bytes),
                        image::ImageFormat::WebP,
                    )
                    .into_decoder()
                    .unwrap();
                    assert_eq!(decoder.dimensions(), (1, 1));
                    let failure = validate_frame_headers(&bytes).unwrap_err();
                    assert!(!failure.retryable_same_deployment && !failure.failover_eligible);
                    assert!(
                        super::super::inline_image("image/webp", &STANDARD.encode(&bytes)).is_err()
                    );
                }
            }
        }
    }

    #[test]
    fn ordinary_lossless_webp_still_decodes() {
        let mut bytes = Vec::new();
        image::codecs::webp::WebPEncoder::new_lossless(&mut bytes)
            .write_image(&[255, 0, 0, 255], 1, 1, image::ExtendedColorType::Rgba8)
            .unwrap();
        assert!(super::super::inline_image("image/webp", &STANDARD.encode(&bytes)).is_ok());
        for animated in [false, true] {
            let mut canvas = [0; 10];
            canvas[0] = if animated { 0x12 } else { 0x10 };
            let mut body = b"WEBP".to_vec();
            body.extend(chunk(b"VP8X", &canvas));
            if animated {
                body.extend(chunk(b"ANIM", &[0; 6]));
                let mut frame = vec![0; 16];
                frame.extend_from_slice(&bytes[12..]);
                body.extend(chunk(b"ANMF", &frame));
            } else {
                body.extend_from_slice(&bytes[12..]);
            }
            assert!(super::super::inline_image(
                "image/webp",
                &STANDARD.encode(chunk(b"RIFF", &body))
            )
            .is_ok());
        }
    }

    #[test]
    fn riff_length_cannot_hide_an_unvalidated_trailing_frame() {
        let mut bytes = mismatched_image(false, false, 16_383);
        bytes[4..8].copy_from_slice(&4_u32.to_le_bytes());
        assert!(validate_frame_headers(&bytes).is_err());
        assert!(super::super::inline_image("image/webp", &STANDARD.encode(bytes)).is_err());
    }
}
