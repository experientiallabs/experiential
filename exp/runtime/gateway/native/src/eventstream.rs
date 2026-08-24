//! AWS binary event-stream decoding (`application/vnd.amazon.eventstream`).
//!
//! One message is: an 8-byte prelude (big-endian total length and headers
//! length) plus its CRC32, the headers block, the payload, and a trailing
//! CRC32 over everything before it. Decoded messages surface as the shared
//! `SseEvent` frame shape: `event` carries the `:event-type` header (or the
//! `:exception-type` header for exception messages) and `data` carries the
//! UTF-8 JSON payload, so dialect normalizers consume one frame vocabulary
//! for both SSE and event-stream providers.

use crate::sse::SseEvent;

/// Hard ceiling on one event-stream message, mirroring the AWS encoding's
/// own 16 MiB payload bound so a corrupt length cannot balloon the buffer.
const MAXIMUM_MESSAGE_BYTES: usize = 16 * 1024 * 1024 + 4096;

/// Minimum legal message: prelude (8) + prelude CRC (4) + message CRC (4).
const MINIMUM_MESSAGE_BYTES: usize = 16;

/// Compute the IEEE CRC32 (the zlib polynomial) of one byte slice.
fn crc32(bytes: &[u8]) -> u32 {
    let mut state = !0u32;
    for byte in bytes {
        state ^= u32::from(*byte);
        for _ in 0..8 {
            let mask = 0u32.wrapping_sub(state & 1);
            state = (state >> 1) ^ (0xedb8_8320 & mask);
        }
    }
    !state
}

fn read_u32(bytes: &[u8]) -> u32 {
    u32::from_be_bytes([bytes[0], bytes[1], bytes[2], bytes[3]])
}

/// One decoded string header of interest.
fn parse_headers(mut block: &[u8]) -> Result<Vec<(String, String)>, String> {
    let mut headers = Vec::new();
    while !block.is_empty() {
        let name_length = usize::from(block[0]);
        block = &block[1..];
        if block.len() < name_length + 1 {
            return Err("provider event-stream header is truncated".to_string());
        }
        let name = String::from_utf8(block[..name_length].to_vec())
            .map_err(|_| "provider event-stream header name is not UTF-8".to_string())?;
        block = &block[name_length..];
        let value_type = block[0];
        block = &block[1..];
        let (value, consumed) = match value_type {
            // Boolean true / false carry no value bytes.
            0 | 1 => (None, 0usize),
            // 1-, 2-, 4-, and 8-byte integers and the 8-byte timestamp.
            2 => (None, 1),
            3 => (None, 2),
            4 => (None, 4),
            5 | 8 => (None, 8),
            // Length-prefixed byte arrays and strings.
            6 | 7 => {
                if block.len() < 2 {
                    return Err("provider event-stream header is truncated".to_string());
                }
                let length = usize::from(u16::from_be_bytes([block[0], block[1]]));
                if block.len() < 2 + length {
                    return Err("provider event-stream header is truncated".to_string());
                }
                let text = if value_type == 7 {
                    Some(
                        String::from_utf8(block[2..2 + length].to_vec()).map_err(|_| {
                            "provider event-stream header value is not UTF-8".to_string()
                        })?,
                    )
                } else {
                    None
                };
                (text, 2 + length)
            }
            // 16-byte UUID.
            9 => (None, 16),
            _ => return Err("provider event-stream header type is unsupported".to_string()),
        };
        if block.len() < consumed {
            return Err("provider event-stream header is truncated".to_string());
        }
        block = &block[consumed..];
        if let Some(value) = value {
            headers.push((name, value));
        }
    }
    Ok(headers)
}

/// Incremental AWS event-stream frame decoder over provider response chunks.
#[derive(Debug, Default)]
pub struct EventStreamDecoder {
    buffer: Vec<u8>,
}

impl EventStreamDecoder {
    pub fn new() -> Self {
        Self::default()
    }

    /// Feed one network chunk, returning every complete message it closes.
    pub fn feed(&mut self, chunk: &[u8]) -> Result<Vec<SseEvent>, String> {
        self.buffer.extend_from_slice(chunk);
        let mut frames = Vec::new();
        loop {
            if self.buffer.len() < MINIMUM_MESSAGE_BYTES {
                if self.buffer.len() >= 4 {
                    // Validate the claimed length early so garbage cannot
                    // stall the stream waiting for bytes that never come.
                    let total = read_u32(&self.buffer) as usize;
                    if !(MINIMUM_MESSAGE_BYTES..=MAXIMUM_MESSAGE_BYTES).contains(&total) {
                        return Err("provider event-stream message length is invalid".to_string());
                    }
                }
                return Ok(frames);
            }
            let total = read_u32(&self.buffer) as usize;
            if !(MINIMUM_MESSAGE_BYTES..=MAXIMUM_MESSAGE_BYTES).contains(&total) {
                return Err("provider event-stream message length is invalid".to_string());
            }
            if self.buffer.len() < total {
                return Ok(frames);
            }
            let message: Vec<u8> = self.buffer.drain(..total).collect();
            frames.push(decode_message(&message)?);
        }
    }

    /// Close the stream; buffered bytes that never formed a message fail it.
    pub fn finish(&mut self) -> Result<Option<SseEvent>, String> {
        if self.buffer.is_empty() {
            return Ok(None);
        }
        Err("provider stream ended inside an event-stream message".to_string())
    }
}

/// Decode one complete framed message into the shared frame shape.
fn decode_message(message: &[u8]) -> Result<SseEvent, String> {
    let headers_length = read_u32(&message[4..8]) as usize;
    let prelude_crc = read_u32(&message[8..12]);
    if crc32(&message[..8]) != prelude_crc {
        return Err("provider event-stream prelude checksum failed".to_string());
    }
    let message_crc = read_u32(&message[message.len() - 4..]);
    if crc32(&message[..message.len() - 4]) != message_crc {
        return Err("provider event-stream message checksum failed".to_string());
    }
    let payload_start = 12 + headers_length;
    if payload_start > message.len() - 4 {
        return Err("provider event-stream headers overflow the message".to_string());
    }
    let headers = parse_headers(&message[12..payload_start])?;
    let payload = String::from_utf8(message[payload_start..message.len() - 4].to_vec())
        .map_err(|_| "provider event-stream payload is not UTF-8".to_string())?;
    let mut message_type = None;
    let mut event_type = None;
    let mut exception_type = None;
    for (name, value) in headers {
        match name.as_str() {
            ":message-type" => message_type = Some(value),
            ":event-type" => event_type = Some(value),
            ":exception-type" => exception_type = Some(value),
            _ => {}
        }
    }
    let event = match message_type.as_deref() {
        Some("exception") => exception_type,
        _ => event_type.or(exception_type),
    };
    Ok(SseEvent {
        event,
        data: payload,
    })
}

/// Encode one event-stream message for fixtures and tests: the exact framing
/// the decoder consumes, so golden streams are built from readable parts.
#[cfg(test)]
pub fn encode_message(headers: &[(&str, &str)], payload: &[u8]) -> Vec<u8> {
    let mut header_block = Vec::new();
    for (name, value) in headers {
        header_block.push(name.len() as u8);
        header_block.extend_from_slice(name.as_bytes());
        header_block.push(7u8);
        header_block.extend_from_slice(&(value.len() as u16).to_be_bytes());
        header_block.extend_from_slice(value.as_bytes());
    }
    let total = 12 + header_block.len() + payload.len() + 4;
    let mut message = Vec::with_capacity(total);
    message.extend_from_slice(&(total as u32).to_be_bytes());
    message.extend_from_slice(&(header_block.len() as u32).to_be_bytes());
    let prelude_crc = crc32(&message[..8]);
    message.extend_from_slice(&prelude_crc.to_be_bytes());
    message.extend_from_slice(&header_block);
    message.extend_from_slice(payload);
    let message_crc = crc32(&message);
    message.extend_from_slice(&message_crc.to_be_bytes());
    message
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn crc32_matches_the_known_check_value() {
        // The classic CRC-32 check vector.
        assert_eq!(crc32(b"123456789"), 0xcbf4_3926);
    }

    #[test]
    fn feed_decodes_split_messages_and_typed_headers() {
        let event = encode_message(
            &[(":message-type", "event"), (":event-type", "messageStart")],
            br#"{"role":"assistant"}"#,
        );
        let exception = encode_message(
            &[
                (":message-type", "exception"),
                (":exception-type", "throttlingException"),
            ],
            br#"{"message":"slow down"}"#,
        );
        let mut stream = Vec::new();
        stream.extend_from_slice(&event);
        stream.extend_from_slice(&exception);
        let mut decoder = EventStreamDecoder::new();
        let mut frames = Vec::new();
        // Feed byte by byte to prove incremental reassembly.
        for byte in stream {
            frames.extend(decoder.feed(&[byte]).expect("valid stream"));
        }
        assert_eq!(frames.len(), 2);
        assert_eq!(frames[0].event.as_deref(), Some("messageStart"));
        assert_eq!(frames[0].data, r#"{"role":"assistant"}"#);
        assert_eq!(frames[1].event.as_deref(), Some("throttlingException"));
        assert_eq!(frames[1].data, r#"{"message":"slow down"}"#);
        assert!(decoder.finish().expect("clean end").is_none());
    }

    #[test]
    fn non_string_headers_are_skipped_without_desync() {
        // Hand-build a message with a boolean and an i32 header before the
        // event type, so skipping fixed-width values is exercised.
        let mut header_block = Vec::new();
        header_block.push(5u8);
        header_block.extend_from_slice(b":flag");
        header_block.push(0u8); // boolean true, no value bytes
        header_block.push(6u8);
        header_block.extend_from_slice(b":count");
        header_block.push(4u8); // i32
        header_block.extend_from_slice(&7i32.to_be_bytes());
        header_block.push(11u8);
        header_block.extend_from_slice(b":event-type");
        header_block.push(7u8);
        header_block.extend_from_slice(&(11u16).to_be_bytes());
        header_block.extend_from_slice(b"messageStop");
        let payload = br#"{"stopReason":"end_turn"}"#;
        let total = 12 + header_block.len() + payload.len() + 4;
        let mut message = Vec::new();
        message.extend_from_slice(&(total as u32).to_be_bytes());
        message.extend_from_slice(&(header_block.len() as u32).to_be_bytes());
        let prelude_crc = crc32(&message[..8]);
        message.extend_from_slice(&prelude_crc.to_be_bytes());
        message.extend_from_slice(&header_block);
        message.extend_from_slice(payload);
        let crc = crc32(&message);
        message.extend_from_slice(&crc.to_be_bytes());

        let mut decoder = EventStreamDecoder::new();
        let frames = decoder.feed(&message).expect("valid message");
        assert_eq!(frames.len(), 1);
        assert_eq!(frames[0].event.as_deref(), Some("messageStop"));
    }

    #[test]
    fn corrupt_checksums_and_lengths_fail_the_stream() {
        let mut corrupt = encode_message(&[(":event-type", "messageStart")], b"{}");
        let last = corrupt.len() - 1;
        corrupt[last] ^= 0xff;
        assert!(EventStreamDecoder::new().feed(&corrupt).is_err());

        let mut bad_prelude = encode_message(&[(":event-type", "messageStart")], b"{}");
        bad_prelude[9] ^= 0xff;
        assert!(EventStreamDecoder::new().feed(&bad_prelude).is_err());

        let mut decoder = EventStreamDecoder::new();
        assert!(decoder.feed(&[0xff, 0xff, 0xff, 0xff]).is_err());
    }

    #[test]
    fn a_stream_cut_mid_message_fails_at_finish() {
        let message = encode_message(&[(":event-type", "messageStart")], b"{}");
        let mut decoder = EventStreamDecoder::new();
        let frames = decoder
            .feed(&message[..message.len() - 2])
            .expect("still buffering");
        assert!(frames.is_empty());
        assert!(decoder.finish().is_err());
    }
}
