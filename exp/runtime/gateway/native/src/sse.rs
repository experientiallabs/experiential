//! Server-sent event decoding, mirroring `_SseDecoder` in
//! `exp.runtime.models.providers.streaming`.

const MAXIMUM_SSE_EVENT_BYTES: usize = 4_000_000;

/// One decoded server-sent event before provider-specific JSON parsing.
#[derive(Debug, Clone, PartialEq)]
pub struct SseEvent {
    pub event: Option<String>,
    pub data: String,
}

/// Incremental SSE frame decoder over provider response chunks.
#[derive(Debug, Default)]
pub struct SseDecoder {
    buffer: Vec<u8>,
    event_name: Option<String>,
    data_lines: Vec<String>,
    current_event_bytes: usize,
}

impl SseDecoder {
    pub fn new() -> Self {
        Self::default()
    }

    /// Feed one network chunk, returning every complete event it closes.
    pub fn feed(&mut self, chunk: &[u8]) -> Result<Vec<SseEvent>, String> {
        self.buffer.extend_from_slice(chunk);
        let mut events = Vec::new();
        while let Some(newline) = self.buffer.iter().position(|byte| *byte == b'\n') {
            let raw_line: Vec<u8> = self.buffer.drain(..=newline).collect();
            let raw_line = &raw_line[..raw_line.len() - 1];
            self.current_event_bytes += raw_line.len() + 1;
            if self.current_event_bytes > MAXIMUM_SSE_EVENT_BYTES {
                return Err("provider stream event exceeds the size limit".to_string());
            }
            let line = decode_line(raw_line)?;
            if line.is_empty() {
                if !self.data_lines.is_empty() {
                    events.push(SseEvent {
                        event: self.event_name.take(),
                        data: self.data_lines.join("\n"),
                    });
                    self.data_lines.clear();
                } else {
                    self.event_name = None;
                }
                self.current_event_bytes = 0;
                continue;
            }
            if line.starts_with(':') {
                continue;
            }
            let (field_name, raw_value) = match line.split_once(':') {
                Some((name, value)) => (name, value),
                None => (line.as_str(), ""),
            };
            let value = raw_value.strip_prefix(' ').unwrap_or(raw_value);
            if field_name == "event" {
                self.event_name = Some(value.to_string());
            } else if field_name == "data" {
                self.data_lines.push(value.to_string());
            }
        }
        if self.current_event_bytes + self.buffer.len() > MAXIMUM_SSE_EVENT_BYTES {
            return Err("provider stream event exceeds the size limit".to_string());
        }
        Ok(events)
    }

    /// Close the stream, returning one trailing unterminated event when present.
    pub fn finish(&mut self) -> Result<Option<SseEvent>, String> {
        if !self.buffer.is_empty() {
            self.current_event_bytes += self.buffer.len();
            if self.current_event_bytes > MAXIMUM_SSE_EVENT_BYTES {
                return Err("provider stream event exceeds the size limit".to_string());
            }
            let trailing: Vec<u8> = std::mem::take(&mut self.buffer);
            let line = decode_line(&trailing)?;
            if let Some(value) = line.strip_prefix("data:") {
                let value = value.strip_prefix(' ').unwrap_or(value);
                self.data_lines.push(value.to_string());
            } else if !line.is_empty() && !line.starts_with(':') {
                return Err("provider stream ended with an incomplete SSE field".to_string());
            }
        }
        if self.data_lines.is_empty() {
            return Ok(None);
        }
        let event = SseEvent {
            event: self.event_name.take(),
            data: self.data_lines.join("\n"),
        };
        self.data_lines.clear();
        Ok(Some(event))
    }
}

fn decode_line(raw_line: &[u8]) -> Result<String, String> {
    let raw_line = raw_line.strip_suffix(b"\r").unwrap_or(raw_line);
    String::from_utf8(raw_line.to_vec())
        .map_err(|_| "provider stream contains invalid UTF-8".to_string())
}
