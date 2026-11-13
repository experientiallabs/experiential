//! RE2 matching over UTF-8 text, including its byte atom and empty-match iteration.

use super::syntax::with_flags;

#[derive(Clone)]
pub(super) enum Matcher {
    Text(regex::Regex),
    Bytes(regex::bytes::Regex),
}

impl Matcher {
    pub fn compile(
        pattern: &str,
        case_insensitive: bool,
        size_limit: usize,
    ) -> Result<Self, String> {
        let translated = with_flags(pattern, case_insensitive).map_err(str::to_owned)?;
        if let Ok(regex) = regex::RegexBuilder::new(&translated)
            .case_insensitive(case_insensitive)
            .size_limit(size_limit)
            .dfa_size_limit(size_limit)
            .build()
        {
            return Ok(Self::Text(regex));
        }
        regex::bytes::RegexBuilder::new(&translated)
            .case_insensitive(case_insensitive)
            .size_limit(size_limit)
            .dfa_size_limit(size_limit)
            .build()
            .map(Self::Bytes)
            .map_err(|_| "expression exceeds the native detector's program bound".to_string())
    }

    /// Keep the wrapper's character-offset semantics for partial UTF-8 byte matches.
    /// Positions past the final character retain a virtual byte offset so redaction
    /// merges the same intervals before clipping ordinary string slices.
    pub fn spans(&self, text: &str, limit: usize) -> Result<Vec<(usize, usize)>, ()> {
        let mut result = Vec::new();
        let mut pos = 0;
        let mut last_offset = 0;
        let mut last_character = 0;
        let mut character_offsets = Vec::new();
        if matches!(self, Self::Bytes(_)) {
            character_offsets.extend(text.char_indices().map(|(i, _)| i));
            character_offsets.push(text.len());
        }
        loop {
            let (raw_end, start, end) = match self {
                Self::Text(regex) => {
                    let Some(found) = regex.find_at(text, pos) else {
                        break;
                    };
                    (found.end(), found.start(), found.end())
                }
                Self::Bytes(regex) => {
                    let Some(captures) = regex.captures_at(text.as_bytes(), pos) else {
                        break;
                    };
                    let full = captures.get(0).unwrap();
                    let mut offsets: Vec<_> = captures
                        .iter()
                        .flatten()
                        .flat_map(|m| [m.start(), m.end()])
                        .collect();
                    offsets.sort_unstable();
                    offsets.dedup();
                    let mut start = 0;
                    let mut end = 0;
                    for offset in offsets {
                        last_character += characters(text.as_bytes(), last_offset, offset);
                        last_offset = offset;
                        let byte = character_offsets.get(last_character).copied().unwrap_or(
                            text.len() + last_character.saturating_sub(character_offsets.len() - 1),
                        );
                        if offset == full.start() {
                            start = byte;
                        }
                        if offset == full.end() {
                            end = byte;
                        }
                    }
                    (full.end(), start, end)
                }
            };
            if result.len() == limit {
                return Err(());
            }
            result.push((start, end));
            if pos == text.len() {
                break;
            }
            if pos == raw_end {
                pos += character_width(text.as_bytes()[pos]);
                pos = pos.min(text.len());
            } else {
                pos = raw_end;
            }
        }
        Ok(result)
    }
}

fn character_width(byte: u8) -> usize {
    match byte {
        0xc0..=0xdf => 2,
        0xe0..=0xef => 3,
        0xf0..=0xf7 => 4,
        _ => 1,
    }
}

fn characters(bytes: &[u8], start: usize, end: usize) -> usize {
    let mut pos = start;
    let mut count = 0;
    while pos < end {
        pos += character_width(bytes[pos]);
        count += 1;
    }
    count
}
