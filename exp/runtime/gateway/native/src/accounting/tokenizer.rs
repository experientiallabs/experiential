//! Bounded reservation counts from the same packaged o200k vocabulary as admission.

use std::collections::{HashMap, HashSet};
use std::sync::{Arc, Mutex, OnceLock};

use crate::errors::PublicError;

const EXACT_CHARACTERS: usize = 1 << 20;
const RESET_NEWLINES: usize = 4096;

fn encoder() -> Result<&'static tiktoken_rs::CoreBPE, PublicError> {
    static ENCODER: OnceLock<Result<tiktoken_rs::CoreBPE, ()>> = OnceLock::new();
    ENCODER
        .get_or_init(|| tiktoken_rs::o200k_base().map_err(|_| ()))
        .as_ref()
        .map_err(|_| PublicError::internal())
}

/// Admission warms the immutable encoder; continuation rounds never fetch ranks.
pub(super) fn warm() -> Result<(), PublicError> {
    encoder().map(|_| ())
}

fn exact(text: &str) -> Result<usize, PublicError> {
    if text.is_empty() {
        return Ok(0);
    }
    encoder()?
        .count(text, &HashSet::new())
        .map_err(|_| PublicError::internal())
}

/// BPE counts a bounded character window; the remaining UTF-8 bytes over-reserve.
pub(super) fn bound(text: &str, letters: &str) -> Result<usize, PublicError> {
    bounded(text, EXACT_CHARACTERS, letters)
}

fn bounded(text: &str, characters: usize, letters: &str) -> Result<usize, PublicError> {
    let Some((edge, _)) = text.char_indices().nth(characters) else {
        return exact(text);
    };
    // Admission freezes its runtime's Unicode version. Rust's Alphabetic and
    // Letter properties can both differ from Python str.isalpha.
    let letter = letter_matcher(letters)?;
    let mut end = edge;
    for _ in 0..RESET_NEWLINES {
        let Some(newline) = text[..end].rfind('\n') else {
            break;
        };
        let next = newline + 1;
        let after = next + text[next..].chars().next().unwrap().len_utf8();
        if letter.is_match(&text[next..after]) {
            return Ok(exact(&text[..next])? + text.len() - next);
        }
        end = newline;
    }
    Ok(text.len())
}

fn letter_matcher(pattern: &str) -> Result<Arc<regex::Regex>, PublicError> {
    static LETTERS: OnceLock<Mutex<HashMap<String, Arc<regex::Regex>>>> = OnceLock::new();
    let mut letters = LETTERS
        .get_or_init(Mutex::default)
        .lock()
        .map_err(|_| PublicError::internal())?;
    if let Some(regex) = letters.get(pattern) {
        return Ok(regex.clone());
    }
    if pattern.len() > 32768 {
        return Err(PublicError::internal());
    }
    let regex = Arc::new(
        regex::RegexBuilder::new(pattern)
            .size_limit(262144)
            .build()
            .map_err(|_| PublicError::internal())?,
    );
    if letters.len() >= 4 {
        letters.clear();
    }
    letters.insert(pattern.to_owned(), regex.clone());
    Ok(regex)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::Value;

    #[test]
    fn bounded_counts_match_packaged_python_encoder() {
        warm().unwrap();
        let letters = include_str!("../../tests/generated/tokenizer_letters.txt").trim();
        for (i, line) in include_str!("../../tests/generated/tokenizer.jsonl")
            .lines()
            .enumerate()
        {
            let case: Value = crate::lossless_json::from_str(line).unwrap();
            let text = case["text"].as_str().unwrap();
            let limit = case["limit"].as_u64().unwrap() as usize;
            assert_eq!(
                bounded(text, limit, letters).unwrap(),
                case["expected"].as_u64().unwrap() as usize,
                "case {i} limit {limit}"
            );
            assert_eq!(
                bound(text, letters).unwrap(),
                case["exact"].as_u64().unwrap() as usize,
                "case {i} exact"
            );
        }
    }
}
