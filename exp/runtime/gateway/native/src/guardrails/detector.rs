//! Bounded deterministic detection and in-place span redaction.
//!
//! This is the native port of the python `regex` guardrail adapter. It
//! compiles one authored rule (custom expressions plus the built-in families)
//! once at policy load and then redacts buffered text without any python
//! call. The same bounds hold as in the python adapter: a 1 MiB subject
//! ceiling and a 4,096 match and candidate ceiling, both fail closed.
//!
//! This module never logs subjects, matches, or replacements.

use regex::{Regex, RegexBuilder};
use serde::Deserialize;

use crate::guardrails::syntax::to_rust_syntax;

/// Largest subject this detector inspects, in UTF-8 bytes.
pub const MAX_TEXT_BYTES: usize = 1_048_576;
/// Largest number of matches and card candidates one subject may produce.
pub const MAX_MATCHES: usize = 4096;
/// Largest authored expression, in UTF-8 bytes.
pub const MAX_PATTERN_BYTES: usize = 1024;
/// Largest number of authored custom expressions in one rule.
pub const MAX_PATTERNS: usize = 32;
/// Compiled-program memory bound, mirroring the python adapter's RE2 option.
pub const COMPILE_SIZE_LIMIT: usize = 262_144;

/// Longest window a card candidate may span: 19 digits and 18 separators,
/// plus the exclusive end.
const CARD_WINDOW: usize = 38;

const EMAIL_PATTERN: &str = r"[A-Za-z0-9.!#$%&'*+/=?^_`{|}~-]+@[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)+";
const CREDIT_CARD_PATTERN: &str = r"\b[0-9](?:[ -]?[0-9]){12,}\b";
const API_KEY_PATTERN: &str = concat!(
    r"\b(?:sk-(?:proj-|ant-api[0-9]+-)?[A-Za-z0-9_-]{20,}",
    r"|gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,}",
    r"|xpl_[A-Za-z0-9_-]{20,}|AKIA[A-Z0-9]{16})\b",
);

/// Return the expression of one built-in family, by its authored name.
fn builtin_pattern(name: &str) -> Option<&'static str> {
    match name {
        "email" => Some(EMAIL_PATTERN),
        "credit_card" => Some(CREDIT_CARD_PATTERN),
        "api_key" => Some(API_KEY_PATTERN),
        _ => None,
    }
}

/// One authored deterministic rule, as handed over from the control plane.
#[derive(Debug, Clone, Deserialize)]
pub struct DetectorSpec {
    #[serde(default)]
    pub patterns: Vec<String>,
    #[serde(default)]
    pub builtin_patterns: Vec<String>,
    pub replacement: String,
}

/// Why one subject could not be inspected. Both variants fail closed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DetectorLimit {
    /// The subject is larger than the inspection ceiling.
    Text,
    /// The subject produced more matches than the ceiling allows.
    Matches,
}

impl DetectorLimit {
    /// The content-free message the python adapter raises for this bound.
    pub fn message(self) -> &'static str {
        match self {
            DetectorLimit::Text => "regex subject exceeds the 1 MiB inspection limit",
            DetectorLimit::Matches => "regex subject exceeds the match limit",
        }
    }
}

/// One compiled expression plus whether it needs Luhn candidate validation.
struct CompiledPattern {
    expression: Regex,
    card: bool,
}

/// A compiled authored rule: every expression plus the literal replacement.
pub struct Detector {
    patterns: Vec<CompiledPattern>,
    replacement: String,
}

/// Compile one expression under the shared program-size bound.
fn compile(pattern: &str) -> Result<Regex, String> {
    let translated = to_rust_syntax(pattern).map_err(|reason| reason.to_string())?;
    RegexBuilder::new(&translated)
        .size_limit(COMPILE_SIZE_LIMIT)
        .build()
        .map_err(|_| "expression is not supported by the native detector".to_string())
}

/// Whether a 13 to 19 digit candidate is a nonuniform valid Luhn number.
fn valid_card(digits: &[u8]) -> bool {
    if !(13..=19).contains(&digits.len()) {
        return false;
    }
    if digits.iter().all(|digit| *digit == digits[0]) {
        return false;
    }
    let mut total: u32 = 0;
    for (index, digit) in digits.iter().rev().enumerate() {
        let value = u32::from(*digit);
        let doubled = if index % 2 == 1 { value * 2 } else { value };
        total += if doubled > 9 { doubled - 9 } else { doubled };
    }
    total.is_multiple_of(10)
}

impl Detector {
    /// Compile one authored rule outside the request path.
    ///
    /// Returns a content-free reason when the rule is unusable natively. The
    /// caller then keeps the python adapter for that rule.
    pub fn compile(spec: &DetectorSpec) -> Result<Self, String> {
        if spec.patterns.len() > MAX_PATTERNS {
            return Err("too many custom expressions".to_string());
        }
        if spec.replacement.is_empty() || spec.replacement.chars().count() > 128 {
            return Err("replacement must contain 1 to 128 characters".to_string());
        }
        let mut patterns = Vec::with_capacity(spec.patterns.len() + spec.builtin_patterns.len());
        for pattern in &spec.patterns {
            if pattern.is_empty() || pattern.len() > MAX_PATTERN_BYTES {
                return Err("each expression must contain 1 to 1024 UTF-8 bytes".to_string());
            }
            patterns.push(CompiledPattern {
                expression: compile(pattern)?,
                card: false,
            });
        }
        for name in &spec.builtin_patterns {
            let pattern = builtin_pattern(name).ok_or("unknown builtin family")?;
            patterns.push(CompiledPattern {
                expression: compile(pattern)?,
                card: name == "credit_card",
            });
        }
        if patterns.is_empty() {
            return Err("rule has no expressions".to_string());
        }
        Ok(Self {
            patterns,
            replacement: spec.replacement.clone(),
        })
    }

    /// Return the rewritten subject, or `None` when nothing matched.
    ///
    /// Overlapping spans are merged before replacement so one rule cannot
    /// leak the tail of another rule's match.
    pub fn redact(&self, text: &str) -> Result<Option<String>, DetectorLimit> {
        let spans = self.spans(text)?;
        if spans.is_empty() {
            return Ok(None);
        }
        let mut rewritten = String::with_capacity(text.len());
        let mut position = 0usize;
        for (start, end) in spans {
            rewritten.push_str(&text[position..start]);
            rewritten.push_str(&self.replacement);
            position = end;
        }
        rewritten.push_str(&text[position..]);
        Ok(Some(rewritten))
    }

    /// Whether the subject matches at all, without building a replacement.
    pub fn matches(&self, text: &str) -> Result<bool, DetectorLimit> {
        Ok(!self.spans(text)?.is_empty())
    }

    /// Return the merged byte spans this rule redacts, in order.
    fn spans(&self, text: &str) -> Result<Vec<(usize, usize)>, DetectorLimit> {
        if text.len() > MAX_TEXT_BYTES {
            return Err(DetectorLimit::Text);
        }
        let bytes = text.as_bytes();
        let mut spans: Vec<(usize, usize)> = Vec::new();
        let mut matches = 0usize;
        for pattern in &self.patterns {
            for found in pattern.expression.find_iter(text) {
                matches += 1;
                if matches > MAX_MATCHES {
                    return Err(DetectorLimit::Matches);
                }
                let (start, end) = (found.start(), found.end());
                if start == end {
                    continue;
                }
                if !pattern.card {
                    spans.push((start, end));
                    continue;
                }
                self.card_spans(bytes, start, end, &mut matches, &mut spans)?;
            }
        }
        Ok(merge(spans))
    }

    /// Collect valid card spans inside one credit-card match.
    ///
    /// Spaces and hyphens separate cards as well as groups inside one card,
    /// so every 13 to 19 digit span that starts and ends on a group boundary
    /// is a candidate. An uninterrupted digit group is never split.
    fn card_spans(
        &self,
        bytes: &[u8],
        start: usize,
        end: usize,
        matches: &mut usize,
        spans: &mut Vec<(usize, usize)>,
    ) -> Result<(), DetectorLimit> {
        for first in start..end {
            if !bytes[first].is_ascii_digit() {
                continue;
            }
            if first > start && bytes[first - 1].is_ascii_digit() {
                continue;
            }
            let mut digits: Vec<u8> = Vec::with_capacity(19);
            for last in first..end.min(first + CARD_WINDOW) {
                if !bytes[last].is_ascii_digit() {
                    continue;
                }
                if digits.len() == 19 {
                    break;
                }
                digits.push(bytes[last] - b'0');
                let boundary =
                    last + 1 == end || bytes[last + 1] == b' ' || bytes[last + 1] == b'-';
                if digits.len() >= 13 && boundary {
                    *matches += 1;
                    if *matches > MAX_MATCHES {
                        return Err(DetectorLimit::Matches);
                    }
                    if valid_card(&digits) {
                        spans.push((first, last + 1));
                    }
                }
            }
        }
        Ok(())
    }
}

/// Sort and union overlapping or touching spans.
fn merge(mut spans: Vec<(usize, usize)>) -> Vec<(usize, usize)> {
    spans.sort_unstable();
    let mut merged: Vec<(usize, usize)> = Vec::with_capacity(spans.len());
    for (start, end) in spans {
        match merged.last_mut() {
            Some(last) if start <= last.1 => last.1 = last.1.max(end),
            _ => merged.push((start, end)),
        }
    }
    merged
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build one detector from custom expressions and built-in families.
    fn detector(patterns: &[&str], builtins: &[&str], replacement: &str) -> Detector {
        Detector::compile(&DetectorSpec {
            patterns: patterns.iter().map(|item| item.to_string()).collect(),
            builtin_patterns: builtins.iter().map(|item| item.to_string()).collect(),
            replacement: replacement.to_string(),
        })
        .expect("the rule compiles")
    }

    #[test]
    fn email_matches_are_replaced_literally() {
        let rule = detector(&[], &["email"], "[REDACTED]");
        assert_eq!(
            rule.redact("write to ada@example.com now").unwrap(),
            Some("write to [REDACTED] now".to_string())
        );
    }

    #[test]
    fn clean_text_is_left_alone() {
        let rule = detector(&[], &["email"], "[REDACTED]");
        assert_eq!(rule.redact("nothing to see").unwrap(), None);
    }

    #[test]
    fn multibyte_subjects_keep_their_code_points() {
        let rule = detector(&[], &["email"], "[REDACTED]");
        assert_eq!(
            rule.redact("naïve 🙂 ada@example.com 🙂 ok").unwrap(),
            Some("naïve 🙂 [REDACTED] 🙂 ok".to_string())
        );
    }

    #[test]
    fn valid_cards_are_redacted_in_both_formats() {
        let rule = detector(&[], &["credit_card"], "[CARD]");
        assert_eq!(
            rule.redact("pay 4111 1111 1111 1111 today").unwrap(),
            Some("pay [CARD] today".to_string())
        );
        assert_eq!(
            rule.redact("pay 4111111111111111 today").unwrap(),
            Some("pay [CARD] today".to_string())
        );
    }

    #[test]
    fn adjacent_cards_are_each_redacted() {
        let rule = detector(&[], &["credit_card"], "[CARD]");
        assert_eq!(
            rule.redact("4111111111111111 5500005555555559").unwrap(),
            Some("[CARD] [CARD]".to_string())
        );
    }

    #[test]
    fn invalid_luhn_uniform_and_overlong_numbers_are_kept() {
        let rule = detector(&[], &["credit_card"], "[CARD]");
        assert_eq!(rule.redact("4111111111111112").unwrap(), None);
        assert_eq!(rule.redact("0000000000000000").unwrap(), None);
        assert_eq!(rule.redact("41111111111111111111").unwrap(), None);
    }

    #[test]
    fn the_python_card_corpus_redacts_identically() {
        let rule = detector(&[], &["credit_card"], "[REDACTED]");
        let corpus = [
            ("4111 1111 1111 1111", Some("[REDACTED]")),
            ("4111-1111-1111-1111", Some("[REDACTED]")),
            ("4111111111111111", Some("[REDACTED]")),
            (
                "4111111111111111 5555555555554444",
                Some("[REDACTED] [REDACTED]"),
            ),
            (
                "4111 1111 1111 1111 5555 5555 5555 4444",
                Some("[REDACTED]"),
            ),
            (
                "4111-1111-1111-1111 378282246310005",
                Some("[REDACTED] [REDACTED]"),
            ),
            (
                "4111111111111111, 5555555555554444",
                Some("[REDACTED], [REDACTED]"),
            ),
            ("4111111111111112", None),
            ("0000000000000000", None),
            ("141111111111111111111", None),
            ("4111 1111 1111 1111 0000", Some("[REDACTED] 0000")),
        ];
        for (subject, expected) in corpus {
            assert_eq!(
                rule.redact(subject).unwrap(),
                expected.map(str::to_string),
                "subject shape differed"
            );
        }
    }

    #[test]
    fn dense_card_candidates_fail_closed() {
        let rule = detector(&[], &["credit_card"], "[REDACTED]");
        assert_eq!(
            rule.redact(&"4111 ".repeat(5000)),
            Err(DetectorLimit::Matches)
        );
    }

    #[test]
    fn the_replacement_is_literal_and_never_expands_a_group() {
        let rule = detector(&["abc", "cde"], &[], r"\1");
        assert_eq!(rule.redact("xabcdey").unwrap(), Some(r"x\1y".to_string()));
    }

    #[test]
    fn api_keys_are_redacted() {
        let rule = detector(&[], &["api_key"], "[KEY]");
        let subject = "use sk-abcdefghijklmnopqrstuvwx plus AKIAABCDEFGHIJKLMNOP";
        assert_eq!(
            rule.redact(subject).unwrap(),
            Some("use [KEY] plus [KEY]".to_string())
        );
    }

    #[test]
    fn overlapping_rules_merge_before_replacement() {
        let rule = detector(&["abcd", "cdef"], &[], "X");
        assert_eq!(rule.redact("abcdef").unwrap(), Some("X".to_string()));
    }

    #[test]
    fn custom_perl_classes_stay_ascii_only() {
        let rule = detector(&[r"\bid-\d+\b"], &[], "[ID]");
        assert_eq!(
            rule.redact("id-42 and id-7").unwrap(),
            Some("[ID] and [ID]".to_string())
        );
        // An arabic-indic digit is not an ASCII digit, so RE2 and this
        // detector both leave it alone.
        assert_eq!(rule.redact("id-٤٢").unwrap(), None);
    }

    #[test]
    fn oversized_subjects_and_dense_matches_fail_closed() {
        let rule = detector(&["a"], &[], "X");
        let oversized = "a".repeat(MAX_TEXT_BYTES + 1);
        assert_eq!(rule.redact(&oversized), Err(DetectorLimit::Text));
        let dense = "a".repeat(MAX_MATCHES + 1);
        assert_eq!(rule.redact(&dense), Err(DetectorLimit::Matches));
    }

    #[test]
    fn adversarial_nesting_completes() {
        let rule = detector(&["(a+)+$"], &[], "X");
        let subject = "a".repeat(40);
        assert!(rule.redact(&subject).unwrap().is_some());
    }

    #[test]
    fn unusable_rules_are_declined() {
        let spec = DetectorSpec {
            patterns: vec![r"(a)\1".to_string()],
            builtin_patterns: vec![],
            replacement: "X".to_string(),
        };
        assert!(Detector::compile(&spec).is_err());
        let unknown = DetectorSpec {
            patterns: vec![],
            builtin_patterns: vec!["nope".to_string()],
            replacement: "X".to_string(),
        };
        assert!(Detector::compile(&unknown).is_err());
    }

    #[test]
    fn matches_reports_without_rewriting() {
        let rule = detector(&[], &["email"], "[REDACTED]");
        assert!(rule.matches("ada@example.com").unwrap());
        assert!(!rule.matches("plain").unwrap());
    }
}
