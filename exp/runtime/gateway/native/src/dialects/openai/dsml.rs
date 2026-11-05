//! A guard that never relays DeepSeek DSML tool-call markup as answer text.
//!
//! DeepSeek models call tools in their own markup (DSML): a
//! `<｜DSML｜function_calls>` block of `<｜DSML｜invoke name="…">` elements,
//! each holding `<｜DSML｜parameter name="…" string="true|false">` values
//! (`｜` is U+FF5C). A vLLM server's tool parser turns well-formed DSML into
//! `tool_calls`; markup it does not recognise passes through as plain
//! `content`. At extreme context the model writes malformed DSML (a space
//! after `｜DSML｜`, `calls` for `function_calls`, a lost opening), and the
//! caller received the raw markup as text: the tool call was lost and the
//! markup landed in the agent's history, where it was imitated
//! (deepseek-v4.1-flash on self-hosted vLLM lanes, 2026-10-22).
//!
//! The guard sits between the OpenAI-compatible content deltas and the text
//! events. Ordinary text passes straight through; a `<` is held back only
//! while what follows could still open a DSML tag (optionally after
//! duplicated `<` or whitespace, optionally closing), and released the moment
//! it cannot, so code (`a < b`) and HTML (`<div>`) flow as before. Once a
//! DSML tag is confirmed, the span is captured until its closing
//! `calls`/`function_calls` tag (or the stream's end) and parsed leniently:
//! complete invokes become ordinary tool calls; anything else fails the
//! request with a typed `malformed_response`, and the markup is dropped.

use serde_json::{Map, Value};

/// The DSML namespace marker every tag carries after its `<` (or `</`).
const MARKER: &str = "｜DSML｜";

/// The most characters a candidate span may hold before the marker must
/// have appeared: bounds how long ordinary text after a `<` is held.
const CANDIDATE_LIMIT: usize = 64;

/// The most bytes one captured DSML span may hold (tool arguments included).
pub(crate) const SPAN_LIMIT: usize = 1 << 20;

/// The caller-facing reason when captured markup cannot be turned into calls.
pub(crate) const MALFORMED_TOOL_CALL: &str =
    "the model emitted a malformed tool call; retry the request";

/// Which DSML parse step refused a captured span. Logged as a content-free
/// reason code (never the markup) so a lane failing with
/// `MALFORMED_TOOL_CALL` can be diagnosed from the worker log: the caller
/// still sees only the one malformed-tool-call reason.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Reject {
    /// The captured span outgrew `SPAN_LIMIT`.
    SpanTooLarge,
    /// Something other than a tag where a tag was expected.
    ExpectedTag,
    /// A tag opener without the DSML marker.
    MissingMarker,
    /// A tag that never closed with `>`.
    UnclosedTag,
    /// A tag with no name.
    EmptyTagName,
    /// An attribute without `=`.
    AttributeMissingEquals,
    /// An attribute value not opened with `"`.
    AttributeUnquoted,
    /// An attribute value never closed with `"`.
    AttributeUnterminated,
    /// A parameter value with no closing parameter tag.
    ParameterUnclosed,
    /// Where an invoke was expected, a closing or other tag.
    ExpectedInvoke,
    /// An invoke without a non-empty `name`.
    InvokeNameMissing,
    /// Inside an invoke, a tag that is neither a parameter nor its close.
    ExpectedParameter,
    /// A parameter without a non-empty `name`.
    ParameterNameMissing,
    /// A `string="false"` parameter whose value is not JSON.
    ParameterNotJson,
    /// Text after the outer block's closing tag.
    TrailingAfterClose,
}

impl Reject {
    /// The stable reason code written to the log.
    pub(crate) fn code(self) -> &'static str {
        match self {
            Self::SpanTooLarge => "span_too_large",
            Self::ExpectedTag => "expected_tag",
            Self::MissingMarker => "missing_marker",
            Self::UnclosedTag => "unclosed_tag",
            Self::EmptyTagName => "empty_tag_name",
            Self::AttributeMissingEquals => "attribute_missing_equals",
            Self::AttributeUnquoted => "attribute_unquoted",
            Self::AttributeUnterminated => "attribute_unterminated",
            Self::ParameterUnclosed => "parameter_unclosed",
            Self::ExpectedInvoke => "expected_invoke",
            Self::InvokeNameMissing => "invoke_name_missing",
            Self::ExpectedParameter => "expected_parameter",
            Self::ParameterNameMissing => "parameter_name_missing",
            Self::ParameterNotJson => "parameter_not_json",
            Self::TrailingAfterClose => "trailing_after_close",
        }
    }
}

/// Log one refused span by step and size only, then answer the caller-facing
/// reason. `closed` says whether the outer closing tag was seen (false: the
/// span was still open at the stream's end); `calls` counts the invokes that
/// parsed before the refusal.
fn reject(step: Reject, span_bytes: usize, closed: bool, calls: usize) -> &'static str {
    let line = serde_json::json!({
        "event": "dsml_tool_call_rejected",
        "step": step.code(),
        "span_bytes": span_bytes,
        "closed": closed,
        "calls_parsed": calls,
    });
    eprintln!("exp-gateway-native: {line}");
    MALFORMED_TOOL_CALL
}

/// One tool call assembled from a complete DSML invoke.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct DsmlCall {
    pub(crate) name: String,
    pub(crate) arguments: Map<String, Value>,
}

/// What the guard hands back for one piece of content.
#[derive(Debug, Default, PartialEq)]
pub(crate) struct Released {
    /// Text that can no longer be markup, in order.
    pub(crate) text: String,
    /// Calls parsed from a closed DSML block, after `text`.
    pub(crate) calls: Vec<DsmlCall>,
    /// Text that followed a closed DSML block, after `calls`.
    pub(crate) trailing_text: String,
}

/// How a candidate span stands toward the marker.
#[derive(Debug, PartialEq, Eq)]
enum Candidate {
    /// Still a possible prefix of a DSML tag opening.
    Prefix,
    /// A DSML tag opening is confirmed.
    Marker,
    /// It can no longer open a DSML tag.
    Not,
}

/// Judge `span` (which starts with `<`): `<`, then any run of `<` and
/// whitespace, then an optional `/` and whitespace, then the marker.
fn candidate(span: &str) -> Candidate {
    let mut rest = span;
    if !rest.starts_with('<') {
        return Candidate::Not;
    }
    rest = rest.trim_start_matches(|c: char| c == '<' || c.is_whitespace());
    if let Some(after) = rest.strip_prefix('/') {
        rest = after.trim_start();
    }
    if rest.starts_with(MARKER) {
        Candidate::Marker
    } else if MARKER.starts_with(rest) && span.chars().count() <= CANDIDATE_LIMIT {
        Candidate::Prefix
    } else {
        Candidate::Not
    }
}

/// The incremental holdback and capture state for one stream's content.
#[derive(Debug, Default)]
pub(crate) struct DsmlGuard {
    /// Text after a `<` that may still open a DSML tag.
    candidate: String,
    /// A confirmed DSML span being captured, `None` outside one.
    span: Option<String>,
}

impl DsmlGuard {
    /// Whether anything is held back (a candidate or a captured span).
    pub(crate) fn holding(&self) -> bool {
        !self.candidate.is_empty() || self.span.is_some()
    }

    /// Feed one content delta; returns what can be released now.
    ///
    /// Errors with the malformed-tool-call reason when a closed DSML block
    /// does not parse, or when a span outgrows `SPAN_LIMIT`.
    pub(crate) fn push(&mut self, text: &str) -> Result<Released, &'static str> {
        let mut released = Released::default();
        for character in text.chars() {
            if let Some(span) = self.span.as_mut() {
                span.push(character);
                if span.len() > SPAN_LIMIT {
                    return Err(reject(Reject::SpanTooLarge, span.len(), false, 0));
                }
                // Both outer closings end in `calls>`: only then is the
                // (normalizing) block check worth running.
                if character == '>' && span.ends_with("calls>") && closes_block(span) {
                    let block = self.span.take().unwrap_or_default();
                    released.calls.extend(parse_block(&block, true)?);
                }
                continue;
            }
            let output = if released.calls.is_empty() {
                &mut released.text
            } else {
                &mut released.trailing_text
            };
            if self.candidate.is_empty() {
                if character == '<' {
                    self.candidate.push(character);
                } else {
                    output.push(character);
                }
                continue;
            }
            self.candidate.push(character);
            self.settle_candidate(output);
        }
        Ok(released)
    }

    /// Re-judge the candidate after a character was appended: release what
    /// cannot be markup, keep a prefix, or open a span on a confirmed marker.
    fn settle_candidate(&mut self, released: &mut String) {
        loop {
            match candidate(&self.candidate) {
                Candidate::Prefix => return,
                Candidate::Marker => {
                    self.span = Some(std::mem::take(&mut self.candidate));
                    return;
                }
                Candidate::Not => {
                    // Release up to the next `<` (which may start a fresh
                    // candidate) and judge what follows it again.
                    match self.candidate[1..].find('<').map(|at| at + 1) {
                        Some(at) => {
                            released.push_str(&self.candidate[..at]);
                            self.candidate = self.candidate[at..].to_string();
                        }
                        None => {
                            released.push_str(&std::mem::take(&mut self.candidate));
                            return;
                        }
                    }
                }
            }
        }
    }

    /// End of content: release a candidate that never became markup; an open
    /// span must parse as a whole block or the request fails.
    pub(crate) fn finish(&mut self) -> Result<Released, &'static str> {
        let mut released = Released {
            text: std::mem::take(&mut self.candidate),
            ..Released::default()
        };
        if let Some(span) = self.span.take() {
            released.calls = parse_block(&span, false)?;
        }
        Ok(released)
    }
}

/// Normalize the lenient spellings inside each tag opener: whitespace between
/// `<` (or `</`) and the marker, and whitespace after the marker. Text
/// outside tag openers (parameter values) is never touched.
fn normalize(span: &str) -> String {
    let mut out = String::with_capacity(span.len());
    let mut rest = span;
    while let Some(at) = rest.find(MARKER) {
        let (head, tail) = rest.split_at(at);
        let opener = head.trim_end();
        let (opener, closing) = match opener.strip_suffix('/') {
            Some(before) => (before.trim_end(), true),
            None => (opener, false),
        };
        match opener.strip_suffix('<') {
            Some(before) => {
                out.push_str(before);
                out.push_str(if closing { "</" } else { "<" });
            }
            None => out.push_str(head),
        }
        out.push_str(MARKER);
        rest = tail[MARKER.len()..].trim_start();
    }
    out.push_str(rest);
    out
}

/// Whether `span` ends with the outer block's closing tag.
fn closes_block(span: &str) -> bool {
    let normalized = normalize(span);
    let normalized = normalized.trim_end();
    normalized.ends_with("</｜DSML｜function_calls>") || normalized.ends_with("</｜DSML｜calls>")
}

/// One parsed DSML tag.
struct Tag {
    closing: bool,
    name: String,
    attributes: Vec<(String, String)>,
}

/// A cursor over normalized DSML.
struct Cursor<'a> {
    rest: &'a str,
}

impl Cursor<'_> {
    /// The next tag, after whitespace and duplicated `<`; anything else is
    /// malformed.
    fn tag(&mut self) -> Result<Tag, Reject> {
        let mut rest = self
            .rest
            .trim_start()
            .strip_prefix('<')
            .ok_or(Reject::ExpectedTag)?;
        while let Some(again) = rest.trim_start().strip_prefix('<') {
            rest = again;
        }
        let (closing, rest) = match rest.strip_prefix('/') {
            Some(rest) => (true, rest),
            None => (false, rest),
        };
        let rest = rest.strip_prefix(MARKER).ok_or(Reject::MissingMarker)?;
        let end = rest.find('>').ok_or(Reject::UnclosedTag)?;
        let (body, after) = (&rest[..end], &rest[end + 1..]);
        let name_end = body
            .find(|c: char| !(c.is_ascii_alphanumeric() || c == '_'))
            .unwrap_or(body.len());
        let name = body[..name_end].to_string();
        if name.is_empty() {
            return Err(Reject::EmptyTagName);
        }
        let mut attributes = Vec::new();
        let mut attrs = body[name_end..].trim_start();
        while !attrs.is_empty() {
            let eq = attrs.find('=').ok_or(Reject::AttributeMissingEquals)?;
            let key = attrs[..eq].trim().to_string();
            let value = attrs[eq + 1..]
                .trim_start()
                .strip_prefix('"')
                .ok_or(Reject::AttributeUnquoted)?;
            let close = value.find('"').ok_or(Reject::AttributeUnterminated)?;
            attributes.push((key, value[..close].to_string()));
            attrs = value[close + 1..].trim_start();
        }
        self.rest = after;
        Ok(Tag {
            closing,
            name,
            attributes,
        })
    }

    /// The raw parameter value up to its closing tag, consuming the tag.
    fn value(&mut self) -> Result<&str, Reject> {
        const CLOSE: &str = "</｜DSML｜parameter>";
        let end = self.rest.find(CLOSE).ok_or(Reject::ParameterUnclosed)?;
        let value = &self.rest[..end];
        self.rest = &self.rest[end + CLOSE.len()..];
        Ok(value)
    }
}

/// Whether `name` is the outer block's tag.
fn is_calls(name: &str) -> bool {
    matches!(name, "function_calls" | "calls")
}

/// The attribute named `key`, if present.
fn attribute<'a>(tag: &'a Tag, key: &str) -> Option<&'a str> {
    tag.attributes
        .iter()
        .find(|(name, _)| name == key)
        .map(|(_, value)| value.as_str())
}

/// Parse one captured DSML block into calls: an optional opening
/// `calls`/`function_calls`, one or more complete invokes, and the closing
/// outer tag (or the span's end). Every invoke needs a name and every
/// parameter a name and a closing tag; `string="false"` values must be JSON.
/// Anything else (a lost opening, a stray closing tag) is malformed: a
/// partial call set is never executed.
///
/// A refusal logs its parse step (`reject`); `closed` says whether the outer
/// closing tag ended the span.
fn parse_block(raw: &str, closed: bool) -> Result<Vec<DsmlCall>, &'static str> {
    let mut calls = Vec::new();
    parse_steps(raw, &mut calls).map_err(|step| reject(step, raw.len(), closed, calls.len()))?;
    Ok(calls)
}

/// The parse proper; completed invokes accumulate in `calls`.
fn parse_steps(raw: &str, calls: &mut Vec<DsmlCall>) -> Result<(), Reject> {
    let normalized = normalize(raw);
    let mut cursor = Cursor { rest: &normalized };
    let mut tag = cursor.tag()?;
    if !tag.closing && is_calls(&tag.name) {
        tag = cursor.tag()?;
    }
    loop {
        if tag.closing || tag.name != "invoke" {
            return Err(Reject::ExpectedInvoke);
        }
        let name = attribute(&tag, "name")
            .filter(|name| !name.is_empty())
            .ok_or(Reject::InvokeNameMissing)?
            .to_string();
        let mut arguments = Map::new();
        loop {
            let next = cursor.tag()?;
            if next.closing && next.name == "invoke" {
                break;
            }
            if next.closing || next.name != "parameter" {
                return Err(Reject::ExpectedParameter);
            }
            let key = attribute(&next, "name")
                .filter(|key| !key.is_empty())
                .ok_or(Reject::ParameterNameMissing)?
                .to_string();
            let raw_value = cursor.value()?;
            let value = match attribute(&next, "string") {
                Some("true") => Value::String(raw_value.to_string()),
                Some("false") => {
                    serde_json::from_str(raw_value.trim()).map_err(|_| Reject::ParameterNotJson)?
                }
                _ => serde_json::from_str(raw_value.trim())
                    .unwrap_or_else(|_| Value::String(raw_value.to_string())),
            };
            arguments.insert(key, value);
        }
        calls.push(DsmlCall { name, arguments });
        if cursor.rest.trim().is_empty() {
            return Ok(());
        }
        tag = cursor.tag()?;
        if tag.closing && is_calls(&tag.name) {
            return if cursor.rest.trim().is_empty() {
                Ok(())
            } else {
                Err(Reject::TrailingAfterClose)
            };
        }
    }
}

#[cfg(test)]
#[path = "dsml_tests.rs"]
mod tests;
