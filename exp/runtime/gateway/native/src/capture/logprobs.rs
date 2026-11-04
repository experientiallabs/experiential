//! Gateway-requested Chat token probabilities, retained for capture only.
//!
//! When the host enables `capture_logprobs`, an eligible OpenAI-compatible
//! rung of a captured request is dialed with `logprobs: true` although the
//! caller never asked. The normalizer then parses the provider's probabilities
//! into a side buffer instead of the event stream, so what the caller receives
//! (bytes, usage, settlement) is exactly the no-injection answer; only the
//! winning attempt's buffer reaches the capture record. A provider that refuses
//! the field is re-dialed once without it inside the same physical attempt, and
//! a rung whose plain re-dial then opens is never injected again by this worker.

use std::collections::HashSet;
use std::sync::{LazyLock, Mutex};

use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::errors::{Failure, FailureClass};
use crate::logprobs::{ChoiceLogprobsDelta, TokenLogprob};

/// Encoded-size bound on one attempt's retained probabilities; beyond it the
/// buffer stops growing and the capture record says it was truncated.
pub(crate) const MAXIMUM_CAPTURED_LOGPROB_BYTES: usize = 4 * 1024 * 1024;

/// Bound on probabilities a relay holds before its attempt wins: commitment
/// normally follows the first text frame, so only a short withheld prefix
/// (refusal failover, metadata) can accumulate here, outside the collector.
pub(crate) const MAXIMUM_HELD_LOGPROB_BYTES: usize = 64 * 1024;

/// Room left for the record's request scope, metrics and keys when sizing
/// retained probabilities against the destination's record ceiling.
pub(crate) const RECORD_ENVELOPE_BYTES: usize = 16 * 1024;

/// Probabilities describe the whole answer, so they are kept only beside a
/// complete, successful captured response: none, a failed exchange, a
/// truncated prefix or a stream the client left early drops them. A
/// response still held as wire bytes is judged once delivery decodes it.
/// Returns the bytes released from the entry's charge.
pub(crate) fn align_with_response(record: &mut super::record::Record, wire_pending: bool) -> usize {
    if wire_pending {
        return 0;
    }
    let complete = match &record.response {
        Some(super::record::Response::Json { status, body, .. }) => {
            (200..300).contains(status) && !reports_failure(body)
        }
        Some(super::record::Response::Sse {
            status,
            frames,
            truncated,
            client_disconnected,
            ..
        }) => {
            (200..300).contains(status)
                && !truncated
                && !client_disconnected
                && !frames.iter().any(reports_failure)
        }
        // Frames are on disk; the stream facts decide, as the relay saw them.
        Some(super::record::Response::Spilled(spilled)) => {
            (200..300).contains(&spilled.status)
                && !spilled.truncated
                && !spilled.client_disconnected
        }
        None => false,
    };
    if complete {
        return 0;
    }
    release(&mut record.provider_logprobs)
}

/// A protocol body or frame that reports a failure (an `error` object, or a
/// Responses/Messages error or failed event) rather than an answer.
fn reports_failure(value: &Value) -> bool {
    value.get("error").is_some_and(|error| !error.is_null())
        || matches!(
            value.get("type").and_then(Value::as_str),
            Some("error" | "response.failed")
        )
}

/// Clear a sidecar and return the bytes charged for it, so the caller can
/// release them from the entry's pending charge.
pub(crate) fn release(sidecar: &mut Option<Captured>) -> usize {
    sidecar.take().map_or(0, |captured| captured.heap_bytes())
}

/// Whether any rung of this admitted route could be dialed with capture-only
/// probabilities; a route with none never reserves a sidecar.
pub(crate) fn route_may_inject(route: &[crate::waterfall::DeploymentWire]) -> bool {
    route.iter().any(|wire| {
        wire.capture_logprobs
            && !wire.billing_customer_managed
            && wire.dialect == "openai_compatible"
            && wire.upstream_body.is_none()
            && !refused(&rung_key(wire))
    })
}

/// Room the encoded record leaves for probabilities after the full request,
/// the largest admissible response and the record envelope.
pub(crate) fn record_room(maximum_record: usize, request: usize, response: usize) -> usize {
    maximum_record
        .saturating_sub(request)
        .saturating_sub(response)
        .saturating_sub(RECORD_ENVELOPE_BYTES)
}

/// Bound on remembered refusing rungs so a churning catalog cannot grow it.
const MAXIMUM_REFUSED_RUNGS: usize = 4096;

static REFUSED: LazyLock<Mutex<HashSet<String>>> = LazyLock::new(Default::default);

/// The rung's identity for refusal memory: a catalog-local deployment id is not
/// unique across catalogs or connections, so the endpoint, wire model and a
/// digest of the dial headers (credential and connection) join it. Only the
/// digest is kept, never a header value.
pub(crate) fn rung_key(wire: &crate::waterfall::DeploymentWire) -> String {
    use sha2::{Digest, Sha256};
    let mut headers: Vec<_> = wire.headers.iter().collect();
    headers.sort();
    let mut digest = Sha256::new();
    for (name, value) in headers {
        digest.update(name.as_bytes());
        digest.update([0]);
        digest.update(value.as_bytes());
        digest.update([0]);
    }
    format!(
        "{}\n{}\n{}\n{:x}",
        wire.deployment_id,
        wire.url,
        wire.model_id,
        digest.finalize()
    )
}

/// Whether this worker has seen the rung refuse injected probabilities.
pub(crate) fn refused(rung: &str) -> bool {
    REFUSED
        .lock()
        .map(|refused| refused.contains(rung))
        .unwrap_or(true)
}

/// Stop injecting on a rung whose plain re-dial opened after a refusal.
pub(crate) fn remember_refusal(rung: &str) {
    if let Ok(mut refused) = REFUSED.lock() {
        // A full memory starts over rather than ignoring current refusals:
        // forgotten rungs cost one more plain re-dial each, never an error.
        if refused.len() >= MAXIMUM_REFUSED_RUNGS {
            refused.clear();
        }
        refused.insert(rung.to_owned());
    }
}

/// The dialed payload with probabilities requested, or `None` when the payload
/// is not an object or already carries a caller probability control.
pub(crate) fn with_logprobs(payload: &Value) -> Option<Value> {
    let object = payload.as_object()?;
    if object.contains_key("logprobs") || object.contains_key("top_logprobs") {
        return None;
    }
    let mut injected = object.clone();
    injected.insert("logprobs".to_owned(), Value::Bool(true));
    Some(Value::Object(injected))
}

/// A refused open that a plain re-dial might answer: only request-shaped
/// refusals not attributed to something else (an encrypted replay, another
/// named parameter), never credential, quota, throttle or transport failures.
pub(crate) fn may_be_refusal(failure: &Failure) -> bool {
    matches!(
        failure.failure_class,
        FailureClass::InvalidRequest | FailureClass::UnsupportedCapability
    ) && !failure.encrypted_reasoning_rejected
        // A refusal the provider pins on another parameter is not ours.
        && failure
            .rejected_parameter
            .as_deref()
            .is_none_or(|param| param == "logprobs" || param == "top_logprobs")
}

/// The text one frame emits on each channel (empty when none), and the frame
/// facts that decide where tokens which emit no text may sit.
#[derive(Clone, Debug, Default)]
pub(crate) struct FrameOutput {
    pub content: String,
    pub refusal: String,
    /// Reasoning text the provider emitted in this frame (`reasoning_content`
    /// or `reasoning`). A self-hosted reasoning model (vLLM with a reasoning
    /// parser) reports these tokens' probabilities on the `content` channel,
    /// before the answer's.
    pub reasoning: String,
    /// The frame carries a finish reason: tokens after its text (a stop or
    /// end-of-turn token the provider does not render) end the sequence.
    pub finishes: bool,
    /// The frame emits output this check cannot spell (tool or function call
    /// arguments): any records on it are a hole, as before.
    pub opaque: bool,
}

fn record_bytes(record: &TokenLogprob) -> &[u8] {
    record.bytes.as_deref().unwrap_or(record.token.as_bytes())
}

/// Bytes the records may run ahead of the emitted text. An OpenAI-compatible
/// server detokenizes incrementally and may hold back a token's text (a quote,
/// a space, a partial UTF-8 sequence) until the next frame, so alignment is
/// judged on the whole sequence, not frame by frame; a lead past this bound is
/// text the gateway never saw, a hole.
const MAXIMUM_RECORD_LEAD_BYTES: usize = 1024;

/// How one capture-only frame's records relate to the text emitted so far.
#[derive(Debug, PartialEq, Eq)]
enum Alignment {
    /// The sequence still spells the emitted text: keep the frame.
    Retain,
    /// No records and no text that needed them.
    Empty,
    /// A missing, surplus or wrong-channel record: retention ends.
    Hole,
}

/// The attempt's alignment state: record bytes not yet matched by emitted
/// text, token by token, and where in the answer the sequence stands.
#[derive(Debug, Default)]
pub(crate) struct Position {
    pending: std::collections::VecDeque<Vec<u8>>,
    /// Bytes of the front pending token already matched.
    offset: usize,
    pending_bytes: usize,
    /// Answer (content) text has been emitted.
    answer_started: bool,
    /// The one silent delimiter before the answer has been skipped.
    delimiter_seen: bool,
    /// Whether this provider prices reasoning tokens (decided by the first
    /// reasoning frame: records present or not).
    reasoning_priced: Option<bool>,
}

impl Position {
    /// Match `text` against the front of the pending record bytes. With
    /// `delimiter`, one whole token that does not start the text may be
    /// skipped first (a template's end-of-thinking token, rendered nowhere).
    fn consume(&mut self, text: &[u8], delimiter: bool) -> bool {
        let mut position = 0;
        let mut may_skip = delimiter;
        while position < text.len() {
            let Some(front) = self.pending.front() else {
                // Text the records never spelled.
                return false;
            };
            let rest = &front[self.offset..];
            let take = rest.len().min(text.len() - position);
            if rest[..take] != text[position..position + take] {
                if may_skip && self.offset == 0 && position == 0 {
                    let skipped = self.pending.pop_front().map_or(0, |token| token.len());
                    self.pending_bytes -= skipped;
                    self.delimiter_seen = true;
                    may_skip = false;
                    continue;
                }
                return false;
            }
            position += take;
            self.offset += take;
            if self.offset == front.len() {
                let done = self.pending.pop_front().map_or(0, |token| token.len());
                self.pending_bytes -= done;
                self.offset = 0;
            }
        }
        true
    }

    /// Align one frame: append its records, then match its reasoning text and
    /// its answer text, in that order. Tokens that emit no text are accepted at
    /// two places only: once before the first answer text (the delimiter), and
    /// still unmatched when the frame finishes (stop or end-of-turn tokens).
    fn align(&mut self, delta: &ChoiceLogprobsDelta, output: &FrameOutput) -> Alignment {
        let (content, refusal) = delta.logprobs.as_ref().map_or((None, None), |logprobs| {
            (logprobs.content.as_deref(), logprobs.refusal.as_deref())
        });
        let records = content.unwrap_or(&[]);
        if refusal.is_some_and(|r| !r.is_empty()) || (output.opaque && !records.is_empty()) {
            // Refusal records beside answer text are on the wrong channel; tool
            // call arguments are output this check cannot spell.
            return Alignment::Hole;
        }
        if !output.reasoning.is_empty() && self.reasoning_priced.is_none() {
            self.reasoning_priced = Some(!records.is_empty());
        }
        if records.is_empty() && output.content.is_empty() {
            // Reasoning text alone is matched only when the provider prices it.
            if self.reasoning_priced == Some(true) && !output.reasoning.is_empty() {
                return if self.consume(output.reasoning.as_bytes(), false) {
                    Alignment::Empty
                } else {
                    Alignment::Hole
                };
            }
            return Alignment::Empty;
        }
        for record in records {
            let bytes = record_bytes(record);
            if !bytes.is_empty() {
                self.pending_bytes += bytes.len();
                self.pending.push_back(bytes.to_vec());
            }
        }
        if self.reasoning_priced == Some(true) && !self.consume(output.reasoning.as_bytes(), false)
        {
            return Alignment::Hole;
        }
        if !output.content.is_empty() {
            let delimiter = !self.answer_started && !self.delimiter_seen;
            if !self.consume(output.content.as_bytes(), delimiter) {
                return Alignment::Hole;
            }
            self.answer_started = true;
        }
        if output.finishes {
            self.pending.clear();
            self.pending_bytes = 0;
            self.offset = 0;
        } else if self.pending_bytes > MAXIMUM_RECORD_LEAD_BYTES {
            return Alignment::Hole;
        }
        if records.is_empty() {
            Alignment::Empty
        } else {
            Alignment::Retain
        }
    }
}

/// How a normalizer treats Chat probabilities on its wire.
#[derive(Debug, Default)]
pub(crate) enum ChatProbabilities {
    /// Not requested: any probabilities the provider sends are ignored.
    #[default]
    Off,
    /// The caller asked: probabilities are events of the caller's answer.
    Caller,
    /// The gateway asked: probabilities go to this side buffer only.
    Capture(Buffer),
}

impl ChatProbabilities {
    pub(crate) fn requested(by_caller: bool) -> Self {
        if by_caller {
            Self::Caller
        } else {
            Self::Off
        }
    }

    pub(crate) fn requested_by_caller(&self) -> bool {
        matches!(self, Self::Caller)
    }

    /// Whether capture-only probabilities are being retained (the decoder does
    /// no per-frame work for them otherwise).
    pub(crate) fn captures(&self) -> bool {
        matches!(self, Self::Capture(_))
    }

    /// Retain a capture-only choice channel. A frame that carries output text
    /// but no usable probabilities, or a malformed channel, leaves a hole: the
    /// prefix is kept and marked incomplete.
    pub(crate) fn retain(
        &mut self,
        value: Option<&Value>,
        choice_index: u32,
        output: &FrameOutput,
    ) {
        if let Self::Capture(buffer) = self {
            match crate::logprobs::parse(value, choice_index) {
                // A frame enters the record only when it is aligned with the
                // captured answer; otherwise retention ends before it. Refusal
                // text is not rendered on every surface (Messages drops it), so
                // a refusal frame always ends retention.
                Ok(parsed) if output.refusal.is_empty() => {
                    let delta = parsed.unwrap_or(ChoiceLogprobsDelta {
                        choice_index,
                        logprobs: None,
                    });
                    match buffer.position.align(&delta, output) {
                        Alignment::Retain => buffer.push(delta),
                        Alignment::Empty => {}
                        Alignment::Hole => buffer.truncated = true,
                    }
                }
                Ok(_) => buffer.truncated = true,
                Err(_) => buffer.truncated = true,
            }
        }
    }

    pub(crate) fn take(&mut self) -> Option<(Vec<ChoiceLogprobsDelta>, bool)> {
        match self {
            Self::Capture(buffer) => Some(buffer.take()),
            _ => None,
        }
    }
}

/// The relay's hold on one dial's capture-only probabilities: deltas wait here
/// until the winning attempt's capture observer attaches.
#[derive(Debug, Default)]
pub(crate) struct Held {
    pub injected: bool,
    pub pending: Vec<ChoiceLogprobsDelta>,
    pending_bytes: usize,
    truncated: bool,
}

impl Held {
    /// Forward newly parsed deltas to the observer, or hold them until it attaches.
    pub(crate) fn forward(
        &mut self,
        taken: Option<(Vec<ChoiceLogprobsDelta>, bool)>,
        observer: Option<&super::reasoning::Observer>,
    ) {
        let Some((deltas, truncated)) = taken else {
            return;
        };
        match observer {
            Some(observer) => {
                self.truncated |= truncated;
                observer.logprobs(deltas, self.truncated);
            }
            None => {
                // Held outside the collector's budget until the winner attaches,
                // so the hold has its own small bound; beyond it, truncate.
                for delta in deltas {
                    let bytes = delta.retained_bytes();
                    if self.truncated
                        || self.pending_bytes.saturating_add(bytes) > MAXIMUM_HELD_LOGPROB_BYTES
                    {
                        self.truncated = true;
                        break;
                    }
                    self.pending_bytes += bytes;
                    self.pending.push(delta);
                }
                // The incoming mark only ends the prefix kept above.
                self.truncated |= truncated;
            }
        }
    }

    /// Mark the winning record and hand it everything held so far.
    pub(crate) fn attach(&mut self, observer: &super::reasoning::Observer) {
        if self.injected {
            observer.logprobs_injected();
            observer.logprobs(std::mem::take(&mut self.pending), self.truncated);
        }
    }
}

/// One attempt's bounded side buffer, filled by the normalizer.
#[derive(Debug, Default)]
pub(crate) struct Buffer {
    pub deltas: Vec<ChoiceLogprobsDelta>,
    pub bytes: usize,
    pub truncated: bool,
    /// Alignment state across the attempt's frames.
    pub position: Position,
}

impl Buffer {
    /// Retain one parsed delta unless it would exceed the attempt bound.
    pub(crate) fn push(&mut self, delta: ChoiceLogprobsDelta) {
        if self.truncated {
            return;
        }
        let bytes = delta.retained_bytes();
        if self.bytes.saturating_add(bytes) > MAXIMUM_CAPTURED_LOGPROB_BYTES {
            self.truncated = true;
            return;
        }
        self.bytes += bytes;
        self.deltas.push(delta);
    }

    /// Hand over the retained deltas; the byte bound stays cumulative for the attempt.
    pub(crate) fn take(&mut self) -> (Vec<ChoiceLogprobsDelta>, bool) {
        (std::mem::take(&mut self.deltas), self.truncated)
    }
}

/// Replace NUL in token texts (record and alternatives); report whether any was.
fn storable(record: &mut TokenLogprob) -> bool {
    let mut replaced = false;
    let mut clean = |text: &mut String| {
        if text.contains('\0') {
            *text = text.replace('\0', "\u{fffd}");
            replaced = true;
        }
    };
    clean(&mut record.token);
    for candidate in &mut record.top_logprobs {
        clean(&mut candidate.token);
    }
    replaced
}

/// The capture record's probabilities: present only when the gateway requested
/// them, so a caller-requested answer (already in the response frames) and every
/// record without injection serialize exactly as before.
#[derive(Clone, Debug, Default, Deserialize, Serialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub(crate) struct Captured {
    /// Always true: the caller never asked for these probabilities.
    pub logprobs_injected: bool,
    /// A bound (the attempt's or the collector's) stopped retaining them.
    pub truncated: bool,
    pub content: Vec<TokenLogprob>,
    pub refusal: Vec<TokenLogprob>,
    /// A token text carried NUL, which JSONB cannot store; it was replaced with
    /// U+FFFD in the text, and the record's `bytes` (when the provider sent
    /// them) keep the exact token.
    #[serde(default)]
    pub nul_replaced: bool,
    /// Retained-size estimate charged against the collector's budgets.
    #[serde(skip)]
    pub bytes: usize,
}

impl Captured {
    #[cfg(test)]
    pub(crate) fn injected() -> Self {
        Self {
            logprobs_injected: true,
            ..Self::default()
        }
    }

    pub(crate) fn heap_bytes(&self) -> usize {
        std::mem::size_of::<Self>() + self.bytes
    }

    /// Append deltas while `allowed(extra)` admits their size; otherwise mark truncated.
    pub(crate) fn extend(
        &mut self,
        deltas: Vec<ChoiceLogprobsDelta>,
        truncated: bool,
        mut allowed: impl FnMut(usize) -> bool,
    ) -> usize {
        let mut added = 0usize;
        // Keep the prefix that fits; an upstream truncation only marks the end.
        for delta in deltas {
            if self.truncated {
                break;
            }
            let bytes = delta.retained_bytes();
            if !allowed(added.saturating_add(bytes)) {
                self.truncated = true;
                break;
            }
            added += bytes;
            if let Some(logprobs) = delta.logprobs {
                for (channel, records) in [
                    (&mut self.content, logprobs.content),
                    (&mut self.refusal, logprobs.refusal),
                ] {
                    for mut record in records.unwrap_or_default() {
                        self.nul_replaced |= storable(&mut record);
                        channel.push(record);
                    }
                }
            }
        }
        self.bytes += added;
        self.truncated |= truncated;
        added
    }
}

#[cfg(test)]
#[path = "logprobs_test.rs"]
mod tests;
