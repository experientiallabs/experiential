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
    /// or `reasoning`, then typed thinking parts). A self-hosted reasoning model (vLLM with a reasoning
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

/// Every reading of the attempt's sequence still consistent with the emitted
/// text. Two decisions cannot always be made on the frame that raises them
/// (whether reasoning is priced, and whether an answer opening with a prefix
/// of the front token skips it as the delimiter), so each keeps both readings
/// until later text rules one out. Each decision is taken at most once per
/// stream, so at most four readings exist; identical readings merge.
#[derive(Debug)]
pub(crate) struct Readings(Vec<Position>);

impl Default for Readings {
    fn default() -> Self {
        Self(vec![Position::default()])
    }
}

impl Readings {
    /// Align one frame under every reading, keeping those it does not break.
    /// The frame is a hole only when no reading survives it.
    fn align(&mut self, delta: &ChoiceLogprobsDelta, output: &FrameOutput) -> Alignment {
        let mut survivors: Vec<Position> = Vec::new();
        let mut alignment = Alignment::Hole;
        for reading in std::mem::take(&mut self.0) {
            for (candidate, result) in reading.branches(delta, output) {
                if result != Alignment::Hole && !survivors.contains(&candidate) {
                    alignment = result;
                    survivors.push(candidate);
                }
            }
        }
        self.0 = survivors;
        alignment
    }

    /// Whether every surviving reading still holds records no text spelled.
    fn all_pending(&self) -> bool {
        self.0.iter().all(|reading| !reading.pending.is_empty())
    }
}

/// One reading's alignment state: record bytes not yet matched by emitted
/// text, token by token, and where in the answer the sequence stands.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
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
    /// reasoning frame: its own records, or records queued ahead of its text).
    reasoning_priced: Option<bool>,
    /// A finishing frame ended the sequence.
    finished: bool,
}

impl Position {
    /// Drop the front pending token, whole.
    fn pop_token(&mut self) {
        if let Some(token) = self.pending.pop_front() {
            self.pending_bytes -= token.len();
        }
        self.offset = 0;
    }

    /// Skip the front pending token as the one silent delimiter.
    fn skip_delimiter(&mut self) {
        self.pop_token();
        self.delimiter_seen = true;
    }

    /// Match `text` against the front of the pending record bytes. With
    /// `delimiter`, one whole token that does not start the text may be
    /// skipped first (a template's end-of-thinking token, rendered nowhere).
    /// `span_skip` picks the reading where text that is a strict prefix of
    /// the front token skips it as that delimiter.
    fn consume(&mut self, text: &[u8], delimiter: bool, span_skip: bool) -> bool {
        let mut position = 0;
        let mut may_skip = delimiter;
        if may_skip && span_skip && self.offset == 0 && self.delimiter_spans(text) {
            // The answer opens with a strict prefix of the pending delimiter
            // (`<` before `</think>`) that the next tokens spell whole: the
            // front token is the silent delimiter, not a token emitted in part.
            self.skip_delimiter();
            may_skip = false;
        }
        while position < text.len() {
            let Some(front) = self.pending.front() else {
                // Text the records never spelled.
                return false;
            };
            if front.is_empty() {
                // A token with no bytes inside text: only the delimiter may
                // be silent, and only where the answer opens.
                if may_skip && position == 0 {
                    self.skip_delimiter();
                    may_skip = false;
                    continue;
                }
                return false;
            }
            let rest = &front[self.offset..];
            let take = rest.len().min(text.len() - position);
            if rest[..take] != text[position..position + take] {
                if may_skip && self.offset == 0 && position == 0 {
                    self.skip_delimiter();
                    may_skip = false;
                    continue;
                }
                return false;
            }
            position += take;
            self.offset += take;
            if self.offset == front.len() {
                self.pop_token();
            }
        }
        true
    }

    /// Whether `text` is a strict prefix of the front pending token while the
    /// tokens after it spell `text` whole, so the front token is a silent
    /// delimiter rather than a token whose text is only partly emitted.
    fn delimiter_spans(&self, text: &[u8]) -> bool {
        let Some(front) = self.pending.front() else {
            return false;
        };
        if text.is_empty() || text.len() >= front.len() || !front.starts_with(text) {
            return false;
        }
        let mut position = 0;
        for token in self.pending.iter().skip(1) {
            if token.is_empty() {
                return false;
            }
            let take = token.len().min(text.len() - position);
            if token[..take] != text[position..position + take] {
                return false;
            }
            position += take;
            if position == text.len() {
                return true;
            }
        }
        false
    }

    /// Align one frame: append its records, then match its reasoning text and
    /// its answer text, in that order. Tokens that emit no text are accepted at
    /// two places only: once before the first answer text (the delimiter), and
    /// on the finishing frame itself (stop or end-of-turn tokens). Returns
    /// every reading of this position after the frame, with its result.
    fn branches(
        mut self,
        delta: &ChoiceLogprobsDelta,
        output: &FrameOutput,
    ) -> Vec<(Self, Alignment)> {
        let (content, refusal) = delta.logprobs.as_ref().map_or((None, None), |logprobs| {
            (logprobs.content.as_deref(), logprobs.refusal.as_deref())
        });
        let records = content.unwrap_or(&[]);
        if refusal.is_some_and(|r| !r.is_empty())
            || (output.opaque && (!records.is_empty() || !self.pending.is_empty()))
        {
            // Refusal records beside answer text are on the wrong channel; tool
            // call arguments and images are output this check cannot spell, so
            // records on the frame or still waiting for text cannot be placed.
            return Vec::new();
        }
        if self.finished
            && (output.opaque
                || !records.is_empty()
                || !output.content.is_empty()
                || !output.reasoning.is_empty())
        {
            // The sequence ended on its finishing frame; nothing follows it.
            return Vec::new();
        }
        if self.reasoning_priced.is_none()
            && !output.reasoning.is_empty()
            && (!records.is_empty() || !self.pending.is_empty())
        {
            // The first reasoning frame's records (its own, or queued ahead of
            // it) may be reasoning tokens, or answer tokens when the provider
            // does not price reasoning: both readings continue.
            let mut priced = self.clone();
            priced.reasoning_priced = Some(true);
            let mut readings = priced.split_on_delimiter(delta, output, false);
            // Unpriced: the records lead an answer and must spell it whole
            // when it arrives, with no silent delimiter among them.
            self.reasoning_priced = Some(false);
            self.delimiter_seen = true;
            if !(output.finishes && output.content.is_empty()) {
                readings.extend(self.split_on_delimiter(delta, output, true));
            }
            return readings;
        }
        self.split_on_delimiter(delta, output, false)
    }

    /// Align the frame, keeping both readings of an answer that opens with a
    /// strict prefix of the front token: the delimiter skipped, or that token
    /// emitted in part.
    fn split_on_delimiter(
        mut self,
        delta: &ChoiceLogprobsDelta,
        output: &FrameOutput,
        answer_only: bool,
    ) -> Vec<(Self, Alignment)> {
        let mut readings = Vec::with_capacity(2);
        if !self.answer_started && !self.delimiter_seen && !output.content.is_empty() {
            let mut spanning = self.clone();
            let alignment = spanning.align_frame(delta, output, answer_only, true);
            readings.push((spanning, alignment));
        }
        let alignment = self.align_frame(delta, output, answer_only, false);
        readings.push((self, alignment));
        readings
    }

    /// Align one frame under the current pricing verdict. `answer_only` marks
    /// the frame that just decided the reasoning is unpriced: its records lead
    /// the answer rather than sitting beside unpriced reasoning. `span_skip`
    /// picks the delimiter reading of an answer opening with a prefix of the
    /// front token.
    fn align_frame(
        &mut self,
        delta: &ChoiceLogprobsDelta,
        output: &FrameOutput,
        answer_only: bool,
        span_skip: bool,
    ) -> Alignment {
        let records = delta
            .logprobs
            .as_ref()
            .and_then(|logprobs| logprobs.content.as_deref())
            .unwrap_or(&[]);
        if !output.reasoning.is_empty() && self.reasoning_priced.is_none() {
            // Records that arrived on earlier text-less frames (a token whose
            // text the detokenizer held back) price the reasoning too.
            self.reasoning_priced = Some(!records.is_empty() || !self.pending.is_empty());
        }
        if !output.reasoning.is_empty() {
            match self.reasoning_priced {
                // Records beside reasoning text, on this frame or queued ahead
                // of it, once the reasoning was found unpriced: the reasoning
                // before them was never matched, so the sequence cannot be
                // shown whole.
                Some(false)
                    if !answer_only
                        && output.content.is_empty()
                        && (!records.is_empty() || !self.pending.is_empty()) =>
                {
                    return Alignment::Hole;
                }
                // Priced reasoning after the answer started is out of order.
                Some(true) if self.answer_started => return Alignment::Hole,
                _ => {}
            }
        }
        if records.is_empty() && output.content.is_empty() && !output.finishes {
            // Reasoning text alone is matched only when the provider prices it.
            if self.reasoning_priced == Some(true) && !output.reasoning.is_empty() {
                return if self.consume(output.reasoning.as_bytes(), false, false) {
                    Alignment::Empty
                } else {
                    Alignment::Hole
                };
            }
            return Alignment::Empty;
        }
        // A token with no bytes stays in the queue as an empty entry, so its
        // position decides: the delimiter where the answer opens, or a stop
        // token after the finishing frame's text, else a hole.
        for record in records {
            let bytes = record_bytes(record);
            self.pending_bytes += bytes.len();
            self.pending.push_back(bytes.to_vec());
        }
        let added = records.len();
        if self.reasoning_priced == Some(true)
            && !self.consume(output.reasoning.as_bytes(), false, false)
        {
            return Alignment::Hole;
        }
        if !output.content.is_empty() {
            // Under an unpriced verdict a frame that still carries reasoning
            // text cannot spend the delimiter: a token skipped there could be
            // that reasoning's own.
            let unpriced_reasoning =
                self.reasoning_priced == Some(false) && !output.reasoning.is_empty();
            let delimiter = !self.answer_started && !self.delimiter_seen && !unpriced_reasoning;
            if !self.consume(output.content.as_bytes(), delimiter, span_skip) {
                return Alignment::Hole;
            }
            self.answer_started = true;
        }
        if output.finishes {
            // Silent stop tokens sit on the finishing frame itself, whole. A
            // token only partly emitted, or a record from an earlier frame
            // still unmatched, never spelled its text, unless it is the one
            // delimiter an answer never followed.
            let earlier = self.pending.len().saturating_sub(added);
            let delimiter = usize::from(!self.answer_started && !self.delimiter_seen);
            if self.offset != 0 || earlier > delimiter {
                return Alignment::Hole;
            }
            self.pending.clear();
            self.pending_bytes = 0;
            self.offset = 0;
            self.finished = true;
        } else if self.pending_bytes - self.offset > MAXIMUM_RECORD_LEAD_BYTES {
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
                // Once retention has ended, later frames are not aligned:
                // nothing they add could enter the record.
                _ if buffer.truncated => {}
                Ok(parsed) if output.refusal.is_empty() => {
                    let delta = parsed.unwrap_or(ChoiceLogprobsDelta {
                        choice_index,
                        logprobs: None,
                    });
                    match buffer.readings.align(&delta, output) {
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

    /// The stream ended (`[DONE]`): records still waiting for text never
    /// spelled any, so a sequence that did not finish on a finishing frame
    /// is incomplete.
    pub(crate) fn end_stream(&mut self) {
        if let Self::Capture(buffer) = self {
            if buffer.readings.all_pending() {
                buffer.truncated = true;
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
    /// Alignment readings across the attempt's frames.
    pub readings: Readings,
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
