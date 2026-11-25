//! Content-free record facts and storage-plan spans for a batched destination.
//!
//! A destination that frames capture records without decoding them still has
//! to know who a record belongs to, whether it carries a response, and (to
//! honor a storage plan that keeps the prompt or the response out) where those
//! parts sit. With facts output the collector hands it a small JSON document
//! beside the record bytes, computed natively while the record is encoded, so
//! the destination never parses the record to answer those questions.
//!
//! The facts are content-free: identities, protocol, model and deployment
//! ids, response status flags, sizes and byte spans. Spans are `[start, end)`
//! byte offsets into the record bytes handed beside them:
//!
//! * `messages`: the value of `request.context.request.messages`. A plan that
//!   keeps the prompt out replaces it with `[]`.
//! * `source_json`, `provider_context` (in the request context) and
//!   `wire_request` (in `transport`): whole members, including one separating
//!   comma, deleted with the prompt. The two context members are deleted
//!   together, as the prompt plan does.
//! * `response`: the value of the record's `response`, replaced by `null`
//!   when a plan keeps the response out, together with the response-derived
//!   fields the collector itself clears when response retention is denied:
//!   `response_fields` (provider reasoning and its source, completed tool
//!   calls, metrics, the Gemini parts' source) are replaced by `null`,
//!   `gemini_thought_parts` by `[]`, and the `provider_logprobs` member is
//!   deleted.
//!
//! Spans never overlap, and splicing them all in descending start order yields
//! exactly the record a decoded rewrite would produce.

use std::cell::{Cell, RefCell};

use serde::ser::{SerializeMap, SerializeStruct};
use serde::{Serialize, Serializer};
use serde_json::Value;

use super::record::{Protocol, Record, Request, Response, Scope};
use super::split::SplitRecord;

/// Version of the facts document; a destination checks it before trusting spans.
pub(crate) const FACTS_VERSION: u32 = 1;

/// Upper bound on one facts document: four validated identities of at most
/// 512 bytes, escaped, plus model and deployment ids and fixed fields. The
/// encoder stops at the bound, so a larger document (an oversized deployment
/// id) refuses its record without being materialized.
pub(crate) const MAXIMUM_FACTS_BYTES: usize = 16 * 1024;

type Span = [usize; 2];

/// The parts a storage plan may drop, as byte spans of one encoded record.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize)]
pub(crate) struct Spans {
    pub messages: Option<Span>,
    pub source_json: Option<Span>,
    pub provider_context: Option<Span>,
    pub wire_request: Option<Span>,
    pub response: Option<Span>,
    pub response_fields: Vec<Span>,
    pub gemini_thought_parts: Option<Span>,
    pub provider_logprobs: Option<Span>,
}

#[derive(Clone, Copy)]
pub(super) enum ValueSlot {
    Messages,
    Response,
    ResponseField,
    GeminiThoughtParts,
}

#[derive(Clone, Copy)]
enum Member {
    SourceJson,
    ProviderContext,
    WireRequest,
}

/// A recorded member span; `first` marks a member written without a leading comma.
#[derive(Clone, Copy, Default)]
struct MemberSpan {
    start: usize,
    end: usize,
    first: bool,
}

/// Shared between the bounded writer (which publishes its length) and the
/// serializer wrappers below (which read it around the values they wrap).
#[derive(Default)]
pub(crate) struct Probe {
    pub(super) position: Cell<usize>,
    values: RefCell<Spans>,
    members: RefCell<[Option<MemberSpan>; 3]>,
}

impl Probe {
    pub(super) fn request<'a>(&'a self, request: &'a Request) -> ProbedRequest<'a> {
        ProbedRequest {
            request,
            probe: self,
        }
    }

    pub(super) fn response<'a, R: Serialize>(
        &'a self,
        response: &'a Option<R>,
    ) -> Spanned<'a, Option<R>> {
        Spanned {
            value: response,
            probe: self,
            slot: response.is_some().then_some(ValueSlot::Response),
        }
    }

    /// Wrap a response-derived top-level value; `present` is false for an
    /// absent (null or empty) value, which no plan needs to touch.
    pub(super) fn response_field<'a, T: Serialize + ?Sized>(
        &'a self,
        value: &'a T,
        slot: ValueSlot,
        present: bool,
    ) -> Spanned<'a, T> {
        Spanned {
            value,
            probe: self,
            slot: present.then_some(slot),
        }
    }

    /// Record the whole `provider_logprobs` member, written with its leading comma.
    pub(super) fn record_logprobs(&self, start: usize) {
        self.values.borrow_mut().provider_logprobs = Some([start, self.position.get()]);
    }

    pub(super) fn transport<'a>(&'a self, transport: &'a Value) -> ProbedObject<'a> {
        ProbedObject {
            value: transport,
            probe: self,
            kind: ObjectKind::Transport,
        }
    }

    fn record_value(&self, slot: ValueSlot, start: usize, end: usize) {
        let mut spans = self.values.borrow_mut();
        match slot {
            ValueSlot::Messages => spans.messages = Some([start, end]),
            ValueSlot::Response => spans.response = Some([start, end]),
            ValueSlot::ResponseField => spans.response_fields.push([start, end]),
            ValueSlot::GeminiThoughtParts => spans.gemini_thought_parts = Some([start, end]),
        }
    }

    fn record_member(&self, member: Member, span: MemberSpan) {
        self.members.borrow_mut()[member as usize] = Some(span);
    }

    /// Final spans over `bytes`. A member written first in its object takes
    /// the comma that follows it, every other member the comma before it, so
    /// removing one leaves valid JSON. When the two context members are
    /// adjacent and the earlier is first, both would claim the one comma
    /// between them; the later then takes the comma after it instead, so the
    /// spans stay disjoint and removing both leaves valid JSON.
    fn finish(&self, bytes: &[u8]) -> Spans {
        let mut spans = self.values.borrow().clone();
        let members = *self.members.borrow();
        let trailing = |end: usize| {
            if bytes.get(end) == Some(&b',') {
                end + 1
            } else {
                end
            }
        };
        let close = |span: Option<MemberSpan>| {
            span.map(|span| {
                let end = if span.first {
                    trailing(span.end)
                } else {
                    span.end
                };
                [span.start, end]
            })
        };
        let mut context = [
            close(members[Member::SourceJson as usize]),
            close(members[Member::ProviderContext as usize]),
        ];
        if let [Some(a), Some(b)] = &mut context {
            let (earlier, later) = if a[0] < b[0] { (a, b) } else { (b, a) };
            if later[0] < earlier[1] {
                later[0] = earlier[1];
                later[1] = trailing(later[1]);
            }
        }
        [spans.source_json, spans.provider_context] = context;
        spans.wire_request = close(members[Member::WireRequest as usize]);
        spans
    }
}

/// Records the byte span of the value it wraps.
pub(super) struct Spanned<'a, T: ?Sized> {
    value: &'a T,
    probe: &'a Probe,
    slot: Option<ValueSlot>,
}

impl<T: Serialize + ?Sized> Serialize for Spanned<'_, T> {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        let start = self.probe.position.get();
        let written = self.value.serialize(serializer)?;
        if let Some(slot) = self.slot {
            self.probe
                .record_value(slot, start, self.probe.position.get());
        }
        Ok(written)
    }
}

/// The request, field for field as its derived serialization writes it.
pub(super) struct ProbedRequest<'a> {
    request: &'a Request,
    probe: &'a Probe,
}

impl Serialize for ProbedRequest<'_> {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        let request = self.request;
        let mut state = serializer.serialize_struct("Request", 5)?;
        state.serialize_field("request_id", &request.request_id)?;
        state.serialize_field("scope", &request.scope)?;
        state.serialize_field("protocol", &request.protocol)?;
        state.serialize_field("model_id", &request.model_id)?;
        state.serialize_field(
            "context",
            &ProbedObject {
                value: &request.context,
                probe: self.probe,
                kind: ObjectKind::Context,
            },
        )?;
        state.end()
    }
}

#[derive(Clone, Copy)]
enum ObjectKind {
    /// The request context: `request` nests the messages; two repeated members.
    Context,
    /// The context's `request`: the messages value.
    ContextRequest,
    /// The transport: the redacted wire request member.
    Transport,
}

/// A JSON object written entry by entry exactly as `Value` writes it.
pub(super) struct ProbedObject<'a> {
    value: &'a Value,
    probe: &'a Probe,
    kind: ObjectKind,
}

impl Serialize for ProbedObject<'_> {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        let Value::Object(entries) = self.value else {
            return self.value.serialize(serializer);
        };
        let mut map = serializer.serialize_map(Some(entries.len()))?;
        for (index, (key, value)) in entries.iter().enumerate() {
            let member = match (self.kind, key.as_str()) {
                (ObjectKind::Context, "source_json") => Some(Member::SourceJson),
                (ObjectKind::Context, "provider_context") => Some(Member::ProviderContext),
                (ObjectKind::Transport, "wire_request") => Some(Member::WireRequest),
                _ => None,
            };
            let start = self.probe.position.get();
            map.serialize_key(key)?;
            match (self.kind, key.as_str()) {
                (ObjectKind::Context, "request") if value.is_object() => {
                    map.serialize_value(&ProbedObject {
                        value,
                        probe: self.probe,
                        kind: ObjectKind::ContextRequest,
                    })?;
                }
                (ObjectKind::ContextRequest, "messages") => {
                    map.serialize_value(&Spanned {
                        value,
                        probe: self.probe,
                        slot: Some(ValueSlot::Messages),
                    })?;
                }
                _ => map.serialize_value(value)?,
            }
            if let Some(member) = member {
                self.probe.record_member(
                    member,
                    MemberSpan {
                        start,
                        end: self.probe.position.get(),
                        first: index == 0,
                    },
                );
            }
        }
        map.end()
    }
}

/// One record serialized through the probe.
struct ProbedRecord<'a> {
    record: &'a Record,
    completion: bool,
    probe: &'a Probe,
}

impl Serialize for ProbedRecord<'_> {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        self.record
            .serialize_update(serializer, self.completion, Some(self.probe))
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
pub(crate) struct ResponseFacts {
    kind: &'static str,
    status: u16,
    truncated: bool,
    client_disconnected: bool,
}

impl ResponseFacts {
    /// Flags of a response as delivered in a split skeleton, which may record
    /// loss (an unreadable or cut spilled stream) the in-memory facts lack.
    pub(crate) fn of_skeleton(response: &Value) -> Option<Self> {
        let kind = match response.get("kind")?.as_str()? {
            "json" => "json",
            "sse" => "sse",
            _ => return None,
        };
        let flag = |name: &str| response.get(name).and_then(Value::as_bool) == Some(true);
        Some(Self {
            kind,
            status: u16::try_from(response.get("status")?.as_u64()?).ok()?,
            truncated: flag("truncated"),
            client_disconnected: flag("client_disconnected"),
        })
    }

    fn of(response: &Response) -> Self {
        match response {
            Response::Json { status, .. } => Self {
                kind: "json",
                status: *status,
                truncated: false,
                client_disconnected: false,
            },
            Response::Sse {
                status,
                truncated,
                client_disconnected,
                ..
            } => Self {
                kind: "sse",
                status: *status,
                truncated: *truncated,
                client_disconnected: *client_disconnected,
            },
            Response::Spilled(spilled) => Self {
                kind: if spilled.sse { "sse" } else { "json" },
                status: spilled.status,
                truncated: spilled.truncated,
                client_disconnected: spilled.client_disconnected,
            },
        }
    }
}

/// The facts document; field names are the destination's contract.
#[derive(Serialize)]
struct Facts<'a> {
    facts_version: u32,
    schema_version: u32,
    request_id: &'a str,
    scope: &'a Scope,
    protocol: Protocol,
    model_id: Option<&'a str>,
    deployment_id: Option<&'a str>,
    captured_at: f64,
    /// Whether the record carries the request's response.
    complete: bool,
    response: Option<ResponseFacts>,
    /// Whether the record is a split skeleton (its parts are on disk).
    split: bool,
    /// Bytes of the record (or of the split skeleton) handed beside the facts.
    record_bytes: usize,
    /// Byte spans over the record; absent for a split skeleton.
    spans: Option<Spans>,
}

fn facts(
    record: &Record,
    schema_version: u32,
    response: Option<ResponseFacts>,
    split: bool,
    record_bytes: usize,
    spans: Option<Spans>,
) -> Option<Vec<u8>> {
    let complete = response.is_some();
    let facts = &Facts {
        facts_version: FACTS_VERSION,
        schema_version,
        request_id: &record.request.request_id,
        scope: &record.request.scope,
        protocol: record.request.protocol,
        model_id: record.request.model_id.as_deref(),
        deployment_id: record.deployment_id.as_deref(),
        captured_at: record.captured_at,
        complete,
        response,
        split,
        record_bytes,
        spans,
    };
    super::budget::encode(facts, MAXIMUM_FACTS_BYTES).map(String::into_bytes)
}

/// Encode a whole record exactly as `Record::encode` (or `encode_update` when
/// `completion_references`) would, and its facts with spans over those bytes.
pub(crate) fn encode_with_facts(
    record: &Record,
    maximum_bytes: usize,
    completion_references: bool,
) -> Result<(String, Vec<u8>), Unencodable> {
    if !record.valid() {
        return Err(Unencodable::Facts);
    }
    let completion = completion_references && record.checkpointed && record.response.is_some();
    let probe = Probe::default();
    let encoded = super::budget::encode_tracked(
        &ProbedRecord {
            record,
            completion,
            probe: &probe,
        },
        maximum_bytes,
        Some(&probe.position),
    )
    .ok_or(Unencodable::Record)?;
    let spans = probe.finish(encoded.as_bytes());
    let schema_version = if completion { 2 } else { record.schema_version };
    let facts = facts(
        record,
        schema_version,
        record.response.as_ref().map(ResponseFacts::of),
        false,
        encoded.len(),
        Some(spans),
    )
    .ok_or(Unencodable::Facts)?;
    Ok((encoded, facts))
}

/// Why `encode_with_facts` refused a record. Only an oversized record may be
/// handed over split instead; an invalid record or oversized facts (whose
/// identities would not fit a split record's facts either) are refused.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Unencodable {
    /// The record exceeds the encoded-record bound.
    Record,
    /// The record is invalid, or its facts exceed `MAXIMUM_FACTS_BYTES`.
    Facts,
}

/// Facts for a split record's skeleton; spans are not computed over a skeleton.
pub(crate) fn split_facts(record: &Record, split: &SplitRecord) -> Option<Vec<u8>> {
    facts(
        record,
        record.schema_version,
        split.response,
        true,
        split.skeleton.len(),
        None,
    )
}

#[cfg(test)]
#[path = "facts_test.rs"]
mod tests;
