//! Gateway-executed tool search: the data-plane half of the feature.
//!
//! A caller with many tools defers most of them and declares a tool-search
//! tool. On a route whose model cannot search natively, admission (python)
//! sends the model the loaded tools plus ONE gateway-owned function tool
//! (`Admission.tool_search.tool_name`). This module owns everything the data
//! plane does with that tool:
//!
//! * the relay-side [`ToolSearchWithholder`] swallows every call to the
//!   gateway tool (start, argument deltas, completion) so the caller never
//!   sees it, and hands the accumulated calls to the waterfall;
//! * the waterfall asks the control plane to run the search and rebuild the
//!   rung's dispatch (`round_argument` / [`ToolSearchRoundReply`]), then
//!   dials the same depth again;
//! * the completed rounds render ahead of the answer on each surface:
//!   Anthropic `server_tool_use` + `tool_search_tool_result` blocks, OpenAI
//!   Responses `tool_search_call` + `tool_search_output` hosted items, and
//!   the `tool_search_requests` meter on every usage object. Chat renders
//!   no items, only the meter.
//!
//! Every helper here is a no-op when the request ran no search, so a
//! response whose admission carried no `tool_search` keeps its exact bytes.

use std::collections::HashSet;

use serde::Deserialize;
use serde_json::{json, Value};

use crate::admission::Admission;
use crate::encode::{compact_json, stable_public_id, ChatSseEncoder};
use crate::encode_messages::{AggregatedMessage, MessagesSseEncoder};
use crate::encode_responses::{
    completed_responses_body_with_gateway_tools, AggregatedResponses, ResponsesSseEncoder,
};
use crate::errors::{Failure, FailureClass, PublicError};
use crate::events::{CompletedToolCall, Event, Usage};
use crate::waterfall::{CommittedAttempt, DeploymentWire, Won};
use crate::web_search::{
    completed_messages_body_with_web_search, configure_messages_encoder, WebSearchAdmission,
};

/// The Anthropic server tool name a BM25 (natural-language) round renders as.
pub const TOOL_SEARCH_BM25_TOOL_NAME: &str = "tool_search_tool_bm25";

/// The Anthropic server tool name a regex round renders as.
pub const TOOL_SEARCH_REGEX_TOOL_NAME: &str = "tool_search_tool_regex";

/// The `x-experiential-ignored-parameters` entry disclosing a search call the
/// model made in the same turn as other output, which the gateway dropped.
pub const TOOL_SEARCH_DROPPED_DISCLOSURE: &str = "tool_search->dropped(after_output)";

/// Rounds per request when the admission names no bound.
pub const DEFAULT_MAX_ROUNDS: u32 = 3;

fn default_max_rounds() -> u32 {
    DEFAULT_MAX_ROUNDS
}

/// The admission's account of the gateway-run search: the function tool the
/// model was given, how many rounds this request may run, how many tools
/// were deferred, and which caller shape asked for it.
#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
pub struct ToolSearchAdmission {
    pub tool_name: String,
    #[serde(default = "default_max_rounds")]
    pub max_rounds: u32,
    #[serde(default)]
    pub deferred: u32,
    #[serde(default)]
    pub surface_shape: String,
}

/// One completed search round as the control plane reports it: the model's
/// call id, its query or pattern, and the deferred tools it loaded.
#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
pub struct ToolSearchRound {
    pub call_id: String,
    #[serde(default)]
    pub query: Option<String>,
    #[serde(default)]
    pub pattern: Option<String>,
    #[serde(default)]
    pub matched: Vec<String>,
    /// The matched tools' full declarations, when the control plane sends
    /// them; the Responses `tool_search_output.tools` array prefers these
    /// over names alone.
    #[serde(default)]
    pub matched_tools: Vec<Value>,
    /// The caller's own declaration name (a versioned Anthropic type keeps
    /// its version), rendered as the Messages `server_tool_use.name`.
    #[serde(default)]
    pub declared_name: Option<String>,
}

impl ToolSearchRound {
    /// Whether the round searched by regular expression (else BM25).
    pub fn is_regex(&self) -> bool {
        self.pattern.is_some() && self.query.is_none()
    }

    /// The Anthropic server tool name this round renders as.
    pub fn anthropic_tool_name(&self) -> &str {
        match self.declared_name.as_deref() {
            Some(name) if !name.is_empty() && name.starts_with("tool_search") => name,
            _ if self.is_regex() => TOOL_SEARCH_REGEX_TOOL_NAME,
            _ => TOOL_SEARCH_BM25_TOOL_NAME,
        }
    }

    /// The Anthropic `server_tool_use.input`: `{"pattern": …}` or `{"query": …}`.
    pub fn anthropic_input(&self) -> Value {
        match (&self.pattern, &self.query) {
            (Some(pattern), None) => json!({"pattern": pattern}),
            (_, query) => json!({"query": query.clone().unwrap_or_default()}),
        }
    }

    /// The Responses `tool_search_call.arguments`: `{"pattern": …}` or
    /// `{"goal": …}`.
    pub fn responses_arguments(&self) -> Value {
        match (&self.pattern, &self.query) {
            (Some(pattern), None) => json!({"pattern": pattern}),
            (_, query) => json!({"goal": query.clone().unwrap_or_default()}),
        }
    }

    /// The Responses `tool_search_output.tools` array: the full declarations
    /// when known, else one `{"type":"function","name":…}` per matched name.
    pub fn responses_tools(&self) -> Vec<Value> {
        if !self.matched_tools.is_empty() {
            return self.matched_tools.clone();
        }
        self.matched
            .iter()
            .map(|name| json!({"type": "function", "name": name}))
            .collect()
    }
}

/// One call to the gateway's search tool the relay withheld from the caller.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WithheldSearchCall {
    pub index: u32,
    pub call_id: String,
    pub name: String,
    pub raw_arguments: String,
}

/// Relay-side filter that swallows every event of a call to the gateway's
/// search tool and accumulates the completed calls. Every other event passes
/// through untouched, and with no tool name set nothing is ever withheld.
#[derive(Default)]
pub struct ToolSearchWithholder {
    tool_name: Option<String>,
    /// Started-but-unfinished search calls by provider tool index.
    open: HashSet<u32>,
    completed: Vec<WithheldSearchCall>,
    /// Bytes of raw arguments across the withheld calls.
    withheld_bytes: usize,
    /// The model exceeded the bounds; the round is refused, never trimmed.
    overflowed: bool,
}

/// Most search calls one dial may withhold; more is a runaway model, not a search.
pub const MAXIMUM_WITHHELD_SEARCH_CALLS: usize = 8;

/// Most raw-argument bytes the withheld calls may hold together; the same
/// order as the withheld-refusal buffer, and what bounds the bridge payload.
pub const MAXIMUM_WITHHELD_SEARCH_BYTES: usize = 65_536;

impl ToolSearchWithholder {
    /// Name the gateway's search tool; `None` (or empty) disables withholding.
    pub fn set_tool_name(&mut self, tool_name: Option<String>) {
        self.tool_name = tool_name.filter(|name| !name.is_empty());
    }

    /// Pass one event through, or swallow it when it belongs to a search
    /// call (`None`).
    pub fn filter(&mut self, event: Event) -> Option<Event> {
        let Some(tool_name) = self.tool_name.as_deref() else {
            return Some(event);
        };
        match &event {
            Event::ToolCallStarted { index, name, .. } if name == tool_name => {
                self.open.insert(*index);
                None
            }
            Event::ToolArgumentsDelta { index, .. } if self.open.contains(index) => None,
            Event::ToolCallCompleted { index, call }
                if self.open.contains(index) || call.name == tool_name =>
            {
                self.open.remove(index);
                self.withheld_bytes = self.withheld_bytes.saturating_add(call.raw_arguments.len());
                if self.completed.len() >= MAXIMUM_WITHHELD_SEARCH_CALLS
                    || self.withheld_bytes > MAXIMUM_WITHHELD_SEARCH_BYTES
                {
                    // Still swallowed (the caller never sees the gateway tool),
                    // but the dial fails closed instead of running the round.
                    self.overflowed = true;
                    return None;
                }
                self.completed.push(WithheldSearchCall {
                    index: *index,
                    call_id: call.call_id.clone(),
                    name: call.name.clone(),
                    raw_arguments: call.raw_arguments.clone(),
                });
                None
            }
            _ => Some(event),
        }
    }

    /// How many completed search calls are withheld and not yet taken.
    pub fn withheld_count(&self) -> usize {
        self.completed.len()
    }

    /// Whether any search call was seen at all: started (still open) or
    /// completed. An open call the terminal never completed is still a call
    /// the caller must be told was dropped.
    pub fn withheld_any(&self) -> bool {
        !self.open.is_empty() || !self.completed.is_empty()
    }

    /// Whether the model exceeded the withheld-call bounds on this dial.
    pub fn overflowed(&self) -> bool {
        self.overflowed
    }

    /// Hand over the withheld completed calls, leaving none behind.
    pub fn take_withheld(&mut self) -> Vec<WithheldSearchCall> {
        std::mem::take(&mut self.completed)
    }
}

/// The failure a dial ends with when the model flooded the gateway's search
/// tool: the gateway's own bound, so neither retryable nor failover-eligible.
pub fn withheld_overflow_failure() -> Failure {
    Failure::new(
        FailureClass::Internal,
        "gateway tool search calls exceeded the per-dial bound",
    )
}

/// The settle-shaped usage object (`null` without a provider report).
pub(crate) fn usage_json(usage: Option<&Usage>) -> Value {
    match usage {
        Some(usage) => json!({
            "input_tokens": usage.input_tokens,
            "output_tokens": usage.output_tokens,
            "cached_input_tokens": usage.cached_input_tokens,
            "cache_creation_input_tokens": usage.cache_creation_input_tokens,
            "cache_creation_1h_input_tokens": usage.cache_creation_1h_input_tokens,
            "reasoning_tokens": usage.reasoning_tokens,
        }),
        None => Value::Null,
    }
}

/// The `tool_search_round` bridge argument: which rung and round, the
/// withheld calls verbatim, and the usage the search-call turn was billed.
pub(crate) fn round_argument(
    request_id: &str,
    route_depth: usize,
    round: u32,
    calls: &[WithheldSearchCall],
    usage: Option<&Usage>,
) -> String {
    let calls: Vec<Value> = calls
        .iter()
        .map(|call| {
            json!({
                "call_id": call.call_id,
                "name": call.name,
                "arguments": call.raw_arguments,
            })
        })
        .collect();
    compact_json(&json!({
        "request_id": request_id,
        "route_depth": route_depth,
        "round": round,
        "calls": calls,
        "usage": usage_json(usage),
    }))
}

/// The control plane's answer to one `tool_search_round`: the rebuilt wire
/// for the same depth (conversation extended with the call and its result,
/// matched tools loaded), the rounds it ran, and whether the round budget is
/// spent (the gateway tool was removed, so the model must answer).
#[derive(Debug, Deserialize)]
pub(crate) struct ToolSearchRoundReply {
    pub(crate) wire: DeploymentWire,
    #[serde(default)]
    pub(crate) rounds: Vec<ToolSearchRound>,
    #[serde(default)]
    pub(crate) exhausted: bool,
}

/// The Messages server-tool index reserved for round `round`: the top of the
/// range minus one per round, below the web search's `u32::MAX`, so no
/// provider-issued block (indexed from zero) can collide.
pub fn messages_round_index(round: usize) -> u32 {
    u32::MAX - 1 - round as u32
}

/// The stable `server_tool_use` block id of round `round`.
pub fn tool_search_tool_use_id(request_id: &str, round: usize) -> String {
    stable_public_id("srvtoolu", &format!("{request_id}:tool_search:{round}"))
}

/// The Anthropic `tool_search_tool_result` block for one round: one
/// `tool_reference` per matched tool (an empty list when nothing matched).
pub fn tool_search_result_block(round: &ToolSearchRound, tool_use_id: &str) -> Value {
    let references: Vec<Value> = round
        .matched
        .iter()
        .map(|name| json!({"type": "tool_reference", "tool_name": name}))
        .collect();
    json!({
        "type": "tool_search_tool_result",
        "tool_use_id": tool_use_id,
        "content": {
            "type": "tool_search_tool_search_result",
            "tool_references": references,
        },
    })
}

/// The normalized events rendering every round as Anthropic blocks, in the
/// same server-tool shape the web-search prelude uses (start, whole input as
/// one delta, completion, verbatim result block), fed ahead of every provider
/// event.
pub fn messages_prelude_events(rounds: &[ToolSearchRound], request_id: &str) -> Vec<Event> {
    let mut events = Vec::with_capacity(rounds.len() * 4);
    for (position, round) in rounds.iter().enumerate() {
        let index = messages_round_index(position);
        let call_id = tool_search_tool_use_id(request_id, position);
        let name = round.anthropic_tool_name().to_string();
        let input = compact_json(&round.anthropic_input());
        events.push(Event::ServerToolUseStarted {
            index,
            call_id: call_id.clone(),
            name: name.clone(),
        });
        events.push(Event::ServerToolArgumentsDelta {
            index,
            delta: input.clone(),
        });
        events.push(Event::ServerToolUseCompleted {
            index,
            call: CompletedToolCall {
                call_id: call_id.clone(),
                name,
                namespace: None,
                caller: None,
                provider_item_id: None,
                provider_status: None,
                raw_arguments: input,
                custom: false,
            },
        });
        events.push(Event::ServerToolResult {
            index,
            block: compact_json(&tool_search_result_block(round, &call_id)),
        });
    }
    events
}

/// What the Messages encoder needs of the search: the round count for its
/// usage meters and the block events it feeds itself at start.
#[derive(Debug, Clone)]
pub struct MessagesToolSearch {
    pub requests: u32,
    pub events: Vec<Event>,
}

/// The Messages encoder's search state, `None` when no round ran.
pub fn messages_tool_search(
    rounds: &[ToolSearchRound],
    request_id: &str,
) -> Option<MessagesToolSearch> {
    if rounds.is_empty() {
        return None;
    }
    Some(MessagesToolSearch {
        requests: rounds.len() as u32,
        events: messages_prelude_events(rounds, request_id),
    })
}

/// Add `tool_search_requests` to Anthropic's `server_tool_use` meter on one
/// Messages usage object, beside a web-search meter already there.
pub fn annotate_messages_tool_search_usage(mut usage: Value, requests: Option<u32>) -> Value {
    if let (Some(requests), Some(entries)) = (requests, usage.as_object_mut()) {
        let meter = entries
            .entry("server_tool_use".to_string())
            .or_insert_with(|| json!({}));
        if let Some(meter) = meter.as_object_mut() {
            meter.insert("tool_search_requests".to_string(), json!(requests));
        }
    }
    usage
}

/// Add `tool_search_requests` to the OpenAI-shaped `server_tool_use_details`
/// meter on one Chat or Responses usage object; zero adds nothing and a
/// `null` usage stays `null`.
pub fn annotate_tool_search_usage_details(usage: &mut Value, requests: u32) {
    if requests == 0 {
        return;
    }
    if let Some(entries) = usage.as_object_mut() {
        let meter = entries
            .entry("server_tool_use_details".to_string())
            .or_insert_with(|| json!({}));
        if let Some(meter) = meter.as_object_mut() {
            meter.insert("tool_search_requests".to_string(), json!(requests));
        }
    }
}

/// Meter the rounds on one aggregated Chat Completion's usage object.
pub fn annotate_chat_completion_tool_search(body: &mut Value, requests: u32) {
    if let Some(usage) = body.get_mut("usage") {
        annotate_tool_search_usage_details(usage, requests);
    }
}

/// The Responses hosted-item indexes reserved for round `round`: the call
/// item and the output item, two per round below the top of the range.
pub fn responses_round_indexes(round: usize) -> (u32, u32) {
    let round = round as u32;
    (u32::MAX - 1 - 2 * round, u32::MAX - 2 - 2 * round)
}

/// The stable `tool_search_call` item id of round `round`.
pub fn tool_search_call_item_id(request_id: &str, round: usize) -> String {
    stable_public_id("tsc", &format!("{request_id}:tool_search:{round}"))
}

/// The stable `tool_search_output` item id of round `round`.
pub fn tool_search_output_item_id(request_id: &str, round: usize) -> String {
    stable_public_id("tso", &format!("{request_id}:tool_search:{round}"))
}

/// The normalized hosted-item events rendering every round on the Responses
/// surface: a completed `tool_search_call` then its `tool_search_output`,
/// each opened and closed with the same verbatim item, ahead of every
/// provider item.
pub fn responses_prelude_events(rounds: &[ToolSearchRound], request_id: &str) -> Vec<Event> {
    let mut events = Vec::with_capacity(rounds.len() * 4);
    for (position, round) in rounds.iter().enumerate() {
        let (call_index, output_index) = responses_round_indexes(position);
        let call_item_id = tool_search_call_item_id(request_id, position);
        let output_item_id = tool_search_output_item_id(request_id, position);
        let call_item = compact_json(&json!({
            "type": "tool_search_call",
            "id": call_item_id,
            "call_id": round.call_id,
            "status": "completed",
            "execution": "server",
            "arguments": round.responses_arguments(),
        }));
        let output_item = compact_json(&json!({
            "type": "tool_search_output",
            "id": output_item_id,
            "call_id": round.call_id,
            "status": "completed",
            "execution": "server",
            "tools": round.responses_tools(),
        }));
        for (index, item_id, item_type, item) in [
            (call_index, &call_item_id, "tool_search_call", &call_item),
            (
                output_index,
                &output_item_id,
                "tool_search_output",
                &output_item,
            ),
        ] {
            events.push(Event::HostedToolItemStarted {
                output_index: index,
                item_id: item_id.clone(),
                item_type: item_type.to_string(),
                item: item.clone(),
            });
            events.push(Event::HostedToolItemCompleted {
                output_index: index,
                item_id: item_id.clone(),
                item_type: item_type.to_string(),
                item: item.clone(),
            });
        }
    }
    events
}

/// What the Responses encoder needs of the search: the round count for its
/// usage meter and the hosted-item events it feeds itself at start.
#[derive(Debug, Clone)]
pub struct ResponsesToolSearch {
    pub requests: u32,
    pub events: Vec<Event>,
}

/// The Responses encoder's search state, `None` when no round ran.
pub fn responses_tool_search(
    rounds: &[ToolSearchRound],
    request_id: &str,
) -> Option<ResponsesToolSearch> {
    if rounds.is_empty() {
        return None;
    }
    Some(ResponsesToolSearch {
        requests: rounds.len() as u32,
        events: responses_prelude_events(rounds, request_id),
    })
}

/// `completed_messages_body_with_web_search` with the search rounds rendered
/// ahead of every provider block (after the web search, which ran before
/// dispatch) and metered on the usage object. The synthesized server tools
/// are the gateway's own work, billed through `tool_search_requests`, so
/// they never join the provider `tool_names` the ledger prices per call.
#[allow(clippy::too_many_arguments)]
pub fn completed_messages_body_with_gateway_tools(
    request_id: &str,
    model: &str,
    events: &[Event],
    ignored_parameters: &[String],
    reasoning_content_carrier: Option<&str>,
    reasoning_output_exposed: bool,
    web_search: Option<&WebSearchAdmission>,
    tool_search: Option<&MessagesToolSearch>,
) -> Result<AggregatedMessage, PublicError> {
    let Some(tool_search) = tool_search else {
        return completed_messages_body_with_web_search(
            request_id,
            model,
            events,
            ignored_parameters,
            reasoning_content_carrier,
            reasoning_output_exposed,
            web_search,
        );
    };
    let mut prefixed = tool_search.events.clone();
    prefixed.extend(events.iter().cloned());
    let mut aggregated = completed_messages_body_with_web_search(
        request_id,
        model,
        &prefixed,
        ignored_parameters,
        reasoning_content_carrier,
        reasoning_output_exposed,
        web_search,
    )?;
    for name in [TOOL_SEARCH_BM25_TOOL_NAME, TOOL_SEARCH_REGEX_TOOL_NAME] {
        let provider_named = events.iter().any(|event| {
            matches!(event, Event::ServerToolUseCompleted { call, .. } if call.name == name)
        });
        if !provider_named {
            aggregated.tool_names.retain(|tool| tool != name);
        }
    }
    if let Some(usage) = aggregated.body.get_mut("usage") {
        let metered = annotate_messages_tool_search_usage(usage.take(), Some(tool_search.requests));
        *usage = metered;
    }
    Ok(aggregated)
}

/// Whether the withheld-search facts of one committed attempt call for the
/// dropped-call disclosure: a search call that shared its turn with other
/// output, at commit or swallowed since.
fn dropped_search_call(committed: &CommittedAttempt) -> bool {
    committed.tool_search_dropped_after_output || committed.relay.withheld_search_call_seen()
}

fn disclose_dropped(ignored_parameters: &mut Vec<String>) {
    if !ignored_parameters
        .iter()
        .any(|entry| entry == TOOL_SEARCH_DROPPED_DISCLOSURE)
    {
        ignored_parameters.push(TOOL_SEARCH_DROPPED_DISCLOSURE.to_string());
    }
}

/// Move the waterfall's tool-search facts onto the admission the response
/// surfaces render from: the completed rounds, and the disclosure of a search
/// call dropped because the rung had already committed on other output.
pub(crate) fn adopt_outcome(admission: &mut Admission, won: &mut Won) {
    match won {
        Won::Committed(committed) => {
            admission.tool_search_rounds = std::mem::take(&mut committed.tool_search_rounds);
            if dropped_search_call(committed) {
                disclose_dropped(&mut admission.ignored_parameters);
            }
        }
        Won::Settled(settled) => {
            admission.tool_search_rounds = std::mem::take(&mut settled.tool_search_rounds);
        }
        Won::Failed(_) => {}
    }
}

/// After a committed attempt was drained to completion: disclose a search
/// call the relay swallowed after the commit, before the body is built.
pub(crate) fn disclose_after_collection(admission: &mut Admission, committed: &CommittedAttempt) {
    if dropped_search_call(committed) {
        disclose_dropped(&mut admission.ignored_parameters);
    }
}

/// Configure one Chat encoder for the admission's gateway-run tools.
pub(crate) fn configure_chat_encoder(encoder: &mut ChatSseEncoder, admission: &Admission) {
    encoder.set_web_search(admission.web_search.clone());
    encoder.set_tool_search_requests(admission.tool_search_rounds.len() as u32);
}

/// Meter the admission's search rounds on one aggregated Chat Completion,
/// after the web search annotated it.
pub(crate) fn annotate_chat_completion_for(body: &mut Value, admission: &Admission) {
    if let Some(web_search) = admission.web_search.as_ref() {
        crate::web_search::annotate_chat_completion(body, web_search);
    }
    annotate_chat_completion_tool_search(body, admission.tool_search_rounds.len() as u32);
}

/// Configure one Messages encoder for the admission's gateway-run tools.
pub(crate) fn configure_messages_encoder_for(
    encoder: &mut MessagesSseEncoder,
    admission: &Admission,
) {
    configure_messages_encoder(
        encoder,
        admission.web_search.as_ref(),
        &admission.request_id,
    );
    encoder.set_tool_search(messages_tool_search(
        &admission.tool_search_rounds,
        &admission.request_id,
    ));
}

/// Aggregate one Messages turn for the admission, rendering its web search
/// and search rounds ahead of the answer.
pub(crate) fn completed_messages_body_for(
    admission: &Admission,
    events: &[Event],
    reasoning_content_carrier: Option<&str>,
    reasoning_output_exposed: bool,
) -> Result<AggregatedMessage, PublicError> {
    completed_messages_body_with_gateway_tools(
        &admission.request_id,
        &admission.alias,
        events,
        &admission.ignored_parameters,
        reasoning_content_carrier,
        reasoning_output_exposed,
        admission.web_search.as_ref(),
        messages_tool_search(&admission.tool_search_rounds, &admission.request_id).as_ref(),
    )
}

/// Configure one Responses encoder for the admission's gateway-run tools.
pub(crate) fn configure_responses_encoder(
    encoder: &mut ResponsesSseEncoder,
    admission: &Admission,
) {
    encoder.set_web_search(admission.web_search.clone());
    encoder.set_tool_search(responses_tool_search(
        &admission.tool_search_rounds,
        &admission.request_id,
    ));
}

/// Aggregate one Responses turn for the admission, rendering its search
/// rounds as hosted items ahead of the answer and citing its web search.
pub(crate) fn completed_responses_body_for(
    admission: &Admission,
    created_at: i64,
    events: &[Event],
    reasoning_content_carrier: Option<&str>,
) -> Result<AggregatedResponses, PublicError> {
    completed_responses_body_with_gateway_tools(
        &admission.request_id,
        &admission.alias,
        created_at,
        admission.envelope.clone().unwrap_or_default(),
        events,
        reasoning_content_carrier,
        admission.web_search.as_ref(),
        responses_tool_search(&admission.tool_search_rounds, &admission.request_id).as_ref(),
    )
}

/// Encode admitted Responses events with their unchanged gateway tool configuration.
pub(crate) fn encode_responses_sse(
    admission: &Admission,
    created_at: i64,
    events: &[Event],
    reasoning_content_carrier: Option<&str>,
) -> Result<Vec<u8>, PublicError> {
    let envelope = admission.envelope.clone().unwrap_or_default();
    let mut encoder = ResponsesSseEncoder::new(
        &admission.request_id,
        &admission.alias,
        created_at,
        envelope,
    );
    configure_responses_encoder(&mut encoder, admission);
    if let Some(carrier) = reasoning_content_carrier {
        encoder.set_reasoning_content_carrier(carrier.to_string())?;
    }
    let mut body = Vec::new();
    for frame in encoder.start()? {
        body.extend_from_slice(frame.as_bytes());
    }
    for event in events {
        for frame in encoder.feed(event)? {
            body.extend_from_slice(frame.as_bytes());
        }
    }
    Ok(body)
}

#[cfg(test)]
#[path = "tool_search_tests.rs"]
mod tests;
