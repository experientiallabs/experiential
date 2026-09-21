//! Rendering of one gateway-executed web search on the three public surfaces.
//!
//! The control plane may run ONE web search before dispatch for a request
//! that asked for web search on a route that cannot serve it natively. It
//! injects the results into the prompt and tells the data plane about the
//! search through `Admission.web_search`; this module turns that fact into
//! the surface-native shapes -- Chat/Responses URL citations over the final
//! assistant text, the Anthropic `server_tool_use` and
//! `web_search_tool_result` blocks, and the request count on every usage
//! object -- without touching a byte of a response whose admission carried
//! no search.

use serde::Deserialize;
use serde_json::{json, Value};

use crate::encode::{compact_json, stable_public_id};
use crate::encode_messages::{
    completed_messages_body_with_reasoning, AggregatedMessage, MessagesSseEncoder,
};
use crate::errors::PublicError;
use crate::events::{CompletedToolCall, Event};

/// The Anthropic server tool name the synthesized blocks carry.
pub const WEB_SEARCH_TOOL_NAME: &str = "web_search";

/// The reserved tool index the synthesized Messages server-tool events use.
/// A provider stream indexes its blocks from zero, so the top of the range
/// can never collide with a provider-issued server tool.
pub const WEB_SEARCH_TOOL_INDEX: u32 = u32::MAX;

/// One ranked search result the control plane injected into the prompt.
#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
pub struct WebSearchSource {
    pub url: String,
    pub title: String,
}

/// The admission's account of the gateway-executed search: the query it ran,
/// how many searches it billed, and the ranked results it injected.
#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
pub struct WebSearchAdmission {
    #[serde(default)]
    pub query: String,
    #[serde(default)]
    pub requests: u32,
    #[serde(default)]
    pub results: Vec<WebSearchSource>,
}

/// Which public citation object shape to render.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CitationShape {
    /// Chat Completions: `{"type":"url_citation","url_citation":{...}}`.
    Chat,
    /// Responses: the flat `{"type":"url_citation","url":...,...}` object.
    Responses,
}

/// Every citation of a source inside `text`, as citation objects in `shape`
/// with `start_index`/`end_index` counted in Unicode scalar values (end
/// exclusive), sorted by `start_index`. A source is cited by a literal
/// occurrence of its URL or by the bracketed result number the prompt frame
/// gave it (`[2]`, `[1, 3]`, `[2][4]`: 1-based ranks, one citation per
/// number, spanning the bracket group). A source never cited yields nothing;
/// an empty URL and an out-of-range number match nothing.
pub fn url_citations(text: &str, sources: &[WebSearchSource], shape: CitationShape) -> Vec<Value> {
    // Byte offset of every char start plus the text end, so a matched byte
    // offset maps to its char index by binary search.
    let mut char_starts: Vec<usize> = text.char_indices().map(|(offset, _)| offset).collect();
    char_starts.push(text.len());
    let char_index = |byte_offset: usize| char_starts.partition_point(|start| *start < byte_offset);
    let mut spans: Vec<(usize, usize, &WebSearchSource)> = Vec::new();
    for source in sources.iter().filter(|source| !source.url.is_empty()) {
        for (byte_offset, matched) in text.match_indices(source.url.as_str()) {
            spans.push((
                char_index(byte_offset),
                char_index(byte_offset + matched.len()),
                source,
            ));
        }
    }
    for (start, end, rank) in numbered_markers(text) {
        if let Some(source) = rank
            .checked_sub(1)
            .and_then(|index| sources.get(index))
            .filter(|source| !source.url.is_empty())
        {
            spans.push((char_index(start), char_index(end), source));
        }
    }
    spans.sort_by_key(|(start, _, _)| *start);
    spans
        .into_iter()
        .map(|(start, end, source)| citation(source, start, end, shape))
        .collect()
}

/// The bracketed result numbers in `text`: `(start_byte, end_byte, rank)` per
/// number, the span covering the whole bracket group. A group is `[` then one
/// or more decimal numbers separated by commas and spaces then `]`, so a
/// markdown link's `[label]` never qualifies; a number above 999 is ignored
/// since no frame carries that many results.
fn numbered_markers(text: &str) -> Vec<(usize, usize, usize)> {
    let mut markers = Vec::new();
    let mut search_from = 0;
    while let Some(open) = text[search_from..].find('[') {
        let open = search_from + open;
        search_from = open + 1;
        let Some(close) = text[open + 1..].find(']') else {
            break;
        };
        let close = open + 1 + close;
        let inner = &text[open + 1..close];
        if inner.is_empty()
            || !inner
                .bytes()
                .all(|byte| byte.is_ascii_digit() || byte == b',' || byte == b' ')
        {
            continue;
        }
        let ranks: Vec<usize> = inner
            .split(',')
            .map(str::trim)
            .filter(|part| !part.is_empty())
            .filter_map(|part| part.parse::<usize>().ok())
            .filter(|rank| (1..=999).contains(rank))
            .collect();
        if ranks.is_empty() {
            continue;
        }
        markers.extend(ranks.into_iter().map(|rank| (open, close + 1, rank)));
        search_from = close + 1;
    }
    markers
}

fn citation(source: &WebSearchSource, start: usize, end: usize, shape: CitationShape) -> Value {
    let fields = json!({
        "url": source.url,
        "title": source.title,
        "start_index": start,
        "end_index": end,
    });
    match shape {
        CitationShape::Chat => json!({"type": "url_citation", "url_citation": fields}),
        CitationShape::Responses => {
            let mut flat = json!({"type": "url_citation"});
            if let (Some(target), Value::Object(entries)) = (flat.as_object_mut(), fields) {
                target.extend(entries);
            }
            flat
        }
    }
}

/// The OpenAI-shaped server tool meter: `{"web_search_requests": N}`.
pub fn server_tool_use_details(requests: u32) -> Value {
    json!({"web_search_requests": requests})
}

/// Add `server_tool_use_details` to one Chat or Responses usage object.
/// A `null` usage (no provider report) stays `null`; an absent search adds
/// nothing, so unsearched responses keep their exact bytes.
pub fn annotate_usage_details(usage: &mut Value, web_search: Option<&WebSearchAdmission>) {
    if let (Some(web_search), Some(entries)) = (web_search, usage.as_object_mut()) {
        entries.insert(
            "server_tool_use_details".to_string(),
            server_tool_use_details(web_search.requests),
        );
    }
}

/// Add Anthropic's `server_tool_use` meter to one Messages usage object.
pub fn annotate_messages_usage(mut usage: Value, requests: Option<u32>) -> Value {
    if let (Some(requests), Some(entries)) = (requests, usage.as_object_mut()) {
        entries.insert(
            "server_tool_use".to_string(),
            server_tool_use_details(requests),
        );
    }
    usage
}

/// Attach the search to one aggregated Chat Completion: the message gains
/// `annotations` (an array, empty when no result URL appears in the text)
/// and the usage object gains the request meter. A failed aggregation has
/// no body object and is left alone.
pub fn annotate_chat_completion(body: &mut Value, web_search: &WebSearchAdmission) {
    let Some(message) = body
        .get_mut("choices")
        .and_then(|choices| choices.get_mut(0))
        .and_then(|choice| choice.get_mut("message"))
    else {
        return;
    };
    let text = message
        .get("content")
        .and_then(Value::as_str)
        .unwrap_or_default();
    let annotations = url_citations(text, &web_search.results, CitationShape::Chat);
    if let Some(entries) = message.as_object_mut() {
        entries.insert("annotations".to_string(), Value::Array(annotations));
    }
    if let Some(usage) = body.get_mut("usage") {
        annotate_usage_details(usage, Some(web_search));
    }
}

/// The Chat encoder's view of the search: the admission plus the assistant
/// text accumulated from the live stream, so the terminal can cite it.
pub struct ChatWebSearch {
    admission: WebSearchAdmission,
    text: String,
}

impl ChatWebSearch {
    pub fn new(admission: WebSearchAdmission) -> Self {
        Self {
            admission,
            text: String::new(),
        }
    }

    /// Accumulate the assistant text a Chat stream returns to the caller.
    pub fn observe(&mut self, event: &Event) {
        if let Event::TextDelta(delta) | Event::ProviderTextDelta { delta, .. } = event {
            self.text.push_str(delta);
        }
    }

    pub fn admission(&self) -> &WebSearchAdmission {
        &self.admission
    }

    /// Chat-shaped citations over everything observed so far.
    pub fn annotations(&self) -> Vec<Value> {
        url_citations(&self.text, &self.admission.results, CitationShape::Chat)
    }
}

/// The stable `server_tool_use` block id for one request.
pub fn web_search_tool_use_id(request_id: &str) -> String {
    stable_public_id("srvtoolu", request_id)
}

/// The Anthropic `web_search_tool_result` block for the injected results.
/// Nothing is encrypted and no page age is known, so those fields carry the
/// documented empty values rather than being omitted.
pub fn web_search_result_block(web_search: &WebSearchAdmission, tool_use_id: &str) -> Value {
    let content: Vec<Value> = web_search
        .results
        .iter()
        .map(|source| {
            json!({
                "type": "web_search_result",
                "url": source.url,
                "title": source.title,
                "encrypted_content": "",
                "page_age": Value::Null,
            })
        })
        .collect();
    json!({
        "type": "web_search_tool_result",
        "tool_use_id": tool_use_id,
        "content": content,
    })
}

/// The normalized events that render the search as Anthropic blocks: one
/// `server_tool_use` (start, its whole input as one delta, completion) and
/// one whole `web_search_tool_result`. Fed to the Messages encoder or
/// aggregation ahead of every provider event, they precede every provider
/// block in the same valid Anthropic order a native rung streams.
pub fn web_search_prelude_events(web_search: &WebSearchAdmission, request_id: &str) -> Vec<Event> {
    let call_id = web_search_tool_use_id(request_id);
    let input = compact_json(&json!({"query": web_search.query}));
    vec![
        Event::ServerToolUseStarted {
            index: WEB_SEARCH_TOOL_INDEX,
            call_id: call_id.clone(),
            name: WEB_SEARCH_TOOL_NAME.to_string(),
        },
        Event::ServerToolArgumentsDelta {
            index: WEB_SEARCH_TOOL_INDEX,
            delta: input.clone(),
        },
        Event::ServerToolUseCompleted {
            index: WEB_SEARCH_TOOL_INDEX,
            call: CompletedToolCall {
                call_id: call_id.clone(),
                name: WEB_SEARCH_TOOL_NAME.to_string(),
                namespace: None,
                caller: None,
                provider_item_id: None,
                provider_status: None,
                raw_arguments: input,
                custom: false,
            },
        },
        Event::ServerToolResult {
            index: WEB_SEARCH_TOOL_INDEX,
            block: compact_json(&web_search_result_block(web_search, &call_id)),
        },
    ]
}

/// What the Messages encoder needs of the search: the request meter for its
/// usage objects and the synthesized block events it feeds itself at start.
#[derive(Debug, Clone)]
pub struct MessagesWebSearch {
    pub requests: u32,
    pub events: Vec<Event>,
}

impl MessagesWebSearch {
    pub fn new(web_search: &WebSearchAdmission, request_id: &str) -> Self {
        Self {
            requests: web_search.requests,
            events: web_search_prelude_events(web_search, request_id),
        }
    }
}

/// Build the Messages encoder's search state from an admission, `None` when
/// the gateway ran no search.
pub fn messages_web_search(
    web_search: Option<&WebSearchAdmission>,
    request_id: &str,
) -> Option<MessagesWebSearch> {
    web_search.map(|web_search| MessagesWebSearch::new(web_search, request_id))
}

/// Configure one Messages encoder for the admission's search, if any.
pub fn configure_messages_encoder(
    encoder: &mut MessagesSseEncoder,
    web_search: Option<&WebSearchAdmission>,
    request_id: &str,
) {
    encoder.set_web_search(messages_web_search(web_search, request_id));
}

/// `completed_messages_body_with_reasoning` with the search rendered ahead
/// of every provider block and metered on the usage object. The synthesized
/// server tool is the gateway's own work, billed through
/// `web_search_requests`, so it never joins the provider `tool_names` the
/// ledger prices per call.
pub fn completed_messages_body_with_web_search(
    request_id: &str,
    model: &str,
    events: &[Event],
    ignored_parameters: &[String],
    reasoning_content_carrier: Option<&str>,
    reasoning_output_exposed: bool,
    web_search: Option<&WebSearchAdmission>,
) -> Result<AggregatedMessage, PublicError> {
    let Some(web_search) = web_search else {
        return completed_messages_body_with_reasoning(
            request_id,
            model,
            events,
            ignored_parameters,
            reasoning_content_carrier,
            reasoning_output_exposed,
        );
    };
    let mut prefixed = web_search_prelude_events(web_search, request_id);
    prefixed.extend(events.iter().cloned());
    let mut aggregated = completed_messages_body_with_reasoning(
        request_id,
        model,
        &prefixed,
        ignored_parameters,
        reasoning_content_carrier,
        reasoning_output_exposed,
    )?;
    let provider_searched = events.iter().any(|event| {
        matches!(event, Event::ServerToolUseCompleted { call, .. } if call.name == WEB_SEARCH_TOOL_NAME)
    });
    if !provider_searched {
        aggregated
            .tool_names
            .retain(|name| name != WEB_SEARCH_TOOL_NAME);
    }
    if let Some(usage) = aggregated.body.get_mut("usage") {
        let metered = annotate_messages_usage(usage.take(), Some(web_search.requests));
        *usage = metered;
    }
    Ok(aggregated)
}

#[cfg(test)]
#[path = "web_search_tests.rs"]
mod tests;
