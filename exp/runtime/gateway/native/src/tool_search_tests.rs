//! Unit tests for gateway-executed tool search: the admission shape, the
//! relay withholder, the bridge round argument and reply, and the Messages,
//! Responses, and Chat renderings with and without rounds.

use super::*;
use crate::encode_responses::{
    completed_responses_body_with_web_search, ResponsesEnvelope, ResponsesSseEncoder,
};
use crate::waterfall::SettledAttempt;
use crate::web_search::WebSearchSource;

fn round(
    call_id: &str,
    query: Option<&str>,
    pattern: Option<&str>,
    matched: &[&str],
) -> ToolSearchRound {
    ToolSearchRound {
        call_id: call_id.to_string(),
        query: query.map(str::to_string),
        pattern: pattern.map(str::to_string),
        matched: matched.iter().map(|name| name.to_string()).collect(),
        matched_tools: Vec::new(),
    }
}

fn started(index: u32, call_id: &str, name: &str) -> Event {
    Event::ToolCallStarted {
        custom: false,
        index,
        call_id: call_id.to_string(),
        name: name.to_string(),
        namespace: None,
        caller: None,
    }
}

fn arguments(index: u32, delta: &str) -> Event {
    Event::ToolArgumentsDelta {
        index,
        delta: delta.to_string(),
    }
}

fn completed(index: u32, call_id: &str, name: &str, raw_arguments: &str) -> Event {
    Event::ToolCallCompleted {
        index,
        call: CompletedToolCall {
            call_id: call_id.to_string(),
            name: name.to_string(),
            namespace: None,
            caller: None,
            provider_item_id: None,
            provider_status: None,
            raw_arguments: raw_arguments.to_string(),
            custom: false,
        },
    }
}

fn usage() -> Usage {
    Usage {
        input_tokens: Some(12),
        output_tokens: Some(7),
        ..Usage::default()
    }
}

fn frame_payload(frame: &str) -> Value {
    let data = frame
        .lines()
        .find_map(|line| line.strip_prefix("data: "))
        .expect("data line");
    serde_json::from_str(data).expect("frame payload is JSON")
}

fn frames_debug(events: &[Event]) -> Vec<String> {
    events.iter().map(|event| format!("{event:?}")).collect()
}

fn web_search() -> WebSearchAdmission {
    WebSearchAdmission {
        query: "stable Python".to_string(),
        requests: 1,
        results: vec![WebSearchSource {
            url: "https://python.org/".to_string(),
            title: "Python".to_string(),
        }],
    }
}

#[test]
fn admission_decodes_the_contract_shape_and_defaults_the_round_budget() {
    let full: ToolSearchAdmission = serde_json::from_value(json!({
        "tool_name": "tool_search",
        "max_rounds": 2,
        "deferred": 42,
        "surface_shape": "messages_bm25",
    }))
    .expect("decodes");
    assert_eq!(full.max_rounds, 2);
    assert_eq!(full.deferred, 42);
    assert_eq!(full.surface_shape, "messages_bm25");
    let sparse: ToolSearchAdmission =
        serde_json::from_value(json!({"tool_name": "gateway_tool_search"})).expect("defaults");
    assert_eq!(sparse.tool_name, "gateway_tool_search");
    assert_eq!(sparse.max_rounds, DEFAULT_MAX_ROUNDS);
    assert_eq!(sparse.deferred, 0);
    assert!(serde_json::from_value::<ToolSearchAdmission>(json!({})).is_err());
}

#[test]
fn withholder_swallows_only_the_named_tool_and_passes_everything_else_untouched() {
    let mut withholder = ToolSearchWithholder::default();
    withholder.set_tool_name(Some("tool_search".to_string()));
    let stream = vec![
        started(0, "call_x", "tool_search"),
        arguments(0, "{\"query\":"),
        started(1, "call_y", "get_weather"),
        arguments(1, "{\"city\":\"Oslo\"}"),
        Event::TextDelta("hi".to_string()),
        arguments(0, "\"weather\"}"),
        completed(0, "call_x", "tool_search", "{\"query\":\"weather\"}"),
        completed(1, "call_y", "get_weather", "{\"city\":\"Oslo\"}"),
        Event::Usage(usage()),
        Event::Completed,
    ];
    let expected = frames_debug(&[
        started(1, "call_y", "get_weather"),
        arguments(1, "{\"city\":\"Oslo\"}"),
        Event::TextDelta("hi".to_string()),
        completed(1, "call_y", "get_weather", "{\"city\":\"Oslo\"}"),
        Event::Usage(usage()),
        Event::Completed,
    ]);
    let mut passed = Vec::new();
    for event in stream {
        // Before the completion lands, the started call is seen but not
        // yet actionable.
        if let Some(kept) = withholder.filter(event) {
            passed.push(kept);
        }
    }
    assert_eq!(frames_debug(&passed), expected);
    assert_eq!(withholder.withheld_count(), 1);
    assert!(withholder.withheld_any());
    let taken = withholder.take_withheld();
    assert_eq!(
        taken,
        vec![WithheldSearchCall {
            index: 0,
            call_id: "call_x".to_string(),
            name: "tool_search".to_string(),
            raw_arguments: "{\"query\":\"weather\"}".to_string(),
        }]
    );
    assert_eq!(withholder.withheld_count(), 0);
    assert!(!withholder.withheld_any());
}

#[test]
fn withholder_without_a_tool_name_withholds_nothing_and_an_open_call_counts_as_seen() {
    let mut inert = ToolSearchWithholder::default();
    let stream = vec![
        started(0, "call_x", "tool_search"),
        arguments(0, "{}"),
        completed(0, "call_x", "tool_search", "{}"),
    ];
    let mut passed = Vec::new();
    for event in stream.clone() {
        passed.extend(inert.filter(event));
    }
    assert_eq!(frames_debug(&passed), frames_debug(&stream));
    assert_eq!(inert.withheld_count(), 0);
    assert!(!inert.withheld_any());
    // An empty name is the same as none.
    inert.set_tool_name(Some(String::new()));
    assert!(inert.filter(started(0, "call_x", "tool_search")).is_some());

    let mut open = ToolSearchWithholder::default();
    open.set_tool_name(Some("tool_search".to_string()));
    assert!(open.filter(started(3, "call_z", "tool_search")).is_none());
    assert!(open.filter(arguments(3, "{")).is_none());
    assert_eq!(
        open.withheld_count(),
        0,
        "an unfinished call is not actionable"
    );
    assert!(
        open.withheld_any(),
        "but it was seen, so a commit discloses it"
    );
}

#[test]
fn round_argument_carries_the_calls_verbatim_and_the_settle_shaped_usage() {
    let calls = vec![WithheldSearchCall {
        index: 0,
        call_id: "call_x".to_string(),
        name: "tool_search".to_string(),
        raw_arguments: "{\"query\":\"weather\"}".to_string(),
    }];
    let argument: Value =
        serde_json::from_str(&round_argument("req-1", 2, 1, &calls, Some(&usage())))
            .expect("argument is JSON");
    assert_eq!(
        argument,
        json!({
            "request_id": "req-1",
            "route_depth": 2,
            "round": 1,
            "calls": [{"call_id": "call_x", "name": "tool_search", "arguments": "{\"query\":\"weather\"}"}],
            "usage": {
                "input_tokens": 12,
                "output_tokens": 7,
                "cached_input_tokens": Value::Null,
                "cache_creation_input_tokens": Value::Null,
                "cache_creation_1h_input_tokens": Value::Null,
                "reasoning_tokens": Value::Null,
            },
        })
    );
    let unreported: Value =
        serde_json::from_str(&round_argument("req-1", 0, 1, &[], None)).expect("argument is JSON");
    assert_eq!(unreported["usage"], Value::Null);
    assert_eq!(unreported["calls"], json!([]));
}

#[test]
fn round_reply_decodes_the_wire_the_rounds_and_the_exhausted_flag() {
    let reply: ToolSearchRoundReply = serde_json::from_value(json!({
        "wire": {
            "provider": "openai",
            "deployment_id": "a",
            "dialect": "openai_compatible",
            "url": "https://provider.test/v1/chat/completions",
            "headers": {},
            "timeout_seconds": 10.0,
            "idempotency_key": "op-a",
            "upstream_payload": {"messages": []},
        },
        "rounds": [{"call_id": "call_x", "query": "weather", "pattern": null, "matched": ["get_weather"]}],
        "exhausted": true,
    }))
    .expect("reply decodes");
    assert_eq!(reply.wire.deployment_id, "a");
    assert_eq!(reply.rounds.len(), 1);
    assert_eq!(reply.rounds[0].query.as_deref(), Some("weather"));
    assert_eq!(reply.rounds[0].matched, vec!["get_weather".to_string()]);
    assert!(reply.rounds[0].matched_tools.is_empty());
    assert!(reply.exhausted);
    let sparse: ToolSearchRoundReply = serde_json::from_value(json!({"wire": {
        "provider": "openai", "deployment_id": "a", "dialect": "openai_compatible",
        "url": "https://provider.test", "headers": {}, "timeout_seconds": 1.0, "idempotency_key": "k",
    }}))
    .expect("sparse reply decodes");
    assert!(sparse.rounds.is_empty());
    assert!(!sparse.exhausted);
}

#[test]
fn messages_prelude_renders_each_round_as_server_tool_use_and_result_under_reserved_indexes() {
    let rounds = [
        round(
            "call_x",
            Some("weather"),
            None,
            &["get_weather", "get_forecast"],
        ),
        round("call_y", None, Some("^db_"), &[]),
    ];
    let events = messages_prelude_events(&rounds, "request-abc");
    assert_eq!(events.len(), 8);
    match &events[0] {
        Event::ServerToolUseStarted {
            index,
            call_id,
            name,
        } => {
            assert_eq!(*index, u32::MAX - 1);
            assert_eq!(call_id, &tool_search_tool_use_id("request-abc", 0));
            assert!(call_id.starts_with("srvtoolu_"));
            assert_eq!(name, TOOL_SEARCH_BM25_TOOL_NAME);
        }
        other => panic!("unexpected first event {other:?}"),
    }
    match (&events[1], &events[2]) {
        (
            Event::ServerToolArgumentsDelta { delta, .. },
            Event::ServerToolUseCompleted { call, .. },
        ) => {
            assert_eq!(delta, "{\"query\":\"weather\"}");
            assert_eq!(delta, &call.raw_arguments);
        }
        other => panic!("unexpected middle events {other:?}"),
    }
    match &events[3] {
        Event::ServerToolResult { index, block } => {
            assert_eq!(*index, u32::MAX - 1);
            let block: Value = serde_json::from_str(block).expect("block is JSON");
            assert_eq!(
                block,
                json!({
                    "type": "tool_search_tool_result",
                    "tool_use_id": tool_search_tool_use_id("request-abc", 0),
                    "content": {
                        "type": "tool_search_tool_search_result",
                        "tool_references": [
                            {"type": "tool_reference", "tool_name": "get_weather"},
                            {"type": "tool_reference", "tool_name": "get_forecast"},
                        ],
                    },
                })
            );
        }
        other => panic!("unexpected result event {other:?}"),
    }
    // The regex round: its own reserved index, the regex tool, an empty list.
    match (&events[4], &events[5], &events[7]) {
        (
            Event::ServerToolUseStarted { index, name, .. },
            Event::ServerToolArgumentsDelta { delta, .. },
            Event::ServerToolResult { block, .. },
        ) => {
            assert_eq!(*index, u32::MAX - 2);
            assert_eq!(name, TOOL_SEARCH_REGEX_TOOL_NAME);
            assert_eq!(delta, "{\"pattern\":\"^db_\"}");
            let block: Value = serde_json::from_str(block).expect("block is JSON");
            assert_eq!(block["content"]["tool_references"], json!([]));
        }
        other => panic!("unexpected regex round events {other:?}"),
    }
    assert_ne!(
        tool_search_tool_use_id("request-abc", 0),
        tool_search_tool_use_id("request-abc", 1)
    );
    assert!(messages_tool_search(&[], "request-abc").is_none());
}

#[test]
fn messages_encoder_streams_the_rounds_ahead_of_the_answer_and_meters_them() {
    let rounds = [round("call_x", Some("weather"), None, &["get_weather"])];
    let mut encoder = MessagesSseEncoder::new("request-abc", "alias");
    encoder.set_tool_search(messages_tool_search(&rounds, "request-abc"));
    let start = encoder.start().expect("starts");
    let payloads: Vec<Value> = start.iter().map(|frame| frame_payload(frame)).collect();
    let blocks: Vec<&Value> = payloads
        .iter()
        .filter(|payload| payload["type"] == json!("content_block_start"))
        .map(|payload| &payload["content_block"])
        .collect();
    assert_eq!(blocks.len(), 2);
    assert_eq!(
        blocks[0],
        &json!({
            "type": "server_tool_use",
            "id": tool_search_tool_use_id("request-abc", 0),
            "name": "tool_search_tool_bm25",
            "input": {},
        })
    );
    assert_eq!(blocks[1]["type"], json!("tool_search_tool_result"));
    assert_eq!(
        blocks[1]["content"]["tool_references"][0]["tool_name"],
        json!("get_weather")
    );
    let input_delta = payloads
        .iter()
        .find(|payload| payload["type"] == json!("content_block_delta"))
        .expect("the input streams as one delta");
    assert_eq!(
        input_delta["delta"]["partial_json"],
        json!("{\"query\":\"weather\"}")
    );
    assert_eq!(
        payloads[0]["message"]["usage"]["server_tool_use"],
        json!({"tool_search_requests": 1})
    );
    assert!(
        !encoder.has_content_blocks(),
        "the gateway's own blocks never count as provider output"
    );
    let mut frames = encoder
        .feed(&Event::TextDelta("hi".to_string()))
        .expect("text");
    frames.extend(encoder.feed(&Event::Usage(usage())).expect("usage"));
    frames.extend(encoder.feed(&Event::Completed).expect("terminal"));
    assert!(encoder.has_content_blocks());
    let delta = frames
        .iter()
        .map(|frame| frame_payload(frame))
        .find(|payload| payload["type"] == json!("message_delta"))
        .expect("message_delta");
    assert_eq!(delta["delta"]["stop_reason"], json!("end_turn"));
    assert_eq!(
        delta["usage"]["server_tool_use"]["tool_search_requests"],
        json!(1)
    );
}

#[test]
fn completed_messages_body_renders_rounds_after_the_web_search_and_meters_both() {
    let rounds = [round("call_x", Some("weather"), None, &["get_weather"])];
    let events = [
        Event::TextDelta("hi".to_string()),
        Event::Usage(usage()),
        Event::Completed,
    ];
    let aggregated = completed_messages_body_with_gateway_tools(
        "request-abc",
        "alias",
        &events,
        &[],
        None,
        false,
        Some(&web_search()),
        messages_tool_search(&rounds, "request-abc").as_ref(),
    )
    .expect("aggregates");
    let content = aggregated.body["content"].as_array().expect("content");
    let types: Vec<&str> = content
        .iter()
        .map(|block| block["type"].as_str().expect("type"))
        .collect();
    assert_eq!(
        types,
        vec![
            "server_tool_use",
            "web_search_tool_result",
            "server_tool_use",
            "tool_search_tool_result",
            "text",
        ]
    );
    assert_eq!(content[2]["name"], json!("tool_search_tool_bm25"));
    assert_eq!(content[2]["input"], json!({"query": "weather"}));
    assert_eq!(
        aggregated.body["usage"]["server_tool_use"],
        json!({"web_search_requests": 1, "tool_search_requests": 1})
    );
    assert_eq!(aggregated.body["stop_reason"], json!("end_turn"));
    assert!(
        aggregated.tool_names.is_empty(),
        "gateway tools never join the priced tool names: {:?}",
        aggregated.tool_names
    );

    let only_rounds = completed_messages_body_with_gateway_tools(
        "request-abc",
        "alias",
        &events,
        &[],
        None,
        false,
        None,
        messages_tool_search(&rounds, "request-abc").as_ref(),
    )
    .expect("aggregates");
    assert_eq!(
        only_rounds.body["usage"]["server_tool_use"],
        json!({"tool_search_requests": 1})
    );
}

#[test]
fn responses_prelude_renders_hosted_call_and_output_items_ahead_of_the_answer() {
    let mut with_declarations = round("call_x", Some("weather"), None, &["get_weather"]);
    with_declarations.matched_tools = vec![json!({
        "type": "function",
        "name": "get_weather",
        "description": "Current weather",
        "parameters": {"type": "object", "properties": {}},
    })];
    let rounds = [
        with_declarations.clone(),
        round("call_y", None, Some("^db_"), &["db_query", "db_list"]),
    ];
    let events = responses_prelude_events(&rounds, "request-abc");
    assert_eq!(events.len(), 8);
    let indexes: Vec<u32> = events
        .iter()
        .filter_map(|event| match event {
            Event::HostedToolItemStarted { output_index, .. } => Some(*output_index),
            _ => None,
        })
        .collect();
    assert_eq!(
        indexes,
        vec![u32::MAX - 1, u32::MAX - 2, u32::MAX - 3, u32::MAX - 4]
    );

    let stream = [
        Event::TextDelta("hi".to_string()),
        Event::Usage(usage()),
        Event::Completed,
    ];
    let aggregated = completed_responses_body_with_gateway_tools(
        "request-abc",
        "alias",
        7,
        ResponsesEnvelope::default(),
        &stream,
        None,
        None,
        responses_tool_search(&rounds, "request-abc").as_ref(),
    )
    .expect("aggregates");
    let output = aggregated.body["output"].as_array().expect("output");
    let types: Vec<&str> = output
        .iter()
        .map(|item| item["type"].as_str().expect("type"))
        .collect();
    assert_eq!(
        types,
        vec![
            "tool_search_call",
            "tool_search_output",
            "tool_search_call",
            "tool_search_output",
            "message",
        ]
    );
    assert_eq!(
        output[0],
        json!({
            "type": "tool_search_call",
            "id": tool_search_call_item_id("request-abc", 0),
            "call_id": "call_x",
            "status": "completed",
            "execution": "server",
            "arguments": {"goal": "weather"},
        })
    );
    assert!(output[0]["id"].as_str().expect("id").starts_with("tsc_"));
    assert!(output[1]["id"].as_str().expect("id").starts_with("tso_"));
    // Full declarations when the control plane sent them, names otherwise.
    assert_eq!(output[1]["tools"], json!(with_declarations.matched_tools));
    assert_eq!(output[2]["arguments"], json!({"pattern": "^db_"}));
    assert_eq!(
        output[3]["tools"],
        json!([
            {"type": "function", "name": "db_query"},
            {"type": "function", "name": "db_list"},
        ])
    );
    assert_eq!(output[4]["content"][0]["text"], json!("hi"));
    assert_eq!(
        aggregated.body["usage"]["server_tool_use_details"],
        json!({"tool_search_requests": 2})
    );
    assert!(
        aggregated.tool_names.is_empty(),
        "hosted search items never join the priced tool names: {:?}",
        aggregated.tool_names
    );
}

#[test]
fn responses_encoder_streams_the_items_at_start_and_meters_the_web_search_beside_them() {
    let rounds = [round("call_x", Some("weather"), None, &["get_weather"])];
    let mut encoder =
        ResponsesSseEncoder::new("request-abc", "alias", 7, ResponsesEnvelope::default());
    encoder.set_web_search(Some(web_search()));
    encoder.set_tool_search(responses_tool_search(&rounds, "request-abc"));
    let start = encoder.start().expect("starts");
    let types: Vec<String> = start
        .iter()
        .map(|frame| {
            frame_payload(frame)["type"]
                .as_str()
                .expect("type")
                .to_string()
        })
        .collect();
    assert_eq!(
        types,
        vec![
            "response.created",
            "response.in_progress",
            "response.output_item.added",
            "response.output_item.done",
            "response.output_item.added",
            "response.output_item.done",
        ]
    );
    assert_eq!(frame_payload(&start[2])["output_index"], json!(0));
    assert_eq!(
        frame_payload(&start[2])["item"]["type"],
        json!("tool_search_call")
    );
    assert_eq!(frame_payload(&start[4])["output_index"], json!(1));
    let mut frames = encoder
        .feed(&Event::TextDelta("hi".to_string()))
        .expect("text");
    frames.extend(encoder.feed(&Event::Usage(usage())).expect("usage"));
    frames.extend(encoder.feed(&Event::Completed).expect("terminal"));
    let terminal = frame_payload(frames.last().expect("terminal frame"));
    assert_eq!(terminal["type"], json!("response.completed"));
    assert_eq!(
        terminal["response"]["usage"]["server_tool_use_details"],
        json!({"web_search_requests": 1, "tool_search_requests": 1})
    );
    assert_eq!(terminal["response"]["output"][2]["type"], json!("message"));
}

#[test]
fn chat_meters_the_rounds_on_usage_and_renders_nothing_else() {
    let mut encoder = ChatSseEncoder::new_with_ignored("request-abc", "alias", 7, true, Vec::new());
    encoder.set_tool_search_requests(2);
    encoder.start().expect("starts");
    let mut frames = encoder
        .feed(&Event::TextDelta("hi".to_string()))
        .expect("text");
    frames.extend(encoder.feed(&Event::Usage(usage())).expect("usage"));
    frames.extend(encoder.feed(&Event::Completed).expect("terminal"));
    let usage_chunk = frames
        .iter()
        .filter(|frame| frame.contains("\"usage\":"))
        .map(|frame| frame_payload(frame))
        .find(|payload| payload["choices"] == json!([]))
        .expect("usage chunk");
    assert_eq!(
        usage_chunk["usage"]["server_tool_use_details"],
        json!({"tool_search_requests": 2})
    );

    let mut body =
        json!({"choices": [{"message": {"content": "hi"}}], "usage": {"prompt_tokens": 1}});
    annotate_chat_completion_tool_search(&mut body, 1);
    assert_eq!(
        body["usage"]["server_tool_use_details"],
        json!({"tool_search_requests": 1})
    );
    assert!(body["choices"][0]["message"].get("annotations").is_none());
    let mut untouched = json!({"usage": {"prompt_tokens": 1}});
    annotate_chat_completion_tool_search(&mut untouched, 0);
    assert_eq!(untouched, json!({"usage": {"prompt_tokens": 1}}));
    let mut null_usage = json!({"usage": Value::Null});
    annotate_chat_completion_tool_search(&mut null_usage, 3);
    assert_eq!(null_usage["usage"], Value::Null);
}

#[test]
fn absent_tool_search_leaves_every_surface_byte_identical() {
    let events = [
        Event::TextDelta("hi".to_string()),
        Event::Usage(usage()),
        Event::Completed,
    ];
    let render_messages = |configure: bool| -> Vec<String> {
        let mut encoder = MessagesSseEncoder::new("request-abc", "alias");
        if configure {
            encoder.set_tool_search(None);
        }
        let mut frames = encoder.start().expect("starts");
        for event in &events {
            frames.extend(encoder.feed(event).expect("encodes"));
        }
        frames
    };
    assert_eq!(render_messages(false), render_messages(true));
    let render_responses = |configure: bool| -> Vec<String> {
        let mut encoder =
            ResponsesSseEncoder::new("request-abc", "alias", 7, ResponsesEnvelope::default());
        if configure {
            encoder.set_tool_search(None);
        }
        let mut frames = encoder.start().expect("starts");
        for event in &events {
            frames.extend(encoder.feed(event).expect("encodes"));
        }
        frames
    };
    assert_eq!(render_responses(false), render_responses(true));
    let render_chat = |configure: bool| -> Vec<String> {
        let mut encoder =
            ChatSseEncoder::new_with_ignored("request-abc", "alias", 7, true, Vec::new());
        if configure {
            encoder.set_tool_search_requests(0);
        }
        let mut frames = encoder.start().expect("starts");
        for event in &events {
            frames.extend(encoder.feed(event).expect("encodes"));
        }
        frames
    };
    assert_eq!(render_chat(false), render_chat(true));

    let plain = completed_messages_body_with_web_search(
        "request-abc",
        "alias",
        &events,
        &[],
        None,
        false,
        Some(&web_search()),
    )
    .expect("aggregates");
    let with_none = completed_messages_body_with_gateway_tools(
        "request-abc",
        "alias",
        &events,
        &[],
        None,
        false,
        Some(&web_search()),
        None,
    )
    .expect("aggregates");
    assert_eq!(plain.body, with_none.body);
    assert_eq!(plain.tool_names, with_none.tool_names);
    let plain = completed_responses_body_with_web_search(
        "request-abc",
        "alias",
        7,
        ResponsesEnvelope::default(),
        &events,
        None,
        None,
    )
    .expect("aggregates");
    let with_none = completed_responses_body_with_gateway_tools(
        "request-abc",
        "alias",
        7,
        ResponsesEnvelope::default(),
        &events,
        None,
        None,
        None,
    )
    .expect("aggregates");
    assert_eq!(plain.body, with_none.body);
    assert!(plain.body["usage"].get("server_tool_use_details").is_none());
}

#[test]
fn usage_meters_extend_an_existing_server_tool_object_and_never_invent_one() {
    let metered = annotate_messages_tool_search_usage(
        json!({"input_tokens": 1, "server_tool_use": {"web_search_requests": 1}}),
        Some(2),
    );
    assert_eq!(
        metered["server_tool_use"],
        json!({"web_search_requests": 1, "tool_search_requests": 2})
    );
    let fresh = annotate_messages_tool_search_usage(json!({"input_tokens": 1}), Some(1));
    assert_eq!(fresh["server_tool_use"], json!({"tool_search_requests": 1}));
    let untouched = annotate_messages_tool_search_usage(json!({"input_tokens": 1}), None);
    assert_eq!(untouched, json!({"input_tokens": 1}));
    assert_eq!(
        annotate_messages_tool_search_usage(Value::Null, Some(1)),
        Value::Null
    );
    let mut details = json!({"server_tool_use_details": {"web_search_requests": 1}});
    annotate_tool_search_usage_details(&mut details, 3);
    assert_eq!(
        details["server_tool_use_details"],
        json!({"web_search_requests": 1, "tool_search_requests": 3})
    );
}

fn admission() -> Admission {
    serde_json::from_value(json!({
        "request_id": "request-abc",
        "alias": "alias",
        "alias_revision_id": "rev",
        "stream": false,
        "include_usage": false,
        "exact_model_id": "model",
        "route_reason": "test",
        "route": [],
        "maximum_total_attempts": 8,
        "maximum_same_deployment_attempts": 2,
        "tool_search": {"tool_name": "tool_search"},
    }))
    .expect("admission decodes")
}

#[test]
fn adopt_outcome_moves_a_settled_attempts_rounds_onto_the_admission() {
    let mut admission = admission();
    assert_eq!(
        admission
            .tool_search
            .as_ref()
            .map(|search| search.max_rounds),
        Some(DEFAULT_MAX_ROUNDS)
    );
    assert!(admission.tool_search_rounds.is_empty());
    let mut won = Won::Settled(SettledAttempt {
        depth: 0,
        events: vec![Event::Completed],
        encrypted_reasoning_stripped: false,
        empty_completion: false,
        tool_search_rounds: vec![round("call_x", Some("weather"), None, &["get_weather"])],
    });
    adopt_outcome(&mut admission, &mut won);
    assert_eq!(admission.tool_search_rounds.len(), 1);
    assert_eq!(admission.tool_search_rounds[0].call_id, "call_x");
    assert!(admission.ignored_parameters.is_empty());
    match &won {
        Won::Settled(settled) => assert!(settled.tool_search_rounds.is_empty()),
        _ => panic!("the outcome keeps its shape"),
    }
    let mut failed = Won::Failed(PublicError::internal());
    adopt_outcome(&mut admission, &mut failed);
    assert_eq!(admission.tool_search_rounds.len(), 1);
    // The Messages surface renders the adopted rounds from the admission.
    let aggregated = completed_messages_body_for(&admission, &[Event::Completed], None, false)
        .expect("aggregates");
    assert_eq!(
        aggregated.body["content"][0]["type"],
        json!("server_tool_use")
    );
    assert_eq!(
        aggregated.body["usage"]["server_tool_use"]["tool_search_requests"],
        json!(1)
    );
}
