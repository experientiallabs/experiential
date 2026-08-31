//! Inline tests for `encode_messages`, split into a submodule file so the
//! implementation stays within the repository line budget.

use super::*;
use crate::events::CompletedToolCall;

#[test]
fn usage_reports_cached_reads_out_of_the_input_total() {
    let usage = Usage {
        input_tokens: Some(10),
        output_tokens: Some(4),
        cached_input_tokens: Some(3),
        reasoning_tokens: None,
    };
    assert_eq!(
        messages_usage(Some(&usage)),
        json!({"input_tokens": 7, "output_tokens": 4, "cache_read_input_tokens": 3})
    );
    assert_eq!(
        messages_usage(None),
        json!({"input_tokens": 0, "output_tokens": 0})
    );
}

#[test]
fn error_body_folds_param_and_maps_status_first() {
    let mut error = PublicError::new(
        400,
        "invalid_parameter",
        "Invalid value.",
        "invalid_request_error",
    );
    error.param = Some("top_k".to_string());
    assert_eq!(
        anthropic_error_body(&error),
        json!({
            "type": "error",
            "error": {
                "type": "invalid_request_error",
                "message": "Invalid value. (param: top_k)",
            },
        })
    );
    let throttled = PublicError::new(429, "unavailable_route", "Throttled.", "api_error");
    assert_eq!(
        anthropic_error_body(&throttled)["error"]["type"],
        json!("rate_limit_error")
    );
}

#[test]
fn completed_body_orders_text_before_tool_use_blocks() {
    let events = vec![
        Event::TextDelta("hi".to_string()),
        Event::ToolCallStarted {
            index: 0,
            call_id: "call-1".to_string(),
            name: "search".to_string(),
        },
        Event::ToolArgumentsDelta {
            index: 0,
            delta: "{\"b\":1,\"a\":2}".to_string(),
        },
        Event::ToolCallCompleted {
            index: 0,
            call: CompletedToolCall {
                call_id: "call-1".to_string(),
                name: "search".to_string(),
                provider_item_id: None,
                provider_status: None,
                raw_arguments: "{\"b\":1,\"a\":2}".to_string(),
                custom: false,
            },
        },
        Event::Completed,
    ];
    let aggregated = completed_messages_body("request-abc", "coding", &events).expect("aggregates");
    assert!(aggregated.failure.is_none());
    assert_eq!(aggregated.body["stop_reason"], json!("tool_use"));
    assert_eq!(aggregated.body["content"][0]["type"], json!("text"));
    // preserve_order keeps the provider's key order in the parsed input.
    assert_eq!(
        compact_json(&aggregated.body["content"][1]["input"]),
        "{\"b\":1,\"a\":2}"
    );
    assert_eq!(aggregated.tool_names, vec!["search".to_string()]);
}

#[test]
fn completed_body_preserves_interleaved_block_order() {
    let events = vec![
        Event::ToolCallStarted {
            index: 0,
            call_id: "call-1".to_string(),
            name: "search".to_string(),
        },
        Event::ToolCallCompleted {
            index: 0,
            call: CompletedToolCall {
                call_id: "call-1".to_string(),
                name: "search".to_string(),
                provider_item_id: None,
                provider_status: None,
                raw_arguments: "{}".to_string(),
                custom: false,
            },
        },
        Event::TextDelta("after ".to_string()),
        Event::TextDelta("the tool".to_string()),
        Event::Completed,
    ];
    let aggregated = completed_messages_body("request-abc", "coding", &events).expect("aggregates");
    assert_eq!(aggregated.body["content"][0]["type"], json!("tool_use"));
    assert_eq!(
        aggregated.body["content"][1],
        json!({"type": "text", "text": "after the tool"})
    );
}

#[test]
fn deferred_tool_completion_keeps_the_started_block_position() {
    // OpenAI-compatible streams complete every tool only at [DONE], so
    // text may arrive between the tool's arguments and its completion.
    let events = vec![
        Event::ToolCallStarted {
            index: 0,
            call_id: "call-1".to_string(),
            name: "search".to_string(),
        },
        Event::ToolArgumentsDelta {
            index: 0,
            delta: "{}".to_string(),
        },
        Event::TextDelta("after".to_string()),
        Event::ToolCallCompleted {
            index: 0,
            call: CompletedToolCall {
                call_id: "call-1".to_string(),
                name: "search".to_string(),
                provider_item_id: None,
                provider_status: None,
                raw_arguments: "{}".to_string(),
                custom: false,
            },
        },
        Event::Completed,
    ];
    let aggregated = completed_messages_body("request-abc", "coding", &events).expect("aggregates");
    assert_eq!(aggregated.body["content"][0]["type"], json!("tool_use"));
    assert_eq!(
        aggregated.body["content"][1],
        json!({"type": "text", "text": "after"})
    );
    let mut encoder = MessagesSseEncoder::new("request-abc", "coding");
    let mut frames = encoder.start().expect("starts");
    for event in &events {
        frames.extend(
            encoder
                .feed(event)
                .expect("streams the deferred completion"),
        );
    }
    assert!(frames.last().expect("terminal").contains("message_stop"));
}

#[test]
fn interleaved_parallel_tools_stream_strictly_sequential_blocks() {
    // Tool A streams live through the interleaving; tool B's fragment
    // buffers and flushes as one delta after A's block closes.
    let events = vec![
        Event::ToolCallStarted {
            index: 0,
            call_id: "call-a".to_string(),
            name: "alpha".to_string(),
        },
        Event::ToolArgumentsDelta {
            index: 0,
            delta: "{\"a\": ".to_string(),
        },
        Event::ToolCallStarted {
            index: 1,
            call_id: "call-b".to_string(),
            name: "beta".to_string(),
        },
        Event::ToolArgumentsDelta {
            index: 1,
            delta: "{\"b\": 2}".to_string(),
        },
        Event::ToolArgumentsDelta {
            index: 0,
            delta: "1}".to_string(),
        },
        Event::ToolCallCompleted {
            index: 0,
            call: CompletedToolCall {
                call_id: "call-a".to_string(),
                name: "alpha".to_string(),
                provider_item_id: None,
                provider_status: None,
                raw_arguments: "{\"a\": 1}".to_string(),
                custom: false,
            },
        },
        Event::ToolCallCompleted {
            index: 1,
            call: CompletedToolCall {
                call_id: "call-b".to_string(),
                name: "beta".to_string(),
                provider_item_id: None,
                provider_status: None,
                raw_arguments: "{\"b\": 2}".to_string(),
                custom: false,
            },
        },
        Event::Completed,
    ];
    let mut encoder = MessagesSseEncoder::new("request-abc", "coding");
    let mut frames = encoder.start().expect("starts");
    for event in &events {
        frames.extend(encoder.feed(event).expect("streams the interleaving"));
    }
    let names: Vec<&str> = frames
        .iter()
        .map(|frame| {
            frame
                .lines()
                .next()
                .and_then(|line| line.strip_prefix("event: "))
                .expect("named frame")
        })
        .collect();
    assert_eq!(
        names,
        vec![
            "message_start",
            "ping",
            "content_block_start",
            "content_block_delta",
            "content_block_delta",
            "content_block_stop",
            "content_block_start",
            "content_block_delta",
            "content_block_stop",
            "message_delta",
            "message_stop",
        ]
    );
    // The buffered tool-B fragment flushes as one input_json_delta on
    // Anthropic block index 1 after block 0 closes.
    assert!(frames[7].contains("\"index\":1"));
    assert!(frames[7].contains("{\\\"b\\\": 2}"));
    let aggregated = completed_messages_body("request-abc", "coding", &events).expect("aggregates");
    assert_eq!(aggregated.body["content"][0]["id"], json!("call-a"));
    assert_eq!(aggregated.body["content"][1]["id"], json!("call-b"));
}

#[test]
fn refusal_content_aggregates_as_a_sanitized_failure() {
    let events = vec![Event::RefusalDelta("no".to_string()), Event::Completed];
    let aggregated = completed_messages_body("request-abc", "coding", &events).expect("aggregates");
    let failure = aggregated.failure.expect("refusal failure");
    assert_eq!(failure.failure_class, FailureClass::Refusal);
}

#[test]
fn thinking_blocks_stream_in_valid_anthropic_order() {
    let events = vec![
        Event::ThinkingDelta {
            index: 0,
            delta: "step ".to_string(),
        },
        Event::ThinkingDelta {
            index: 0,
            delta: "one".to_string(),
        },
        Event::ThinkingSignature {
            index: 0,
            signature: "sig==".to_string(),
        },
        Event::RedactedThinking {
            index: 1,
            data: "opaque==".to_string(),
        },
        Event::TextDelta("answer".to_string()),
        Event::Completed,
    ];
    let mut encoder = MessagesSseEncoder::new("request-abc", "coding");
    let mut frames = encoder.start().expect("starts");
    for event in &events {
        frames.extend(encoder.feed(event).expect("streams thinking"));
    }
    let names: Vec<&str> = frames
        .iter()
        .map(|frame| {
            frame
                .lines()
                .next()
                .and_then(|line| line.strip_prefix("event: "))
                .expect("named frame")
        })
        .collect();
    assert_eq!(
        names,
        vec![
            "message_start",
            "ping",
            "content_block_start",
            "content_block_delta",
            "content_block_delta",
            "content_block_delta",
            "content_block_stop",
            "content_block_start",
            "content_block_stop",
            "content_block_start",
            "content_block_delta",
            "content_block_stop",
            "message_delta",
            "message_stop",
        ]
    );
    // The thinking block opens with the SDK-required empty fields, streams
    // its text, and closes with one signature_delta before its stop.
    assert!(frames[2].contains("{\"type\":\"thinking\",\"thinking\":\"\",\"signature\":\"\"}"));
    assert!(frames[3].contains("{\"type\":\"thinking_delta\",\"thinking\":\"step \"}"));
    assert!(frames[5].contains("{\"type\":\"signature_delta\",\"signature\":\"sig==\"}"));
    assert!(frames[7].contains("{\"type\":\"redacted_thinking\",\"data\":\"opaque==\"}"));
    assert!(frames[10].contains("{\"type\":\"text_delta\",\"text\":\"answer\"}"));
}

#[test]
fn completed_body_carries_thinking_blocks_verbatim_and_in_order() {
    let events = vec![
        Event::ThinkingDelta {
            index: 0,
            delta: "step one".to_string(),
        },
        Event::ThinkingSignature {
            index: 0,
            signature: "sig==".to_string(),
        },
        Event::RedactedThinking {
            index: 1,
            data: "opaque==".to_string(),
        },
        Event::TextDelta("answer".to_string()),
        Event::Completed,
    ];
    let aggregated = completed_messages_body("request-abc", "coding", &events).expect("aggregates");
    assert!(aggregated.failure.is_none());
    assert_eq!(
        aggregated.body["content"],
        json!([
            {"type": "thinking", "thinking": "step one", "signature": "sig=="},
            {"type": "redacted_thinking", "data": "opaque=="},
            {"type": "text", "text": "answer"},
        ])
    );
    assert_eq!(aggregated.body["stop_reason"], json!("end_turn"));
}

#[test]
fn interleaved_thinking_between_tool_blocks_keeps_sequential_indices() {
    // Interleaved thinking may arrive between tool calls; blocks stay
    // strictly sequential in start order.
    let events = vec![
        Event::ThinkingDelta {
            index: 0,
            delta: "plan".to_string(),
        },
        Event::ThinkingSignature {
            index: 0,
            signature: "sig-a".to_string(),
        },
        Event::ToolCallStarted {
            index: 0,
            call_id: "call-1".to_string(),
            name: "search".to_string(),
        },
        Event::ToolArgumentsDelta {
            index: 0,
            delta: "{}".to_string(),
        },
        Event::ToolCallCompleted {
            index: 0,
            call: CompletedToolCall {
                call_id: "call-1".to_string(),
                name: "search".to_string(),
                provider_item_id: None,
                provider_status: None,
                raw_arguments: "{}".to_string(),
                custom: false,
            },
        },
        Event::Completed,
    ];
    let mut encoder = MessagesSseEncoder::new("request-abc", "coding");
    let mut frames = encoder.start().expect("starts");
    for event in &events {
        frames.extend(encoder.feed(event).expect("streams interleaving"));
    }
    assert!(frames.last().expect("terminal").contains("message_stop"));
    let aggregated = completed_messages_body("request-abc", "coding", &events).expect("aggregates");
    assert_eq!(aggregated.body["content"][0]["type"], json!("thinking"));
    assert_eq!(aggregated.body["content"][1]["type"], json!("tool_use"));
    assert_eq!(aggregated.body["stop_reason"], json!("tool_use"));
}

#[test]
fn server_tool_blocks_stream_whole_and_paused_turns_keep_pause_turn() {
    // The live web_search shape (captured 2026-08-31): a server_tool_use
    // block, its whole result block, then cited answer text; a paused turn
    // must reach the caller as the provider's own pause_turn stop reason.
    let search = json!({
        "type": "server_tool_use",
        "id": "srvtoolu_1",
        "name": "web_search",
        "input": {"query": "current UTC date"},
    });
    let result = json!({
        "type": "web_search_tool_result",
        "tool_use_id": "srvtoolu_1",
        "caller": {"type": "direct"},
        "content": [{"type": "web_search_result", "title": "UTC", "url": "https://utc.test"}],
    });
    let mut encoder = MessagesSseEncoder::new("req-1", "sonnet");
    let mut frames = encoder.start().expect("start");
    frames.extend(
        encoder
            .feed(&Event::ServerToolBlock {
                index: 0,
                block: search.clone(),
            })
            .expect("search block"),
    );
    frames.extend(
        encoder
            .feed(&Event::ServerToolBlock {
                index: 1,
                block: result.clone(),
            })
            .expect("result block"),
    );
    frames.extend(
        encoder
            .feed(&Event::TextDelta("Today is ...".to_string()))
            .expect("text"),
    );
    frames.extend(encoder.feed(&Event::Paused).expect("paused terminal"));
    let joined = frames.join("");
    let search_start = format!(
        "event: content_block_start\ndata: {}\n\n",
        compact_json(&json!({
            "type": "content_block_start",
            "index": 0,
            "content_block": search,
        }))
    );
    let result_start = format!(
        "event: content_block_start\ndata: {}\n\n",
        compact_json(&json!({
            "type": "content_block_start",
            "index": 1,
            "content_block": result,
        }))
    );
    assert!(joined.contains(&search_start));
    assert!(joined.contains(&result_start));
    assert!(joined.contains("\"stop_reason\":\"pause_turn\""));
    // A pure server-tool turn is not a client tool_use stop.
    assert!(!joined.contains("\"stop_reason\":\"tool_use\""));

    let aggregated = completed_messages_body(
        "req-1",
        "sonnet",
        &[
            Event::ServerToolBlock {
                index: 0,
                block: search.clone(),
            },
            Event::ServerToolBlock {
                index: 1,
                block: result.clone(),
            },
            Event::TextDelta("Today is ...".to_string()),
            Event::Paused,
        ],
    )
    .expect("aggregate");
    assert!(aggregated.failure.is_none());
    assert_eq!(aggregated.body["stop_reason"], json!("pause_turn"));
    assert_eq!(
        aggregated.body["content"],
        json!([search, result, {"type": "text", "text": "Today is ..."}])
    );
}
