//! Unit tests for gateway-executed web search rendering: citation offsets,
//! the Anthropic block prelude, and the Responses/Messages encoders' shapes
//! with and without a search on the admission.

use super::*;
use crate::encode_messages::MessagesSseEncoder;
use crate::encode_responses::{
    completed_responses_body, completed_responses_body_with_web_search, ResponsesEnvelope,
    ResponsesSseEncoder,
};
use crate::events::Usage;

fn admission() -> WebSearchAdmission {
    WebSearchAdmission {
        query: "current stable Python".to_string(),
        requests: 1,
        results: vec![
            WebSearchSource {
                url: "https://python.org/".to_string(),
                title: "Python".to_string(),
            },
            WebSearchSource {
                url: "https://docs.rs/".to_string(),
                title: "Docs.rs".to_string(),
            },
            WebSearchSource {
                url: "https://unmentioned.example/".to_string(),
                title: "Unmentioned".to_string(),
            },
        ],
    }
}

fn usage() -> Usage {
    Usage {
        input_tokens: Some(12),
        output_tokens: Some(7),
        ..Usage::default()
    }
}

fn frame_names(frames: &[String]) -> Vec<&str> {
    frames
        .iter()
        .map(|frame| {
            frame
                .lines()
                .next()
                .and_then(|line| line.strip_prefix("event: "))
                .expect("named frame")
        })
        .collect()
}

fn frame_payload(frame: &str) -> Value {
    let data = frame
        .lines()
        .find_map(|line| line.strip_prefix("data: "))
        .expect("data line");
    serde_json::from_str(data).expect("frame payload is JSON")
}

#[test]
fn url_citations_count_every_occurrence_in_char_offsets_sorted_by_start() {
    let text =
        "Ünïcödé → see https://docs.rs/ and https://python.org/ then https://docs.rs/ again.";
    let citations = url_citations(text, &admission().results, CitationShape::Responses);
    let titles: Vec<&str> = citations
        .iter()
        .map(|citation| citation["title"].as_str().expect("title"))
        .collect();
    assert_eq!(titles, vec!["Docs.rs", "Python", "Docs.rs"]);
    let mut previous_start = 0;
    for citation in &citations {
        let start = citation["start_index"].as_u64().expect("start") as usize;
        let end = citation["end_index"].as_u64().expect("end") as usize;
        assert!(start >= previous_start, "sorted by start");
        previous_start = start;
        let cited: String = text.chars().skip(start).take(end - start).collect();
        assert_eq!(cited, citation["url"].as_str().expect("url"));
    }
    // Offsets are scalar values, not bytes: the multibyte prefix shifts them.
    let first_byte = text.find("https://docs.rs/").expect("present");
    assert!(citations[0]["start_index"].as_u64().expect("start") < first_byte as u64);
}

#[test]
fn url_citations_resolve_bracketed_result_numbers_to_their_ranked_source() {
    let text = "Antonelli won [1, 3]. Norris was third [2][3], see [2].";
    let citations = url_citations(text, &admission().results, CitationShape::Chat);
    let titles: Vec<&str> = citations
        .iter()
        .map(|citation| citation["url_citation"]["title"].as_str().expect("title"))
        .collect();
    assert_eq!(
        titles,
        vec!["Python", "Unmentioned", "Docs.rs", "Unmentioned", "Docs.rs"]
    );
    for citation in &citations {
        let start = citation["url_citation"]["start_index"]
            .as_u64()
            .expect("start") as usize;
        let end = citation["url_citation"]["end_index"].as_u64().expect("end") as usize;
        let span: String = text.chars().skip(start).take(end - start).collect();
        assert!(
            span.starts_with('[') && span.ends_with(']'),
            "spans the group: {span}"
        );
    }
    // A URL citation and a numbered citation of the same source both count.
    let mixed = url_citations(
        "see https://docs.rs/ and [2]",
        &admission().results,
        CitationShape::Responses,
    );
    assert_eq!(mixed.len(), 2);
    assert!(mixed
        .iter()
        .all(|citation| citation["title"] == json!("Docs.rs")));
}

#[test]
fn url_citations_ignore_markdown_labels_and_out_of_range_numbers() {
    let sources = admission().results;
    for text in [
        "[python.org](https://example.com/) is a link label",
        "out of range [4] and [0] and [1000]",
        "not numbers [1a] [a, 2] [] [ , ]",
        "unclosed [1",
    ] {
        assert!(
            url_citations(text, &sources, CitationShape::Chat).is_empty(),
            "{text}"
        );
    }
    // An empty URL at a cited rank yields nothing for that rank alone.
    let mut blank_second = sources.clone();
    blank_second[1].url = String::new();
    let citations = url_citations("[1, 2]", &blank_second, CitationShape::Chat);
    assert_eq!(citations.len(), 1);
    assert_eq!(citations[0]["url_citation"]["title"], json!("Python"));
}

#[test]
fn url_citations_skip_unmatched_and_empty_sources() {
    let sources = vec![
        WebSearchSource {
            url: String::new(),
            title: "empty".to_string(),
        },
        WebSearchSource {
            url: "https://nowhere.example/".to_string(),
            title: "nowhere".to_string(),
        },
    ];
    assert!(url_citations("plain text", &sources, CitationShape::Chat).is_empty());
    assert!(url_citations("", &admission().results, CitationShape::Chat).is_empty());
}

#[test]
fn citation_shapes_nest_for_chat_and_flatten_for_responses() {
    let text = "go https://python.org/";
    let sources = &admission().results[..1];
    assert_eq!(
        url_citations(text, sources, CitationShape::Chat),
        vec![json!({
            "type": "url_citation",
            "url_citation": {
                "url": "https://python.org/",
                "title": "Python",
                "start_index": 3,
                "end_index": 22,
            },
        })]
    );
    assert_eq!(
        url_citations(text, sources, CitationShape::Responses),
        vec![json!({
            "type": "url_citation",
            "url": "https://python.org/",
            "title": "Python",
            "start_index": 3,
            "end_index": 22,
        })]
    );
}

#[test]
fn admission_decodes_the_contract_shape_and_defaults_missing_fields() {
    let decoded: WebSearchAdmission = serde_json::from_value(json!({
        "query": "q",
        "requests": 1,
        "results": [{"url": "https://a.example/", "title": "A"}],
    }))
    .expect("decodes");
    assert_eq!(decoded.requests, 1);
    assert_eq!(decoded.results[0].title, "A");
    let sparse: WebSearchAdmission = serde_json::from_value(json!({})).expect("defaults");
    assert_eq!(sparse.requests, 0);
    assert!(sparse.results.is_empty());
}

#[test]
fn annotate_chat_completion_adds_an_annotations_array_and_the_usage_meter() {
    let mut body = json!({
        "choices": [{"message": {"role": "assistant", "content": "no links here"}}],
        "usage": {"prompt_tokens": 1},
    });
    annotate_chat_completion(&mut body, &admission());
    assert_eq!(body["choices"][0]["message"]["annotations"], json!([]));
    assert_eq!(
        body["usage"]["server_tool_use_details"],
        json!({"web_search_requests": 1})
    );

    let mut cited = json!({
        "choices": [{"message": {"content": "see https://docs.rs/"}}],
        "usage": Value::Null,
    });
    annotate_chat_completion(&mut cited, &admission());
    assert_eq!(
        cited["choices"][0]["message"]["annotations"][0]["url_citation"]["title"],
        json!("Docs.rs")
    );
    // No provider report stays `null`; the meter never invents a usage object.
    assert_eq!(cited["usage"], Value::Null);
    let mut failed = Value::Null;
    annotate_chat_completion(&mut failed, &admission());
    assert_eq!(failed, Value::Null);
}

#[test]
fn prelude_events_render_the_two_anthropic_blocks_under_a_stable_id() {
    let events = web_search_prelude_events(&admission(), "request-abc");
    let expected_id = stable_public_id("srvtoolu", "request-abc");
    assert_eq!(events.len(), 4);
    match &events[0] {
        Event::ServerToolUseStarted {
            index,
            call_id,
            name,
        } => {
            assert_eq!(*index, WEB_SEARCH_TOOL_INDEX);
            assert_eq!(call_id, &expected_id);
            assert_eq!(name, "web_search");
        }
        other => panic!("unexpected first event {other:?}"),
    }
    let (delta, raw) = match (&events[1], &events[2]) {
        (
            Event::ServerToolArgumentsDelta { delta, .. },
            Event::ServerToolUseCompleted { call, .. },
        ) => (delta.clone(), call.raw_arguments.clone()),
        other => panic!("unexpected middle events {other:?}"),
    };
    assert_eq!(delta, raw, "the whole input rides one delta");
    assert_eq!(
        serde_json::from_str::<Value>(&raw).expect("input JSON"),
        json!({"query": "current stable Python"})
    );
    match &events[3] {
        Event::ServerToolResult { block, .. } => {
            let block: Value = serde_json::from_str(block).expect("block JSON");
            assert_eq!(block["type"], json!("web_search_tool_result"));
            assert_eq!(block["tool_use_id"], json!(expected_id));
            assert_eq!(
                block["content"][0],
                json!({
                    "type": "web_search_result",
                    "url": "https://python.org/",
                    "title": "Python",
                    "encrypted_content": "",
                    "page_age": Value::Null,
                })
            );
            assert_eq!(block["content"].as_array().expect("content").len(), 3);
        }
        other => panic!("unexpected result event {other:?}"),
    }
}

fn messages_events() -> Vec<Event> {
    vec![
        Event::TextDelta("See https://python.org/".to_string()),
        Event::Usage(usage()),
        Event::Completed,
    ]
}

#[test]
fn completed_messages_body_leads_with_the_search_blocks_and_meters_usage() {
    let aggregated = completed_messages_body_with_web_search(
        "request-abc",
        "coding",
        &messages_events(),
        &[],
        None,
        false,
        Some(&admission()),
    )
    .expect("aggregates");
    let content = aggregated.body["content"].as_array().expect("content");
    assert_eq!(content.len(), 3);
    assert_eq!(content[0]["type"], json!("server_tool_use"));
    assert_eq!(content[0]["name"], json!("web_search"));
    assert_eq!(
        content[0]["input"],
        json!({"query": "current stable Python"})
    );
    assert_eq!(content[1]["type"], json!("web_search_tool_result"));
    assert_eq!(content[1]["tool_use_id"], content[0]["id"]);
    assert_eq!(
        content[2],
        json!({"type": "text", "text": "See https://python.org/"})
    );
    assert_eq!(aggregated.body["stop_reason"], json!("end_turn"));
    assert_eq!(
        aggregated.body["usage"]["server_tool_use"],
        json!({"web_search_requests": 1})
    );
    assert_eq!(aggregated.body["usage"]["input_tokens"], json!(12));
    // The gateway's own search is billed through the request meter, never as
    // a provider tool call.
    assert!(aggregated.tool_names.is_empty());
}

#[test]
fn completed_messages_body_without_a_search_is_unchanged() {
    let plain = completed_messages_body_with_reasoning(
        "request-abc",
        "coding",
        &messages_events(),
        &[],
        None,
        false,
    )
    .expect("aggregates");
    let routed = completed_messages_body_with_web_search(
        "request-abc",
        "coding",
        &messages_events(),
        &[],
        None,
        false,
        None,
    )
    .expect("aggregates");
    assert_eq!(compact_json(&plain.body), compact_json(&routed.body));
    assert!(!compact_json(&plain.body).contains("server_tool_use"));
}

#[test]
fn messages_stream_leads_with_the_search_blocks_in_valid_order_and_meters_usage() {
    let mut encoder = MessagesSseEncoder::new("request-abc", "coding");
    configure_messages_encoder(&mut encoder, Some(&admission()), "request-abc");
    let mut frames = encoder.start().expect("starts");
    assert_eq!(
        frame_names(&frames),
        vec![
            "message_start",
            "ping",
            "content_block_start",
            "content_block_delta",
            "content_block_stop",
            "content_block_start",
        ]
    );
    // Only the gateway's own blocks so far: the provider has rendered nothing.
    assert!(!encoder.has_content_blocks());
    let start = frame_payload(&frames[0]);
    assert_eq!(
        start["message"]["usage"]["server_tool_use"],
        json!({"web_search_requests": 1})
    );
    let tool_start = frame_payload(&frames[2]);
    assert_eq!(
        tool_start["content_block"]["type"],
        json!("server_tool_use")
    );
    assert_eq!(
        tool_start["content_block"]["id"],
        json!(stable_public_id("srvtoolu", "request-abc"))
    );
    assert_eq!(
        frame_payload(&frames[3])["delta"]["type"],
        json!("input_json_delta")
    );
    assert_eq!(
        frame_payload(&frames[5])["content_block"]["type"],
        json!("web_search_tool_result")
    );
    for event in &messages_events() {
        frames.extend(encoder.feed(event).expect("streams"));
    }
    assert!(encoder.has_content_blocks());
    assert_eq!(
        frame_names(&frames)[6..],
        [
            "content_block_stop",
            "content_block_start",
            "content_block_delta",
            "content_block_stop",
            "message_delta",
            "message_stop",
        ]
    );
    let delta = frame_payload(&frames[10]);
    assert_eq!(delta["delta"]["stop_reason"], json!("end_turn"));
    assert_eq!(
        delta["usage"]["server_tool_use"],
        json!({"web_search_requests": 1})
    );
    assert_eq!(delta["usage"]["output_tokens"], json!(7));
}

#[test]
fn messages_stream_without_a_search_is_byte_identical() {
    let mut plain = MessagesSseEncoder::new("request-abc", "coding");
    let mut configured = MessagesSseEncoder::new("request-abc", "coding");
    configure_messages_encoder(&mut configured, None, "request-abc");
    let mut plain_frames = plain.start().expect("starts");
    let mut configured_frames = configured.start().expect("starts");
    for event in &messages_events() {
        plain_frames.extend(plain.feed(event).expect("streams"));
        configured_frames.extend(configured.feed(event).expect("streams"));
    }
    assert_eq!(plain_frames, configured_frames);
    assert!(!plain_frames.join("").contains("server_tool_use"));
}

fn responses_events() -> Vec<Event> {
    vec![
        Event::TextDelta("Read https://docs.rs/ and https://python.org/".to_string()),
        Event::Usage(usage()),
        Event::Completed,
    ]
}

#[test]
fn responses_synthetic_message_emits_annotation_frames_before_text_done() {
    let mut encoder = ResponsesSseEncoder::new(
        "request-1",
        "coding",
        1_700_000_000,
        ResponsesEnvelope::default(),
    );
    encoder.set_web_search(Some(admission()));
    let mut frames = encoder.start().expect("starts");
    for event in &responses_events() {
        frames.extend(encoder.feed(event).expect("encodes"));
    }
    let names: Vec<&str> = frames
        .iter()
        .map(|frame| {
            frame
                .lines()
                .next()
                .and_then(|line| line.strip_prefix("event: "))
        })
        .map(|name| name.expect("named frame"))
        .collect();
    let done = names
        .iter()
        .position(|name| *name == "response.output_text.done")
        .expect("text done frame");
    assert_eq!(
        &names[done - 2..done],
        &[
            "response.output_text.annotation.added",
            "response.output_text.annotation.added",
        ]
    );
    let first = frame_payload(&frames[done - 2]);
    assert_eq!(first["annotation_index"], json!(0));
    assert_eq!(first["content_index"], json!(0));
    assert_eq!(first["output_index"], json!(0));
    assert_eq!(
        first["item_id"],
        json!(stable_public_id("msg", "request-1"))
    );
    assert_eq!(
        first["annotation"],
        json!({
            "type": "url_citation",
            "url": "https://docs.rs/",
            "title": "Docs.rs",
            "start_index": 5,
            "end_index": 21,
        })
    );
    assert_eq!(
        frame_payload(&frames[done - 1])["annotation_index"],
        json!(1)
    );
    let completed = frame_payload(frames.last().expect("terminal"));
    let part = &completed["response"]["output"][0]["content"][0];
    assert_eq!(
        part["annotations"].as_array().expect("annotations").len(),
        2
    );
    assert_eq!(part["annotations"][1]["title"], json!("Python"));
    assert_eq!(
        completed["response"]["usage"]["server_tool_use_details"],
        json!({"web_search_requests": 1})
    );
    assert_eq!(completed["response"]["usage"]["input_tokens"], json!(12));
    // The content_part.done part carries the same annotations.
    assert_eq!(
        frame_payload(&frames[done + 1])["part"]["annotations"]
            .as_array()
            .expect("annotations")
            .len(),
        2
    );
}

#[test]
fn responses_aggregate_cites_the_search_and_stays_unchanged_without_one() {
    let cited = completed_responses_body_with_web_search(
        "request-1",
        "coding",
        1_700_000_000,
        ResponsesEnvelope::default(),
        &responses_events(),
        None,
        Some(&admission()),
    )
    .expect("aggregates");
    let annotations = &cited.body["output"][0]["content"][0]["annotations"];
    assert_eq!(annotations.as_array().expect("annotations").len(), 2);
    assert_eq!(annotations[0]["type"], json!("url_citation"));
    assert_eq!(
        cited.body["usage"]["server_tool_use_details"],
        json!({"web_search_requests": 1})
    );
    let plain = completed_responses_body(
        "request-1",
        "coding",
        1_700_000_000,
        ResponsesEnvelope::default(),
        &responses_events(),
    )
    .expect("aggregates");
    let routed = completed_responses_body_with_web_search(
        "request-1",
        "coding",
        1_700_000_000,
        ResponsesEnvelope::default(),
        &responses_events(),
        None,
        None,
    )
    .expect("aggregates");
    assert_eq!(compact_json(&plain.body), compact_json(&routed.body));
    assert_eq!(
        plain.body["output"][0]["content"][0]["annotations"],
        json!([])
    );
    assert!(plain.body["usage"].get("server_tool_use_details").is_none());
}

#[test]
fn responses_provider_keyed_messages_keep_only_provider_annotations() {
    let mut encoder = ResponsesSseEncoder::new(
        "request-1",
        "coding",
        1_700_000_000,
        ResponsesEnvelope::default(),
    );
    encoder.set_web_search(Some(admission()));
    let events = vec![
        Event::ProviderTextDelta {
            output_index: 0,
            item_id: "msg_provider".to_string(),
            delta: "Read https://docs.rs/".to_string(),
        },
        Event::Usage(usage()),
        Event::Completed,
    ];
    let mut frames = encoder.start().expect("starts");
    for event in &events {
        frames.extend(encoder.feed(event).expect("encodes"));
    }
    let joined = frames.join("");
    assert!(!joined.contains("response.output_text.annotation.added"));
    let completed = frame_payload(frames.last().expect("terminal"));
    assert_eq!(
        completed["response"]["output"][0]["content"][0]["annotations"],
        json!([])
    );
    // The meter is a request fact and rides usage regardless of the rung.
    assert_eq!(
        completed["response"]["usage"]["server_tool_use_details"],
        json!({"web_search_requests": 1})
    );
}
