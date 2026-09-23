//! Relay tests: first-token stamping, stop-sequence cuts, customer-owned
//! credential failures, first-byte stalls, and the tool-search withholder
//! at the relay's yield point.

use super::*;
use crate::dialects::Dialect;
use crate::events::Event;
use futures_util::stream;

#[tokio::test]
async fn first_token_at_is_stamped_on_the_first_output_delta() {
    // A content delta then the OpenAI terminal sentinel.
    let frames = vec![
        Ok::<_, reqwest::Error>(Bytes::from(
            "data: {\"choices\":[{\"delta\":{\"content\":\"hi\"}}]}\n\n",
        )),
        Ok::<_, reqwest::Error>(Bytes::from("data: [DONE]\n\n")),
    ];
    let mut relay = UpstreamRelay::from_stream(
        stream::iter(frames).boxed(),
        Dialect::OpenAiCompatible,
        Instant::now() + Duration::from_secs(5),
    );
    assert!(
        relay.first_token_at().is_none(),
        "no first-token time before any event is yielded"
    );
    let deadline = Instant::now() + Duration::from_secs(30);
    let per_chunk = Duration::from_secs(5);
    let first = relay
        .next_event(deadline, per_chunk, Instant::now())
        .await
        .expect("the stream yields")
        .expect("an event is produced");
    assert!(
        matches!(&first, Event::TextDelta(text) if text == "hi"),
        "the first output event is the content delta"
    );
    assert!(
        relay.first_token_at().is_some(),
        "the first content delta stamps time-to-first-token"
    );
}

#[tokio::test]
async fn stop_sequences_cut_the_relayed_text_and_keep_usage_and_settlement_exact() {
    // A Chat-compatible stream stands in for any dialect: "</block>" spans
    // two content deltas, more text follows it, then usage and the
    // provider's own terminal arrive. The guard cuts at the match, drops
    // the trailing text, still yields the usage, and replaces the terminal.
    let frames = vec![
        Ok::<_, reqwest::Error>(Bytes::from(
            "data: {\"choices\":[{\"delta\":{\"content\":\"allow</bl\"}}]}\n\n",
        )),
        Ok::<_, reqwest::Error>(Bytes::from(
            "data: {\"choices\":[{\"delta\":{\"content\":\"ock>ignored\"}}]}\n\n",
        )),
        Ok::<_, reqwest::Error>(Bytes::from(
            "data: {\"choices\":[],\"usage\":{\"prompt_tokens\":5,\"completion_tokens\":3}}\n\n",
        )),
        Ok::<_, reqwest::Error>(Bytes::from(
            "data: {\"choices\":[{\"delta\":{},\"finish_reason\":\"stop\"}]}\n\n",
        )),
        Ok::<_, reqwest::Error>(Bytes::from("data: [DONE]\n\n")),
    ];
    let mut relay = UpstreamRelay::from_stream(
        stream::iter(frames).boxed(),
        Dialect::OpenAiCompatible,
        Instant::now() + Duration::from_secs(5),
    );
    relay.set_stop_sequences(["</block>"]);
    let deadline = Instant::now() + Duration::from_secs(30);
    let per_chunk = Duration::from_secs(5);
    let mut events = Vec::new();
    loop {
        match relay.next_event(deadline, per_chunk, Instant::now()).await {
            Ok(Some(event)) => {
                let terminal = event.is_terminal();
                events.push(event);
                if terminal {
                    break;
                }
            }
            Ok(None) => panic!("the stream must end on a terminal"),
            Err(failure) => panic!("unexpected failure: {failure:?}"),
        }
    }
    let text: String = events
        .iter()
        .filter_map(|event| match event {
            Event::TextDelta(text) => Some(text.as_str()),
            _ => None,
        })
        .collect();
    assert_eq!(text, "allow", "text stops exactly before the sequence");
    assert!(
        events
            .iter()
            .any(|event| matches!(event, Event::Usage(usage) if usage.has_token_counts())),
        "usage still reaches settlement after the cut"
    );
    assert!(
        matches!(events.last(), Some(Event::StoppedAtSequence(sequence)) if sequence == "</block>"),
        "the terminal names the matched sequence"
    );
}

#[tokio::test]
async fn a_customer_managed_relay_re_owns_a_declared_credential_failure_after_output() {
    // Text has already streamed (the attempt is committed) when the
    // provider declares a 401 in-stream: the customer still gets their
    // own message, not the house "ask the gateway operator" one.
    let frames = vec![
        Ok::<_, reqwest::Error>(Bytes::from(
            "data: {\"choices\":[{\"delta\":{\"content\":\"hi\"}}]}\n\n",
        )),
        Ok::<_, reqwest::Error>(Bytes::from(
            "data: {\"error\":{\"code\":401,\"message\":\"Incorrect API key provided\"}}\n\n",
        )),
    ];
    let mut relay = UpstreamRelay::from_stream(
        stream::iter(frames).boxed(),
        Dialect::OpenAiCompatible,
        Instant::now() + Duration::from_secs(5),
    );
    relay.set_customer_managed_provider(Some("openai".to_string()));
    let deadline = Instant::now() + Duration::from_secs(30);
    let per_chunk = Duration::from_secs(5);
    let first = relay
        .next_event(deadline, per_chunk, Instant::now())
        .await
        .expect("yields")
        .expect("event");
    assert!(matches!(&first, Event::TextDelta(text) if text == "hi"));
    let second = relay
        .next_event(deadline, per_chunk, Instant::now())
        .await
        .expect("yields")
        .expect("event");
    match second {
        Event::Failed(failure) => {
            assert_eq!(failure.failure_class, FailureClass::ProviderAuthentication);
            assert!(failure.customer_owned);
            assert!(failure
                .safe_message
                .contains("your connected openai credential"));
            assert_eq!(failure.public_error().status_code, 400);
        }
        other => panic!("expected the re-owned failure, got {other:?}"),
    }
}

#[test]
fn a_first_byte_stall_fails_over_without_redialing_the_dead_lane() {
    let failure = first_byte_timeout_failure();
    assert_eq!(failure.failure_class, FailureClass::Timeout);
    assert!(failure.failover_eligible);
    // A stalled lane is skipped, not redialed: redialing it would only
    // stall again for another window.
    assert!(!failure.retryable_same_deployment);
}

#[tokio::test]
async fn a_gemini_partial_then_abnormal_frame_ends_incomplete_not_failed() {
    // A Gemini content frame, its usage, then a structurally malformed frame
    // (a non-string text part). The relay must route the abnormal end
    // through recovery: yield the content, fold the usage, and end on an
    // Incomplete terminal instead of surfacing the malformed failure.
    let frames = vec![
        Ok::<_, reqwest::Error>(Bytes::from(
            "data: {\"candidates\":[{\"content\":{\"parts\":[{\"text\":\"partial\"}]}}]}\n\n",
        )),
        Ok::<_, reqwest::Error>(Bytes::from(
            "data: {\"usageMetadata\":{\"promptTokenCount\":9,\"candidatesTokenCount\":3}}\n\n",
        )),
        Ok::<_, reqwest::Error>(Bytes::from(
            "data: {\"candidates\":[{\"content\":{\"parts\":[{\"text\":5}]}}]}\n\n",
        )),
    ];
    let mut relay = UpstreamRelay::from_stream(
        stream::iter(frames).boxed(),
        Dialect::GeminiGenerateContent,
        Instant::now() + Duration::from_secs(5),
    );
    let deadline = Instant::now() + Duration::from_secs(30);
    let per_chunk = Duration::from_secs(5);
    let mut seen: Vec<Event> = Vec::new();
    loop {
        let event = relay
            .next_event(deadline, per_chunk, Instant::now())
            .await
            .expect("recovery yields events, never the malformed failure");
        match event {
            Some(event) => {
                let terminal = event.is_terminal();
                seen.push(event);
                if terminal {
                    break;
                }
            }
            None => panic!("the recovered terminal must arrive before EOF"),
        }
    }
    assert!(
        matches!(seen.first(), Some(Event::TextDelta(text)) if text == "partial"),
        "the partial content is delivered"
    );
    assert!(
        seen.iter().any(|event| matches!(event, Event::Usage(_))),
        "last-seen usage is folded so delivered tokens bill"
    );
    assert!(
        matches!(seen.last(), Some(Event::Incomplete)),
        "the turn ends incomplete, not failed"
    );
}

#[tokio::test]
async fn invalid_gemini_images_keep_their_nonretryable_failure_before_and_after_text() {
    for preceding_text in [false, true] {
        for (image, message) in [
            (
                serde_json::json!({"mimeType": "image/png", "data": "iVBORw0KGgo="}),
                "provider returned an invalid generated image",
            ),
            (
                serde_json::json!({"mimeType": "image/png"}),
                "Gemini image requires base64 data",
            ),
            (
                serde_json::json!({"data": "iVBORw0KGgo="}),
                "Gemini image requires a media type",
            ),
        ] {
            let mut frames = Vec::new();
            if preceding_text {
                frames.push(Ok::<_, reqwest::Error>(Bytes::from(
                    "data: {\"candidates\":[{\"content\":{\"parts\":[{\"text\":\"partial\"}]}}]}\n\n",
                )));
            }
            let payload = serde_json::json!({
                "candidates": [{"content": {"parts": [{"inlineData": image}]}}],
                "usageMetadata": {"promptTokenCount": 9, "candidatesTokenCount": 100}
            });
            frames.push(Ok(Bytes::from(format!("data: {payload}\n\n"))));
            let started = Instant::now();
            let deadline = started + Duration::from_secs(30);
            let per_chunk = Duration::from_secs(5);
            let mut relay = UpstreamRelay::from_stream(
                stream::iter(frames).boxed(),
                Dialect::GeminiGenerateContent,
                deadline,
            );
            relay.allow_image_output();
            if preceding_text {
                assert!(matches!(
                    relay.next_event(deadline, per_chunk, started).await.unwrap(),
                    Some(Event::TextDelta(text)) if text == "partial"
                ));
            }
            let failure = relay
                .next_event(deadline, per_chunk, started)
                .await
                .expect_err("an invalid image must not recover to a partial completion");
            assert_eq!(failure.failure_class, FailureClass::MalformedResponse);
            assert_eq!(failure.safe_message, message);
            assert!(!failure.retryable_same_deployment);
            assert!(!failure.failover_eligible);
            assert_eq!(
                relay.usage_before_failure(None).unwrap().output_tokens,
                Some(100)
            );
        }
    }
}

#[tokio::test]
async fn a_stalled_first_byte_trips_the_ttft_bound_not_the_chunk_timeout() {
    // A provider that opened the stream but never sends a byte must fail
    // over in about the time-to-first-byte window, not the (far larger)
    // per-chunk deployment timeout.
    let never = stream::pending::<reqwest::Result<Bytes>>().boxed();
    let time_to_first_byte = Duration::from_millis(80);
    let mut relay = UpstreamRelay::from_stream(
        never,
        Dialect::OpenAiCompatible,
        Instant::now() + time_to_first_byte,
    );
    let request_deadline = Instant::now() + Duration::from_secs(120);
    let per_chunk_timeout = Duration::from_secs(35);

    let started = Instant::now();
    let outcome = relay
        .next_event(request_deadline, per_chunk_timeout, started)
        .await;
    let elapsed = started.elapsed();

    assert!(
        elapsed < Duration::from_secs(2),
        "expected a fail-fast time-to-first-byte trip, waited {elapsed:?}"
    );
    let failure = outcome.expect_err("a never-yielding stream must not succeed");
    assert_eq!(failure.failure_class, FailureClass::Timeout);
    assert!(
        failure.failover_eligible,
        "a first-byte stall must advance to the next deployment"
    );
}

/// One OpenAI-compatible tool-call turn calling the gateway's search tool,
/// with its usage report, in the frame shapes the chat wire streams.
fn search_call_frames() -> Vec<Result<Bytes, reqwest::Error>> {
    vec![
        Ok(Bytes::from(
            "data: {\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":0,\"id\":\"call_x\",\
             \"type\":\"function\",\"function\":{\"name\":\"tool_search\",\"arguments\":\"\"}}]}}]}\n\n",
        )),
        Ok(Bytes::from(
            "data: {\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":0,\"function\":\
             {\"arguments\":\"{\\\"query\\\":\\\"weather\\\"}\"}}]}}]}\n\n",
        )),
        Ok(Bytes::from(
            "data: {\"choices\":[],\"usage\":{\"prompt_tokens\":5,\"completion_tokens\":3}}\n\n",
        )),
        Ok(Bytes::from(
            "data: {\"choices\":[{\"delta\":{},\"finish_reason\":\"tool_calls\"}]}\n\n",
        )),
        Ok(Bytes::from("data: [DONE]\n\n")),
    ]
}

async fn drain(relay: &mut UpstreamRelay) -> Vec<Event> {
    let deadline = Instant::now() + Duration::from_secs(30);
    let per_chunk = Duration::from_secs(5);
    let mut events = Vec::new();
    loop {
        match relay.next_event(deadline, per_chunk, Instant::now()).await {
            Ok(Some(event)) => {
                let terminal = event.is_terminal();
                events.push(event);
                if terminal {
                    return events;
                }
            }
            Ok(None) => panic!("the stream must end on a terminal"),
            Err(failure) => panic!("unexpected failure: {failure:?}"),
        }
    }
}

#[tokio::test]
async fn the_tool_search_tool_is_withheld_at_the_relay_and_handed_over() {
    let mut relay = UpstreamRelay::from_stream(
        stream::iter(search_call_frames()).boxed(),
        Dialect::OpenAiCompatible,
        Instant::now() + Duration::from_secs(5),
    );
    relay.set_tool_search_tool_name(Some("tool_search".to_string()));
    let events = drain(&mut relay).await;
    assert!(
        !events.iter().any(|event| matches!(
            event,
            Event::ToolCallStarted { .. }
                | Event::ToolArgumentsDelta { .. }
                | Event::ToolCallCompleted { .. }
        )),
        "no search-call event reaches the caller: {events:?}"
    );
    assert!(
        events
            .iter()
            .any(|event| matches!(event, Event::Usage(usage) if usage.input_tokens == Some(5))),
        "usage still reaches settlement"
    );
    assert!(matches!(events.last(), Some(Event::Completed)));
    assert!(
        relay.first_token_at().is_none(),
        "a withheld call is not a visible token"
    );
    assert_eq!(relay.withheld_search_call_count(), 1);
    assert!(relay.withheld_search_call_seen());
    let calls = relay.take_withheld_search_calls();
    assert_eq!(calls.len(), 1);
    assert_eq!(calls[0].call_id, "call_x");
    assert_eq!(calls[0].name, "tool_search");
    assert_eq!(calls[0].raw_arguments, "{\"query\":\"weather\"}");
    assert_eq!(relay.withheld_search_call_count(), 0);

    // Without a named tool the very same stream yields the call intact.
    let mut plain = UpstreamRelay::from_stream(
        stream::iter(search_call_frames()).boxed(),
        Dialect::OpenAiCompatible,
        Instant::now() + Duration::from_secs(5),
    );
    let events = drain(&mut plain).await;
    assert!(matches!(
        events.first(),
        Some(Event::ToolCallStarted { name, .. }) if name == "tool_search"
    ));
    assert!(events.iter().any(|event| matches!(
        event,
        Event::ToolCallCompleted { call, .. } if call.raw_arguments == "{\"query\":\"weather\"}"
    )));
    assert_eq!(plain.withheld_search_call_count(), 0);
    assert!(!plain.withheld_search_call_seen());
}

#[tokio::test]
async fn ready_bytes_cannot_bypass_an_expired_first_token_deadline() {
    let frames = stream::iter(vec![Ok::<_, reqwest::Error>(Bytes::from(
        "data: {\"choices\":[{\"delta\":{\"content\":\"too late\"}}]}\n\n",
    ))])
    .boxed();
    let mut relay = UpstreamRelay::from_stream(
        frames,
        Dialect::OpenAiCompatible,
        Instant::now() - Duration::from_secs(1),
    );
    let failure = relay
        .next_event(
            Instant::now() + Duration::from_secs(30),
            Duration::from_secs(10),
            Instant::now(),
        )
        .await
        .expect_err("an immediately ready read cannot outlive its deadline");
    assert_eq!(failure.failure_class, FailureClass::Timeout);
}

#[tokio::test]
async fn committed_keepalives_do_not_renew_generation_progress() {
    let chunks = stream::once(async {
        Ok::<_, reqwest::Error>(Bytes::from(
            "data: {\"choices\":[{\"delta\":{\"content\":\"hi\"}}]}\n\n",
        ))
    })
    .chain(stream::unfold(0, |count| async move {
        if count >= 30 {
            return None;
        }
        tokio::time::sleep(Duration::from_millis(10)).await;
        Some((Ok(Bytes::from(": keepalive\n\n")), count + 1))
    }))
    .chain(stream::pending())
    .boxed();
    let mut relay = UpstreamRelay::from_stream(
        chunks,
        Dialect::OpenAiCompatible,
        Instant::now() + Duration::from_secs(1),
    );
    let deadline = Instant::now() + Duration::from_secs(5);
    let idle = Duration::from_millis(80);
    relay
        .next_event(deadline, idle, Instant::now())
        .await
        .unwrap();
    relay.commit();
    let started = Instant::now();
    let failure = relay
        .next_event(deadline, idle, started)
        .await
        .expect_err("pings are not generation progress");
    assert!(started.elapsed() < Duration::from_millis(250));
    assert_eq!(failure.failure_class, FailureClass::Transport);
    assert!(!failure.retryable_same_deployment);
}

#[tokio::test]
async fn always_ready_ping_flood_stops_at_the_absolute_deadline() {
    let chunks =
        stream::repeat_with(|| Ok::<_, reqwest::Error>(Bytes::from_static(b": ping\n\n"))).boxed();
    let mut relay = UpstreamRelay::from_stream(
        chunks,
        Dialect::OpenAiCompatible,
        Instant::now() + Duration::from_millis(30),
    );
    let started = Instant::now();
    let failure = relay
        .next_event(
            started + Duration::from_secs(2),
            Duration::from_secs(1),
            started,
        )
        .await
        .expect_err("an always-ready flood must not starve its timer");
    assert_eq!(failure.failure_class, FailureClass::Timeout);
    assert!(started.elapsed() < Duration::from_millis(300));
}

#[tokio::test]
async fn active_output_still_obeys_the_hard_total_deadline() {
    let chunks = stream::unfold((), |()| async {
        tokio::time::sleep(Duration::from_millis(10)).await;
        Some((
            Ok::<_, reqwest::Error>(Bytes::from_static(
                b"data: {\"choices\":[{\"delta\":{\"content\":\"token\"}}]}\n\n",
            )),
            (),
        ))
    })
    .boxed();
    let started = Instant::now();
    let mut relay = UpstreamRelay::from_stream(
        chunks,
        Dialect::OpenAiCompatible,
        started + Duration::from_millis(50),
    );
    let deadline = started + Duration::from_millis(200);
    let mut tokens = 0;
    loop {
        match relay
            .next_event(deadline, Duration::from_millis(100), started)
            .await
        {
            Ok(Some(Event::TextDelta(_))) => {
                tokens += 1;
                relay.commit();
            }
            Err(failure) => {
                assert_eq!(failure.failure_class, FailureClass::Timeout);
                assert_eq!(failure.safe_message, "gateway execution deadline exceeded");
                break;
            }
            other => panic!("unexpected outcome: {other:?}"),
        }
    }
    assert!(
        tokens > 5,
        "active output survives the first-token allowance"
    );
    assert!(started.elapsed() < Duration::from_millis(500));
}

#[tokio::test]
async fn structural_commit_keeps_first_progress_allowance_and_drains_received_terminal() {
    let frames = stream::once(async { Ok::<_, reqwest::Error>(Bytes::from_static(
        b"data: {\"type\":\"response.output_item.added\",\"output_index\":0,\"item\":{\"type\":\"message\",\"id\":\"msg_1\",\"role\":\"assistant\",\"content\":[]}}\n\n",
    )) }).chain(stream::once(async {
        tokio::time::sleep(Duration::from_millis(120)).await;
        Ok::<_, reqwest::Error>(Bytes::from_static(
            b"data: {\"type\":\"response.output_text.delta\",\"output_index\":0,\"content_index\":0,\"item_id\":\"msg_1\",\"delta\":\"hi\"}\n\ndata: {\"type\":\"response.completed\",\"response\":{\"status\":\"completed\",\"usage\":{\"input_tokens\":2,\"output_tokens\":1}}}\n\n",
        ))
    })).boxed();
    let started = Instant::now();
    let mut relay = UpstreamRelay::from_stream(
        frames,
        Dialect::OpenAiResponses,
        started + Duration::from_millis(300),
    );
    let deadline = started + Duration::from_secs(2);
    let idle = Duration::from_millis(60);
    assert!(matches!(
        relay.next_event(deadline, idle, started).await.unwrap(),
        Some(Event::ProviderOutputItemStarted { .. })
    ));
    relay.commit();
    assert!(matches!(
        relay.next_event(deadline, idle, started).await.unwrap(),
        Some(Event::ProviderTextDelta { .. })
    ));
    // The terminal already arrived in the same chunk as the text. A slow
    // consumer must still receive its usage and terminal, not a false stall.
    tokio::time::sleep(Duration::from_millis(100)).await;
    let mut terminal = false;
    while let Some(event) = relay.next_event(deadline, idle, started).await.unwrap() {
        if event.is_terminal() {
            terminal = true;
            break;
        }
    }
    assert!(terminal);
}

#[tokio::test]
async fn omitted_thinking_waits_for_signature_under_full_first_token_allowance() {
    let frames = stream::once(async { Ok::<_, reqwest::Error>(Bytes::from_static(
        b"event: content_block_start\ndata: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"thinking\",\"thinking\":\"\"}}\n\nevent: content_block_delta\ndata: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"thinking_delta\",\"thinking\":\"\"}}\n\n",
    )) }).chain(stream::once(async {
        tokio::time::sleep(Duration::from_millis(120)).await;
        Ok::<_, reqwest::Error>(Bytes::from_static(
            b"event: content_block_delta\ndata: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"signature_delta\",\"signature\":\"opaque\"}}\n\n",
        ))
    })).boxed();
    let started = Instant::now();
    let mut relay = UpstreamRelay::from_stream(
        frames,
        Dialect::AnthropicMessages,
        started + Duration::from_millis(300),
    );
    let event = relay
        .next_event(
            started + Duration::from_secs(2),
            Duration::from_millis(60),
            started,
        )
        .await
        .unwrap()
        .unwrap();
    assert!(matches!(event, Event::ThinkingSignature { .. }));
    assert!(relay.first_token_at().is_none());
}

#[tokio::test]
async fn provider_tool_phase_preserves_byte_idle_then_resumes_progress_idle() {
    let scenarios = [
        (Dialect::AnthropicMessages,
         "event: content_block_start\ndata: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"server_tool_use\",\"id\":\"srv_1\",\"name\":\"web_search\",\"input\":{}}}\n\n",
         "event: content_block_start\ndata: {\"type\":\"content_block_start\",\"index\":1,\"content_block\":{\"type\":\"web_search_tool_result\",\"tool_use_id\":\"srv_1\",\"content\":[]}}\n\n"),
        (Dialect::OpenAiResponses,
         "data: {\"type\":\"response.output_item.added\",\"output_index\":0,\"item\":{\"type\":\"web_search_call\",\"id\":\"ws_1\",\"status\":\"in_progress\"}}\n\n",
         "data: {\"type\":\"response.output_item.done\",\"output_index\":0,\"item\":{\"type\":\"web_search_call\",\"id\":\"ws_1\",\"status\":\"completed\"}}\n\n"),
    ];
    for (dialect, start, result) in scenarios {
        let chunks = stream::once(async move { Ok::<_, reqwest::Error>(Bytes::from(start)) })
            .chain(stream::unfold(0, move |count| async move {
                if count == 25 {
                    return None;
                }
                tokio::time::sleep(Duration::from_millis(10)).await;
                Some((
                    Ok(Bytes::from(if count == 12 { result } else { ": ping\n\n" })),
                    count + 1,
                ))
            }))
            .chain(stream::pending())
            .boxed();
        let started = Instant::now();
        let deadline = started + Duration::from_secs(2);
        let idle = Duration::from_millis(60);
        let mut relay =
            UpstreamRelay::from_stream(chunks, dialect, started + Duration::from_millis(100));
        relay
            .next_event(deadline, idle, started)
            .await
            .unwrap()
            .unwrap();
        relay.commit();
        let result = relay
            .next_event(deadline, idle, started)
            .await
            .unwrap()
            .unwrap();
        assert!(matches!(
            result,
            Event::ServerToolResult { .. } | Event::HostedToolItemCompleted { .. }
        ));
        assert!(started.elapsed() > Duration::from_millis(100));
        let idle_started = Instant::now();
        let failure = relay.next_event(deadline, idle, started).await.unwrap_err();
        assert!(idle_started.elapsed() < Duration::from_millis(120));
        assert!(failure.safe_message.contains("stopped making progress"));
    }
}

#[tokio::test]
async fn failure_usage_prefers_latest_cumulative_counts() {
    let chunks = stream::iter(vec![Ok::<_, reqwest::Error>(Bytes::from_static(
        b"data: {\"choices\":[],\"usage\":{\"prompt_tokens\":3,\"completion_tokens\":2}}\n\ndata: {\"choices\":[],\"usage\":{\"prompt_tokens\":7,\"completion_tokens\":9}}\n\n",
    ))]).chain(stream::pending()).boxed();
    let started = Instant::now();
    let mut relay = UpstreamRelay::from_stream(
        chunks,
        Dialect::OpenAiCompatible,
        started + Duration::from_millis(30),
    );
    let carried = Usage {
        input_tokens: Some(10),
        output_tokens: Some(20),
        ..Default::default()
    };
    relay
        .next_event(
            started + Duration::from_secs(2),
            Duration::from_secs(1),
            started,
        )
        .await
        .unwrap_err();
    let observed = relay.usage_before_failure(Some(carried)).unwrap();
    assert_eq!(observed.input_tokens, Some(7));
    assert_eq!(observed.output_tokens, Some(9));
    let again = relay.usage_before_failure(Some(observed)).unwrap();
    assert_eq!(again.input_tokens, Some(7));
    assert_eq!(again.output_tokens, Some(9));
}

#[test]
fn failure_without_current_usage_keeps_the_attempt_total_unknown() {
    let relay = UpstreamRelay::from_stream(
        stream::pending().boxed(),
        Dialect::OpenAiCompatible,
        Instant::now(),
    );
    assert!(relay.usage_before_failure(None).is_none());
    // A separate physical attempt cannot borrow another dial's usage.
    assert!(relay.usage_before_failure(None).is_none());
}

#[test]
fn empty_private_reasoning_is_neither_progress_nor_commit() {
    let event = Event::ReasoningContentDelta {
        route_sha256: "a".repeat(64),
        delta: String::new(),
    };
    assert!(!event.is_generation_progress());
    assert!(!crate::waterfall::is_semantic(&event));
}

#[tokio::test]
async fn responses_stop_drain_counts_hidden_generation_but_not_keepalives() {
    for continuing_text in [true, false] {
        let chunks = stream::once(async { Ok::<_, reqwest::Error>(Bytes::from_static(
            b"data: {\"type\":\"response.output_text.delta\",\"output_index\":0,\"content_index\":0,\"item_id\":\"msg_1\",\"delta\":\"answer<STOP>hidden\"}\n\n",
        )) }).chain(stream::unfold(0, move |count| async move {
            if count == 12 { return None; }
            tokio::time::sleep(Duration::from_millis(15)).await;
            let chunk: &'static [u8] = if count == 11 {
                b"data: {\"type\":\"response.completed\",\"response\":{\"status\":\"completed\",\"usage\":{\"input_tokens\":7,\"output_tokens\":9}}}\n\n"
            } else if continuing_text {
                b"data: {\"type\":\"response.output_text.delta\",\"output_index\":0,\"content_index\":0,\"item_id\":\"msg_1\",\"delta\":\"hidden\"}\n\n"
            } else {
                b": ping\n\n"
            };
            Some((Ok(Bytes::from_static(chunk)), count + 1))
        })).boxed();
        let started = Instant::now();
        let deadline = started + Duration::from_secs(2);
        let idle = Duration::from_millis(70);
        let mut relay = UpstreamRelay::from_stream(
            chunks,
            Dialect::OpenAiResponses,
            started + Duration::from_millis(100),
        );
        relay.set_stop_sequences(["<STOP>"]);
        assert!(matches!(
            relay.next_event(deadline, idle, started).await.unwrap(),
            Some(Event::ProviderOutputItemStarted { .. })
        ));
        relay.commit();
        assert!(
            matches!(relay.next_event(deadline, idle, started).await.unwrap(), Some(Event::ProviderTextDelta { delta, .. }) if delta == "answer")
        );
        if continuing_text {
            let mut usage = None;
            loop {
                let event = relay
                    .next_event(deadline, idle, started)
                    .await
                    .unwrap()
                    .unwrap();
                match event {
                    Event::Usage(report) => usage = Some(report),
                    Event::StoppedAtSequence(sequence) => {
                        assert_eq!(sequence, "<STOP>");
                        break;
                    }
                    Event::ProviderTextDelta { .. } | Event::TextDelta(_) => {
                        panic!("post-stop text escaped")
                    }
                    _ => {}
                }
            }
            let usage = usage.unwrap();
            assert_eq!(usage.input_tokens, Some(7));
            assert_eq!(usage.output_tokens, Some(9));
            assert!(started.elapsed() > idle);
        } else {
            let failure = relay.next_event(deadline, idle, started).await.unwrap_err();
            assert!(failure.safe_message.contains("stopped making progress"));
            assert!(started.elapsed() < Duration::from_millis(160));
        }
    }
}

fn keepalive_then_pending() -> BoxStream<'static, reqwest::Result<Bytes>> {
    // Headers already arrived (the relay is built from the body stream); the
    // body opens with SSE keepalive comments -- bytes that decode to no event
    // -- and then never sends a token.
    stream::iter(vec![
        Ok::<_, reqwest::Error>(Bytes::from(": keepalive\n\n")),
        Ok::<_, reqwest::Error>(Bytes::from(": keepalive\n\n")),
    ])
    .chain(stream::pending())
    .boxed()
}

#[tokio::test]
async fn keepalive_comments_do_not_satisfy_the_first_token_bound() {
    // 2026-09-19: a lane answered headers and keepalives at once, then stalled
    // ~2 minutes before its first token; the old bound was satisfied by the
    // first body byte and the request sat on the per-chunk timeout instead.
    let time_to_first_token = Duration::from_millis(80);
    let mut relay = UpstreamRelay::from_stream(
        keepalive_then_pending(),
        Dialect::OpenAiCompatible,
        Instant::now() + time_to_first_token,
    );
    let request_deadline = Instant::now() + Duration::from_secs(120);
    let per_chunk_timeout = Duration::from_secs(35);

    let started = Instant::now();
    let outcome = relay
        .next_event(request_deadline, per_chunk_timeout, started)
        .await;
    let elapsed = started.elapsed();

    assert!(
        elapsed < Duration::from_secs(2),
        "keepalives must not buy the provider the per-chunk timeout, waited {elapsed:?}"
    );
    let failure = outcome.expect_err("a stream of comments and no token must not succeed");
    assert_eq!(failure.failure_class, FailureClass::Timeout);
    assert!(
        failure.failover_eligible && !failure.retryable_same_deployment,
        "a first-token stall fails over without redialing the stalled lane"
    );
}

#[tokio::test]
async fn a_role_only_frame_does_not_satisfy_the_first_token_bound() {
    // OpenAI-compatible streams open with a delta carrying only the role; it
    // decodes to no semantic event, so the bound stays armed through it.
    let frames = stream::iter(vec![Ok::<_, reqwest::Error>(Bytes::from(
        "data: {\"choices\":[{\"delta\":{\"role\":\"assistant\"}}]}\n\n",
    ))])
    .chain(stream::pending())
    .boxed();
    let mut relay = UpstreamRelay::from_stream(
        frames,
        Dialect::OpenAiCompatible,
        Instant::now() + Duration::from_millis(80),
    );
    let started = Instant::now();
    let outcome = relay
        .next_event(
            Instant::now() + Duration::from_secs(120),
            Duration::from_secs(35),
            started,
        )
        .await;
    assert!(started.elapsed() < Duration::from_secs(2));
    let failure = outcome.expect_err("a role-only frame then silence is a stall");
    assert_eq!(failure.failure_class, FailureClass::Timeout);
    assert!(failure.failover_eligible);
}

#[tokio::test]
async fn a_semantic_event_the_waterfall_did_not_commit_leaves_the_bound_armed() {
    // Under refusal failover the waterfall withholds refusal deltas without
    // committing; the relay must not disarm on its own, so a provider that
    // stalls behind such an event still trips the fail-fast bound.
    let frames = stream::iter(vec![Ok::<_, reqwest::Error>(Bytes::from(
        "data: {\"choices\":[{\"delta\":{\"content\":\"hi\"}}]}\n\n",
    ))])
    .chain(stream::pending())
    .boxed();
    let mut relay = UpstreamRelay::from_stream(
        frames,
        Dialect::OpenAiCompatible,
        Instant::now() + Duration::from_millis(80),
    );
    let request_deadline = Instant::now() + Duration::from_secs(120);
    let started = Instant::now();
    let first = relay
        .next_event(request_deadline, Duration::from_secs(35), started)
        .await
        .expect("the event arrives")
        .expect("an event is produced");
    assert!(matches!(&first, Event::TextDelta(_)));
    // No commit() call: the waterfall withheld it.
    let failure = relay
        .next_event(request_deadline, Duration::from_secs(35), started)
        .await
        .expect_err("silence behind an uncommitted event is still a first-token stall");
    assert!(started.elapsed() < Duration::from_secs(2));
    assert_eq!(failure.failure_class, FailureClass::Timeout);
    assert!(failure.failover_eligible);
}

#[tokio::test]
async fn translated_custom_start_commits_before_deferred_wrapper_input() {
    let frames = stream::iter(vec![Ok::<_, reqwest::Error>(Bytes::from(
        "data: {\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":0,\"id\":\"call\",\"type\":\"function\",\"function\":{\"name\":\"patch\",\"arguments\":\"\"}}]}}]}\n\n",
    ))]).chain(stream::pending()).boxed();
    let mut relay = UpstreamRelay::from_stream(
        frames,
        Dialect::OpenAiCompatible,
        Instant::now() + Duration::from_millis(80),
    );
    relay.set_native_tool_translation(NativeToolTranslation::from([(
        "patch".into(),
        ("patch".into(), None, true),
    )]));
    let deadline = Instant::now() + Duration::from_secs(2);
    let started = Instant::now();
    let event = relay
        .next_event(deadline, Duration::from_millis(300), started)
        .await
        .unwrap()
        .unwrap();
    assert!(matches!(event, Event::ToolCallStarted { custom: true, .. }));
    assert!(crate::waterfall::is_semantic(&event));
    relay.commit();
    let failure = relay
        .next_event(deadline, Duration::from_millis(300), started)
        .await
        .expect_err("deferred custom input stalls only on the post-commit chunk bound");
    assert_eq!(failure.failure_class, FailureClass::Transport);
}

#[tokio::test]
async fn the_first_token_disarms_the_bound_and_the_chunk_timeout_takes_over() {
    // Keepalives, then a content token inside the bound: the token is yielded,
    // and a later stall is paced by the per-chunk timeout (a transport
    // failure at that horizon), never re-judged by the first-token bound.
    let frames = stream::iter(vec![
        Ok::<_, reqwest::Error>(Bytes::from(": keepalive\n\n")),
        Ok::<_, reqwest::Error>(Bytes::from(
            "data: {\"choices\":[{\"delta\":{\"content\":\"hi\"}}]}\n\n",
        )),
    ])
    .chain(stream::pending())
    .boxed();
    let mut relay = UpstreamRelay::from_stream(
        frames,
        Dialect::OpenAiCompatible,
        Instant::now() + Duration::from_millis(80),
    );
    let request_deadline = Instant::now() + Duration::from_secs(120);
    let per_chunk_timeout = Duration::from_millis(300);
    let started = Instant::now();
    let first = relay
        .next_event(request_deadline, per_chunk_timeout, started)
        .await
        .expect("the token arrives inside the bound")
        .expect("an event is produced");
    assert!(matches!(&first, Event::TextDelta(text) if text == "hi"));
    assert!(relay.first_token_at().is_some());
    // The waterfall commits on that first semantic event.
    relay.commit();

    let stall_started = Instant::now();
    let failure = relay
        .next_event(request_deadline, per_chunk_timeout, started)
        .await
        .expect_err("silence after the first token stalls on the chunk timeout");
    let waited = stall_started.elapsed();
    assert!(
        waited >= Duration::from_millis(250),
        "after the first token the per-chunk timeout paces reads, waited only {waited:?}"
    );
    assert_eq!(
        failure.failure_class,
        FailureClass::Transport,
        "a post-token stall is the ordinary chunk stall, not a first-token failure"
    );
}

#[test]
fn separately_reserved_repair_cannot_inherit_prior_cache_write_ttl() {
    let first = crate::settlement::Observation::default();
    first.record(&Event::Usage(Usage {
        input_tokens: Some(30),
        output_tokens: Some(7),
        cache_creation_input_tokens: Some(10),
        cache_creation_1h_input_tokens: Some(4),
        ..Usage::default()
    }));
    let next_attempt = crate::settlement::Observation::default();
    next_attempt.record(&Event::Usage(Usage {
        input_tokens: Some(20),
        output_tokens: Some(3),
        cache_creation_input_tokens: Some(20),
        ..Usage::default()
    }));
    assert_eq!(
        first
            .snapshot()
            .usage
            .unwrap()
            .cache_creation_1h_input_tokens,
        Some(4)
    );
    assert_eq!(
        next_attempt
            .snapshot()
            .usage
            .unwrap()
            .cache_creation_1h_input_tokens,
        None
    );
}
