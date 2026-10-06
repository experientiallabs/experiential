//! Review regressions for consumer pauses, provider work and retained usage.

use super::*;
use futures_util::stream;

#[tokio::test]
async fn downstream_pause_does_not_expire_provider_idle() {
    let frames = stream::iter([
        Ok::<_, reqwest::Error>(Bytes::from_static(
            b"data: {\"choices\":[{\"delta\":{\"content\":\"first\"}}]}\n\n",
        )),
        Ok(Bytes::from_static(
            b"data: {\"choices\":[{\"delta\":{\"content\":\"second\"}}]}\n\n",
        )),
    ])
    .boxed();
    let start = Instant::now();
    let mut relay = UpstreamRelay::from_stream(
        frames,
        Dialect::OpenAiCompatible,
        start + Duration::from_secs(1),
    );
    let deadline = start + Duration::from_secs(2);
    let idle = Duration::from_millis(20);
    relay.next_event(deadline, idle, start).await.unwrap();
    relay.commit();
    tokio::time::sleep(Duration::from_millis(60)).await;
    assert!(
        matches!(relay.next_event(deadline, idle, start).await.unwrap(), Some(Event::TextDelta(text)) if text == "second")
    );
}

#[tokio::test]
async fn anthropic_message_delta_usage_survives_stall_before_message_stop() {
    let frames = stream::iter([
        Ok::<_, reqwest::Error>(Bytes::from_static(
            b"data: {\"type\":\"message_start\",\"message\":{\"usage\":{\"input_tokens\":7,\"output_tokens\":1}}}\n\n",
        )),
        Ok(Bytes::from_static(
            b"data: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"end_turn\"},\"usage\":{\"input_tokens\":11,\"output_tokens\":9}}\n\n",
        )),
    ]).chain(stream::pending()).boxed();
    let start = Instant::now();
    let mut relay = UpstreamRelay::from_stream(
        frames,
        Dialect::AnthropicMessages,
        start + Duration::from_millis(30),
    );
    let deadline = start + Duration::from_secs(1);
    let Some(Event::Usage(initial)) = relay
        .next_event(deadline, Duration::from_secs(1), start)
        .await
        .unwrap()
    else {
        panic!("initial usage");
    };
    relay
        .next_event(deadline, Duration::from_secs(1), start)
        .await
        .unwrap_err();
    let latest = relay.usage_before_failure(Some(initial)).unwrap();
    assert_eq!(latest.input_tokens, Some(11));
    assert_eq!(latest.output_tokens, Some(9));
}

#[tokio::test]
async fn remote_mcp_listing_keeps_byte_idle_until_its_result() {
    let frames = stream::once(async { Ok::<_, reqwest::Error>(Bytes::from_static(
        b"data: {\"type\":\"response.output_item.added\",\"output_index\":0,\"item\":{\"id\":\"mcpl_1\",\"type\":\"mcp_list_tools\",\"server_label\":\"remote\",\"tools\":[]}}\n\n",
    )) }).chain(stream::unfold(0, |count| async move {
        if count == 8 { return None; }
        tokio::time::sleep(Duration::from_millis(10)).await;
        let frame: &'static [u8] = if count == 7 {
            b"data: {\"type\":\"response.output_item.done\",\"output_index\":0,\"item\":{\"id\":\"mcpl_1\",\"type\":\"mcp_list_tools\",\"server_label\":\"remote\",\"tools\":[]}}\n\n"
        } else { b": ping\n\n" };
        Some((Ok(Bytes::from_static(frame)), count + 1))
    })).chain(stream::pending()).boxed();
    let start = Instant::now();
    let mut relay = UpstreamRelay::from_stream(
        frames,
        Dialect::OpenAiResponses,
        start + Duration::from_millis(50),
    );
    let deadline = start + Duration::from_secs(1);
    let idle = Duration::from_millis(40);
    relay.next_event(deadline, idle, start).await.unwrap();
    relay.commit();
    assert!(matches!(
        relay.next_event(deadline, idle, start).await.unwrap(),
        Some(Event::HostedToolItemCompleted { .. })
    ));
    assert!(relay
        .next_event(deadline, idle, start)
        .await
        .unwrap_err()
        .safe_message
        .contains("stopped making progress"));
}

/// One Anthropic SSE frame as the relay receives it.
fn anthropic_frame(payload: &'static str) -> reqwest::Result<Bytes> {
    Ok(Bytes::from(format!("data: {payload}\n\n")))
}

/// An Anthropic Write call whose `content` argument arrives after `silence`.
fn buffered_write_call(silence: Duration) -> BoxStream<'static, reqwest::Result<Bytes>> {
    stream::iter([
        anthropic_frame(r#"{"type":"message_start","message":{"usage":{"input_tokens":7,"output_tokens":1}}}"#),
        anthropic_frame(r#"{"type":"content_block_start","index":0,"content_block":{"type":"tool_use","id":"toolu_1","name":"Write","input":{}}}"#),
        anthropic_frame(r#"{"type":"content_block_delta","index":0,"delta":{"type":"input_json_delta","partial_json":"{\"file_path\": \"a.html\""}}"#),
    ])
    .chain(stream::once(async move {
        tokio::time::sleep(silence).await;
        anthropic_frame(r#"{"type":"content_block_delta","index":0,"delta":{"type":"input_json_delta","partial_json":", \"content\": \"<html/>\"}"}}"#)
    }))
    .chain(stream::iter([
        anthropic_frame(r#"{"type":"content_block_stop","index":0}"#),
        anthropic_frame(r#"{"type":"content_block_start","index":1,"content_block":{"type":"text","text":""}}"#),
        anthropic_frame(r#"{"type":"content_block_delta","index":1,"delta":{"type":"text_delta","text":"done"}}"#),
    ]))
    .chain(stream::pending())
    .boxed()
}

/// Drain events until `stop` matches one, returning the first failure if any.
async fn drain_until(
    relay: &mut UpstreamRelay,
    deadline: Instant,
    idle: Duration,
    start: Instant,
    stop: impl Fn(&Event) -> bool,
) -> Result<(), Failure> {
    loop {
        let event = relay
            .next_event(deadline, idle, start)
            .await?
            .expect("open stream");
        if matches!(event, Event::ToolCallStarted { .. }) {
            relay.commit();
        }
        if stop(&event) {
            return Ok(());
        }
    }
}

#[tokio::test]
async fn anthropic_buffered_tool_argument_outlasts_the_generation_idle_window() {
    let start = Instant::now();
    let idle = Duration::from_millis(20);
    let mut relay = UpstreamRelay::from_stream(
        buffered_write_call(idle * 4),
        Dialect::AnthropicMessages,
        start + Duration::from_secs(1),
    );
    let deadline = start + Duration::from_secs(5);
    drain_until(&mut relay, deadline, idle, start, |event| {
        matches!(event, Event::ToolCallCompleted { .. })
    })
    .await
    .expect("a silent buffered argument is not a stall");
    drain_until(
        &mut relay,
        deadline,
        idle,
        start,
        |event| matches!(event, Event::TextDelta(text) if text == "done"),
    )
    .await
    .unwrap();
    // The widened window closes with the call: later silence is a stall again.
    let failure = relay.next_event(deadline, idle, start).await.unwrap_err();
    assert!(failure.safe_message.contains("stopped making progress"));
    assert!(start.elapsed() < Duration::from_secs(1));
}

#[tokio::test]
async fn anthropic_buffered_tool_argument_is_still_bounded() {
    let start = Instant::now();
    let idle = Duration::from_millis(10);
    let mut relay = UpstreamRelay::from_stream(
        buffered_write_call(Duration::from_secs(5)),
        Dialect::AnthropicMessages,
        start + Duration::from_secs(1),
    );
    let deadline = start + Duration::from_secs(5);
    let failure = drain_until(&mut relay, deadline, idle, start, |_| false)
        .await
        .unwrap_err();
    assert!(failure.safe_message.contains("stopped making progress"));
    let waited = start.elapsed();
    assert!(waited >= idle * 10, "waited {waited:?}");
    assert!(waited < Duration::from_secs(1), "waited {waited:?}");
}

#[tokio::test]
async fn other_dialects_keep_the_generation_idle_window_for_tool_arguments() {
    let frames = stream::iter([Ok::<_, reqwest::Error>(Bytes::from_static(
        b"data: {\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":0,\"id\":\"call_1\",\"type\":\"function\",\"function\":{\"name\":\"Write\",\"arguments\":\"{\\\"a\\\"\"}}]}}]}\n\n",
    ))])
    .chain(stream::pending())
    .boxed();
    let start = Instant::now();
    let idle = Duration::from_millis(10);
    let mut relay = UpstreamRelay::from_stream(
        frames,
        Dialect::OpenAiCompatible,
        start + Duration::from_secs(1),
    );
    let deadline = start + Duration::from_secs(5);
    let failure = drain_until(&mut relay, deadline, idle, start, |_| false)
        .await
        .unwrap_err();
    assert!(failure.safe_message.contains("stopped making progress"));
    assert!(start.elapsed() < idle * 10);
}
