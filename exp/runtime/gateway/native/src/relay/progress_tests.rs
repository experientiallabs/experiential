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
