//! Bounded Gemini trailer reads preserve the one declared outcome and latest meter.

use super::*;
use futures_util::stream;
use serde_json::json;

fn frame(value: serde_json::Value) -> reqwest::Result<Bytes> {
    Ok(Bytes::from(format!("data: {value}\n\n")))
}

fn finish(usage: bool) -> reqwest::Result<Bytes> {
    let mut value =
        json!({"candidates":[{"content":{"parts":[{"text":"hi"}]},"finishReason":"STOP"}]});
    if usage {
        value["usageMetadata"] = json!({"promptTokenCount":7,"candidatesTokenCount":2});
    }
    frame(value)
}

fn trailer() -> reqwest::Result<Bytes> {
    frame(
        json!({"usageMetadata":{"promptTokenCount":7,"candidatesTokenCount":2,"cachedContentTokenCount":3}}),
    )
}

async fn drain(relay: &mut UpstreamRelay, deadline: Instant, phase: Duration) -> Vec<Event> {
    let mut events = Vec::new();
    while let Some(event) = relay
        .next_event(deadline, phase, Instant::now())
        .await
        .unwrap()
    {
        events.push(event);
    }
    assert_eq!(events.iter().filter(|e| e.is_terminal()).count(), 1);
    assert!(matches!(events.last(), Some(Event::Completed)));
    events
}

#[tokio::test]
async fn gemini_immediate_eof_adds_no_static_drain_delay() {
    let now = Instant::now();
    let mut relay = UpstreamRelay::from_stream(
        stream::iter([finish(false), trailer()]).boxed(),
        Dialect::GeminiGenerateContent,
        now + Duration::from_secs(5),
    );
    let events = tokio::time::timeout(
        Duration::from_millis(100),
        drain(
            &mut relay,
            now + Duration::from_secs(5),
            Duration::from_secs(3),
        ),
    )
    .await
    .expect("ready EOF must not wait for the drain allowance");
    assert!(
        matches!(&events[1], Event::Usage(u) if u.input_tokens == Some(7) && u.cached_input_tokens == Some(3))
    );
}

#[tokio::test]
async fn gemini_trailer_before_bound_is_kept_but_after_bound_is_not_polled() {
    for (delay_ms, expected_cache) in [(60, Some(3)), (140, None)] {
        let source = stream::iter([finish(false)])
            .chain(stream::once(async move {
                tokio::time::sleep(Duration::from_millis(delay_ms)).await;
                trailer()
            }))
            .chain(stream::pending());
        let now = Instant::now();
        let mut relay = UpstreamRelay::from_stream(
            source.boxed(),
            Dialect::GeminiGenerateContent,
            now + Duration::from_secs(5),
        );
        let events = drain(
            &mut relay,
            now + Duration::from_secs(5),
            Duration::from_millis(100),
        )
        .await;
        let cache = events.iter().find_map(|e| match e {
            Event::Usage(u) => u.cached_input_tokens,
            _ => None,
        });
        assert_eq!(cache, expected_cache);
        assert!(now.elapsed() < Duration::from_millis(500));
    }
}

#[tokio::test]
async fn gemini_keepalives_and_ready_content_never_renew_the_drain_bound() {
    let source = stream::iter([finish(true)]).chain(stream::repeat_with(|| {
        Ok(Bytes::from_static(b": ping\n\n"))
    }));
    let now = Instant::now();
    let mut relay = UpstreamRelay::from_stream(
        source.boxed(),
        Dialect::GeminiGenerateContent,
        now + Duration::from_secs(5),
    );
    let events = drain(
        &mut relay,
        now + Duration::from_secs(5),
        Duration::from_millis(30),
    )
    .await;
    assert!(matches!(&events[1], Event::Usage(u) if u.input_tokens == Some(7)));
    assert!(now.elapsed() < Duration::from_millis(500));
}

#[tokio::test]
async fn gemini_request_deadline_caps_the_metadata_window() {
    let source = stream::iter([finish(true)]).chain(stream::pending());
    let now = Instant::now();
    let mut relay = UpstreamRelay::from_stream(
        source.boxed(),
        Dialect::GeminiGenerateContent,
        now + Duration::from_secs(5),
    );
    let events = drain(
        &mut relay,
        now + Duration::from_millis(30),
        Duration::from_secs(5),
    )
    .await;
    assert!(matches!(&events[1], Event::Usage(u) if u.output_tokens == Some(2)));
    assert!(now.elapsed() < Duration::from_millis(500));
}

#[tokio::test]
async fn gemini_cancelled_read_closes_transport_and_keeps_known_meter_once() {
    let source = stream::iter([finish(true), trailer()]).chain(stream::pending());
    let now = Instant::now();
    let observation = crate::settlement::Observation::default();
    let mut relay = UpstreamRelay::from_stream(
        source.boxed(),
        Dialect::GeminiGenerateContent,
        now + Duration::from_secs(5),
    );
    relay.set_observation(observation.clone());
    let deadline = now + Duration::from_secs(5);
    let phase = Duration::from_secs(2);
    assert!(matches!(
        relay.next_event(deadline, phase, now).await.unwrap(),
        Some(Event::TextDelta(_))
    ));
    assert!(tokio::time::timeout(
        Duration::from_millis(20),
        relay.next_event(deadline, phase, now)
    )
    .await
    .is_err());
    assert_eq!(
        observation.snapshot().usage.unwrap().cached_input_tokens,
        Some(3)
    );
    assert!(
        observation.snapshot().terminal.is_none(),
        "usage must not freeze before trailer drain"
    );
    relay.close_transport();
    let events = drain(&mut relay, deadline, phase).await;
    assert_eq!(
        events
            .iter()
            .filter(|e| matches!(e, Event::Usage(_)))
            .count(),
        1
    );
    let snapshot = observation.snapshot();
    assert!(matches!(snapshot.terminal, Some(Event::Completed)));
    assert_eq!(snapshot.usage.unwrap().cached_input_tokens, Some(3));
    relay.close_transport();
    assert!(relay
        .next_event(deadline, phase, now)
        .await
        .unwrap()
        .is_none());
}

#[tokio::test]
async fn gemini_post_finish_transport_break_keeps_completed_and_known_usage() {
    let error = reqwest::Client::new()
        .get("not a url")
        .send()
        .await
        .unwrap_err();
    let source = stream::iter([finish(true), trailer(), Err(error)]);
    let now = Instant::now();
    let mut relay = UpstreamRelay::from_stream(
        source.boxed(),
        Dialect::GeminiGenerateContent,
        now + Duration::from_secs(5),
    );
    let events = drain(
        &mut relay,
        now + Duration::from_secs(5),
        Duration::from_secs(2),
    )
    .await;
    assert!(matches!(&events[1], Event::Usage(u) if u.cached_input_tokens == Some(3)));
}
