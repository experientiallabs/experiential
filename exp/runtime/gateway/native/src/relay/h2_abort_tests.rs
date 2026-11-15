use super::*;
use crate::dialects::Dialect;
use crate::events::Event;

fn sse_chunk(text: &str) -> Bytes {
    Bytes::from(format!(
        "data: {{\"choices\":[{{\"delta\":{{\"content\":\"{text}\"}}}}]}}\n\n"
    ))
}

/// Serve one h2c response that streams two deltas, then abort it the
/// given way after a pacing delay so the client is mid-body when the
/// abort lands (mirroring a proxy whose upstream dies mid-response).
async fn relay_over_h2(reset_stream: bool) -> (Vec<Event>, Failure) {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
        .await
        .expect("bind");
    let address = listener.local_addr().expect("addr");
    tokio::spawn(async move {
        let (socket, _peer) = listener.accept().await.expect("accept");
        let mut connection = h2::server::handshake(socket).await.expect("handshake");
        let accepted = connection.accept().await;
        // The connection must keep being polled for handshake frames and
        // window updates to reach the peer.
        let driver = tokio::spawn(async move {
            let _ = futures_util::future::poll_fn(|cx| connection.poll_closed(cx)).await;
        });
        if let Some(Ok((_request, mut respond))) = accepted {
            let response = http::Response::builder()
                .status(200)
                .header("content-type", "text/event-stream")
                .body(())
                .expect("response");
            let mut stream = respond.send_response(response, false).expect("headers");
            stream.send_data(sse_chunk("hi"), false).expect("data one");
            stream
                .send_data(sse_chunk("there"), false)
                .expect("data two");
            tokio::time::sleep(Duration::from_millis(300)).await;
            if reset_stream {
                // The exact shape a fronting proxy (Caddy) produces when
                // its own upstream aborts: the h2 stream resets.
                stream.send_reset(h2::Reason::INTERNAL_ERROR);
                let _ = driver.await;
            } else {
                // Whole-connection abort: every multiplexed stream on the
                // connection severs at once.
                driver.abort();
                drop(stream);
            }
        }
    });
    let client = reqwest::Client::builder()
        .http2_prior_knowledge()
        .build()
        .expect("client");
    let response = client
        .post(format!("http://{address}/v1/chat/completions"))
        .body("{}")
        .send()
        .await
        .expect("send");
    let mut relay = UpstreamRelay::new(
        response,
        Dialect::OpenAiCompatible,
        Instant::now() + Duration::from_secs(5),
    );
    let deadline = Instant::now() + Duration::from_secs(10);
    let per_chunk = Duration::from_secs(5);
    let mut events = Vec::new();
    loop {
        match relay.next_event(deadline, per_chunk, Instant::now()).await {
            Ok(Some(event)) => events.push(event),
            Ok(None) => panic!("an aborted stream must surface a failure, not clean EOF"),
            Err(failure) => return (events, failure),
        }
    }
}

#[tokio::test]
async fn an_h2_stream_reset_mid_stream_classifies_and_never_panics() {
    // Production wire fact (verified live 2026-09-03): the house-lane
    // proxy negotiates ALPN h2, so its "aborting with incomplete
    // response" reaches this relay as RST_STREAM, never h1 truncation.
    let (events, failure) = relay_over_h2(true).await;
    assert!(
        matches!(events.as_slice(), [Event::TextDelta(a), Event::TextDelta(b)] if a == "hi" && b == "there"),
        "delivered deltas precede the abort: {events:?}"
    );
    assert_eq!(failure.failure_class, FailureClass::Transport);
    assert!(failure.failover_eligible, "an aborted rung fails over");
    // The engine's account of the mid-stream break rides to the ledger.
    let detail = failure
        .provider_detail
        .as_deref()
        .expect("transport detail");
    assert!(detail.starts_with("stream "), "{detail}");
}

#[tokio::test]
async fn an_h2_connection_drop_mid_stream_classifies_and_never_panics() {
    let (events, failure) = relay_over_h2(false).await;
    assert_eq!(
        events.len(),
        2,
        "delivered deltas precede the abort: {events:?}"
    );
    assert_eq!(failure.failure_class, FailureClass::Transport);
    assert!(failure.failover_eligible);
}
