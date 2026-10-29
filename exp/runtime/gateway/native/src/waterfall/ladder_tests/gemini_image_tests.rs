//! A signed Gemini image commits its rung: its 1 to 2 MB signature evidence
//! never counts against the precommit withhold byte bound.

use super::*;

const PNG: &str =
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAIAAACQd1PeAAAADElEQVR4nGP4z8AAAAMBAQDJ/pLvAAAAAElFTkSuQmCC";

/// Serve one Gemini event stream (no `[DONE]`) to a single connection.
async fn spawn_gemini_rung(frames: Vec<Value>) -> String {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
        .await
        .expect("bind");
    let address = listener.local_addr().expect("address");
    tokio::spawn(async move {
        let (mut socket, _) = listener.accept().await.expect("accept");
        read_request(&mut socket).await;
        let body: String = frames
            .iter()
            .map(|frame| format!("data: {frame}\n\n"))
            .collect();
        let response = format!(
            "HTTP/1.1 200 OK\r\ncontent-type: text/event-stream\r\n\
             content-length: {}\r\nconnection: close\r\n\r\n{body}",
            body.len(),
        );
        socket.write_all(response.as_bytes()).await.expect("write");
        let _ = socket.shutdown().await;
    });
    format!("http://{address}/v1beta/models/image:streamGenerateContent?alt=sse")
}

#[test]
fn a_signed_gemini_image_commits_and_meters_its_image_output_tokens() {
    block_on(async {
        let harness = Harness::new();
        let signature = "s".repeat(4 * MAXIMUM_WITHHELD_REFUSAL_BYTES);
        let url = spawn_gemini_rung(vec![
            json!({"candidates":[{"content":{"parts":[{"inlineData":{"mimeType":"image/png","data":PNG},"thoughtSignature":signature}]}}]}),
            json!({"candidates":[{"content":{"parts":[{"text":""}]},"finishReason":"STOP"}],"usageMetadata":{"promptTokenCount":9,"candidatesTokenCount":1174,"thoughtsTokenCount":32,"candidatesTokensDetails":[{"modality":"IMAGE","tokenCount":1120}]}}),
        ])
        .await;
        let mut rung = wire("a", &url, 0);
        rung.provider = "gemini".to_string();
        rung.dialect = "gemini_generate_content".to_string();
        rung.image_output = true;
        rung.upstream_payload = json!({"contents":[{"role":"user","parts":[{"text":"Draw"}]}]});
        let (won, guard) = harness.run(&[rung], None, Duration::from_secs(10)).await;
        let Won::Committed(mut committed) = won else {
            panic!("the delivered image commits the rung");
        };
        assert!(
            matches!(committed.prefix.last(), Some(Event::Image(image)) if image.ends_with(PNG))
        );
        let prefix = committed.prefix.clone();
        let events = crate::relay::collect_committed(
            &mut committed,
            Instant::now() + Duration::from_secs(10),
            Duration::from_secs(10),
            Instant::now(),
        )
        .await
        .expect("the committed stream completes");
        assert!(prefix.iter().chain(&events).any(|event| matches!(
            event,
            Event::GeminiThoughtPart(part) if part["thoughtSignature"] == signature.as_str()
        )));
        let usage = events
            .iter()
            .find_map(|event| match event {
                Event::Usage(usage) => Some(usage.clone()),
                _ => None,
            })
            .expect("the meter arrives");
        assert_eq!(usage.image_output_tokens, Some(1120));
        assert_eq!(usage.output_tokens, Some(1206));
        let won = Won::Committed(committed);
        drop(finish(guard, won).await);
    });
}

#[test]
fn signed_thought_evidence_before_the_image_does_not_overflow_the_withhold_bound() {
    block_on(async {
        let harness = Harness::new();
        let signature = "t".repeat(4 * MAXIMUM_WITHHELD_REFUSAL_BYTES);
        let url = spawn_gemini_rung(vec![
            json!({"candidates":[{"content":{"parts":[{"text":"","thoughtSignature":signature}]}}]}),
            json!({"candidates":[{"content":{"parts":[{"thought":true,"inlineData":{"mimeType":"image/png","data":PNG},"thoughtSignature":signature}]}}]}),
            json!({"candidates":[{"content":{"parts":[{"inlineData":{"mimeType":"image/png","data":PNG}}]},"finishReason":"STOP"}],"usageMetadata":{"promptTokenCount":9,"candidatesTokenCount":55,"thoughtsTokenCount":100,"candidatesTokensDetails":[{"modality":"IMAGE","tokenCount":1120}]}}),
        ])
        .await;
        let mut rung = wire("a", &url, 0);
        rung.provider = "gemini".to_string();
        rung.dialect = "gemini_generate_content".to_string();
        rung.image_output = true;
        rung.upstream_payload = json!({"contents":[{"role":"user","parts":[{"text":"Draw"}]}]});
        let (won, guard) = harness.run(&[rung], None, Duration::from_secs(10)).await;
        let Won::Committed(committed) = won else {
            panic!("withheld thought evidence must not fail the rung before its image");
        };
        let evidence = committed
            .prefix
            .iter()
            .filter(|event| matches!(event, Event::GeminiThoughtPart(_)))
            .count();
        assert_eq!(evidence, 2);
        assert!(
            matches!(committed.prefix.last(), Some(Event::Image(image)) if image.ends_with(PNG))
        );
        drop(finish(guard, Won::Committed(committed)).await);
    });
}
