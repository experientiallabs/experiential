//! Frame-level evidence for graceful connection retirement health hints.

use super::*;

#[tokio::test]
async fn only_remote_graceful_goaway_sets_the_health_hint() {
    use tokio::io::{AsyncReadExt, AsyncWriteExt};

    for (kind, reason, retirement) in [
        (7_u8, 0_u8, true),
        (7, 1, false),
        (3, 0, false),
        (3, 7, false),
    ] {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let server = tokio::spawn(async move {
            let (mut socket, _) = listener.accept().await.unwrap();
            let mut preface = [0; 24];
            socket.read_exact(&mut preface).await.unwrap();
            socket
                .write_all(&[0, 0, 0, 4, 0, 0, 0, 0, 0])
                .await
                .unwrap();
            loop {
                let mut header = [0; 9];
                socket.read_exact(&mut header).await.unwrap();
                let length = (usize::from(header[0]) << 16)
                    | (usize::from(header[1]) << 8)
                    | usize::from(header[2]);
                assert!(length <= 65_536);
                let mut payload = vec![0; length];
                socket.read_exact(&mut payload).await.unwrap();
                if header[3] == 4 && header[4] & 1 == 0 {
                    socket
                        .write_all(&[0, 0, 0, 4, 1, 0, 0, 0, 0])
                        .await
                        .unwrap();
                }
                if header[3] == 1 {
                    let frame = if kind == 7 {
                        // GOAWAY explicitly excludes this stream. Accounting
                        // still cannot infer that from reqwest's error alone.
                        vec![0, 0, 8, 7, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, reason]
                    } else {
                        vec![
                            0, 0, 4, 3, 0, header[5], header[6], header[7], header[8], 0, 0, 0,
                            reason,
                        ]
                    };
                    socket.write_all(&frame).await.unwrap();
                    // Keep the connection alive until the client observes the
                    // protocol frame, rather than replacing it with a socket EOF.
                    std::future::pending::<()>().await;
                }
            }
        });
        let client = client_builder(Duration::from_secs(1))
            .http2_prior_knowledge()
            .build()
            .unwrap();
        let payload = serde_json::json!({"model": "fixture"});
        let failure = classify_open(
            client
                .post(format!("http://{address}/v1/chat/completions"))
                .json(&payload)
                .send(),
            &payload,
            Duration::from_secs(2),
            Dialect::OpenAiCompatible,
            false,
        )
        .await
        .unwrap_err();
        server.abort();
        assert_eq!(failure.failure_class, FailureClass::Transport);
        assert_eq!(failure.graceful_transport_retirement, retirement);
        assert!(!failure.decision_provider_rejected);
    }
}
