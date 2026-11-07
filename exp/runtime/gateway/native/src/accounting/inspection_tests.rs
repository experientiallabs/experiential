//! Comparison with independent policy ordering and deterministic release decisions.

use std::time::{Duration, Instant};

use serde_json::{json, Value};

use super::inspection::{Check, Inspection};

#[test]
fn native_output_checks_match_python_ordered_policy_and_streaming_decisions() {
    for (stream, lines) in [
        (
            false,
            include_str!("../../tests/generated/inspection.jsonl"),
        ),
        (
            true,
            include_str!("../../tests/generated/inspection_stream.jsonl"),
        ),
    ] {
        for (i, line) in lines.lines().enumerate() {
            let case: Value = crate::lossless_json::from_str(line).unwrap();
            let checks: Vec<Check> =
                crate::lossless_json::from_value(case["checks"].clone()).unwrap();
            let owner = Inspection::new(&checks, Instant::now() + Duration::from_secs(30));
            let actual = if stream {
                owner.segment(&case["input"])
            } else {
                owner.output(&case["input"])
            }
            .unwrap();
            assert_eq!(
                actual["action"], case["expected"]["action"],
                "stream={stream} case {i}"
            );
            for key in ["replacement_text", "release", "pending", "flagged"] {
                assert_eq!(
                    actual[key], case["expected"][key],
                    "stream={stream} case {i} field={key}"
                );
            }
            for key in ["failure_class", "safe_message", "safe_details"] {
                assert_eq!(
                    actual["failure"][key], case["expected"]["failure"][key],
                    "stream={stream} case {i} failure={key}"
                );
            }
        }
    }
}

fn http_check(url: &str, protected: bool, limit: u64) -> Check {
    crate::lossless_json::from_value(
        json!({"policy": {"protected": protected, "max_response_bytes": 1024},
        "check": {"check_id": "check", "capability": "pii", "action": "block", "timeout_ms": 1000},
        "adapter": {"kind": "http_json", "url": url, "max_response_bytes": limit,
            "authorization": null, "transport": {"public_only": false,
                "headers": {"authorization": "Bearer synthetic-fixture"}}}}),
    )
    .unwrap()
}

#[test]
fn native_classifier_http_is_bounded_cookie_free_and_never_redirects_or_attaches_python() {
    use std::io::{Read, Write};
    use std::net::TcpListener;
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let url = format!("http://{}/inspect", listener.local_addr().unwrap());
    let server = std::thread::spawn(move || {
        for (status, body) in [
            (200, "{\"flagged\":false}"),
            (200, "{\"flagged\":true}"),
            (200, "{\"flagged\":false,\"replacement_text\":\"invalid\"}"),
            (302, ""),
            (200, "{\"flagged\":false}"),
        ] {
            let (mut stream, _) = listener.accept().unwrap();
            stream
                .set_read_timeout(Some(Duration::from_secs(5)))
                .unwrap();
            let mut bytes = Vec::new();
            let end = loop {
                let mut buffer = [0; 512];
                let count = stream.read(&mut buffer).unwrap();
                assert!(count > 0);
                bytes.extend_from_slice(&buffer[..count]);
                if let Some(i) = bytes.windows(4).position(|w| w == b"\r\n\r\n") {
                    break i + 4;
                }
            };
            let header = String::from_utf8(bytes[..end].to_vec())
                .unwrap()
                .to_lowercase();
            assert!(!header.contains("\r\ncookie:"));
            assert!(header.contains("authorization: bearer synthetic-fixture"));
            let length: usize = header
                .lines()
                .find_map(|line| line.strip_prefix("content-length: "))
                .unwrap()
                .parse()
                .unwrap();
            while bytes.len() < end + length {
                let mut buffer = [0; 512];
                let count = stream.read(&mut buffer).unwrap();
                assert!(count > 0);
                bytes.extend_from_slice(&buffer[..count]);
            }
            let payload: Value =
                crate::lossless_json::from_slice(&bytes[end..end + length]).unwrap();
            assert_eq!(payload["completion"]["text"], "sample");
            assert_eq!(payload["stage"], "output");
            write!(stream, "HTTP/1.1 {status} Test\r\nContent-Length: {}\r\nContent-Type: application/json\r\nSet-Cookie: classifier=secret\r\nLocation: /must-not-follow\r\nConnection: close\r\n\r\n{body}", body.len()).unwrap();
        }
    });
    pyo3::Python::initialize();
    pyo3::Python::attach(|_| {
        let (send, receive) = std::sync::mpsc::channel();
        let worker = std::thread::spawn(move || {
            let mut actions = Vec::new();
            for (protected, limit) in [
                (true, 512),
                (true, 512),
                (false, 512),
                (true, 512),
                (true, 1),
            ] {
                let inspection = Inspection::new(
                    &[http_check(&url, protected, limit)],
                    Instant::now() + Duration::from_secs(5),
                );
                actions
                    .push(inspection.output(&json!({"text": "sample"})).unwrap()["action"].clone());
            }
            send.send(actions).unwrap();
        });
        assert_eq!(
            receive
                .recv_timeout(Duration::from_secs(8))
                .expect("classifier execution must not acquire the GIL"),
            vec![
                json!("allow"),
                json!("block"),
                json!("allow"),
                json!("error"),
                json!("error")
            ]
        );
        worker.join().unwrap();
    });
    server.join().unwrap();
}

#[test]
fn classifier_public_egress_rejects_private_literal_destinations_before_connecting() {
    for url in [
        "http://127.0.0.1/inspect",
        "http://[::1]/inspect",
        "http://169.254.169.254/inspect",
        "https://10.0.0.1/inspect",
        "http://2130706433/inspect",
    ] {
        assert!(crate::upstream::classifier_url(url, true).is_err(), "{url}");
    }
    assert!(
        crate::upstream::classifier_url("http://user:pass@example.test/inspect", false).is_err()
    );
    assert!(crate::upstream::classifier_url("http://127.0.0.1/inspect", false).is_ok());
}
