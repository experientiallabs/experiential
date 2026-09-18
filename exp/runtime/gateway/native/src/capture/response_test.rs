use super::*;
use crate::capture::collector::Configuration;
use crate::capture::delivery::{Limits, Sink};
use crate::capture::record::{Protocol, Record, Request, Scope};
use bytes::Bytes;
use http_body_util::BodyExt;
use serde_json::json;
use std::convert::Infallible;
use std::sync::mpsc;
use std::time::{Duration, Instant};

struct MemorySink(mpsc::Sender<Record>);
impl Sink for MemorySink {
    fn write(&mut self, encoded: &str) -> Result<(), ()> {
        self.0
            .send(serde_json::from_str(encoded).map_err(|_| ())?)
            .map_err(|_| ())
    }
}

fn collector(maximum_response_bytes: usize) -> (Arc<Collector>, mpsc::Receiver<Record>) {
    let (sender, receiver) = mpsc::channel();
    let collector = Arc::new(
        Collector::new(
            Configuration {
                delivery: Limits {
                    maximum_records: 8,
                    maximum_bytes: 65536,
                    maximum_record_bytes: 8192,
                },
                maximum_pending_records: 8,
                maximum_pending_bytes: 65536,
                maximum_request_bytes: 4096,
                maximum_response_bytes,
                ttl_seconds: 30,
                settlement_required: false,
            },
            MemorySink(sender),
        )
        .unwrap(),
    );
    assert!(collector.begin(Request {
        request_id: "request".into(),
        scope: Scope {
            organization_id: "org".into(),
            identity_id: "identity".into(),
            application_id: "alias".into()
        },
        protocol: Protocol::ChatCompletions,
        model_id: Some("model".into()),
        context: json!({"schema_version":1,"request":{"messages":[]}}),
    }));
    (collector, receiver)
}

fn record(collector: &Collector, receiver: mpsc::Receiver<Record>) -> Record {
    assert!(collector.close_until(Instant::now() + Duration::from_secs(1)));
    let records: Vec<Record> = receiver.try_iter().collect();
    assert_eq!(records.len(), 1);
    records.into_iter().next().unwrap()
}

#[tokio::test]
async fn json_capture_preserves_wire_bytes_and_normalizes_only_the_stored_copy() {
    let (collector, receiver) = collector(4096);
    let original = br#"{"id":"completion","a\u0000":1,"a\ufffd":2,"text":"a\u0000b"}"#;
    let response = Response::builder()
        .header("content-type", "application/json")
        .header("x-gateway-deployment", "deployment")
        .body(Body::from(original.as_slice()))
        .unwrap();
    let captured = capture_response(Some(collector.clone()), "request", response);
    assert_eq!(captured.headers()["x-gateway-deployment"], "deployment");
    let actual = captured.into_body().collect().await.unwrap().to_bytes();
    assert_eq!(actual.as_ref(), original);
    let record = record(&collector, receiver);
    let Some(CapturedResponse::Json { body, .. }) = record.response else {
        panic!()
    };
    assert_eq!(body["a\u{fffd}"], 2);
    assert_eq!(body["a\u{fffd}~1"], 1);
    assert_eq!(body["text"], "a\u{fffd}b");
}

#[test]
fn sse_parser_preserves_multiline_crlf_done_and_ignores_unfinished_event() {
    let frames = data_frames(
        b": comment\r\ndata: {\r\ndata: \"ok\": true}\r\n\r\ndata:[DONE]\n\ndata: partial",
    );
    assert_eq!(frames, vec![json!({"ok":true}), json!("[DONE]")]);
}

#[tokio::test]
async fn content_length_json_is_complete_even_when_consumer_never_polls_eof() {
    let (collector, receiver) = collector(4096);
    let bytes = Bytes::from_static(br#"{"id":"response","status":"completed"}"#);
    let source = futures_util::stream::iter([Ok::<_, Infallible>(bytes.clone())])
        .chain(futures_util::stream::pending());
    let response = Response::builder()
        .header("content-type", "application/json")
        .header("content-length", bytes.len())
        .body(Body::from_stream(source))
        .unwrap();
    let mut body = capture_response(Some(collector.clone()), "request", response)
        .into_body()
        .into_data_stream();
    assert_eq!(body.next().await.unwrap().unwrap(), bytes);
    drop(body);
    let record = record(&collector, receiver);
    assert!(matches!(
        record.response,
        Some(CapturedResponse::Json { .. })
    ));
}

#[tokio::test]
async fn dropped_stream_keeps_only_whole_observed_frames_and_marks_disconnect() {
    let (collector, receiver) = collector(4096);
    let prefix = Bytes::from_static(b"data: {\"delta\":\"hello\"}\n\n");
    let source = futures_util::stream::iter([Ok::<_, Infallible>(prefix.clone())])
        .chain(futures_util::stream::pending());
    let response = Response::builder()
        .header("content-type", "text/event-stream")
        .body(Body::from_stream(source))
        .unwrap();
    let mut body = capture_response(Some(collector.clone()), "request", response)
        .into_body()
        .into_data_stream();
    assert_eq!(body.next().await.unwrap().unwrap(), prefix);
    drop(body);
    let record = record(&collector, receiver);
    let Some(CapturedResponse::Sse {
        frames,
        client_disconnected,
        truncated,
        ..
    }) = record.response
    else {
        panic!()
    };
    assert_eq!(frames, vec![json!({"delta":"hello"})]);
    assert!(client_disconnected);
    assert!(!truncated);
}

#[tokio::test]
async fn oversized_stream_is_forwarded_in_full_but_capture_is_a_marked_prefix() {
    let (collector, receiver) = collector(256);
    let first = Bytes::from_static(b"data: {\"delta\":\"first\"}\n\n");
    let large = Bytes::from(format!("data: {{\"delta\":\"{}\"}}\n\n", "a".repeat(512)));
    let expected = [first.as_ref(), large.as_ref()].concat();
    let source = futures_util::stream::iter([Ok::<_, Infallible>(first), Ok(large)]);
    let response = Response::builder()
        .header("content-type", "text/event-stream")
        .body(Body::from_stream(source))
        .unwrap();
    let actual = capture_response(Some(collector.clone()), "request", response)
        .into_body()
        .collect()
        .await
        .unwrap()
        .to_bytes();
    assert_eq!(actual.as_ref(), expected);
    let record = record(&collector, receiver);
    let Some(CapturedResponse::Sse {
        frames,
        truncated,
        client_disconnected,
        ..
    }) = record.response
    else {
        panic!()
    };
    assert_eq!(frames, vec![json!({"delta":"first"})]);
    assert!(truncated);
    assert!(!client_disconnected);
}

#[tokio::test]
async fn unregistered_requests_and_failed_responses_never_capture_response_content() {
    let (collector, receiver) = collector(4096);
    let body = capture_response(
        Some(collector.clone()),
        "unknown",
        Response::new(Body::from("not-captured")),
    )
    .into_body()
    .collect()
    .await
    .unwrap()
    .to_bytes();
    assert_eq!(body.as_ref(), b"not-captured");
    let failed = Response::builder()
        .status(429)
        .body(Body::from("failure"))
        .unwrap();
    let body = capture_response(Some(collector.clone()), "request", failed)
        .into_body()
        .collect()
        .await
        .unwrap()
        .to_bytes();
    assert_eq!(body.as_ref(), b"failure");
    assert!(record(&collector, receiver).response.is_none());
}
