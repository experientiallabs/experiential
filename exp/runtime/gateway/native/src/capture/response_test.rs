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
    type Prepared = String;

    fn preparation_bytes(maximum_record_bytes: usize) -> usize {
        maximum_record_bytes
    }

    fn prepare(&self, record: &Record, maximum_bytes: usize) -> Result<Self::Prepared, ()> {
        record.encode(maximum_bytes).ok_or(())
    }

    fn write(&mut self, record: &Self::Prepared) -> Result<(), ()> {
        self.0
            .send(serde_json::from_str(record).map_err(|_| ())?)
            .map_err(|_| ())
    }
}

fn collector(maximum_response_bytes: usize) -> (Arc<Collector>, mpsc::Receiver<Record>) {
    collector_with_relay(maximum_response_bytes, false)
}

fn collector_with_relay(
    maximum_response_bytes: usize,
    relay_metadata: bool,
) -> (Arc<Collector>, mpsc::Receiver<Record>) {
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
                relay_metadata,
                truncate_request: false,
                asynchronous_delivery: true,
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
        context: Arc::new(json!({"schema_version":1,"request":{"messages":[]}})),
    }));
    (collector, receiver)
}

struct HeldSink {
    entered: Option<tokio::sync::oneshot::Sender<()>>,
    resume: mpsc::Receiver<()>,
    records: mpsc::Sender<Record>,
    fail: bool,
}

impl Sink for HeldSink {
    type Prepared = String;

    fn preparation_bytes(maximum_record_bytes: usize) -> usize {
        maximum_record_bytes
    }

    fn prepare(&self, record: &Record, maximum_bytes: usize) -> Result<Self::Prepared, ()> {
        record.encode(maximum_bytes).ok_or(())
    }

    fn write(&mut self, record: &Self::Prepared) -> Result<(), ()> {
        if let Some(entered) = self.entered.take() {
            let _ = entered.send(());
            self.resume
                .recv_timeout(Duration::from_secs(5))
                .map_err(|_| ())?;
        }
        if self.fail {
            self.fail = false;
            Err(())
        } else {
            self.records
                .send(serde_json::from_str(record).map_err(|_| ())?)
                .map_err(|_| ())
        }
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 1)]
async fn stalled_writer_backpressures_only_after_queue_capacity_is_consumed() {
    for fail in [false, true] {
        let (started, entered) = tokio::sync::oneshot::channel();
        let (resume, paused) = mpsc::channel();
        let (records, observed) = mpsc::channel();
        let collector = Arc::new(
            Collector::new(
                Configuration {
                    delivery: Limits {
                        maximum_records: 1,
                        maximum_bytes: 32768,
                        maximum_record_bytes: 8192,
                    },
                    maximum_pending_records: 8,
                    // Eight request trees plus the two admitted response bodies.
                    maximum_pending_bytes: 32768,
                    maximum_request_bytes: 2048,
                    maximum_response_bytes: 16384,
                    ttl_seconds: 30,
                    settlement_required: false,
                    relay_metadata: false,
                    truncate_request: false,
                    asynchronous_delivery: true,
                },
                HeldSink {
                    entered: Some(started),
                    resume: paused,
                    records,
                    fail,
                },
            )
            .unwrap(),
        );
        // Two whole responses fit; a third waits before consuming any bytes.
        let first = collector.body_permit().await.unwrap();
        let second = collector.body_permit().await.unwrap();
        assert!(
            tokio::time::timeout(Duration::from_millis(20), collector.body_permit())
                .await
                .is_err()
        );
        drop(first);
        let third = tokio::time::timeout(Duration::from_secs(1), collector.body_permit())
            .await
            .unwrap()
            .unwrap();
        drop((second, third));
        let mut tasks = Vec::new();
        for index in 0..8 {
            let id = index.to_string();
            assert!(collector.begin(Request {
                request_id: id.clone(),
                scope: Scope {
                    organization_id: "org".into(),
                    identity_id: "identity".into(),
                    application_id: "alias".into()
                },
                protocol: Protocol::ChatCompletions,
                model_id: Some("model".into()),
                context: Arc::new(json!({"schema_version":1,"request":{}})),
            }));
            let owner = collector.clone();
            tasks.push(tokio::spawn(async move {
                let expected =
                    serde_json::to_vec(&json!({"id":id,"text":"x".repeat(2048)})).unwrap();
                let actual = capture_response(
                    Some(owner),
                    &id,
                    Response::new(Body::from(expected.clone())),
                )
                .into_body()
                .collect()
                .await;
                assert_eq!(actual.unwrap().to_bytes().as_ref(), expected.as_slice());
            }));
        }
        tokio::time::timeout(Duration::from_secs(1), entered)
            .await
            .unwrap()
            .unwrap();
        // With one async worker this timer proves delivery does not monopolize it.
        tokio::time::timeout(
            Duration::from_secs(1),
            tokio::time::sleep(Duration::from_millis(30)),
        )
        .await
        .unwrap();
        let completed = tasks.iter().filter(|task| task.is_finished()).count();
        let failures = collector.counts()[3..].to_vec();
        resume.send(()).unwrap();
        for task in tasks {
            tokio::time::timeout(Duration::from_secs(2), task)
                .await
                .unwrap()
                .unwrap();
        }
        assert!(collector.close_until(Instant::now() + Duration::from_secs(1)));
        // The first response leaves without waiting for storage, then the
        // single-slot destination applies bounded backpressure to the rest.
        assert_eq!(completed, 1);
        assert_eq!(failures, [0, 0, 0]);
        let rows: Vec<_> = observed.try_iter().collect();
        assert_eq!(rows.len(), 8);
        assert!(rows
            .iter()
            .all(|record| matches!(record.response, Some(CapturedResponse::Json { .. }))));
        assert_eq!(
            collector.counts(),
            if fail {
                [0, 0, 8, 1, 0, 0]
            } else {
                [0, 0, 8, 0, 0, 0]
            }
        );
    }
}

fn record(collector: &Collector, receiver: mpsc::Receiver<Record>) -> Record {
    assert!(collector.close_until(Instant::now() + Duration::from_secs(1)));
    let records: Vec<Record> = receiver.try_iter().collect();
    assert_eq!(records.len(), 1);
    records.into_iter().next().unwrap()
}

#[tokio::test]
async fn numeric_sources_preserve_wide_integers_in_json_sse_and_wire() {
    const SOURCE: &str =
        r#"{"enum":[1208925819614629174706177,-9223372036854775809],"text":"a\u0000b"}"#;
    for sse in [false, true] {
        let (collector, receiver) = collector_with_relay(8192, true);
        assert!(collector.claim_relay("request"));
        assert!(collector.finish_relay(
            "request",
            super::super::relay::Relay {
                metadata: serde_json::from_value(json!({
                    "wire_request": {"method":"POST", "body_bytes":SOURCE.len()},
                    "headers":[], "timing":{}, "relay_completed":true,
                    "client_disconnected":false
                }))
                .unwrap(),
                body: SOURCE.as_bytes().to_vec(),
            }
        ));
        let content = if sse {
            format!("data: {SOURCE}\n\ndata: [DONE]\n\n")
        } else {
            SOURCE.to_owned()
        };
        let response = capture_response(
            Some(collector.clone()),
            "request",
            Response::builder()
                .header(
                    "content-type",
                    if sse {
                        "text/event-stream"
                    } else {
                        "application/json"
                    },
                )
                .body(Body::from(content.clone()))
                .unwrap(),
        );
        assert_eq!(
            response.into_body().collect().await.unwrap().to_bytes(),
            content.as_bytes()
        );
        let record = record(&collector, receiver);
        assert_eq!(
            record.transport.unwrap()["wire_request"]["body_source_json"],
            SOURCE
        );
        let source = match record.response.unwrap() {
            CapturedResponse::Json { source_json, .. } => source_json.unwrap(),
            CapturedResponse::Sse { source_json, .. } => {
                let frames: Vec<Box<RawValue>> =
                    serde_json::from_str(&source_json.unwrap()).unwrap();
                assert_eq!(frames.len(), 2);
                frames[0].get().to_owned()
            }
        };
        assert_eq!(source, SOURCE);
    }
}

#[tokio::test]
async fn relay_metadata_rendezvous_preserves_wire_and_does_not_duplicate_replays() {
    for metadata_first in [false, true] {
        let (collector, receiver) = collector_with_relay(4096, true);
        assert!(collector.claim_relay("request"));
        assert!(!collector.claim_relay("request"));
        assert!(!collector.claim_relay("unknown"));
        let metadata = || super::super::relay::Relay {
            metadata: serde_json::from_value(json!({
                "wire_request": {"method":"POST", "path":"/v1/chat/completions",
                    "headers":[["authorization","<redacted>"]], "body_bytes":14},
                "headers":[["x-gateway-provider","Experiential Cloud"]],
                "timing":{"total_ms":12.5},
                "relay_completed":true,"client_disconnected":false
            }))
            .unwrap(),
            body: br#"{"raw":"wire"}"#.to_vec(),
        };
        if metadata_first {
            assert!(collector.finish_relay("request", metadata()));
        }
        let body = capture_response(
            Some(collector.clone()),
            "request",
            Response::builder()
                .header("x-request-id", "request")
                .body(Body::from(r#"{"choices":[]}"#))
                .unwrap(),
        )
        .into_body()
        .collect()
        .await
        .unwrap()
        .to_bytes();
        assert_eq!(&body[..], br#"{"choices":[]}"#);
        if !metadata_first {
            assert!(receiver.try_recv().is_err());
            assert!(collector.finish_relay("request", metadata()));
        }
        let record = record(&collector, receiver);
        assert_eq!(
            record.transport.as_ref().unwrap()["wire_request"]["body"],
            json!({"raw":"wire"})
        );
        assert_eq!(
            record.transport.as_ref().unwrap()["timing"]["total_ms"],
            12.5
        );
        assert!(matches!(
            record.response,
            Some(CapturedResponse::Json { .. })
        ));
        assert_eq!(collector.counts()[4..], [0, 0]);
    }
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
    let Some(CapturedResponse::Json {
        body, source_json, ..
    }) = record.response
    else {
        panic!()
    };
    assert_eq!(body["a\u{fffd}"], 2);
    assert_eq!(body["a\u{fffd}~1"], 1);
    assert_eq!(body["text"], "a\u{fffd}b");
    assert_eq!(
        serde_json::from_str::<Value>(&source_json.unwrap()).unwrap(),
        serde_json::from_slice::<Value>(original).unwrap()
    );
}

#[tokio::test]
async fn uncorrelated_provider_error_does_not_wait_for_unavailable_relay_metadata() {
    let (collector, receiver) = collector_with_relay(4096, true);
    let response = Response::builder()
        .status(400)
        .body(Body::from(r#"{"error":"provider rejected"}"#))
        .unwrap();
    let actual = capture_response(Some(collector.clone()), "request", response)
        .into_body()
        .collect()
        .await
        .unwrap()
        .to_bytes();
    assert_eq!(&actual[..], br#"{"error":"provider rejected"}"#);
    let record = record(&collector, receiver);
    assert!(record.transport.is_none());
    assert!(matches!(
        record.response,
        Some(CapturedResponse::Json { status: 400, .. })
    ));
}

#[test]
fn sse_parser_preserves_multiline_crlf_done_and_ignores_unfinished_event() {
    let frames = data_frames(
        b": comment\r\ndata: {\r\ndata: \"ok\": true}\r\n\r\ndata:[DONE]\n\ndata: partial",
    );
    assert_eq!(frames, vec![json!({"ok":true}), json!("[DONE]")]);
}

#[test]
fn ordinary_frames_and_literal_escape_text_need_no_lossless_sidecar() {
    let mut value = json!([{"text":"café 雪", "literal":"\\u0000", "nested":[false,3,null]}]);
    let original = value.clone();
    assert!(lossless_projection(&mut value).is_none());
    assert_eq!(value, original);
}

#[test]
fn nested_nul_keys_and_values_preserve_exact_source_and_do_not_merge_keys() {
    let mut value = json!([{"nested":{"a\0":1,"a\u{fffd}":2,"text":"x\0y"}}]);
    let original = value.clone();
    let source = lossless_projection(&mut value).unwrap();
    assert_eq!(serde_json::from_str::<Value>(&source).unwrap(), original);
    assert_eq!(value[0]["nested"]["a\u{fffd}"], 2);
    assert_eq!(value[0]["nested"]["a\u{fffd}~1"], 1);
    assert_eq!(value[0]["nested"]["text"], "x\u{fffd}y");
}

#[tokio::test]
async fn encoded_sse_budget_includes_lossless_sidecar_and_keeps_exact_prefix() {
    let prefix = CapturedResponse::Sse {
        status: 200,
        frames: vec![json!({"text":"first\u{fffd}"})],
        truncated: true,
        client_disconnected: false,
        source_json: Some(serde_json::to_string(&vec![json!({"text":"first\0"})]).unwrap()),
    };
    let limit = serde_json::to_string(&prefix).unwrap().len();
    let (collector, receiver) = collector(limit);
    let data = b"data: {\"text\":\"first\\u0000\"}\n\ndata: {\"text\":\"second\\u0000\"}\n\ndata: {\"text\":\"third\\u0000\"}\n\n";
    let response = Response::builder()
        .header("content-type", "text/event-stream")
        .body(Body::from(data.as_slice()))
        .unwrap();
    let actual = capture_response(Some(collector.clone()), "request", response)
        .into_body()
        .collect()
        .await
        .unwrap()
        .to_bytes();
    assert_eq!(actual.as_ref(), data);
    let response = record(&collector, receiver).response.unwrap();
    assert_eq!(serde_json::to_string(&response).unwrap().len(), limit);
    let CapturedResponse::Sse {
        frames,
        source_json,
        truncated,
        ..
    } = response
    else {
        panic!()
    };
    assert!(truncated);
    let restored: Vec<Value> = serde_json::from_str(&source_json.unwrap()).unwrap();
    assert_eq!(restored, vec![json!({"text":"first\0"})]);
    assert_eq!(frames, vec![json!({"text":"first\u{fffd}"})]);
}

#[tokio::test]
async fn encoded_sse_budget_includes_wide_numeric_sources_and_keeps_exact_prefix() {
    const SOURCE: &str = r#"{"value":1208925819614629174706177}"#;
    let prefix = CapturedResponse::Sse {
        status: 200,
        frames: vec![serde_json::from_str(SOURCE).unwrap()],
        truncated: true,
        client_disconnected: false,
        source_json: Some(format!("[{SOURCE}]")),
    };
    let limit = serde_json::to_string(&prefix).unwrap().len();
    let (collector, receiver) = collector(limit);
    let data = format!("data: {SOURCE}\n\ndata: {SOURCE}\n\ndata: {SOURCE}\n\n");
    let response = Response::builder()
        .header("content-type", "text/event-stream")
        .body(Body::from(data.clone()))
        .unwrap();
    let actual = capture_response(Some(collector.clone()), "request", response)
        .into_body()
        .collect()
        .await
        .unwrap()
        .to_bytes();
    assert_eq!(actual.as_ref(), data.as_bytes());
    let response = record(&collector, receiver).response.unwrap();
    assert_eq!(serde_json::to_string(&response).unwrap().len(), limit);
    let CapturedResponse::Sse {
        source_json,
        truncated,
        ..
    } = response
    else {
        panic!()
    };
    assert!(truncated);
    assert_eq!(source_json.unwrap(), format!("[{SOURCE}]"));
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
async fn explicit_host_denial_does_not_turn_a_valid_response_into_a_storage_failure() {
    for keep_prompt in [false, true] {
        let (collector, receiver) = collector(4096);
        let bytes = Bytes::from_static(b"data: {\"text\":\"ok\"}\n\n");
        let source = futures_util::stream::iter([Ok::<_, Infallible>(bytes.clone())]);
        let response = Response::builder()
            .header("content-type", "text/event-stream")
            .body(Body::from_stream(source))
            .unwrap();
        let mut body = capture_response(Some(collector.clone()), "request", response)
            .into_body()
            .into_data_stream();
        assert_eq!(body.next().await.unwrap().unwrap(), bytes);
        collector.settle("request", keep_prompt, false);
        assert!(body.next().await.is_none());
        assert!(collector.close_until(Instant::now() + Duration::from_secs(1)));
        let records: Vec<_> = receiver.try_iter().collect();
        assert_eq!(records.len(), usize::from(keep_prompt));
        assert!(records.iter().all(|record| record.response.is_none()));
        assert_eq!(&collector.counts()[3..], &[0, 0, 0]);
    }
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
async fn unregistered_requests_are_excluded_but_admitted_errors_are_evidence() {
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
        .body(Body::from(r#"{"error":{"message":"provider throttled"}}"#))
        .unwrap();
    let body = capture_response(Some(collector.clone()), "request", failed)
        .into_body()
        .collect()
        .await
        .unwrap()
        .to_bytes();
    assert_eq!(
        body.as_ref(),
        br#"{"error":{"message":"provider throttled"}}"#
    );
    let Some(CapturedResponse::Json { status, body, .. }) = record(&collector, receiver).response
    else {
        panic!("admitted provider error was discarded");
    };
    assert_eq!(status, 429);
    assert_eq!(body, json!({"error":{"message":"provider throttled"}}));
}
