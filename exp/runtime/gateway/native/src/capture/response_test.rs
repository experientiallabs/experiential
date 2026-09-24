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

#[test]
fn failed_node_heavy_wire_preparation_owns_only_one_decoded_workspace() {
    use crate::capture::delivery::Delivery;
    use std::collections::{HashMap, HashSet};
    use std::sync::Mutex;

    struct RecoveringBatch {
        seen: Arc<Mutex<HashMap<String, (usize, usize)>>>,
        writes: mpsc::Sender<Vec<String>>,
        first_write: mpsc::Sender<()>,
        release: mpsc::Receiver<()>,
        recover: Arc<AtomicBool>,
        retried_first: bool,
    }
    impl Sink for RecoveringBatch {
        type Prepared = String;

        fn preparation_bytes(maximum: usize) -> usize {
            (maximum + 1024 * 1024) * 5 + 64 * 256
        }

        fn batch_records(&self) -> usize {
            64
        }

        fn batch_bytes(&self) -> usize {
            1024 * 1024
        }

        fn prepared_bytes(&self, value: &String) -> usize {
            value.len()
        }

        fn prepare(&self, record: &Record, maximum: usize) -> Result<String, ()> {
            let id = &record.request.request_id;
            if id != "good" {
                let Some(CapturedResponse::Json { body, .. }) = &record.response else {
                    panic!("actual WireResponse must decode before preparation")
                };
                let nodes = body["nodes"].as_array().unwrap();
                assert_eq!(nodes.len(), 16384);
                assert!(nodes.iter().all(|node| node == &json!(0)));
                assert!(
                    record.encode(maximum).is_none(),
                    "genuine encoded record overflow"
                );
                let mut seen = self.seen.lock().unwrap();
                let entry = seen
                    .entry(id.clone())
                    .or_insert((nodes.as_ptr() as usize, 0));
                assert_eq!(
                    entry.0,
                    nodes.as_ptr() as usize,
                    "retry replaced accepted data"
                );
                entry.1 += 1;
                if !self.recover.load(Ordering::Acquire) {
                    return Err(());
                }
                // A recovered test destination acknowledges a deliberate policy
                // exclusion only after rechecking every retained input element.
            }
            Ok(id.clone())
        }

        fn write(&mut self, _: &String) -> Result<(), ()> {
            unreachable!("batch destination")
        }

        fn write_batch(&mut self, records: &[&String]) -> Vec<bool> {
            if !self.retried_first {
                self.retried_first = true;
                self.first_write.send(()).unwrap();
                self.release.recv_timeout(Duration::from_secs(5)).unwrap();
                return vec![false; records.len()];
            }
            self.writes
                .send(records.iter().map(|value| (*value).clone()).collect())
                .unwrap();
            vec![true; records.len()]
        }
    }

    let (body_owner, _) = collector(65536);
    let seen = Arc::new(Mutex::new(HashMap::new()));
    let recover = Arc::new(AtomicBool::new(false));
    let (written, writes) = mpsc::channel();
    let (started, first_write) = mpsc::channel();
    let (release, held) = mpsc::channel();
    let maximum_bytes = 8 * 1024 * 1024;
    let delivery = Arc::new(
        Delivery::new(
            Limits {
                maximum_records: 64,
                maximum_bytes,
                maximum_record_bytes: 4096,
            },
            RecoveringBatch {
                seen: seen.clone(),
                writes: written,
                first_write: started,
                release: held,
                recover: recover.clone(),
                retried_first: false,
            },
        )
        .unwrap(),
    );
    let input = |id: String| -> Record {
        serde_json::from_value(json!({
            "schema_version":1,"request":{"request_id":id,
                "scope":{"organization_id":"org","identity_id":"identity","application_id":"app"},
                "protocol":"chat_completions","model_id":"root",
                "context":{"schema_version":1,"request":{}}},
            "response":null,"deployment_id":null,"metrics":null,"gemini_thought_parts":[],
            "captured_at":1.0
        }))
        .unwrap()
    };
    let target = delivery.clone();
    let mut producers = vec![std::thread::spawn(move || {
        target.submit_wait(input("good".into()), None)
    })];
    first_write.recv_timeout(Duration::from_secs(2)).unwrap();
    let wire = serde_json::to_vec(&json!({"nodes":vec![0;16384]})).unwrap();
    for index in 0..16 {
        let target = delivery.clone();
        let response = WireResponse {
            bytes: wire.clone(),
            sse: false,
            status: 200,
            truncated: false,
            disconnected: false,
            maximum_bytes: 65536,
            _charge: BodyCharge {
                collector: body_owner.clone(),
                bytes: 0,
                _permit: None,
            },
        };
        producers.push(std::thread::spawn(move || {
            target.submit_wait(input(format!("heavy-{index}")), Some(response))
        }));
    }
    let deadline = Instant::now() + Duration::from_secs(2);
    while delivery.counts()[0] != 17 && Instant::now() < deadline {
        std::thread::yield_now();
    }
    let queued = delivery.counts()[0];
    release.send(()).unwrap();
    let healthy_ack = writes.recv_timeout(Duration::from_secs(2)).unwrap();
    let deadline = Instant::now() + Duration::from_secs(2);
    while seen
        .lock()
        .unwrap()
        .values()
        .map(|(_, attempts)| attempts)
        .sum::<usize>()
        < 3
        && Instant::now() < deadline
    {
        std::thread::yield_now();
    }
    let blocked_workspaces = seen.lock().unwrap().len();
    let blocked_counts = delivery.counts();
    let drained_while_failed = delivery.close_until(Instant::now());
    recover.store(true, Ordering::Release);
    for producer in producers {
        assert!(producer.join().unwrap());
    }
    assert!(delivery.close_until(Instant::now() + Duration::from_secs(2)));
    body_owner.settle("request", false, false);
    assert!(body_owner.close_until(Instant::now() + Duration::from_secs(1)));
    let remaining: Vec<_> = writes.try_iter().flatten().collect();
    assert_eq!(queued, 17);
    assert_eq!(healthy_ack, ["good"]);
    assert_eq!(
        blocked_workspaces, 1,
        "failed batches retained multiple decoded workspaces"
    );
    assert_eq!(blocked_counts[0], 16);
    assert!(blocked_counts[1] <= maximum_bytes as u64);
    assert_eq!(blocked_counts[2], 1);
    assert!(!drained_while_failed);
    assert_eq!(seen.lock().unwrap().len(), 16);
    assert_eq!(remaining.len(), 16);
    assert_eq!(remaining.iter().collect::<HashSet<_>>().len(), 16);
    assert_eq!(delivery.counts()[0..3], [0, 0, 17]);
    assert_eq!(delivery.counts()[4], 0);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 1)]
async fn stalled_writer_backpressures_complete_responses_without_blocking_the_runtime() {
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
        assert!(tasks.iter().all(|task| !task.is_finished()));
        assert_eq!(&collector.counts()[3..], &[0, 0, 0]);
        resume.send(()).unwrap();
        for task in tasks {
            tokio::time::timeout(Duration::from_secs(2), task)
                .await
                .unwrap()
                .unwrap();
        }
        assert!(collector.close_until(Instant::now() + Duration::from_secs(1)));
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
