use super::*;
use crate::capture::collector::{CheckpointFailure, Configuration};
use crate::capture::delivery::{Limits, Sink};
use crate::capture::record::{Protocol, Record, Request, Scope};
use axum::http::StatusCode;
use serde_json::json;
use std::sync::mpsc;
use std::time::Instant;

/// Hand each encoded record to the test as JSON, before its files are released.
struct JsonSink(mpsc::Sender<Value>);

impl Sink for JsonSink {
    type Prepared = String;

    fn preparation_bytes(maximum_record_bytes: usize) -> usize {
        maximum_record_bytes
    }

    fn prepare(&self, record: &Record, maximum_bytes: usize) -> Result<Self::Prepared, ()> {
        record.encode(maximum_bytes).ok_or(())
    }

    fn write(&mut self, record: &Self::Prepared) -> Result<(), ()> {
        let value: Value = serde_json::from_str(record).map_err(|_| ())?;
        // Read every named part while the record still owns its files.
        if let Some(parts) = value["media"].as_array() {
            for part in parts {
                if let Some(path) = part["spill_path"].as_str() {
                    let bytes = std::fs::read(path).map_err(|_| ())?;
                    assert_eq!(bytes.len() as u64, part["bytes"].as_u64().unwrap());
                    assert_eq!(hex(&Sha256::digest(&bytes)), part["sha256"]);
                }
            }
        }
        self.0.send(value).map_err(|_| ())
    }
}

fn config() -> Configuration {
    Configuration {
        delivery: Limits {
            maximum_records: 8,
            maximum_bytes: 1 << 20,
            maximum_record_bytes: 65536,
        },
        maximum_pending_records: 4,
        maximum_pending_bytes: 1 << 20,
        maximum_request_bytes: 65536,
        maximum_response_bytes: 65536,
        ttl_seconds: 30,
        settlement_required: true,
        relay_metadata: false,
        truncate_request: false,
        asynchronous_delivery: false,
        capture_hidden_reasoning: false,
        capture_logprobs: false,
        spill: None,
        checkpoint_failure: CheckpointFailure::Fail,
    }
}

fn collector(name: Option<&str>) -> (Arc<Collector>, mpsc::Receiver<Value>) {
    let (sender, receiver) = mpsc::channel();
    let collector = match name {
        None => Collector::new(config(), JsonSink(sender)).unwrap(),
        Some(name) => {
            let mut config = config();
            let spill = crate::capture::spill::Configuration {
                directory: crate::capture::spill::tests::temporary_directory(name)
                    .to_string_lossy()
                    .into_owned(),
                maximum_bytes: 64 * 1024 * 1024,
                maximum_queued_bytes: 64 * 1024 * 1024,
                maximum_inflight_bytes: 64 * 1024 * 1024,
                part_threshold_bytes: 1024 * 1024,
            };
            let store = crate::capture::spill::SpillStore::new(&spill).unwrap();
            config.spill = Some(spill);
            Collector::with_spill(config, JsonSink(sender), Some(store)).unwrap()
        }
    };
    (Arc::new(collector), receiver)
}

fn request(id: &str, protocol: Protocol, body: Value) -> Request {
    Request {
        request_id: id.to_owned(),
        scope: Scope {
            organization_id: "org".into(),
            identity_id: "identity".into(),
            application_id: "app".into(),
        },
        protocol,
        model_id: Some("model".into()),
        context: Arc::new(json!({"schema_version": 1, "request": body})),
    }
}

fn drain(collector: &Collector, receiver: mpsc::Receiver<Value>) -> Vec<Value> {
    assert!(collector.close_until(Instant::now() + Duration::from_secs(5)));
    receiver.try_iter().collect()
}

fn json_answer(body: Value) -> Response {
    Response::builder()
        .status(StatusCode::OK)
        .header("content-type", "application/json")
        .header("x-gateway-deployment", "deployment-1")
        .body(Body::from(serde_json::to_vec(&body).unwrap()))
        .unwrap()
}

#[tokio::test]
async fn embeddings_vectors_leave_the_body_for_one_spilled_vector_part() {
    let (collector, receiver) = collector(Some("media-embeddings"));
    let input = json!({"input": ["a", "b"], "model": "embed"});
    assert!(collector.begin(request("embed", Protocol::Embeddings, input)));
    let answer = json!({
        "object": "list",
        "data": [
            {"object": "embedding", "index": 0, "embedding": [0.5, 0.25, 1.0]},
            {"object": "embedding", "index": 1, "embedding": [0.0, 0.5, 0.75]}
        ],
        "usage": {"prompt_tokens": 2, "total_tokens": 2}
    });
    let response = capture_buffered(
        Some(collector.clone()),
        "embed",
        MediaSurface::Embeddings,
        json_answer(answer.clone()),
    )
    .await;
    // The caller receives the exact answer.
    let body = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .unwrap();
    assert_eq!(serde_json::from_slice::<Value>(&body).unwrap(), answer);
    collector.settle("embed", true, true);
    let records = drain(&collector, receiver);
    assert_eq!(records.len(), 1);
    let record = &records[0];
    assert_eq!(record["request"]["protocol"], "embeddings");
    assert_eq!(record["deployment_id"], "deployment-1");
    assert_eq!(
        record["vector_summary"],
        json!({"count": 2, "dimensions": 3})
    );
    let stored = &record["response"]["body"];
    assert!(stored["data"][0].get("embedding").is_none());
    assert_eq!(stored["usage"]["total_tokens"], 2);
    let parts = record["media"].as_array().unwrap();
    assert_eq!(parts.len(), 1);
    assert_eq!(parts[0]["part"], "output");
    assert_eq!(parts[0]["kind"], "vector");
    assert_eq!(parts[0]["mime"], "application/json");
    assert!(parts[0]["spill_path"].is_string());
    assert!(record.get("omitted_parts").is_none());
}

#[tokio::test]
async fn images_decode_base64_into_image_parts_in_data_order() {
    let (collector, receiver) = collector(Some("media-images"));
    assert!(collector.begin(request(
        "image",
        Protocol::Images,
        json!({"prompt": "a cat"})
    )));
    let png = b"\x89PNG\r\n\x1a\nfirst";
    let jpeg = b"\xff\xd8\xff\xe0second";
    let encode = |raw: &[u8]| base64::engine::general_purpose::STANDARD.encode(raw);
    let answer = json!({
        "created": 1,
        "data": [
            {"b64_json": encode(png), "revised_prompt": "a cat"},
            {"b64_json": encode(jpeg)}
        ]
    });
    capture_buffered(
        Some(collector.clone()),
        "image",
        MediaSurface::Images,
        json_answer(answer),
    )
    .await;
    collector.settle("image", true, true);
    let record = drain(&collector, receiver).remove(0);
    let stored = &record["response"]["body"];
    assert!(stored["data"][0].get("b64_json").is_none());
    assert_eq!(stored["data"][0]["revised_prompt"], "a cat");
    let parts = record["media"].as_array().unwrap();
    assert_eq!(parts.len(), 2);
    assert_eq!(parts[0]["mime"], "image/png");
    assert_eq!(parts[0]["bytes"], png.len());
    assert_eq!(parts[1]["mime"], "image/jpeg");
    assert!(parts.iter().all(|part| part["kind"] == "image"));
}

#[tokio::test]
async fn speech_audio_becomes_an_output_part_without_a_spill_directory() {
    let (collector, receiver) = collector(None);
    assert!(collector.begin(request(
        "speech",
        Protocol::Speech,
        json!({"input": "hello"})
    )));
    let audio = vec![7u8; 4096];
    let answer = Response::builder()
        .status(StatusCode::OK)
        .header("content-type", "audio/mpeg")
        .body(Body::from(audio.clone()))
        .unwrap();
    let response = capture_buffered(
        Some(collector.clone()),
        "speech",
        MediaSurface::Speech,
        answer,
    )
    .await;
    let body = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .unwrap();
    assert_eq!(body.as_ref(), audio.as_slice());
    collector.settle("speech", true, true);
    let record = drain(&collector, receiver).remove(0);
    assert_eq!(
        record["response"]["body"],
        json!({"content_type": "audio/mpeg", "bytes": 4096})
    );
    let part = &record["media"][0];
    assert_eq!(part["kind"], "audio");
    assert_eq!(part["bytes"], 4096);
    assert_eq!(part["sha256"], hex(&Sha256::digest(&audio)));
    assert!(part["spill_path"].is_null());
}

#[tokio::test]
async fn a_text_transcript_is_kept_as_text_beside_its_input_audio() {
    let (collector, receiver) = collector(Some("media-transcription"));
    assert!(collector.begin(request(
        "transcribe",
        Protocol::Transcription,
        json!({"audio_bytes": 3, "response_format": "text"})
    )));
    attach_inputs(Some(&collector), "transcribe", || {
        vec![(
            MediaKind::Audio,
            "audio/wav".to_owned(),
            bytes::Bytes::from_static(b"RIF"),
        )]
    });
    let answer = Response::builder()
        .status(StatusCode::OK)
        .header("content-type", "text/plain; charset=utf-8")
        .body(Body::from("hello world"))
        .unwrap();
    capture_buffered(
        Some(collector.clone()),
        "transcribe",
        MediaSurface::Transcription,
        answer,
    )
    .await;
    collector.settle("transcribe", true, true);
    let record = drain(&collector, receiver).remove(0);
    assert_eq!(
        record["response"]["body"],
        json!({"content_type": "text/plain", "text": "hello world"})
    );
    assert_eq!(record["media"][0]["part"], "input");
    assert_eq!(record["media"][0]["mime"], "audio/wav");
}

#[tokio::test]
async fn settlement_omits_the_refused_half_and_its_parts() {
    // Response kept, request omitted.
    let (collector, receiver) = collector(Some("media-omit"));
    assert!(collector.begin(request(
        "omit-request",
        Protocol::Transcription,
        json!({"prompt": "secret"})
    )));
    attach_inputs(Some(&collector), "omit-request", || {
        vec![(
            MediaKind::Audio,
            "audio/wav".to_owned(),
            bytes::Bytes::from_static(b"voice"),
        )]
    });
    capture_buffered(
        Some(collector.clone()),
        "omit-request",
        MediaSurface::Transcription,
        json_answer(json!({"text": "hi"})),
    )
    .await;
    collector.settle("omit-request", false, true);
    // Request kept, response omitted.
    assert!(collector.begin(request(
        "omit-response",
        Protocol::Speech,
        json!({"input": "hi"})
    )));
    let audio = Response::builder()
        .status(StatusCode::OK)
        .header("content-type", "audio/mpeg")
        .body(Body::from(vec![1u8; 16]))
        .unwrap();
    capture_buffered(
        Some(collector.clone()),
        "omit-response",
        MediaSurface::Speech,
        audio,
    )
    .await;
    collector.settle("omit-response", true, false);
    let records = drain(&collector, receiver);
    let by_id = |id: &str| {
        records
            .iter()
            .find(|record| record["request"]["request_id"] == id)
            .unwrap()
            .clone()
    };
    let request_omitted = by_id("omit-request");
    assert_eq!(request_omitted["omitted_parts"], json!(["request"]));
    assert_eq!(
        request_omitted["request"]["context"],
        json!({"schema_version": 1, "request": {}})
    );
    assert!(request_omitted.get("media").is_none());
    assert_eq!(request_omitted["response"]["body"]["text"], "hi");
    let response_omitted = by_id("omit-response");
    assert_eq!(response_omitted["omitted_parts"], json!(["response"]));
    assert!(response_omitted["response"].is_null());
    assert!(response_omitted.get("media").is_none());
    assert_eq!(
        response_omitted["request"]["context"]["request"]["input"],
        "hi"
    );
}

#[test]
fn a_provider_job_outlives_the_configured_ttl_and_keeps_its_latest_state() {
    let (collector, receiver) = collector(None);
    let ttl = Duration::from_secs(7200);
    assert!(collector.begin_until(
        request("video", Protocol::Video, json!({"prompt": "waves"})),
        Some(ttl)
    ));
    let job = ProviderJob {
        id: "video_1".into(),
        deadline: 1_900_000_000.0,
    };
    assert!(collector.finish_job(
        "video",
        json!({"id": "video_1", "status": "queued"}),
        job.clone(),
        Some("deployment-v".into()),
    ));
    assert!(collector.finish_job(
        "video",
        json!({"id": "video_1", "status": "completed", "seconds": "4"}),
        job,
        None,
    ));
    collector.settle("video", true, true);
    let record = drain(&collector, receiver).remove(0);
    assert_eq!(record["response"]["body"]["status"], "completed");
    assert_eq!(record["deployment_id"], "deployment-v");
    assert_eq!(
        record["provider_job"],
        json!({"id": "video_1", "deadline": 1_900_000_000.0})
    );
    // Settled and emitted: a late update finds nothing.
    assert!(!collector.finish_job(
        "video",
        json!({"status": "completed"}),
        ProviderJob {
            id: "video_1".into(),
            deadline: 0.0
        },
        None
    ));
}

#[test]
fn an_error_body_is_kept_as_json_with_no_parts() {
    let shaped = shape(
        MediaSurface::Images,
        400,
        "application/json",
        br#"{"error":{"message":"bad"}}"#,
        None,
    )
    .unwrap();
    assert!(shaped.parts.is_empty());
    assert_eq!(shaped.body["error"]["message"], "bad");
}

#[test]
fn base64_embeddings_report_float32_dimensions() {
    let raw = [0u8; 16];
    let encoded = base64::engine::general_purpose::STANDARD.encode(raw);
    let body = serde_json::to_vec(&json!({"data": [{"embedding": encoded}]})).unwrap();
    let shaped = shape(
        MediaSurface::Embeddings,
        200,
        "application/json",
        &body,
        None,
    )
    .unwrap();
    assert_eq!(
        shaped.vector_summary,
        Some(VectorSummary {
            count: 1,
            dimensions: 4
        })
    );
}

#[test]
fn text_records_serialize_no_media_keys() {
    let capture = MediaCapture::default();
    let mut serializer = serde_json::Serializer::new(Vec::new());
    let mut record = serde::Serializer::serialize_struct(&mut serializer, "Record", 0).unwrap();
    capture.serialize_fields(&mut record).unwrap();
    SerializeStruct::end(record).unwrap();
    assert_eq!(serializer.into_inner(), b"{}");
}
