//! The tap with a spill store: nothing is trimmed and capture memory stays bounded.

use super::*;
use crate::capture::collector::Configuration;
use crate::capture::delivery::{Limits, Sink};
use crate::capture::record::{Protocol, Record, Request, Scope};
use crate::capture::spill::{self, SpillStore};
use crate::capture::split::{self, PartKind, SplitRecord};
use bytes::Bytes;
use http_body_util::BodyExt;
use serde_json::json;
use std::sync::mpsc;
use std::time::{Duration, Instant};

enum Prepared {
    Whole(String),
    Split(SplitRecord),
}

/// What the destination received, without loading parts into memory.
struct Delivered {
    record: Option<Record>,
    skeleton: Option<serde_json::Value>,
    parts: Vec<(PartKind, u64, Option<usize>)>,
    reassembled: Option<serde_json::Value>,
}

struct SplitSink {
    store: Arc<SpillStore>,
    delivered: mpsc::Sender<Delivered>,
    reassemble: bool,
}

impl Sink for SplitSink {
    type Prepared = Prepared;

    fn preparation_bytes(maximum_record_bytes: usize) -> usize {
        maximum_record_bytes
    }

    fn prepare(&self, record: &Record, maximum_bytes: usize) -> Result<Prepared, ()> {
        if !record.requires_split() {
            if let Some(encoded) = record.encode(maximum_bytes) {
                return Ok(Prepared::Whole(encoded));
            }
        }
        split::split(record, &self.store, maximum_bytes)
            .map(Prepared::Split)
            .ok_or(())
    }

    fn split_raw_above(&self) -> Option<usize> {
        Some(self.store.part_threshold_bytes)
    }

    fn write(&mut self, prepared: &Prepared) -> Result<(), ()> {
        let delivered = match prepared {
            Prepared::Whole(encoded) => Delivered {
                record: Some(serde_json::from_str(encoded).map_err(|_| ())?),
                skeleton: None,
                parts: Vec::new(),
                reassembled: None,
            },
            Prepared::Split(split) => Delivered {
                record: None,
                skeleton: serde_json::from_str(&split.skeleton).ok(),
                parts: split
                    .parts
                    .iter()
                    .map(|part| {
                        let lines = (part.kind == PartKind::Frames).then(|| {
                            let file = std::fs::File::open(part.file.path()).unwrap();
                            std::io::BufRead::lines(std::io::BufReader::new(file)).count()
                        });
                        (part.kind, part.bytes, lines)
                    })
                    .collect(),
                reassembled: self.reassemble.then(|| split::tests::reassemble(split)),
            },
        };
        self.delivered.send(delivered).map_err(|_| ())
    }
}

fn spill_collector(
    name: &str,
    window: usize,
    maximum_spill_bytes: u64,
    reassemble: bool,
) -> (Arc<Collector>, mpsc::Receiver<Delivered>) {
    let spill_config = spill::Configuration {
        directory: spill::tests::temporary_directory(name)
            .to_string_lossy()
            .into_owned(),
        maximum_bytes: maximum_spill_bytes,
        maximum_queued_bytes: 64 * 1024 * 1024,
        maximum_inflight_bytes: 256 * 1024 * 1024,
        part_threshold_bytes: 64 * 1024,
    };
    let store = SpillStore::new(&spill_config).unwrap();
    let (sender, receiver) = mpsc::channel();
    let collector = Collector::with_spill(
        Configuration {
            // The hosted worker's delivery bounds.
            delivery: Limits {
                maximum_records: 64,
                maximum_bytes: 256 * 1024 * 1024,
                maximum_record_bytes: 1024 * 1024,
            },
            maximum_pending_records: 64,
            maximum_pending_bytes: 256 * 1024 * 1024,
            maximum_request_bytes: 128 * 1024 * 1024,
            maximum_response_bytes: window,
            ttl_seconds: 600,
            settlement_required: false,
            relay_metadata: false,
            truncate_request: false,
            asynchronous_delivery: true,
            capture_hidden_reasoning: true,
            capture_logprobs: false,
            spill: Some(spill_config),
        },
        SplitSink {
            store: store.clone(),
            delivered: sender,
            reassemble,
        },
        Some(store),
    )
    .unwrap();
    (Arc::new(collector), receiver)
}

fn begin(collector: &Collector, id: &str, context: serde_json::Value) -> bool {
    collector.begin(Request {
        request_id: id.into(),
        scope: Scope {
            organization_id: "org".into(),
            identity_id: "identity".into(),
            application_id: "alias".into(),
        },
        protocol: Protocol::ChatCompletions,
        model_id: Some("model".into()),
        context: Arc::new(context),
    })
}

fn sse_frame(index: usize, frames: usize, padding: &str) -> Bytes {
    Bytes::from(if index == frames {
        "data: [DONE]\n\n".to_owned()
    } else {
        format!(
            "data: {}\n\n",
            json!({"id": "c", "choices": [{"delta": {"content": format!("{index}{padding}")}}]})
        )
    })
}

/// An SSE body produced chunk by chunk, so the test never holds it whole.
fn sse_body(frames: usize, frame_bytes: usize) -> Body {
    let padding = "p".repeat(frame_bytes);
    let stream =
        futures_util::stream::iter((0..=frames).map(move |index| {
            Ok::<_, std::convert::Infallible>(sse_frame(index, frames, &padding))
        }));
    Body::from_stream(stream)
}

/// The same body arriving like a network transfer: a pause every 64 KB.
fn paced_sse_body(frames: usize, frame_bytes: usize) -> Body {
    let padding = Arc::new("p".repeat(frame_bytes));
    let stream = futures_util::stream::unfold(0usize, move |index| {
        let padding = padding.clone();
        async move {
            if index > frames {
                return None;
            }
            if index % 8 == 0 {
                tokio::time::sleep(Duration::from_millis(1)).await;
            }
            Some((
                Ok::<_, std::convert::Infallible>(sse_frame(index, frames, &padding)),
                index + 1,
            ))
        }
    });
    Body::from_stream(stream)
}

fn sse_response(frames: usize, frame_bytes: usize) -> Response {
    sse_response_from(sse_body(frames, frame_bytes))
}

fn sse_response_from(body: Body) -> Response {
    let mut response = Response::new(body);
    response.headers_mut().insert(
        "content-type",
        axum::http::HeaderValue::from_static("text/event-stream"),
    );
    response
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_stream_far_past_the_window_is_kept_whole_on_disk() {
    let (collector, delivered) = spill_collector("tap-whole", 64 * 1024, 4 << 30, true);
    assert!(begin(
        &collector,
        "request",
        json!({"schema_version": 1, "request": {"messages": [{"role": "user", "content": "go"}]}})
    ));
    let frames = 1500;
    let forwarded = capture_response(
        Some(collector.clone()),
        "request",
        sse_response(frames, 8000),
    )
    .into_body()
    .collect()
    .await
    .unwrap()
    .to_bytes();
    let expected = sse_body(frames, 8000).collect().await.unwrap().to_bytes();
    // The caller's bytes are exactly the provider's.
    assert_eq!(forwarded, expected);
    assert!(collector.close_until(Instant::now() + Duration::from_secs(30)));
    let rows: Vec<Delivered> = delivered.try_iter().collect();
    assert_eq!(rows.len(), 1);
    let rebuilt = rows[0].reassembled.as_ref().unwrap();
    assert_eq!(rebuilt["response"]["kind"], "sse");
    assert_eq!(rebuilt["response"]["truncated"], false);
    let stored = rebuilt["response"]["frames"].as_array().unwrap();
    assert_eq!(stored.len(), frames + 1);
    assert_eq!(stored[frames], "[DONE]");
    assert!(stored[1499]["choices"][0]["delta"]["content"]
        .as_str()
        .unwrap()
        .starts_with("1499"));
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_large_request_is_split_not_trimmed() {
    let (collector, delivered) = spill_collector("tap-request", 64 * 1024, 1 << 30, true);
    let image = "I".repeat(20 * 1024 * 1024);
    let context = json!({"schema_version": 1, "request": {"messages": [
        {"role": "user", "content": [{"type": "image_url", "image_url": {"url": image}}]}
    ]}});
    assert!(begin(&collector, "request", context.clone()));
    let mut response = Response::new(Body::from(r#"{"id":"c","choices":[]}"#));
    response.headers_mut().insert(
        "content-type",
        axum::http::HeaderValue::from_static("application/json"),
    );
    capture_response(Some(collector.clone()), "request", response)
        .into_body()
        .collect()
        .await
        .unwrap();
    assert!(collector.close_until(Instant::now() + Duration::from_secs(30)));
    let rows: Vec<Delivered> = delivered.try_iter().collect();
    assert_eq!(rows.len(), 1);
    let rebuilt = rows[0].reassembled.as_ref().unwrap();
    assert_eq!(rebuilt["request"]["context"], context);
    assert_eq!(rebuilt["response"]["body"]["id"], "c");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_discarded_response_stops_spilling() {
    let (collector, delivered) = spill_collector("tap-discard", 64 * 1024, 4 << 30, false);
    assert!(begin(
        &collector,
        "request",
        json!({"schema_version": 1, "request": {"messages": []}})
    ));
    // A bring-your-own-key lane: the settle keeps the prompt only.
    let tapped = capture_response(Some(collector.clone()), "request", sse_response(400, 8000));
    collector.settle("request", true, false);
    tapped.into_body().collect().await.unwrap();
    assert!(collector.close_until(Instant::now() + Duration::from_secs(30)));
    let rows: Vec<Delivered> = delivered.try_iter().collect();
    assert_eq!(rows.len(), 1);
    let record = rows[0]
        .record
        .as_ref()
        .expect("a prompt-only record is whole");
    assert!(record.response.is_none());
}

/// Peak resident memory of this process, sampled with `ps` (test-only).
fn resident_bytes() -> u64 {
    let output = std::process::Command::new("ps")
        .args(["-o", "rss=", "-p", &std::process::id().to_string()])
        .output()
        .unwrap();
    String::from_utf8_lossy(&output.stdout)
        .trim()
        .parse::<u64>()
        .unwrap_or(0)
        * 1024
}

/// Twenty concurrent 200 MB streams plus 64 MiB requests through one collector.
/// Run with `cargo test --no-default-features --release -- --ignored spill_memory`.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[ignore = "moves about 4 GB through the disk; run by hand"]
async fn spill_memory_stays_bounded_under_twenty_large_streams() {
    let (collector, delivered) = spill_collector("tap-memory", 3_670_016, 16 << 30, false);
    let baseline = resident_bytes();
    let peak = Arc::new(std::sync::atomic::AtomicU64::new(baseline));
    let sampling = Arc::new(std::sync::atomic::AtomicBool::new(true));
    let sampler = {
        let (peak, sampling) = (peak.clone(), sampling.clone());
        std::thread::spawn(move || {
            while sampling.load(Ordering::Acquire) {
                peak.fetch_max(resident_bytes(), Ordering::AcqRel);
                std::thread::sleep(Duration::from_millis(50));
            }
        })
    };
    let started = Instant::now();
    let mut tasks = Vec::new();
    for index in 0..20 {
        let id = format!("stream-{index}");
        assert!(begin(
            &collector,
            &id,
            json!({"schema_version": 1, "request": {"messages": [{"role": "user", "content": "go"}]}})
        ));
        let owner = collector.clone();
        tasks.push(tokio::spawn(async move {
            // 25,600 frames of ~8 KB: about 200 MB per stream. The caller
            // drains it without keeping it, as a real client socket does.
            let mut body = capture_response(
                Some(owner),
                &id,
                sse_response_from(paced_sse_body(25_600, 8000)),
            )
            .into_body()
            .into_data_stream();
            let mut forwarded = 0usize;
            while let Some(chunk) = futures_util::StreamExt::next(&mut body).await {
                forwarded += chunk.unwrap().len();
            }
            forwarded
        }));
    }
    let mut forwarded = 0usize;
    for task in tasks {
        forwarded += task.await.unwrap();
    }
    let streams_peak = peak.load(Ordering::Acquire);
    // Requests at the serving limit, admitted as the pending budget allows.
    let mut admitted_requests = 0;
    for index in 0..20 {
        let mut context = json!({"schema_version": 1, "request": {"messages": [{"role": "user", "content": ""}]}});
        context["request"]["messages"][0]["content"] =
            serde_json::Value::String("R".repeat(64 * 1024 * 1024));
        let id = format!("request-{index}");
        if begin(&collector, &id, context) {
            admitted_requests += 1;
            collector.finish_unexposed(&id);
        }
    }
    assert!(collector.close_until(Instant::now() + Duration::from_secs(600)));
    sampling.store(false, Ordering::Release);
    sampler.join().unwrap();
    let rows: Vec<Delivered> = delivered.try_iter().collect();
    let streamed: Vec<_> = rows
        .iter()
        .filter(|row| {
            row.skeleton
                .as_ref()
                .is_some_and(|skeleton| skeleton["response"]["kind"] == "sse")
        })
        .collect();
    let peak = peak.load(Ordering::Acquire);
    let whole = streamed
        .iter()
        .filter(|row| {
            row.skeleton.as_ref().unwrap()["response"]["truncated"] == false
                && row
                    .parts
                    .iter()
                    .any(|(kind, _, lines)| *kind == PartKind::Frames && *lines == Some(25_601))
        })
        .count();
    let large_requests = rows
        .iter()
        .filter(|row| {
            row.parts
                .iter()
                .any(|(kind, size, _)| *kind == PartKind::Json && *size > 64 * 1024 * 1024)
        })
        .count();
    eprintln!(
        "spill memory: forwarded {} MB in {:?}; baseline RSS {} MB, peak during streams {} MB \
         (delta {} MB), overall peak {} MB; {} of 20 streams delivered, {} whole; \
         {} of 20 large requests admitted, {} delivered whole; {} records delivered",
        forwarded >> 20,
        started.elapsed(),
        baseline >> 20,
        streams_peak >> 20,
        streams_peak.saturating_sub(baseline) >> 20,
        peak >> 20,
        streamed.len(),
        whole,
        admitted_requests,
        large_requests,
        rows.len(),
    );
    assert_eq!((streamed.len(), whole), (20, 20));
    // Back-to-back admissions beyond the background writers' budget are
    // skipped and counted, never trimmed; every admitted one arrives whole.
    assert_eq!(large_requests, admitted_requests);
    assert!(
        admitted_requests >= 10,
        "{admitted_requests} large requests admitted"
    );
    // During the streams the capture holds at most its windows and spill queue.
    assert!(
        streams_peak.saturating_sub(baseline) < 512 * 1024 * 1024,
        "resident memory grew by {} MB during the streams",
        streams_peak.saturating_sub(baseline) >> 20
    );
}

/// Paced like the stress preview's slow callers: about 3 MB/s per stream.
fn slow_sse_body(frames: usize, frame_bytes: usize) -> Body {
    let padding = Arc::new("p".repeat(frame_bytes));
    let stream = futures_util::stream::unfold(0usize, move |index| {
        let padding = padding.clone();
        async move {
            if index > frames {
                return None;
            }
            if index % 8 == 0 {
                tokio::time::sleep(Duration::from_millis(20)).await;
            }
            Some((
                Ok::<_, std::convert::Infallible>(sse_frame(index, frames, &padding)),
                index + 1,
            ))
        }
    });
    Body::from_stream(stream)
}

/// The bounds the platform derives from a 256 MiB capture memory budget:
/// B/16 pending (and B/16 windows), B/8 delivery, spill queue and request writers.
fn budgeted_collector(name: &str) -> (Arc<Collector>, mpsc::Receiver<Delivered>) {
    let spill_config = spill::Configuration {
        directory: spill::tests::temporary_directory(name)
            .to_string_lossy()
            .into_owned(),
        maximum_bytes: 16 << 30,
        maximum_queued_bytes: 32 << 20,
        maximum_inflight_bytes: 32 << 20,
        part_threshold_bytes: 1 << 20,
    };
    let store = SpillStore::new(&spill_config).unwrap();
    let (sender, receiver) = mpsc::channel();
    let collector = Collector::with_spill(
        Configuration {
            delivery: Limits {
                maximum_records: 2000,
                maximum_bytes: 32 << 20,
                maximum_record_bytes: 1 << 20,
            },
            maximum_pending_records: 2048,
            maximum_pending_bytes: 16 << 20,
            maximum_request_bytes: 128 << 20,
            maximum_response_bytes: 1 << 20,
            ttl_seconds: 600,
            settlement_required: false,
            relay_metadata: false,
            truncate_request: false,
            asynchronous_delivery: true,
            capture_hidden_reasoning: true,
            capture_logprobs: false,
            spill: Some(spill_config),
        },
        SplitSink {
            store: store.clone(),
            delivered: sender,
            reassemble: false,
        },
        Some(store),
    )
    .unwrap();
    (Arc::new(collector), receiver)
}

/// The stress preview's failing shape: 24 concurrent 64 MB streams at about
/// 3 MB/s each. Run once per mode, each in its own process, and compare:
/// `EXP_CAPTURE_STRESS=control` (no capture) and `EXP_CAPTURE_STRESS=capture`.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[ignore = "about a minute and 1.5 GB through the disk; run by hand"]
async fn stress_shape_capture_memory_stays_under_budget() {
    let capture = std::env::var("EXP_CAPTURE_STRESS").as_deref() != Ok("control");
    let (collector, delivered) = budgeted_collector("tap-stress");
    let baseline = resident_bytes();
    let peak = Arc::new(std::sync::atomic::AtomicU64::new(baseline));
    let sampling = Arc::new(std::sync::atomic::AtomicBool::new(true));
    let sampler = {
        let (peak, sampling) = (peak.clone(), sampling.clone());
        std::thread::spawn(move || {
            while sampling.load(Ordering::Acquire) {
                peak.fetch_max(resident_bytes(), Ordering::AcqRel);
                std::thread::sleep(Duration::from_millis(50));
            }
        })
    };
    let started = Instant::now();
    let streams = 24;
    // 8,192 frames of ~8 KB: about 64 MB per stream.
    let frames = 8192;
    let mut tasks = Vec::new();
    for index in 0..streams {
        let id = format!("stream-{index}");
        if capture {
            assert!(begin(
                &collector,
                &id,
                json!({"schema_version": 1, "request": {"messages": [{"role": "user", "content": "go"}]}})
            ));
        }
        let owner = capture.then(|| collector.clone());
        tasks.push(tokio::spawn(async move {
            let mut body =
                capture_response(owner, &id, sse_response_from(slow_sse_body(frames, 8000)))
                    .into_body()
                    .into_data_stream();
            let mut hasher = <sha2::Sha256 as sha2::Digest>::new();
            let mut forwarded = 0usize;
            while let Some(chunk) = futures_util::StreamExt::next(&mut body).await {
                let chunk = chunk.unwrap();
                sha2::Digest::update(&mut hasher, &chunk);
                forwarded += chunk.len();
            }
            (forwarded, sha2::Digest::finalize(hasher).to_vec())
        }));
    }
    let mut digests = std::collections::HashSet::new();
    let mut forwarded = 0usize;
    for task in tasks {
        let (bytes, digest) = task.await.unwrap();
        forwarded += bytes;
        digests.insert(digest);
    }
    assert!(collector.close_until(Instant::now() + Duration::from_secs(600)));
    sampling.store(false, Ordering::Release);
    sampler.join().unwrap();
    let rows: Vec<Delivered> = delivered.try_iter().collect();
    let whole = rows
        .iter()
        .filter(|row| {
            row.skeleton.as_ref().is_some_and(|skeleton| {
                skeleton["response"]["truncated"] == false
                    && row.parts.iter().any(|(kind, _, lines)| {
                        *kind == PartKind::Frames && *lines == Some(frames + 1)
                    })
            })
        })
        .count();
    let peak = peak.load(Ordering::Acquire);
    eprintln!(
        "capture stress ({}): forwarded {} MB in {:?}; baseline RSS {} MB, peak {} MB, \
         delta {} MB; {} of {} captured whole; counts {:?}",
        if capture { "capture" } else { "control" },
        forwarded >> 20,
        started.elapsed(),
        baseline >> 20,
        peak >> 20,
        peak.saturating_sub(baseline) >> 20,
        whole,
        streams,
        collector.counts(),
    );
    // Every caller received identical bytes.
    assert_eq!(digests.len(), 1);
    if capture {
        assert_eq!(whole, streams);
        // Capture memory is bounded by its budget whatever the stream sizes.
        assert!(
            peak.saturating_sub(baseline) < 256 << 20,
            "resident memory grew by {} MB",
            peak.saturating_sub(baseline) >> 20
        );
    }
}
