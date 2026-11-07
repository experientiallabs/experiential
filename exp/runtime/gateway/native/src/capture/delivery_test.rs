use super::*;
use std::sync::atomic::AtomicBool;

#[test]
fn delayed_capture_does_not_block_ready_neighbor() {
    struct DeferredSink {
        release: Arc<std::sync::atomic::AtomicBool>,
        attempted: mpsc::Sender<String>,
    }
    impl Sink for DeferredSink {
        fn independent_retries(&self) -> bool {
            true
        }
        type Prepared = String;
        fn preparation_bytes(_: usize) -> usize {
            512
        }
        fn prepare(&self, record: &Record, _: usize) -> Result<String, ()> {
            Ok(record.request.request_id.clone())
        }
        fn write(&mut self, value: &String) -> Result<(), ()> {
            self.attempted.send(value.clone()).unwrap();
            if value == "waiting" && !self.release.load(Ordering::Acquire) {
                Err(())
            } else {
                Ok(())
            }
        }
    }
    let release = Arc::new(std::sync::atomic::AtomicBool::new(false));
    let (attempted, observed) = mpsc::channel();
    let delivery = Delivery::new(
        limits(),
        DeferredSink {
            release: release.clone(),
            attempted,
        },
    )
    .unwrap();
    assert!(delivery.submit(record("waiting")));
    assert_eq!(
        observed.recv_timeout(Duration::from_secs(1)).unwrap(),
        "waiting"
    );
    assert!(delivery.submit(record("ready")));
    let deadline = Instant::now() + Duration::from_millis(500);
    let mut progressed = false;
    while Instant::now() < deadline {
        if observed
            .recv_timeout(Duration::from_millis(30))
            .is_ok_and(|id| id == "ready")
        {
            progressed = true;
            break;
        }
    }
    release.store(true, Ordering::Release);
    assert!(delivery.close_until(Instant::now() + Duration::from_secs(3)));
    assert!(
        progressed,
        "a retry blocked an independently writable capture"
    );
    assert_eq!(delivery.counts()[2], 2);
}

#[test]
fn deferred_retries_keep_batching_and_drain_without_dropping() {
    struct BatchRetry {
        attempts: mpsc::Sender<usize>,
        recovered: Arc<std::sync::atomic::AtomicBool>,
    }
    impl Sink for BatchRetry {
        type Prepared = String;
        fn preparation_bytes(_: usize) -> usize {
            512
        }
        fn prepare(&self, record: &Record, _: usize) -> Result<String, ()> {
            Ok(record.request.request_id.clone())
        }
        fn write(&mut self, _: &String) -> Result<(), ()> {
            unreachable!()
        }
        fn independent_retries(&self) -> bool {
            true
        }
        fn batch_records(&self) -> usize {
            4
        }
        fn batch_bytes(&self) -> usize {
            1024
        }
        fn batch_delay(&self) -> Duration {
            Duration::from_millis(100)
        }
        fn prepared_bytes(&self, value: &String) -> usize {
            value.len()
        }
        fn write_batch(&mut self, values: &[&String]) -> Vec<bool> {
            self.attempts.send(values.len()).unwrap();
            vec![self.recovered.load(Ordering::Acquire); values.len()]
        }
    }
    let (attempts, observed) = mpsc::channel();
    let recovered = Arc::new(std::sync::atomic::AtomicBool::new(false));
    let delivery = Delivery::new(
        Limits {
            maximum_records: 4,
            maximum_bytes: limits().maximum_bytes * 4,
            ..limits()
        },
        BatchRetry {
            attempts,
            recovered: recovered.clone(),
        },
    )
    .unwrap();
    for id in ["a", "b", "c", "d"] {
        assert!(delivery.submit(record(id)));
    }
    assert_eq!(observed.recv_timeout(Duration::from_secs(2)).unwrap(), 4);
    assert_eq!(observed.recv_timeout(Duration::from_secs(2)).unwrap(), 4);
    assert!(!delivery.close_until(Instant::now()));
    recovered.store(true, Ordering::Release);
    assert!(delivery.close_until(Instant::now() + Duration::from_secs(2)));
    assert_eq!(delivery.counts()[0..3], [0, 0, 4]);
    assert_eq!(delivery.counts()[4], 0);
}

#[test]
fn a_new_retry_does_not_wait_for_an_older_records_long_backoff() {
    struct BackoffSink {
        recovered: Arc<std::sync::atomic::AtomicBool>,
        attempts: mpsc::Sender<String>,
        new_attempts: usize,
    }
    impl Sink for BackoffSink {
        type Prepared = String;
        fn preparation_bytes(_: usize) -> usize {
            512
        }
        fn independent_retries(&self) -> bool {
            true
        }
        fn prepare(&self, record: &Record, _: usize) -> Result<String, ()> {
            Ok(record.request.request_id.clone())
        }
        fn write(&mut self, id: &String) -> Result<(), ()> {
            self.attempts.send(id.clone()).unwrap();
            if id == "old" && !self.recovered.load(Ordering::Acquire) {
                return Err(());
            }
            if id == "new" {
                self.new_attempts += 1;
                if self.new_attempts == 1 {
                    return Err(());
                }
            }
            Ok(())
        }
    }
    let recovered = Arc::new(std::sync::atomic::AtomicBool::new(false));
    let (attempts, observed) = mpsc::channel();
    let delivery = Delivery::new(
        limits(),
        BackoffSink {
            recovered: recovered.clone(),
            attempts,
            new_attempts: 0,
        },
    )
    .unwrap();
    assert!(delivery.submit(record("old")));
    for _ in 0..7 {
        assert_eq!(
            observed.recv_timeout(Duration::from_secs(2)).unwrap(),
            "old"
        );
    }
    assert!(delivery.submit(record("new")));
    assert_eq!(
        observed.recv_timeout(Duration::from_secs(1)).unwrap(),
        "new"
    );
    let next = observed.recv_timeout(Duration::from_millis(500));
    recovered.store(true, Ordering::Release);
    assert!(delivery.close_until(Instant::now() + Duration::from_secs(2)));
    assert_eq!(next.unwrap(), "new");
    assert_eq!(delivery.counts()[2], 2);
}

struct PausedSink {
    entered: mpsc::Sender<String>,
    resume: mpsc::Receiver<()>,
    fail: bool,
}

impl Sink for PausedSink {
    type Prepared = String;

    fn preparation_bytes(_maximum_record_bytes: usize) -> usize {
        512
    }

    fn prepare(&self, record: &Record, _maximum_bytes: usize) -> Result<Self::Prepared, ()> {
        Ok(record.request.request_id.clone())
    }

    fn write(&mut self, record: &Self::Prepared) -> Result<(), ()> {
        self.entered.send(record.clone()).map_err(|_| ())?;
        self.resume
            .recv_timeout(Duration::from_secs(5))
            .map_err(|_| ())?;
        if self.fail {
            self.fail = false;
            Err(())
        } else {
            Ok(())
        }
    }
}

fn paused(limits: Limits, fail: bool) -> (Delivery, mpsc::Receiver<String>, mpsc::Sender<()>) {
    let (entered, observer) = mpsc::channel();
    let (resume, paused) = mpsc::channel();
    let delivery = Delivery::new(
        limits,
        PausedSink {
            entered,
            resume: paused,
            fail,
        },
    )
    .unwrap();
    (delivery, observer, resume)
}

fn limits() -> Limits {
    Limits {
        maximum_records: 2,
        maximum_bytes: record("12345678").heap_bytes() * 2 + 512,
        maximum_record_bytes: 1024,
    }
}

fn record(id: &str) -> Record {
    serde_json::from_value(serde_json::json!({
        "schema_version":1,
        "request": {"request_id":id,
            "scope":{"organization_id":"org","identity_id":"identity","application_id":"app"},
            "protocol":"chat_completions","model_id":null,
            "context":{"schema_version":1,"request":{}}},
        "response":null,"provider_reasoning":null,"provider_reasoning_source_json":null,
        "provider_tool_calls_json":null,"deployment_id":null,"captured_at":1.0,
        "metrics":null,"gemini_thought_parts":[],"gemini_thought_parts_source_json":null,
    }))
    .unwrap()
}

#[test]
fn preparation_releases_decoded_trees_but_keeps_admission_charged_until_ack() {
    let (delivery, entered, resume) = paused(limits(), false);
    let value = record("tree");
    let context = Arc::downgrade(&value.request.context);
    assert!(delivery.submit(value));
    entered.recv_timeout(Duration::from_secs(1)).unwrap();
    assert!(context.upgrade().is_none());
    assert_eq!(delivery.counts()[0], 1);
    assert!(delivery.counts()[1] > 0);
    assert!(!delivery.close_until(Instant::now()));
    resume.send(()).unwrap();
    assert!(delivery.close_until(Instant::now() + Duration::from_secs(1)));
    assert_eq!(delivery.counts(), [0, 0, 1, 0, 0]);
}

#[test]
fn preparation_failure_retains_input_for_retry_without_a_false_ack() {
    struct RecoveringPreparation {
        recover: Arc<std::sync::atomic::AtomicBool>,
        attempted: mpsc::Sender<()>,
    }
    impl Sink for RecoveringPreparation {
        type Prepared = String;
        fn preparation_bytes(_: usize) -> usize {
            512
        }
        fn prepare(&self, record: &Record, _: usize) -> Result<String, ()> {
            let _ = self.attempted.send(());
            if self.recover.load(Ordering::Acquire) {
                Ok(record.request.request_id.clone())
            } else {
                Err(())
            }
        }
        fn write(&mut self, value: &String) -> Result<(), ()> {
            assert_eq!(value, "retry");
            Ok(())
        }
    }
    let recover = Arc::new(std::sync::atomic::AtomicBool::new(false));
    let (attempted, observed) = mpsc::channel();
    let delivery = Delivery::new(
        limits(),
        RecoveringPreparation {
            recover: recover.clone(),
            attempted,
        },
    )
    .unwrap();
    let value = record("retry");
    let context = Arc::downgrade(&value.request.context);
    assert!(delivery.submit(value));
    observed.recv_timeout(Duration::from_secs(1)).unwrap();
    assert!(context.upgrade().is_some());
    assert!(!delivery.close_until(Instant::now()));
    assert_eq!(delivery.counts()[0], 1);
    assert_eq!(delivery.counts()[2], 0);
    recover.store(true, Ordering::Release);
    assert!(delivery.close_until(Instant::now() + Duration::from_secs(1)));
    assert!(context.upgrade().is_none());
    assert_eq!(delivery.counts()[2], 1);
    assert_eq!(delivery.counts()[4], 0);
}

#[test]
fn failed_preparation_owns_the_single_workspace_until_recovery() {
    struct Preparing {
        recover: Arc<std::sync::atomic::AtomicBool>,
        first: std::sync::atomic::AtomicBool,
        resume: Mutex<mpsc::Receiver<()>>,
        attempted: mpsc::Sender<String>,
        persisted: mpsc::Sender<String>,
    }
    impl Sink for Preparing {
        type Prepared = String;
        fn preparation_bytes(_: usize) -> usize {
            512
        }
        fn batch_records(&self) -> usize {
            2
        }
        fn batch_bytes(&self) -> usize {
            1024
        }
        fn prepared_bytes(&self, value: &String) -> usize {
            value.len()
        }
        fn prepare(&self, record: &Record, _: usize) -> Result<String, ()> {
            let id = &record.request.request_id;
            self.attempted.send(id.clone()).unwrap();
            if self.first.swap(false, Ordering::AcqRel) {
                self.resume
                    .lock()
                    .unwrap()
                    .recv_timeout(Duration::from_secs(2))
                    .unwrap();
            }
            if id == "failed" && !self.recover.load(Ordering::Acquire) {
                Err(())
            } else {
                Ok(id.clone())
            }
        }
        fn write(&mut self, value: &String) -> Result<(), ()> {
            self.persisted.send(value.clone()).map_err(|_| ())
        }
    }
    let recover = Arc::new(std::sync::atomic::AtomicBool::new(false));
    let (resume, paused) = mpsc::channel();
    let (attempted, observed) = mpsc::channel();
    let (persisted, saved) = mpsc::channel();
    let delivery = Delivery::new(
        limits(),
        Preparing {
            recover: recover.clone(),
            first: std::sync::atomic::AtomicBool::new(true),
            resume: Mutex::new(paused),
            attempted,
            persisted,
        },
    )
    .unwrap();
    assert!(delivery.submit(record("failed")));
    assert_eq!(
        observed.recv_timeout(Duration::from_secs(1)).unwrap(),
        "failed"
    );
    assert!(delivery.submit(record("waiting")));
    resume.send(()).unwrap();
    let until = Instant::now() + Duration::from_millis(80);
    let mut attempts = Vec::new();
    while let Ok(id) = observed.recv_timeout(until.saturating_duration_since(Instant::now())) {
        attempts.push(id);
    }
    recover.store(true, Ordering::Release);
    assert!(delivery.close_until(Instant::now() + Duration::from_secs(2)));
    assert!(!attempts.is_empty());
    assert!(attempts.iter().all(|id| id == "failed"), "{attempts:?}");
    let mut saved: Vec<_> = saved.try_iter().collect();
    saved.sort();
    assert_eq!(saved, ["failed", "waiting"]);
    assert_eq!(delivery.counts()[0], 0);
    assert_eq!(delivery.counts()[2], 2);
    assert_eq!(delivery.counts()[4], 0);
}

#[test]
fn saturated_destination_waits_without_losing_records_or_exceeding_queued_budget() {
    let (delivery, entered, resume) = paused(limits(), false);
    let delivery = Arc::new(delivery);
    assert!(delivery.submit(record("12345678")));
    assert_eq!(
        entered.recv_timeout(Duration::from_secs(1)).unwrap(),
        "12345678"
    );
    assert!(delivery.submit(record("abcdefgh")));
    let (returned, result) = mpsc::channel();
    let producer = delivery.clone();
    let waiting = std::thread::spawn(move || returned.send(producer.submit(record("x"))).unwrap());
    assert!(result.recv_timeout(Duration::from_millis(30)).is_err());
    assert_eq!(
        delivery.counts(),
        [2, limits().maximum_bytes as u64, 0, 0, 0]
    );
    resume.send(()).unwrap();
    assert!(result.recv_timeout(Duration::from_secs(1)).unwrap());
    waiting.join().unwrap();
    assert_eq!(
        entered.recv_timeout(Duration::from_secs(1)).unwrap(),
        "abcdefgh"
    );
    resume.send(()).unwrap();
    assert_eq!(entered.recv_timeout(Duration::from_secs(1)).unwrap(), "x");
    resume.send(()).unwrap();
    assert!(delivery.close_until(Instant::now() + Duration::from_secs(1)));
    assert_eq!(delivery.counts(), [0, 0, 3, 0, 0]);
    assert!(!delivery.submit(record("closed")));
    assert_eq!(delivery.counts(), [0, 0, 3, 0, 1]);
}

#[test]
fn record_count_is_bounded_even_for_tiny_records() {
    let (delivery, entered, resume) = paused(limits(), false);
    let delivery = Arc::new(delivery);
    assert!(delivery.submit(record("a")));
    entered.recv_timeout(Duration::from_secs(1)).unwrap();
    assert!(delivery.submit(record("b")));
    let (returned, result) = mpsc::channel();
    let producer = delivery.clone();
    let waiting = std::thread::spawn(move || returned.send(producer.submit(record("c"))).unwrap());
    assert!(result.recv_timeout(Duration::from_millis(30)).is_err());
    assert_eq!(
        delivery.counts(),
        [2, (record("a").heap_bytes() * 2 + 512) as u64, 0, 0, 0]
    );
    resume.send(()).unwrap();
    assert!(result.recv_timeout(Duration::from_secs(1)).unwrap());
    waiting.join().unwrap();
    entered.recv_timeout(Duration::from_secs(1)).unwrap();
    resume.send(()).unwrap();
    entered.recv_timeout(Duration::from_secs(1)).unwrap();
    resume.send(()).unwrap();
    assert!(delivery.close_until(Instant::now() + Duration::from_secs(1)));
}

#[test]
fn allocated_capacity_not_just_json_length_is_charged() {
    let (delivery, _, _) = paused(limits(), false);
    let mut large_allocation = record("x");
    let mut text = String::with_capacity(limits().maximum_bytes + 1);
    text.push('x');
    large_allocation.provider_reasoning = Some(text);
    assert!(!delivery.submit(large_allocation));
    assert_eq!(delivery.counts(), [0, 0, 0, 0, 1]);
    assert!(delivery.close_until(Instant::now() + Duration::from_secs(1)));
}

#[test]
fn failed_destination_retains_budget_and_retries_the_same_record_after_close_timeout() {
    let (delivery, entered, resume) = paused(limits(), true);
    assert!(delivery.submit(record("private")));
    entered.recv_timeout(Duration::from_secs(1)).unwrap();
    resume.send(()).unwrap();
    assert_eq!(
        entered.recv_timeout(Duration::from_secs(1)).unwrap(),
        "private"
    );
    assert!(!delivery.close_until(Instant::now()));
    assert_eq!(
        delivery.counts(),
        [1, (record("private").heap_bytes() + 512) as u64, 0, 1, 0]
    );
    resume.send(()).unwrap();
    assert!(delivery.close_until(Instant::now() + Duration::from_secs(1)));
    assert_eq!(delivery.counts(), [0, 0, 1, 1, 0]);
}

#[test]
fn committed_write_and_cleanup_failures_have_separate_counters() {
    struct CleanupFailure;
    impl Sink for CleanupFailure {
        type Prepared = ();
        fn preparation_bytes(_: usize) -> usize {
            0
        }
        fn prepare(&self, _: &Record, _: usize) -> Result<(), ()> {
            Ok(())
        }
        fn write(&mut self, _: &()) -> Result<(), ()> {
            Ok(())
        }
        fn take_maintenance_failures(&mut self) -> u64 {
            1
        }
        fn maintain(&mut self) -> Result<(), ()> {
            Err(())
        }
    }
    let delivery = Delivery::new(limits(), CleanupFailure).unwrap();
    assert!(delivery.submit(record("saved")));
    let until = Instant::now() + Duration::from_secs(5);
    while delivery.maintenance_failures() < 2 && Instant::now() < until {
        std::thread::sleep(Duration::from_millis(5));
    }
    assert!(delivery.close_until(until));
    assert_eq!(delivery.counts(), [0, 0, 1, 0, 0]);
    assert!(delivery.maintenance_failures() >= 2);
}

#[test]
fn shutdown_timeout_reports_incomplete_drain_without_purging_accepted_records() {
    let (delivery, entered, resume) = paused(limits(), false);
    let delivery = Arc::new(delivery);
    assert!(delivery.submit(record("active")));
    entered.recv_timeout(Duration::from_secs(1)).unwrap();
    assert!(delivery.submit(record("queued")));
    let (returned, result) = mpsc::channel();
    let producer = delivery.clone();
    let waiting =
        std::thread::spawn(move || returned.send(producer.submit(record("waiting"))).unwrap());
    assert!(result.recv_timeout(Duration::from_millis(30)).is_err());
    assert!(!delivery.close_until(Instant::now()));
    assert!(!delivery.submit(record("late")));
    resume.send(()).unwrap();
    assert!(result.recv_timeout(Duration::from_secs(1)).unwrap());
    waiting.join().unwrap();
    assert_eq!(
        entered.recv_timeout(Duration::from_secs(1)).unwrap(),
        "queued"
    );
    resume.send(()).unwrap();
    assert_eq!(
        entered.recv_timeout(Duration::from_secs(1)).unwrap(),
        "waiting"
    );
    resume.send(()).unwrap();
    assert!(delivery.close_until(Instant::now() + Duration::from_secs(1)));
    assert_eq!(delivery.counts(), [0, 0, 3, 0, 1]);
    assert!(entered.try_recv().is_err());
}

#[test]
fn queued_completion_does_not_wait_for_storage_and_retries_keep_ownership() {
    for fail in [false, true] {
        let (delivery, entered, resume) = paused(limits(), fail);
        let delivery = Arc::new(delivery);
        let (returned, result) = mpsc::channel();
        let producer = delivery.clone();
        let waiting = std::thread::spawn(move || {
            returned
                .send(producer.submit_record(record("ack"), None, None))
                .unwrap();
        });
        entered.recv_timeout(Duration::from_secs(1)).unwrap();
        assert!(result.recv_timeout(Duration::from_secs(1)).unwrap());
        assert_eq!(delivery.counts()[0], 1);
        assert!(!delivery.close_until(Instant::now()));
        resume.send(()).unwrap();
        if fail {
            assert_eq!(entered.recv_timeout(Duration::from_secs(1)).unwrap(), "ack");
            assert_eq!(delivery.counts()[0], 1);
            resume.send(()).unwrap();
        }
        waiting.join().unwrap();
        assert!(delivery.close_until(Instant::now() + Duration::from_secs(1)));
        assert_eq!(delivery.counts()[4], 0);
    }
}

#[test]
fn limits_reject_zero_unbounded_and_incoherent_configuration() {
    for bounds in [
        Limits {
            maximum_records: 0,
            ..limits()
        },
        Limits {
            maximum_records: 4097,
            ..limits()
        },
        Limits {
            maximum_bytes: 7,
            ..limits()
        },
        Limits {
            maximum_bytes: usize::MAX,
            ..limits()
        },
        Limits {
            maximum_record_bytes: 0,
            ..limits()
        },
        Limits {
            maximum_record_bytes: 17 * 1024 * 1024,
            maximum_bytes: 18 * 1024 * 1024,
            ..limits()
        },
    ] {
        assert!(bounds.validate().is_err());
    }
}

#[test]
fn microbatch_waits_for_neighbors_but_flushes_on_count_deadline_and_close() {
    struct BatchSink(mpsc::Sender<Vec<String>>, mpsc::Sender<()>);
    impl Sink for BatchSink {
        type Prepared = String;
        fn preparation_bytes(_: usize) -> usize {
            512
        }
        fn prepare(&self, record: &Record, _: usize) -> Result<String, ()> {
            self.1.send(()).unwrap();
            Ok(record.request.request_id.clone())
        }
        fn write(&mut self, _: &String) -> Result<(), ()> {
            unreachable!()
        }
        fn batch_records(&self) -> usize {
            2
        }
        fn batch_bytes(&self) -> usize {
            1024
        }
        fn batch_delay(&self) -> Duration {
            Duration::from_millis(200)
        }
        fn prepared_bytes(&self, value: &String) -> usize {
            value.len()
        }
        fn write_batch(&mut self, values: &[&String]) -> Vec<bool> {
            self.0
                .send(values.iter().map(|v| (*v).clone()).collect())
                .unwrap();
            vec![true; values.len()]
        }
    }
    let (written, observed) = mpsc::channel();
    let (prepared, ready) = mpsc::channel();
    let delivery = Delivery::new(limits(), BatchSink(written, prepared)).unwrap();
    assert!(delivery.submit(record("first")));
    ready.recv_timeout(Duration::from_secs(1)).unwrap();
    assert!(observed.recv_timeout(Duration::from_millis(20)).is_err());
    assert!(delivery.submit(record("second")));
    assert_eq!(
        observed.recv_timeout(Duration::from_secs(1)).unwrap(),
        ["first", "second"]
    );
    assert!(delivery.submit(record("deadline")));
    assert_eq!(
        observed.recv_timeout(Duration::from_secs(1)).unwrap(),
        ["deadline"]
    );
    assert!(delivery.submit(record("drain")));
    assert!(delivery.close_until(Instant::now() + Duration::from_secs(1)));
    assert_eq!(
        observed.recv_timeout(Duration::from_secs(1)).unwrap(),
        ["drain"]
    );
    assert_eq!(delivery.counts(), [0, 0, 4, 0, 0]);
}

fn record_with_probabilities(id: &str) -> Record {
    let mut value = record(id);
    let mut captured = crate::capture::logprobs::Captured::injected();
    captured.content.push(crate::logprobs::TokenLogprob {
        token: "a".into(),
        logprob: -1.0,
        bytes: None,
        top_logprobs: Vec::new(),
    });
    value.provider_logprobs = Some(captured);
    value
}

/// Prepares a record unless the policy refuses it; writes what it prepared.
struct ProbabilitySink {
    refuse: Box<dyn Fn(&Record, usize) -> bool + Send>,
    calls: Arc<std::sync::atomic::AtomicUsize>,
    written: mpsc::Sender<bool>,
}

impl Sink for ProbabilitySink {
    type Prepared = bool;
    fn preparation_bytes(_: usize) -> usize {
        512
    }
    fn prepare(&self, record: &Record, _: usize) -> Result<bool, ()> {
        let call = self.calls.fetch_add(1, Ordering::AcqRel);
        if (self.refuse)(record, call) {
            Err(())
        } else {
            Ok(record.provider_logprobs.is_some())
        }
    }
    fn write(&mut self, kept: &bool) -> Result<(), ()> {
        let _ = self.written.send(*kept);
        Ok(())
    }
}

fn deliver_probabilities(refuse: Box<dyn Fn(&Record, usize) -> bool + Send>) -> bool {
    let (written, observed) = mpsc::channel();
    let delivery = Delivery::new(
        limits(),
        ProbabilitySink {
            refuse,
            calls: Arc::new(std::sync::atomic::AtomicUsize::new(0)),
            written,
        },
    )
    .unwrap();
    assert!(delivery.submit(record_with_probabilities("probabilities")));
    let kept = observed.recv_timeout(Duration::from_secs(2)).unwrap();
    assert!(delivery.close_until(Instant::now() + Duration::from_secs(1)));
    kept
}

#[test]
fn a_sink_that_refuses_only_the_probabilities_gets_the_exchange_without_them() {
    // The sink's own limit (any payload shape) is exceeded only with the sidecar.
    assert!(!deliver_probabilities(Box::new(|record, _| {
        record.provider_logprobs.is_some()
    })));
}

#[test]
fn a_transient_preparation_failure_keeps_the_probabilities() {
    assert!(deliver_probabilities(Box::new(|_, call| call == 0)));
}

#[test]
fn exhausted_retained_budget_keeps_accepted_neighbor_until_recovery() {
    struct Expanded {
        attempts: mpsc::Sender<String>,
        resume: mpsc::Receiver<()>,
        recovered: Arc<AtomicBool>,
    }
    impl Sink for Expanded {
        type Prepared = String;
        fn preparation_bytes(_: usize) -> usize {
            512
        }
        fn prepare(&self, record: &Record, _: usize) -> Result<String, ()> {
            Ok(record.request.request_id.clone())
        }
        fn independent_retries(&self) -> bool {
            true
        }
        fn retained_prepared_bytes(&self, _: &String, _: usize) -> usize {
            // Model expansion that fits scratch but cannot be retained beside
            // both accepted records in the configured queue budget.
            record("waiting").heap_bytes() + 512
        }
        fn write(&mut self, value: &String) -> Result<(), ()> {
            self.attempts.send(value.clone()).unwrap();
            self.resume.recv_timeout(Duration::from_secs(3)).unwrap();
            if value == "waiting" && !self.recovered.load(Ordering::Acquire) {
                Err(())
            } else {
                Ok(())
            }
        }
    }
    let (attempts, observed) = mpsc::channel();
    let (resume, proceed) = mpsc::channel();
    let recovered = Arc::new(AtomicBool::new(false));
    let bound = limits().maximum_bytes;
    let delivery = Delivery::new(
        limits(),
        Expanded {
            attempts,
            resume: proceed,
            recovered: recovered.clone(),
        },
    )
    .unwrap();
    assert!(delivery.submit(record("waiting")));
    assert_eq!(
        observed.recv_timeout(Duration::from_secs(1)).unwrap(),
        "waiting"
    );
    assert!(delivery.submit(record("ready")));
    resume.send(()).unwrap();
    assert_eq!(
        observed.recv_timeout(Duration::from_secs(1)).unwrap(),
        "waiting"
    );
    assert_eq!(delivery.counts()[0], 2);
    assert!(delivery.counts()[1] <= bound as u64);
    assert_eq!(delivery.counts()[4], 0);
    recovered.store(true, Ordering::Release);
    resume.send(()).unwrap();
    assert_eq!(
        observed.recv_timeout(Duration::from_secs(1)).unwrap(),
        "ready"
    );
    resume.send(()).unwrap();
    assert!(delivery.close_until(Instant::now() + Duration::from_secs(1)));
    assert_eq!(delivery.counts()[0..3], [0, 0, 2]);
    assert_eq!(delivery.counts()[4], 0);
}

#[test]
fn destroying_failed_delivery_stops_worker_and_accounts_for_undelivered_records() {
    struct Outage {
        attempts: mpsc::Sender<()>,
        stopped: mpsc::Sender<()>,
    }
    impl Drop for Outage {
        fn drop(&mut self) {
            let _ = self.stopped.send(());
        }
    }
    impl Sink for Outage {
        type Prepared = String;
        fn preparation_bytes(_: usize) -> usize {
            512
        }
        fn prepare(&self, record: &Record, _: usize) -> Result<String, ()> {
            Ok(record.request.request_id.clone())
        }
        fn independent_retries(&self) -> bool {
            true
        }
        fn write(&mut self, _: &String) -> Result<(), ()> {
            let _ = self.attempts.send(());
            Err(())
        }
    }
    let (attempts, observed) = mpsc::channel();
    let (stopped, exited) = mpsc::channel();
    let delivery = Delivery::new(
        Limits {
            maximum_records: 8,
            maximum_bytes: limits().maximum_bytes * 8,
            ..limits()
        },
        Outage { attempts, stopped },
    )
    .unwrap();
    for id in ["a", "b", "c", "d"] {
        assert!(delivery.submit(record(id)));
    }
    for _ in 0..4 {
        observed.recv_timeout(Duration::from_secs(1)).unwrap();
    }
    assert!(!delivery.close_until(Instant::now() + Duration::from_millis(50)));
    assert_eq!(delivery.counts()[0], 4);
    assert_eq!(delivery.counts()[4], 0);
    let counters = delivery.counters.clone();
    drop(delivery);
    exited
        .recv_timeout(Duration::from_secs(3))
        .expect("retry worker survived its owner");
    assert_eq!(counters.pending.load(Ordering::Acquire), 0);
    assert_eq!(counters.bytes.load(Ordering::Acquire), 0);
    assert_eq!(counters.preparation_bytes.load(Ordering::Acquire), 0);
    assert_eq!(counters.dropped.load(Ordering::Acquire), 4);
    assert_eq!(counters.persisted.load(Ordering::Acquire), 0);
}

#[test]
fn owner_destruction_allows_recovery_after_maximum_retry_backoff() {
    struct Recoverable {
        attempts: mpsc::Sender<()>,
        recovered: Arc<AtomicBool>,
        exited: mpsc::Sender<()>,
    }
    impl Drop for Recoverable {
        fn drop(&mut self) {
            let _ = self.exited.send(());
        }
    }
    impl Sink for Recoverable {
        type Prepared = String;
        fn preparation_bytes(_: usize) -> usize {
            512
        }
        fn prepare(&self, record: &Record, _: usize) -> Result<String, ()> {
            Ok(record.request.request_id.clone())
        }
        fn independent_retries(&self) -> bool {
            true
        }
        fn write(&mut self, _: &String) -> Result<(), ()> {
            let recovered = self.recovered.load(Ordering::Acquire);
            self.attempts.send(()).unwrap();
            if recovered {
                Ok(())
            } else {
                Err(())
            }
        }
    }
    let (attempts, observed) = mpsc::channel();
    let (exited, stopped) = mpsc::channel();
    let recovered = Arc::new(AtomicBool::new(false));
    let delivery = Delivery::new(
        limits(),
        Recoverable {
            attempts,
            recovered: recovered.clone(),
            exited,
        },
    )
    .unwrap();
    let counters = delivery.counters.clone();
    assert!(delivery.submit(record("recover")));
    // Initial attempt plus 25/50/100/200/400/800 ms retries: next delay is 1s.
    for _ in 0..7 {
        observed.recv_timeout(Duration::from_secs(2)).unwrap();
    }
    drop(delivery);
    std::thread::sleep(Duration::from_millis(100));
    recovered.store(true, Ordering::Release);
    stopped.recv_timeout(Duration::from_secs(3)).unwrap();
    assert_eq!(counters.persisted.load(Ordering::Acquire), 1);
    assert_eq!(counters.dropped.load(Ordering::Acquire), 0);
    assert_eq!(counters.pending.load(Ordering::Acquire), 0);
}
