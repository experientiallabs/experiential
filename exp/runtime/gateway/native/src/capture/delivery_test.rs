use super::*;

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
