//! Ownership and closure races for the asynchronous checkpoint lifecycle.

use super::*;

fn config() -> Configuration {
    let mut configuration = super::config();
    configuration.asynchronous_delivery = false;
    configuration
}

#[test]
fn rejected_checkpoint_retains_unexposed_terminal_without_waiting_for_storage() {
    for asynchronous in [false, true] {
        for (keep_response, settle_first) in [(false, false), (true, false), (true, true)] {
            let (entered, started) = mpsc::channel();
            let (records, receiver) = mpsc::channel();
            let released = Arc::new(AtomicBool::new(false));
            let mut configuration = config();
            configuration.asynchronous_delivery = asynchronous;
            configuration.maximum_pending_records = 128;
            configuration.maximum_pending_bytes = 16384;
            let mut collector = Arc::new(
                Collector::new(
                    configuration,
                    PausedSink {
                        entered,
                        released: released.clone(),
                        records: MemorySink(records),
                    },
                )
                .unwrap(),
            );
            assert!(collector.begin(request("rejected")));
            let mut fillers = Vec::new();
            for index in 0..32 {
                let id = format!("pending-{index}");
                if !collector.begin(request(&id)) {
                    break;
                }
                fillers.push(id);
            }
            assert!(!fillers.is_empty());
            // Pin the limit to accepted ownership so checkpoint overhead cannot fit,
            // independently of allocator capacity and target-specific struct sizes.
            let admitted = collector.retained_bytes(&collector.pending.lock().unwrap());
            let configuration = &mut Arc::get_mut(&mut collector).unwrap().config;
            configuration.maximum_pending_bytes = admitted;
            configuration.validate().unwrap();
            assert!(collector.checkpoint_receipt("rejected").is_err());
            let retained = {
                let pending = collector.pending.lock().unwrap();
                let entry = &pending.entries["rejected"];
                assert!(!entry.checkpoint_started);
                assert!(!entry.checkpointing);
                assert!(!entry.record.checkpointed);
                collector.retained_bytes(&pending)
            };
            let (completed, completion) = mpsc::channel();
            let worker = collector.clone();
            let task = std::thread::spawn(move || {
                if settle_first {
                    worker.settle("rejected", true, keep_response);
                }
                worker.finish_unexposed("rejected");
                if !settle_first {
                    worker.settle("rejected", true, keep_response);
                }
                completed.send(()).unwrap();
            });
            assert!(started.recv_timeout(Duration::from_secs(2)).is_ok());
            let returned = completion.recv_timeout(Duration::from_secs(2));
            let count_before_ack = collector.admissions.load(Ordering::Acquire);
            let retained_before_ack = {
                let pending = collector.pending.lock().unwrap();
                assert!(!pending.entries.contains_key("rejected"));
                collector.retained_bytes(&pending)
            };
            assert!(!collector.begin(request("replacement")));
            assert!(receiver.try_recv().is_err());
            collector.settle("rejected", true, keep_response);
            for id in &fillers {
                collector.settle(id, false, false);
            }
            assert!(!collector.close_until(Instant::now()));
            released.store(true, Ordering::Release);
            task.join().unwrap();
            assert!(collector.close_until(Instant::now() + Duration::from_secs(2)));
            assert!(
                returned.is_ok(),
                "terminal retention waited for storage ACK"
            );
            assert_eq!(count_before_ack, fillers.len() + 1);
            assert_eq!(retained_before_ack, retained);
            assert_eq!(collector.admissions.load(Ordering::Acquire), 0);
            assert_eq!(collector.handoff_bytes.load(Ordering::Acquire), 0);
            let persisted: Vec<_> = receiver.try_iter().collect();
            assert_eq!(persisted.len(), 1);
            assert_eq!(persisted[0].request.request_id, "rejected");
            assert!(!persisted[0].checkpointed);
            assert!(persisted[0].response.is_none());
        }
    }
}

#[test]
fn asynchronous_queue_ack_retains_checkpoint_owner_until_storage_ack() {
    let (entered, started) = mpsc::channel();
    let (records, receiver) = mpsc::channel();
    let released = Arc::new(AtomicBool::new(false));
    let mut configuration = config();
    configuration.asynchronous_delivery = true;
    configuration.maximum_pending_records = 1;
    let collector = Collector::new(
        configuration,
        PausedSink {
            entered,
            released: released.clone(),
            records: MemorySink(records),
        },
    )
    .unwrap();
    assert!(collector.begin(request("async-owner")));
    let receipt = collector
        .checkpoint_receipt("async-owner")
        .unwrap()
        .unwrap();
    assert_eq!(receipt.blocking_recv(), Ok(true));
    assert!(started.recv_timeout(Duration::from_secs(2)).is_ok());
    collector.settle("async-owner", false, false);
    assert_eq!(collector.admissions.load(Ordering::Acquire), 1);
    assert!(collector.handoff_bytes.load(Ordering::Acquire) > 0);
    assert!(!collector.begin(request("cannot-reuse")));
    assert!(!collector.close_until(Instant::now()));
    released.store(true, Ordering::Release);
    assert!(collector.close_until(Instant::now() + Duration::from_secs(2)));
    assert_eq!(collector.admissions.load(Ordering::Acquire), 0);
    assert_eq!(receiver.try_iter().count(), 1);
}

struct ReceiveBarrierSink {
    sink: PausedSink,
    entered: mpsc::Sender<()>,
    release: Option<mpsc::Receiver<()>>,
}

impl Sink for ReceiveBarrierSink {
    type Prepared = String;
    fn preparation_bytes(maximum: usize) -> usize {
        maximum
    }
    fn prepare(&self, record: &Record, maximum: usize) -> Result<String, ()> {
        self.sink.prepare(record, maximum)
    }
    fn write(&mut self, value: &String) -> Result<(), ()> {
        self.sink.write(value)
    }
    fn before_receive(&mut self) {
        if let Some(release) = self.release.take() {
            self.entered.send(()).unwrap();
            release.recv_timeout(Duration::from_secs(3)).unwrap();
        }
    }
}

#[test]
fn disconnected_ingress_still_drains_an_accepted_checkpoint() {
    let (parked, waiting) = mpsc::channel();
    let (resume, proceed) = mpsc::channel();
    let (entered, started) = mpsc::channel();
    let (records, receiver) = mpsc::channel();
    let released = Arc::new(AtomicBool::new(false));
    let collector = Collector::new(
        config(),
        ReceiveBarrierSink {
            sink: PausedSink {
                entered,
                released: released.clone(),
                records: MemorySink(records),
            },
            entered: parked,
            release: Some(proceed),
        },
    )
    .unwrap();
    assert!(waiting.recv_timeout(Duration::from_secs(2)).is_ok());
    assert!(collector.begin(request("last-owner")));
    let receipt = collector.checkpoint_receipt("last-owner").unwrap().unwrap();
    let count = collector.admissions.clone();
    let bytes = collector.handoff_bytes.clone();
    collector.settle("last-owner", false, false);
    drop(receipt);
    drop(collector);
    resume.send(()).unwrap();
    assert!(started.recv_timeout(Duration::from_secs(2)).is_ok());
    assert_eq!(count.load(Ordering::Acquire), 1);
    assert!(bytes.load(Ordering::Acquire) > 0);
    released.store(true, Ordering::Release);
    assert_eq!(
        receiver
            .recv_timeout(Duration::from_secs(2))
            .unwrap()
            .request
            .request_id,
        "last-owner"
    );
    let until = Instant::now() + Duration::from_secs(2);
    while count.load(Ordering::Acquire) > 0 && Instant::now() < until {
        std::thread::yield_now();
    }
    assert_eq!(count.load(Ordering::Acquire), 0);
    assert_eq!(bytes.load(Ordering::Acquire), 0);
}

#[test]
fn checkpoint_owns_admission_after_discard_and_cancelled_receipt() {
    for byte_bound in [false, true] {
        for prequeue in [false, true] {
            let (entered, started) = mpsc::channel();
            let (records, receiver) = mpsc::channel();
            let released = Arc::new(AtomicBool::new(false));
            let mut configuration = config();
            configuration.maximum_pending_records = if byte_bound {
                16
            } else if prequeue {
                2
            } else {
                1
            };
            configuration.delivery.maximum_records = 1;
            let mut collector = Arc::new(
                Collector::new(
                    configuration,
                    PausedSink {
                        entered,
                        released: released.clone(),
                        records: MemorySink(records),
                    },
                )
                .unwrap(),
            );
            let second = request("checkpoint");
            let first = if prequeue {
                assert!(collector.begin(request("prior")));
                let receipt = collector.checkpoint_receipt("prior").unwrap().unwrap();
                assert!(started.recv_timeout(Duration::from_secs(2)).is_ok());
                collector.settle("prior", false, false);
                Some(receipt)
            } else {
                None
            };
            assert!(collector.begin(second));
            let receipt = collector.checkpoint_receipt("checkpoint").unwrap().unwrap();
            assert!(collector.checkpoint_receipt("checkpoint").is_err());
            if !prequeue {
                assert!(started.recv_timeout(Duration::from_secs(2)).is_ok());
            }
            let held = collector.handoff_bytes.load(Ordering::Acquire);
            assert!(held > 0);
            if byte_bound {
                Arc::get_mut(&mut collector)
                    .unwrap()
                    .config
                    .maximum_pending_bytes = held;
            }
            collector.settle("checkpoint", false, false);
            drop(receipt);
            assert_eq!(collector.handoff_bytes.load(Ordering::Acquire), held);
            assert!(!collector.begin(request("replacement")));
            assert!(!collector.close_until(Instant::now()));
            assert!(!collector.close_until(Instant::now()));
            released.store(true, Ordering::Release);
            if let Some(receipt) = first {
                assert_eq!(receipt.blocking_recv(), Ok(true));
            }
            assert!(collector.close_until(Instant::now() + Duration::from_secs(2)));
            assert_eq!(collector.admissions.load(Ordering::Acquire), 0);
            assert_eq!(collector.handoff_bytes.load(Ordering::Acquire), 0);
            assert_eq!(receiver.try_iter().count(), if prequeue { 2 } else { 1 });
        }
    }
}

#[test]
fn checkpoint_ack_before_cancel_does_not_pin_later_terminal_policy() {
    let (entered, started) = mpsc::channel();
    let (records, receiver) = mpsc::channel();
    let released = Arc::new(AtomicBool::new(true));
    let collector = Collector::new(
        config(),
        PausedSink {
            entered,
            released: released.clone(),
            records: MemorySink(records),
        },
    )
    .unwrap();
    assert!(collector.begin(request("ack-before-cancel")));
    let receipt = collector
        .checkpoint_receipt("ack-before-cancel")
        .unwrap()
        .unwrap();
    assert_eq!(receipt.blocking_recv(), Ok(true));
    assert!(collector.checkpoint_receipt("ack-before-cancel").is_err());
    released.store(false, Ordering::Release);
    collector.finish_unexposed("ack-before-cancel");
    collector.settle("ack-before-cancel", true, true);
    assert!(started.recv_timeout(Duration::from_secs(2)).is_ok());
    assert_eq!(collector.admissions.load(Ordering::Acquire), 1);
    assert!(!collector.close_until(Instant::now()));
    collector.settle("ack-before-cancel", true, true);
    released.store(true, Ordering::Release);
    assert!(collector.close_until(Instant::now() + Duration::from_secs(2)));
    assert_eq!(receiver.try_iter().count(), 2);
    assert_eq!(collector.handoff_bytes.load(Ordering::Acquire), 0);
}

#[test]
fn old_checkpoint_ack_does_not_mutate_a_reused_request_id() {
    let (entered, started) = mpsc::channel();
    let (records, receiver) = mpsc::channel();
    let released = Arc::new(AtomicBool::new(false));
    let collector = Collector::new(
        config(),
        PausedSink {
            entered,
            released: released.clone(),
            records: MemorySink(records),
        },
    )
    .unwrap();
    assert!(collector.begin(request("same")));
    let receipt = collector.checkpoint_receipt("same").unwrap().unwrap();
    assert!(started.recv_timeout(Duration::from_secs(2)).is_ok());
    collector.settle("same", false, false);
    assert!(collector.begin(request("same")));
    let expires = collector.pending.lock().unwrap().entries["same"].expires;
    released.store(true, Ordering::Release);
    assert_eq!(receipt.blocking_recv(), Ok(true));
    assert_eq!(
        collector.pending.lock().unwrap().entries["same"].expires,
        expires
    );
    assert_eq!(collector.admissions.load(Ordering::Acquire), 1);
    collector.settle("same", false, false);
    assert!(collector.close_until(Instant::now() + Duration::from_secs(2)));
    assert_eq!(receiver.try_iter().count(), 1);
}

#[test]
fn checkpoint_charge_survives_output_removal_before_ack() {
    let (entered, started) = mpsc::channel();
    let (records, receiver) = mpsc::channel();
    let released = Arc::new(AtomicBool::new(false));
    let mut configuration = config();
    configuration.maximum_pending_records = 1;
    let collector = Collector::new(
        configuration,
        PausedSink {
            entered,
            released: released.clone(),
            records: MemorySink(records),
        },
    )
    .unwrap();
    assert!(collector.begin(request("oversized")));
    let receipt = collector.checkpoint_receipt("oversized").unwrap().unwrap();
    assert!(started.recv_timeout(Duration::from_secs(2)).is_ok());
    let held = collector.handoff_bytes.load(Ordering::Acquire);
    let response = Response::Json {
        status: 200,
        body: json!({"large":"x".repeat(8192)}),
        source_json: None,
    };
    assert!(!collector.finish("oversized", Some(response), None));
    assert_eq!(collector.admissions.load(Ordering::Acquire), 1);
    assert_eq!(collector.handoff_bytes.load(Ordering::Acquire), held);
    assert!(!collector.begin(request("new")));
    assert!(!collector.close_until(Instant::now()));
    released.store(true, Ordering::Release);
    assert_eq!(receipt.blocking_recv(), Ok(true));
    assert!(collector.close_until(Instant::now() + Duration::from_secs(2)));
    assert_eq!(collector.handoff_bytes.load(Ordering::Acquire), 0);
    assert_eq!(receiver.try_iter().count(), 1);
}
