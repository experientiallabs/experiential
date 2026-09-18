use super::*;

struct PausedSink {
    entered: mpsc::Sender<String>,
    resume: mpsc::Receiver<()>,
    fail: bool,
}

impl Sink for PausedSink {
    fn write(&mut self, record: &str) -> Result<(), ()> {
        self.entered.send(record.to_owned()).map_err(|_| ())?;
        self.resume
            .recv_timeout(Duration::from_secs(5))
            .map_err(|_| ())?;
        if self.fail {
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
        maximum_bytes: 16,
        maximum_record_bytes: 8,
    }
}

#[test]
fn saturated_destination_never_blocks_serving_or_exceeds_total_byte_budget() {
    let (delivery, entered, resume) = paused(limits(), false);
    assert!(delivery.submit("12345678".to_owned()));
    assert_eq!(
        entered.recv_timeout(Duration::from_secs(1)).unwrap(),
        "12345678"
    );
    assert!(delivery.submit("abcdefgh".to_owned()));
    for _ in 0..1000 {
        assert!(!delivery.submit("x".to_owned()));
    }
    assert_eq!(delivery.counts(), [2, 16, 0, 0, 1000]);
    resume.send(()).unwrap();
    assert_eq!(
        entered.recv_timeout(Duration::from_secs(1)).unwrap(),
        "abcdefgh"
    );
    resume.send(()).unwrap();
    assert!(delivery.close_until(Instant::now() + Duration::from_secs(1)));
    assert_eq!(delivery.counts(), [0, 0, 2, 0, 1000]);
    assert!(!delivery.submit("closed".to_owned()));
    assert_eq!(delivery.counts(), [0, 0, 2, 0, 1001]);
}

#[test]
fn record_count_is_bounded_even_for_tiny_records() {
    let (delivery, entered, resume) = paused(limits(), false);
    assert!(delivery.submit("a".to_owned()));
    entered.recv_timeout(Duration::from_secs(1)).unwrap();
    assert!(delivery.submit("b".to_owned()));
    assert!(!delivery.submit("c".to_owned()));
    assert_eq!(delivery.counts(), [2, 2, 0, 0, 1]);
    resume.send(()).unwrap();
    entered.recv_timeout(Duration::from_secs(1)).unwrap();
    resume.send(()).unwrap();
    assert!(delivery.close_until(Instant::now() + Duration::from_secs(1)));
}

#[test]
fn allocated_capacity_not_just_json_length_is_charged() {
    let (delivery, _, _) = paused(limits(), false);
    let mut large_allocation = String::with_capacity(64);
    large_allocation.push('x');
    assert!(!delivery.submit(large_allocation));
    assert_eq!(delivery.counts(), [0, 0, 0, 0, 1]);
    assert!(delivery.close_until(Instant::now() + Duration::from_secs(1)));
}

#[test]
fn failed_destination_releases_budget_and_records_no_sensitive_error() {
    let (delivery, entered, resume) = paused(limits(), true);
    assert!(delivery.submit("private".to_owned()));
    entered.recv_timeout(Duration::from_secs(1)).unwrap();
    resume.send(()).unwrap();
    assert!(delivery.close_until(Instant::now() + Duration::from_secs(1)));
    assert_eq!(delivery.counts(), [0, 0, 0, 1, 0]);
}

#[test]
fn shutdown_returns_while_sink_is_blocked_and_expires_queued_records() {
    let (delivery, entered, resume) = paused(limits(), false);
    assert!(delivery.submit("active".to_owned()));
    entered.recv_timeout(Duration::from_secs(1)).unwrap();
    assert!(delivery.submit("queued".to_owned()));
    assert!(!delivery.close_until(Instant::now()));
    assert!(!delivery.submit("late".to_owned()));
    resume.send(()).unwrap();
    assert!(delivery.close_until(Instant::now() + Duration::from_secs(1)));
    assert_eq!(delivery.counts(), [0, 0, 1, 0, 2]);
    assert!(entered.try_recv().is_err());
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
            maximum_record_bytes: 9 * 1024 * 1024,
            maximum_bytes: 10 * 1024 * 1024,
            ..limits()
        },
    ] {
        assert!(bounds.validate().is_err());
    }
}
