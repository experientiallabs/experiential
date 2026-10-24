use super::*;
use crate::encode_messages::MessagesSseEncoder;
use crate::events::Event;
use crate::responses_retention::ResponsesRetention;

fn budget(limit: usize) -> &'static RetainedOutputBudget {
    Box::leak(Box::new(RetainedOutputBudget::new(limit)))
}

#[test]
fn a_lease_reserves_up_to_the_budget_and_refuses_past_it() {
    let budget = budget(100);
    with_test_budget(budget, || {
        let mut first = RetainedLease::default();
        let mut second = RetainedLease::default();
        assert!(first.reserve(60));
        assert!(
            !second.reserve(41),
            "a reservation past the budget is refused"
        );
        assert_eq!(second.bytes(), 0, "a refused reservation holds nothing");
        assert!(second.reserve(40));
        assert_eq!(budget.held(), 100);
        first.release(30);
        assert_eq!(budget.held(), 70);
        first.release(1_000);
        assert_eq!(first.bytes(), 0, "a release never returns more than held");
        assert_eq!(budget.held(), 40);
    });
}

#[test]
fn dropping_a_lease_returns_everything_it_held() {
    let budget = budget(100);
    with_test_budget(budget, || {
        let mut lease = RetainedLease::default();
        assert!(lease.reserve(80));
        drop(lease);
        assert_eq!(budget.held(), 0);
    });
}

#[test]
fn concurrent_streams_share_one_budget_and_never_cross_it() {
    let budget = budget(1_000);
    let barrier: &'static std::sync::Barrier = Box::leak(Box::new(std::sync::Barrier::new(8)));
    let handles: Vec<_> = (0..8)
        .map(|_| {
            std::thread::spawn(move || {
                with_test_budget(budget, || {
                    let mut lease = RetainedLease::default();
                    let mut granted = 0;
                    for _ in 0..500 {
                        if lease.reserve(7) {
                            granted += 7;
                        }
                        assert!(budget.held() <= 1_000);
                    }
                    assert_eq!(lease.bytes(), granted);
                    // Every lease is still held here, so the grants coexist.
                    barrier.wait();
                    granted
                })
            })
        })
        .collect();
    let granted: usize = handles
        .into_iter()
        .map(|handle| handle.join().unwrap())
        .sum();
    assert!(granted <= 1_000);
    assert_eq!(budget.held(), 0, "every lease returned its bytes on drop");
}

#[test]
fn a_messages_stream_fails_retryably_when_the_worker_budget_is_full() {
    let budget = budget(64);
    with_test_budget(budget, || {
        let mut occupant = RetainedLease::default();
        assert!(occupant.reserve(60));
        let mut encoder = MessagesSseEncoder::new("req", "model");
        encoder.start().unwrap();
        let event = Event::RedactedThinking {
            index: 0,
            data: "x".repeat(16),
        };
        let error = encoder.feed(&event).expect_err("the worker budget is full");
        assert_eq!(error.status_code, 503, "{error:?}");
        assert_eq!(error.code, "gateway_unavailable");
        drop(occupant);
        let mut encoder = MessagesSseEncoder::new("req", "model");
        encoder.start().unwrap();
        assert!(
            encoder.feed(&event).is_ok(),
            "freed budget serves the stream"
        );
    });
    assert_eq!(budget.held(), 0);
}

#[test]
fn a_responses_continuation_is_dropped_not_retained_past_the_worker_budget() {
    let budget = budget(32);
    with_test_budget(budget, || {
        let mut retention = ResponsesRetention::default();
        retention.track(&Event::TextDelta("y".repeat(20)));
        assert!(!retention.overflowed);
        assert_eq!(budget.held(), 20);
        retention.track(&Event::TextDelta("y".repeat(20)));
        assert!(retention.overflowed, "the continuation is not retained");
        assert_eq!(budget.held(), 0, "an abandoned continuation holds nothing");
    });
}

#[test]
fn the_default_budget_is_off_so_nothing_is_ever_refused() {
    assert_eq!(DEFAULT_WORKER_RETAINED_OUTPUT_BYTES, usize::MAX);
    assert_eq!(crate::server::default_retained_for_tests(), usize::MAX);
    let budget = budget(DEFAULT_WORKER_RETAINED_OUTPUT_BYTES);
    with_test_budget(budget, || {
        let mut leases: Vec<RetainedLease> = (0..64).map(|_| RetainedLease::default()).collect();
        for lease in &mut leases {
            assert!(
                lease.reserve(64 * 1024 * 1024),
                "an unconfigured worker retains as before"
            );
        }
    });
}
