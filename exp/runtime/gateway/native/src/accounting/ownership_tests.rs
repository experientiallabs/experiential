use std::sync::{Arc, Condvar, Mutex};
use std::time::{Duration, Instant};

use serde_json::{json, Value};

use super::ownership::{Driver, Owner};
use crate::errors::PublicError;

#[derive(Default)]
struct Recording {
    calls: Mutex<Vec<(String, Value)>>,
    hold: Mutex<(bool, bool)>,
    wake: Condvar,
    fail_settle: Mutex<bool>,
}

impl Driver for Recording {
    fn call(&self, method: &str, input: &Value) -> Result<Value, PublicError> {
        self.calls
            .lock()
            .unwrap()
            .push((method.into(), input.clone()));
        if method == "start_attempt" {
            let mut hold = self.hold.lock().unwrap();
            hold.1 = true;
            self.wake.notify_all();
            while hold.0 {
                hold = self.wake.wait(hold).unwrap();
            }
        }
        if method == "settle" && *self.fail_settle.lock().unwrap() {
            return Err(PublicError::internal());
        }
        Ok(json!({"terminalized": true, "attempt_id": "attempt"}))
    }
}

fn owner(driver: Arc<Recording>) -> Arc<Owner> {
    Owner::new(
        "request".into(),
        "tenant".into(),
        Instant::now() + Duration::from_secs(60),
        driver,
    )
}

fn terminal(tokens: u64) -> Value {
    json!({"request_id": "request", "attempt_id": "attempt", "outcome": "completed",
        "finalize": true, "usage": {"input_tokens": tokens, "output_tokens": 5}})
}

#[test]
fn a_late_reservation_is_cancelled_before_any_dispatch_reply() {
    let driver = Arc::new(Recording::default());
    driver.hold.lock().unwrap().0 = true;
    let owner = owner(driver.clone());
    let task_owner = owner.clone();
    let task = std::thread::spawn(move || {
        task_owner.call("start_attempt", json!({"request_id": "request"}))
    });
    {
        let mut hold = driver.hold.lock().unwrap();
        let deadline = Instant::now() + Duration::from_secs(2);
        while !hold.1 {
            let now = Instant::now();
            assert!(now < deadline, "reservation did not start");
            hold = driver.wake.wait_timeout(hold, deadline - now).unwrap().0;
        }
    }
    assert_eq!(
        owner
            .call("abandon", json!({"request_id": "request"}))
            .unwrap()["terminalized"],
        false
    );
    driver.hold.lock().unwrap().0 = false;
    driver.wake.notify_all();
    assert_eq!(task.join().unwrap().unwrap()["exhausted"], true);
    assert!(owner.finished());
    assert_eq!(
        driver
            .calls
            .lock()
            .unwrap()
            .iter()
            .map(|(method, _)| method.as_str())
            .collect::<Vec<_>>(),
        ["start_attempt", "abandon"]
    );
}

#[test]
fn failed_settlement_keeps_its_first_meter_across_quote_cancel_and_recovery() {
    let driver = Arc::new(Recording::default());
    let owner = owner(driver.clone());
    owner.call("price", terminal(37)).unwrap();
    *driver.fail_settle.lock().unwrap() = true;
    assert!(owner.call("settle", terminal(999)).is_err());
    assert!(owner.fenced());
    assert!(owner
        .call("abandon", json!({"request_id": "request"}))
        .is_err());
    assert!(!owner.finished());
    *driver.fail_settle.lock().unwrap() = false;
    assert!(owner.recover(Instant::now()).unwrap());
    assert!(!owner.fenced());
    let calls = driver.calls.lock().unwrap();
    assert_eq!(calls.len(), 4);
    assert!(calls.iter().all(|(_, input)| input == &terminal(37)));
    assert!(calls.iter().all(|(method, _)| method != "abandon"));
}

#[test]
fn mismatched_requests_and_attempts_cannot_replace_retained_evidence() {
    let driver = Arc::new(Recording::default());
    let owner = owner(driver.clone());
    let mut input = terminal(3);
    input["request_id"] = json!("someone-else");
    assert!(owner.call("settle", input).is_err());
    owner.call("price", terminal(37)).unwrap();
    let mut input = terminal(5);
    input["attempt_id"] = json!("other-attempt");
    assert!(owner.call("settle", input).is_err());
    assert!(owner.recover(Instant::now()).unwrap());
    assert_eq!(driver.calls.lock().unwrap().last().unwrap().1, terminal(37));
}

#[test]
fn a_meter_after_deadline_recovery_still_reaches_settlement() {
    let driver = Arc::new(Recording::default());
    let owner = owner(driver.clone());
    assert!(owner
        .recover(Instant::now() + Duration::from_secs(66))
        .unwrap());
    assert!(owner.finished());
    owner.call("settle", terminal(42)).unwrap();
    let calls = driver.calls.lock().unwrap();
    assert_eq!(
        calls
            .iter()
            .map(|(method, _)| method.as_str())
            .collect::<Vec<_>>(),
        ["abandon", "settle"]
    );
    assert_eq!(calls[1].1, terminal(42));
}
