//! A caller may drop a serial gate immediately after receiving its decision.

use super::{input::InputGate, input_reports::Factory, input_reports::Reports, inspection::Check};
use crate::errors::PublicError;
use crate::native_callbacks::{CallbackTarget, NativeTarget};
use serde_json::{json, Value};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

#[derive(Default)]
struct Records(Mutex<Vec<Value>>);

impl CallbackTarget for Records {
    fn handles(&self, method: &str) -> bool {
        method == "record_input_moderation"
    }

    fn call(&self, method: &str, argument: &str) -> Result<String, PublicError> {
        assert_eq!(method, "record_input_moderation");
        self.0
            .lock()
            .unwrap()
            .push(serde_json::from_str(argument).unwrap());
        Ok("{}".into())
    }
}

#[test]
fn dropping_a_completed_serial_gate_preserves_every_moderation_report() {
    let workers: Vec<_> = (0..4)
        .map(|worker| {
            std::thread::spawn(move || {
                let records = Arc::new(Records::default());
                let reports = Arc::new(Reports::default());
                let factory = Factory {
                    target: NativeTarget::new(records.clone()),
                    request_id: format!("serial-report-{worker}"),
                    reports: reports.clone(),
                };
                for count in 1..=256 {
                    let check: Check = crate::lossless_json::from_value(json!({
                "policy": {"mode":"enforce", "protected":true, "input_execution":"before_dispatch"},
                "check": {"check_id":"serial", "action":"block", "timeout_ms":1000},
                "adapter": {"kind":"allow"}
            })).unwrap();
                    let gate = InputGate::start(vec![check], 5.0).unwrap();
                    gate.bind_reporter(factory.reporter());
                    assert_eq!(gate.wait()["action"], "allow");
                    drop(gate);
                    assert!(reports.drain(Instant::now() + Duration::from_secs(2)));
                    let recorded = records.0.lock().unwrap();
                    assert_eq!(recorded.len(), count, "completed gate lost its report");
                    assert_eq!(recorded[count - 1]["moderation"]["status"], "allowed");
                    assert_eq!(reports.counters()["dropped"], 0);
                    assert_eq!(reports.counters()["failed"], 0);
                }
            })
        })
        .collect();
    for worker in workers {
        worker.join().unwrap();
    }
}
