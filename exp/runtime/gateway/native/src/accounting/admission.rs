//! Failed admissions use the same native retention and recovery as serving requests.

use serde_json::{json, Value};

use crate::errors::PublicError;
use crate::native_callbacks::NativeTarget;

pub(super) struct Admission {
    pub target: std::sync::Arc<NativeTarget>,
}

impl super::ownership::Driver for Admission {
    fn call(&self, method: &str, input: &Value) -> Result<Value, PublicError> {
        if method != "abandon" {
            return Err(PublicError::internal());
        }
        let mut terminal = json!({"state": "failed", "finalize": true,
            "observations": {}, "service_tier": null});
        if input["failure"].is_object() {
            terminal["failure"] = input["failure"].clone();
            if input["failure"]["failure_class"] == "cancelled" {
                terminal["state"] = json!("cancelled");
            }
        }
        let output = self.target.call(
            "abandon",
            &json!({
                "request_id": input["request_id"], "terminal": terminal,
                "certify_no_effects": input["certify_no_effects"] == true,
                "web_search_requests": input["web_search_requests"].as_u64().unwrap_or(0),
            })
            .to_string(),
        )?;
        crate::lossless_json::from_str(&output).map_err(|_| PublicError::internal())
    }
}

#[cfg(test)]
mod tests {
    use super::super::ownership::{Driver, Owner};
    use super::*;
    use crate::native_callbacks::CallbackTarget;
    use std::sync::{
        atomic::{AtomicBool, Ordering},
        Arc, Mutex,
    };
    use std::time::Instant;

    #[derive(Default)]
    struct Ledger {
        fail: AtomicBool,
        writes: Mutex<Vec<Value>>,
    }
    impl CallbackTarget for Ledger {
        fn handles(&self, method: &str) -> bool {
            method == "abandon"
        }
        fn call(&self, method: &str, argument: &str) -> Result<String, PublicError> {
            assert_eq!(method, "abandon");
            self.writes
                .lock()
                .unwrap()
                .push(crate::lossless_json::from_str(argument).unwrap());
            if self.fail.load(Ordering::SeqCst) {
                return Err(PublicError::internal());
            }
            Ok(json!({"terminalized":true}).to_string())
        }
    }
    #[test]
    fn failed_admission_retains_the_original_meter_and_certificate_until_acknowledged() {
        let ledger = Arc::new(Ledger::default());
        ledger.fail.store(true, Ordering::SeqCst);
        let owner = Owner::new(
            "request".into(),
            "org".into(),
            Instant::now(),
            Arc::new(Admission {
                target: NativeTarget::new(ledger.clone()),
            }),
        );
        let failure = json!({"failure_class":"invalid_request","safe_message":"Invalid model"});
        assert!(owner
            .call(
                "abandon",
                json!({"request_id":"request","failure":failure,
            "certify_no_effects":false,"web_search_requests":3})
            )
            .is_err());
        assert!(owner.fenced());
        assert!(!owner.retired());
        ledger.fail.store(false, Ordering::SeqCst);
        owner.recover(Instant::now()).unwrap();
        assert!(owner.retired());
        assert!(!owner.fenced());
        let writes = ledger.writes.lock().unwrap();
        assert_eq!(writes.len(), 2);
        assert_eq!(writes[0], writes[1]);
        assert_eq!(writes[1]["terminal"]["failure"], failure);
        assert_eq!(writes[1]["web_search_requests"], 3);
        assert_eq!(writes[1]["certify_no_effects"], false);
    }
    #[test]
    fn failed_admission_cannot_start_a_provider_attempt() {
        let ledger = Arc::new(Ledger::default());
        let admission = Admission {
            target: NativeTarget::new(ledger.clone()),
        };
        assert!(admission
            .call("start_attempt", &json!({"request_id":"request"}))
            .is_err());
        assert!(ledger.writes.lock().unwrap().is_empty());
    }
}
