//! Native inspection and capture handles retained with the admitted owner.

use super::input::InputGate;
use crate::capture::{admission, collector::Collector};
use crate::errors::PublicError;
use serde_json::{json, Value};
use std::sync::{
    atomic::{AtomicBool, Ordering},
    Arc, Mutex,
};

#[derive(Default)]
pub(super) struct Resources {
    pub input: Option<Arc<InputGate>>,
    pub(super) reporting: Option<super::input_reports::Factory>,
    pub capture: Option<Arc<Collector>>,
    pub pending_capture: Mutex<Option<Vec<u8>>>,
    reinspection_failure: Mutex<Option<Value>>,
    closed: AtomicBool,
}

impl Resources {
    pub fn new(
        input: Option<Arc<InputGate>>,
        capture: Option<Arc<Collector>>,
        pending: Option<Vec<u8>>,
        request_id: &str,
    ) -> Result<Self, PublicError> {
        if let Some(bytes) = &pending {
            if input.is_none() || capture.is_none() || bytes.len() > 128 * 1024 * 1024 {
                return Err(PublicError::internal());
            }
            #[derive(serde::Deserialize)]
            struct Identity {
                request_id: String,
            }
            let identity: Identity =
                crate::lossless_json::from_slice(bytes).map_err(|_| PublicError::internal())?;
            if identity.request_id != request_id {
                return Err(PublicError::internal());
            }
        }
        Ok(Self {
            input,
            reporting: None,
            capture,
            pending_capture: Mutex::new(pending),
            reinspection_failure: Mutex::new(None),
            closed: AtomicBool::new(false),
        })
    }

    // The caller holds the dispatch lock so registration cannot race terminal cleanup.
    pub fn status(&self) -> Value {
        if self.closed.load(Ordering::Acquire) {
            return json!({"action": "closed"});
        }
        if let Some(failure) = self.reinspection_decision() {
            return failure;
        }
        let Some(input) = &self.input else {
            return json!({"action": "allow"});
        };
        let mut decision = input.decision();
        if decision["action"] == "allow" {
            let Ok(mut pending) = self.pending_capture.lock() else {
                return json!({"action": "closed"});
            };
            if let Some(bytes) = pending.take() {
                if self
                    .capture
                    .as_ref()
                    .is_none_or(|collector| !admission::begin(collector, &bytes))
                {
                    input.cancel(Some(json!({"failure_class": "unavailable",
                        "safe_message": "Traffic capture is unavailable or at capacity. Restore capacity and retry.",
                        "safe_details": {"code": "capture_unavailable"},
                        "retryable_same_deployment": false, "failover_eligible": false})));
                    decision = input.decision();
                }
            }
        }
        decision
    }

    pub fn terminal(&self, terminal: &mut Value) {
        let mut decision = self
            .reinspection_decision()
            .or_else(|| self.input.as_ref().map(|input| input.decision()))
            .unwrap_or_else(|| json!({"action": "allow"}));
        if decision["action"] == "pending" {
            if let Some(input) = &self.input {
                input.cancel(None);
                decision = input.decision();
            }
        }
        // An approval that landed before the cancellation stays approved.
        if decision["action"] == "allow" {
            return;
        }
        terminal["state"] = json!("failed");
        terminal["failure"] = decision["failure"].clone();
        terminal["input_guardrail_denied"] = json!(true);
    }

    pub fn deny_reinspection(&self, failure: Value) -> Result<(), PublicError> {
        let mut denied = self
            .reinspection_failure
            .lock()
            .map_err(|_| PublicError::internal())?;
        denied.get_or_insert_with(|| super::input::denied(json!({"failure": failure})));
        Ok(())
    }

    fn reinspection_decision(&self) -> Option<Value> {
        match self.reinspection_failure.lock() {
            Ok(decision) => decision.clone(),
            Err(_) => Some(super::input::unavailable()),
        }
    }

    pub fn settled(&self, request_id: &str, result: &Value) {
        let Some(capture) = &self.capture else {
            return;
        };
        let (Some(prompt), Some(response)) = (
            result["capture"]["keep_prompt"].as_bool(),
            result["capture"]["keep_response"].as_bool(),
        ) else {
            return;
        };
        let approved = self
            .input
            .as_ref()
            .is_none_or(|input| input.decision()["action"] == "allow");
        capture.settle(
            request_id,
            approved && prompt,
            approved && prompt && response,
        );
    }

    pub fn close(&self, request_id: &str) {
        self.closed.store(true, Ordering::Release);
        if let Some(input) = &self.input {
            if input.decision()["action"] == "pending" {
                input.cancel(None);
            }
            if input.decision()["action"] != "allow" {
                if let Some(capture) = &self.capture {
                    capture.settle(request_id, false, false);
                }
            }
        }
        if let Ok(mut pending) = self.pending_capture.lock() {
            pending.take();
        }
    }
}
