//! Native inspection and capture handles retained with the admitted owner.

use super::input::InputGate;
use crate::capture::{admission, collector::Collector};
use crate::errors::PublicError;
use futures_util::{stream::FuturesUnordered, StreamExt};
use serde_json::{json, Value};
use std::sync::{
    atomic::{AtomicBool, Ordering},
    Arc, Mutex,
};

#[derive(Default)]
pub(crate) struct Resources {
    pub(super) input: Option<Arc<InputGate>>,
    followup: Mutex<Followup>,
    pub(super) reporting: Option<super::input_reports::Factory>,
    pub capture: Option<Arc<Collector>>,
    pub pending_capture: Mutex<Option<Vec<u8>>>,
    capture_failed: AtomicBool,
    reinspection_failure: Mutex<Option<Value>>,
    closed: AtomicBool,
}

#[derive(Default)]
struct Followup {
    active: Vec<Arc<InputGate>>,
    pending: Option<Arc<InputGate>>,
}

impl Resources {
    fn inputs(&self) -> Vec<Arc<InputGate>> {
        let mut inputs: Vec<_> = self.input.iter().cloned().collect();
        if let Ok(followup) = self.followup.lock() {
            inputs.extend(followup.active.iter().cloned());
        }
        inputs
    }

    pub(super) fn stage_reinspection(&self, gate: Arc<InputGate>) -> Result<(), PublicError> {
        let mut followup = self.followup.lock().map_err(|_| PublicError::internal())?;
        if followup.pending.is_some() || followup.active.len() >= 8 {
            return Err(PublicError::internal());
        }
        followup.pending = Some(gate);
        Ok(())
    }

    pub(super) fn activate_reinspection(&self) -> Result<(), PublicError> {
        let mut followup = self.followup.lock().map_err(|_| PublicError::internal())?;
        if let Some(gate) = followup.pending.take() {
            followup.active.push(gate);
        }
        Ok(())
    }

    #[cfg(test)]
    pub(crate) fn test_input(checks: Value, remaining: f64) -> Self {
        Self::new(
            Some(
                InputGate::start(crate::lossless_json::from_value(checks).unwrap(), remaining)
                    .unwrap(),
            ),
            None,
            None,
            "request",
        )
        .unwrap()
    }
    pub async fn wait(&self) -> Value {
        // Every physical input remains authoritative until it completes. Race all
        // pending checks so an earlier round cannot hide a later explicit denial.
        let inputs = self.inputs();
        let mut waiting: FuturesUnordered<_> = inputs
            .iter()
            .filter(|input| !input.observe())
            .map(|input| input.wait_async())
            .collect();
        while let Some(decision) = waiting.next().await {
            if decision["action"] != "allow" {
                return decision;
            }
        }
        self.status()
    }

    pub fn begin_response(&self) {
        let at = std::time::Instant::now();
        for input in self.inputs() {
            input.begin_response(at);
        }
    }

    /// Strict policies protect tool side effects; fail-open parallel policies keep running.
    pub async fn wait_for_tools(&self) -> Value {
        let inputs = self.inputs();
        let mut waiting: FuturesUnordered<_> = inputs
            .iter()
            .filter(|input| !input.observe())
            .map(|input| input.wait_strict())
            .collect();
        while let Some(decision) = waiting.next().await {
            if decision["action"] != "allow" {
                return decision;
            }
        }
        // Capture registration belongs to the dispatch-locked status recheck.
        self.decision()
    }

    #[cfg(test)]
    pub fn grace(&self) -> Option<std::time::Duration> {
        let inputs = self.inputs();
        let mut grace = std::time::Duration::ZERO;
        for input in inputs.iter().filter(|input| !input.observe()) {
            grace = grace.max(input.grace()?);
        }
        Some(grace)
    }

    #[cfg(test)]
    pub fn release_incomplete(&self) {
        for input in self.inputs().iter().filter(|input| !input.observe()) {
            input.release_incomplete("response_ready_timeout");
        }
    }

    fn decision(&self) -> Value {
        if let Some(failure) = self.reinspection_decision() {
            return failure;
        }
        let mut decision = json!({"action": "allow"});
        let mut observations = Vec::new();
        let mut status = "allowed";
        let mut moderation = Value::Null;
        for input in self.inputs() {
            let value = input.decision();
            if let Some(records) = value["moderation"]["observations"].as_array() {
                observations.extend(records.iter().cloned());
            }
            if input.observe() {
                if value["action"] == "pending" && status == "allowed" {
                    status = "observe_pending";
                }
                continue;
            }
            if value["action"] == "error"
                || (value["action"] == "pending" && decision["action"] == "allow")
            {
                decision = value.clone();
            }
            if value["action"] == "error" {
                moderation = value["moderation"].clone();
                status = if value["moderation"]["status"] == "unavailable" {
                    "unavailable"
                } else {
                    "denied"
                };
            } else if decision["action"] != "error"
                && value["moderation"]["status"] == "failed_open"
            {
                status = "failed_open";
                moderation = value["moderation"].clone();
            }
        }
        if !observations.is_empty() || status != "allowed" {
            if !moderation.is_object() {
                moderation = json!({});
            }
            moderation["status"] = json!(status);
            moderation["observations"] = json!(observations);
            decision["moderation"] = moderation;
        }
        if self.capture_failed.load(Ordering::Acquire) {
            decision["failure"] = json!({
                "failure_class": "unavailable",
                "safe_message": "Traffic capture is unavailable or at capacity. Restore capacity and retry.",
                "safe_details": {"code": "capture_unavailable"},
                "retryable_same_deployment": false,
                "failover_eligible": false,
            });
            decision = super::input::denied(decision);
        }
        decision
    }

    pub(super) fn new(
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
            followup: Mutex::new(Followup::default()),
            reporting: None,
            capture,
            pending_capture: Mutex::new(pending),
            capture_failed: AtomicBool::new(false),
            reinspection_failure: Mutex::new(None),
            closed: AtomicBool::new(false),
        })
    }

    // The pending-capture lock serializes registration with terminal cleanup.
    pub fn status(&self) -> Value {
        if self.closed.load(Ordering::Acquire) {
            return json!({"action": "closed"});
        }
        let decision = self.decision();
        if decision["action"] == "allow" {
            let Ok(mut pending) = self.pending_capture.lock() else {
                return json!({"action": "closed"});
            };
            if self.closed.load(Ordering::Acquire) {
                return json!({"action": "closed"});
            }
            if let Some(bytes) = pending.take() {
                if let Some(collector) = &self.capture {
                    // Required capture is independent of moderation fail-open. Retain
                    // its failure through response gating and terminal settlement.
                    if !admission::begin(collector, &bytes) {
                        self.capture_failed.store(true, Ordering::Release);
                    }
                }
            }
            if self.capture_failed.load(Ordering::Acquire) {
                return self.decision();
            }
        }
        decision
    }

    pub fn terminal(&self, terminal: &mut Value) {
        let mut decision = self.decision();
        if decision["action"] == "pending" {
            if terminal["finalize"] == false {
                // An intermediate attempt has no outward response. Its retry
                // keeps every pending decision and the response-ready budget.
                return;
            }
            for input in self.inputs().iter().filter(|input| !input.observe()) {
                input.cancel(None);
            }
            decision = self.decision();
        }
        if !decision["moderation"].is_null() {
            terminal["input_moderation"] = decision["moderation"].clone();
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
        let approved = self.decision()["action"] == "allow";
        capture.settle(
            request_id,
            approved && prompt,
            approved && prompt && response,
        );
    }

    pub fn close(&self, request_id: &str) {
        let mut pending = self.pending_capture.lock().ok();
        self.closed.store(true, Ordering::Release);
        for input in self.inputs().iter().filter(|input| !input.observe()) {
            if input.decision()["action"] == "pending" {
                input.cancel(None);
            }
            if input.decision()["action"] != "allow" {
                if let Some(capture) = &self.capture {
                    capture.settle(request_id, false, false);
                }
            }
        }
        if let Ok(mut followup) = self.followup.lock() {
            if let Some(gate) = followup.pending.take() {
                gate.cancel(None);
            }
        }
        if let Some(pending) = pending.as_mut() {
            pending.take();
        }
    }
}
