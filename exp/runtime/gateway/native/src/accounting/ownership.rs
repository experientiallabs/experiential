//! Request-local execution gates and retained terminal evidence.
//!
//! Intent is recorded before waiting for I/O ownership. Process recovery keeps
//! a strong reference until every required terminal write has completed.

use std::sync::{Arc, Mutex, MutexGuard, TryLockError};
use std::time::{Duration, Instant};

use serde_json::{json, Value};

use crate::errors::PublicError;

pub(super) trait Driver: Send + Sync {
    fn call(&self, method: &str, input: &Value) -> Result<Value, PublicError>;
    fn validate_terminal(&self, _input: &Value) -> Result<(), PublicError> {
        Ok(())
    }
}

#[derive(Default)]
struct State {
    pending: Option<Arc<Value>>,
    abandon: Option<Value>,
    finished: bool,
    failed_write: bool,
    /// A meter observed after the request finished whose write failed. The
    /// data plane hands a finalizing write's retry to this owner, so it stays
    /// owned until recovery delivers it.
    late: Option<Value>,
}

pub(super) struct Owner {
    pub request_id: String,
    pub organization_id: String,
    pub deadline: Instant,
    driver: Arc<dyn Driver>,
    execution: Mutex<()>,
    state: Mutex<State>,
}

impl Owner {
    pub fn new(
        request_id: String,
        organization_id: String,
        deadline: Instant,
        driver: Arc<dyn Driver>,
    ) -> Arc<Self> {
        Arc::new(Self {
            request_id,
            organization_id,
            deadline,
            driver,
            execution: Mutex::new(()),
            state: Mutex::new(State::default()),
        })
    }

    fn state(&self) -> Result<MutexGuard<'_, State>, PublicError> {
        self.state.lock().map_err(|_| PublicError::internal())
    }

    pub fn finished(&self) -> bool {
        self.state.lock().is_ok_and(|state| state.finished)
    }

    /// Finished, and owes no retained write.
    pub fn retired(&self) -> bool {
        self.state
            .lock()
            .is_ok_and(|state| state.finished && state.late.is_none())
    }

    pub fn fenced(&self) -> bool {
        self.state.lock().map_or(true, |state| {
            (!state.finished && (state.failed_write || state.abandon.is_some()))
                || state.late.is_some()
        })
    }

    pub fn needs_recovery(&self, now: Instant) -> bool {
        self.state.lock().map_or(true, |state| {
            (!state.finished
                && (state.failed_write
                    || state.abandon.is_some()
                    || now >= self.deadline + Duration::from_secs(5)))
                || state.late.is_some()
        })
    }

    pub fn call(&self, method: &str, input: Value) -> Result<Value, PublicError> {
        if input["request_id"].as_str() != Some(&self.request_id) {
            return Err(PublicError::internal());
        }
        match method {
            "start_attempt" => self.start(input),
            "price" | "settle" => self.terminal(method, input),
            "abandon" => self.abandon(input),
            // Other native operations keep their own resource locks; they may
            // never execute a Python callback or replace terminal evidence.
            _ => self.driver.call(method, &input),
        }
    }

    fn start(&self, input: Value) -> Result<Value, PublicError> {
        let _execution = self.execution.lock().map_err(|_| PublicError::internal())?;
        {
            let state = self.state()?;
            if state.finished || state.pending.is_some() {
                return Err(recovering());
            }
            if let Some(abandon) = &state.abandon {
                return Ok(exhausted(abandon));
            }
        }
        if Instant::now() >= self.deadline {
            self.note_abandon(timeout(&self.request_id))?;
            self.drain()?;
            return Ok(exhausted(&timeout(&self.request_id)));
        }
        let result = self.driver.call("start_attempt", &input);
        if result.is_err() {
            self.state()?.failed_write = true;
            self.note_abandon(json!({"request_id": self.request_id, "failure": {
                "failure_class": "internal", "safe_message": "gateway could not reserve attempt accounting before dispatch"
            }}))?;
            // An uncertain reservation is owned until cleanup is durable.
            // Preserve its original public error after successful cleanup.
            self.drain()?;
            return result;
        }
        if let Some(refused) = result.as_ref().ok().and_then(|value| value.get("refused")) {
            let error: PublicError = crate::lossless_json::from_value(refused["error"].clone())
                .map_err(|_| PublicError::internal())?;
            // The refusal is the customer's terminal. Keep it as this request's
            // intent before the write, so recovery retries the refusal rather
            // than an uncertain-reservation failure.
            self.note_abandon(
                json!({"request_id": self.request_id, "failure": refused["failure"]}),
            )?;
            self.drain()?;
            return Err(error);
        }
        if Instant::now() >= self.deadline {
            self.note_abandon(timeout(&self.request_id))?;
        }
        // A cancellation never needs the execution lock to become visible.
        // A hold that completed late is closed before a dispatch can escape.
        let abandon = self.state()?.abandon.clone();
        if let Some(abandon) = abandon {
            self.drain()?;
            return Ok(exhausted(&abandon));
        }
        if let Ok(value) = &result {
            if value["exhausted"] == true {
                self.note_abandon(json!({
                    "request_id": self.request_id,
                    "failure": value["failure"],
                    "certify_no_effects": value["certify_no_effects"] == true,
                }))?;
                let terminal = self.drain()?;
                let mut response = value.clone();
                response["known_unbilled"] =
                    json!(terminal["terminalized"] == true && terminal["known_unbilled"] == true);
                return Ok(response);
            }
        }
        result
    }

    fn observe(&self, input: Value) -> Result<Option<Arc<Value>>, PublicError> {
        let attempt = input["attempt_id"]
            .as_str()
            .filter(|id| !id.is_empty())
            .ok_or_else(PublicError::internal)?;
        // Reject malformed provenance before it can win the retained meter.
        if !matches!(
            input["outcome"].as_str(),
            Some("completed" | "incomplete" | "failed")
        ) || input
            .get("finalize")
            .is_some_and(|value| !value.is_boolean())
        {
            return Err(PublicError::internal());
        }
        self.driver.validate_terminal(&input)?;
        let mut state = self.state()?;
        if state.finished {
            return Ok(None);
        }
        if let Some(pending) = &state.pending {
            if pending["attempt_id"].as_str() != Some(attempt) {
                return Err(recovering());
            }
            return Ok(Some(pending.clone()));
        }
        let pending = Arc::new(input);
        state.pending = Some(pending.clone());
        Ok(Some(pending))
    }

    fn terminal(&self, method: &str, input: Value) -> Result<Value, PublicError> {
        let Some(pending) = self.observe(input.clone())? else {
            // A meter that arrives after recovery finished the request is still
            // delivered: a settled attempt answers with its cached result and a
            // cancelled one records the late meter, so served usage stays billed.
            // A failed delivery stays owned until recovery writes it.
            let _execution = self.execution.lock().map_err(|_| PublicError::internal())?;
            let result = self.driver.call(method, &input);
            let mut state = self.state()?;
            if result.is_err() {
                state.late.get_or_insert(input);
            } else if state
                .late
                .as_ref()
                .is_some_and(|late| late["attempt_id"] == input["attempt_id"])
            {
                state.late = None;
            }
            return result;
        };
        let _execution = self.execution.lock().map_err(|_| PublicError::internal())?;
        {
            let state = self.state()?;
            if state.finished {
                drop(state);
                return self.driver.call(method, &input);
            }
            if !state
                .pending
                .as_ref()
                .is_some_and(|value| Arc::ptr_eq(value, &pending))
            {
                return Ok(json!({}));
            }
        }
        let result = self.driver.call(method, &pending);
        if method == "price" {
            // Even a failed quote leaves the original meter for settlement.
            if result.is_err() {
                self.state()?.failed_write = true;
            }
            return result;
        }
        self.committed(&pending, &result)?;
        let output = result?;
        if self.state()?.abandon.is_some() && !self.finished() {
            self.drain()?;
        }
        Ok(output)
    }

    fn committed(
        &self,
        pending: &Arc<Value>,
        result: &Result<Value, PublicError>,
    ) -> Result<(), PublicError> {
        let mut state = self.state()?;
        if result.is_err() {
            state.failed_write = true;
        } else if state
            .pending
            .as_ref()
            .is_some_and(|value| Arc::ptr_eq(value, pending))
        {
            state.pending = None;
            state.failed_write = false;
            state.finished = pending["finalize"].as_bool().unwrap_or(true);
        }
        Ok(())
    }

    fn note_abandon(&self, input: Value) -> Result<(), PublicError> {
        self.state()?.abandon.get_or_insert(input);
        Ok(())
    }

    fn abandon(&self, input: Value) -> Result<Value, PublicError> {
        self.note_abandon(input)?;
        match self.execution.try_lock() {
            Ok(_execution) => self.drain(),
            Err(TryLockError::WouldBlock) => Ok(json!({"terminalized": false})),
            Err(TryLockError::Poisoned(_)) => Err(PublicError::internal()),
        }
    }

    /// Runs only while the caller owns the execution gate.
    fn drain(&self) -> Result<Value, PublicError> {
        let pending = {
            let state = self.state()?;
            if state.finished {
                let late = state.late.clone();
                drop(state);
                // Recovery and shutdown both deliver a retained late meter.
                if let Some(late) = late {
                    self.driver.call("settle", &late)?;
                    self.state()?.late = None;
                }
                return Ok(json!({"terminalized": true}));
            }
            state.pending.clone()
        };
        if let Some(pending) = pending {
            let result = self.driver.call("settle", &pending);
            self.committed(&pending, &result)?;
            result?;
        }
        if self.finished() {
            return Ok(json!({"terminalized": true}));
        }
        let abandon = self.state()?.abandon.clone();
        if let Some(abandon) = abandon {
            let result = self.driver.call("abandon", &abandon);
            let mut state = self.state()?;
            state.failed_write = result.is_err();
            let result = result?;
            // Evidence may arrive while a cancellation is writing. Keep that
            // evidence owned even if the host's cancellation already committed.
            if state.pending.is_some() {
                return Ok(json!({"terminalized": false}));
            }
            state.finished = result["terminalized"].as_bool() == Some(true);
            return Ok(result);
        }
        Ok(json!({"terminalized": false}))
    }

    pub fn recover(&self, now: Instant) -> Result<bool, PublicError> {
        if now >= self.deadline + Duration::from_secs(5) {
            self.note_abandon(timeout(&self.request_id))?;
        }
        match self.execution.try_lock() {
            Ok(_execution) => {
                // Selection can race a completed retry and the next healthy quote.
                if !self.needs_recovery(now) {
                    return Ok(self.retired());
                }
                self.drain()?;
                Ok(self.retired())
            }
            Err(TryLockError::WouldBlock) => Ok(false),
            Err(TryLockError::Poisoned(_)) => Err(PublicError::internal()),
        }
    }
}

fn recovering() -> PublicError {
    PublicError::new(
        503,
        "accounting_unavailable",
        "Request accounting is recovering. Retry later.",
        "api_error",
    )
}

fn timeout(request: &str) -> Value {
    json!({"request_id": request, "failure": {
        "failure_class": "timeout",
        "safe_message": "the request reached the gateway's time limit before any provider finished answering; retry the request"
    }})
}

fn exhausted(abandon: &Value) -> Value {
    json!({"exhausted": true, "failure": abandon.get("failure").cloned().unwrap_or_else(|| json!({
        "failure_class": "cancelled", "safe_message": "gateway request was cancelled"
    }))})
}
