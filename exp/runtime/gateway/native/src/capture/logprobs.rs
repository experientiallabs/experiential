//! Gateway-requested Chat token probabilities, retained for capture only.
//!
//! When the host enables `capture_logprobs`, an eligible OpenAI-compatible
//! rung of a captured request is dialed with `logprobs: true` although the
//! caller never asked. The normalizer then parses the provider's probabilities
//! into a side buffer instead of the event stream, so what the caller receives
//! (bytes, usage, settlement) is exactly the no-injection answer; only the
//! winning attempt's buffer reaches the capture record. A provider that refuses
//! the field is re-dialed once without it inside the same physical attempt, and
//! a rung whose plain re-dial then opens is never injected again by this worker.

use std::collections::HashSet;
use std::sync::{LazyLock, Mutex};

use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::errors::{Failure, FailureClass};
use crate::logprobs::{ChoiceLogprobsDelta, TokenLogprob};

/// Encoded-size bound on one attempt's retained probabilities; beyond it the
/// buffer stops growing and the capture record says it was truncated.
pub(crate) const MAXIMUM_CAPTURED_LOGPROB_BYTES: usize = 4 * 1024 * 1024;

/// Bound on remembered refusing rungs so a churning catalog cannot grow it.
const MAXIMUM_REFUSED_RUNGS: usize = 4096;

static REFUSED: LazyLock<Mutex<HashSet<String>>> = LazyLock::new(Default::default);

/// Whether this worker has seen the rung refuse injected probabilities.
pub(crate) fn refused(deployment_id: &str) -> bool {
    REFUSED
        .lock()
        .map(|refused| refused.contains(deployment_id))
        .unwrap_or(true)
}

/// Stop injecting on a rung whose plain re-dial opened after a refusal.
pub(crate) fn remember_refusal(deployment_id: &str) {
    if let Ok(mut refused) = REFUSED.lock() {
        if refused.len() < MAXIMUM_REFUSED_RUNGS {
            refused.insert(deployment_id.to_owned());
        }
    }
}

/// The dialed payload with probabilities requested, or `None` when the payload
/// is not an object or already carries a caller probability control.
pub(crate) fn with_logprobs(payload: &Value) -> Option<Value> {
    let object = payload.as_object()?;
    if object.contains_key("logprobs") || object.contains_key("top_logprobs") {
        return None;
    }
    let mut injected = object.clone();
    injected.insert("logprobs".to_owned(), Value::Bool(true));
    Some(Value::Object(injected))
}

/// A refused open that a plain re-dial might answer: only request-shaped
/// refusals, never credential, quota, throttle or transport failures.
pub(crate) fn may_be_refusal(failure: &Failure) -> bool {
    matches!(
        failure.failure_class,
        FailureClass::InvalidRequest | FailureClass::UnsupportedCapability
    )
}

/// How a normalizer treats Chat probabilities on its wire.
#[derive(Debug, Default)]
pub(crate) enum ChatProbabilities {
    /// Not requested: any probabilities the provider sends are ignored.
    #[default]
    Off,
    /// The caller asked: probabilities are events of the caller's answer.
    Caller,
    /// The gateway asked: probabilities go to this side buffer only.
    Capture(Buffer),
}

impl ChatProbabilities {
    pub(crate) fn requested(by_caller: bool) -> Self {
        if by_caller {
            Self::Caller
        } else {
            Self::Off
        }
    }

    pub(crate) fn requested_by_caller(&self) -> bool {
        matches!(self, Self::Caller)
    }

    /// Retain a capture-only choice channel; malformed or null channels are dropped.
    pub(crate) fn retain(&mut self, value: Option<&Value>, choice_index: u32) {
        if let Self::Capture(buffer) = self {
            if let Ok(Some(delta)) = crate::logprobs::parse(value, choice_index) {
                if delta.logprobs.is_some() {
                    buffer.push(delta);
                }
            }
        }
    }

    pub(crate) fn take(&mut self) -> Option<(Vec<ChoiceLogprobsDelta>, bool)> {
        match self {
            Self::Capture(buffer) => Some(buffer.take()),
            _ => None,
        }
    }
}

/// The relay's hold on one dial's capture-only probabilities: deltas wait here
/// until the winning attempt's capture observer attaches.
#[derive(Debug, Default)]
pub(crate) struct Held {
    pub injected: bool,
    pub pending: Vec<ChoiceLogprobsDelta>,
    truncated: bool,
}

impl Held {
    /// Forward newly parsed deltas to the observer, or hold them until it attaches.
    pub(crate) fn forward(
        &mut self,
        taken: Option<(Vec<ChoiceLogprobsDelta>, bool)>,
        observer: Option<&super::reasoning::Observer>,
    ) {
        let Some((deltas, truncated)) = taken else {
            return;
        };
        self.truncated |= truncated;
        match observer {
            Some(observer) => observer.logprobs(deltas, self.truncated),
            None => self.pending.extend(deltas),
        }
    }

    /// Mark the winning record and hand it everything held so far.
    pub(crate) fn attach(&mut self, observer: &super::reasoning::Observer) {
        if self.injected {
            observer.logprobs_injected();
            observer.logprobs(std::mem::take(&mut self.pending), self.truncated);
        }
    }
}

/// One attempt's bounded side buffer, filled by the normalizer.
#[derive(Debug, Default)]
pub(crate) struct Buffer {
    pub deltas: Vec<ChoiceLogprobsDelta>,
    pub bytes: usize,
    pub truncated: bool,
}

impl Buffer {
    /// Retain one parsed delta unless it would exceed the attempt bound.
    pub(crate) fn push(&mut self, delta: ChoiceLogprobsDelta) {
        if self.truncated {
            return;
        }
        let bytes = delta.retained_bytes();
        if self.bytes.saturating_add(bytes) > MAXIMUM_CAPTURED_LOGPROB_BYTES {
            self.truncated = true;
            return;
        }
        self.bytes += bytes;
        self.deltas.push(delta);
    }

    /// Hand over the retained deltas; the byte bound stays cumulative for the attempt.
    pub(crate) fn take(&mut self) -> (Vec<ChoiceLogprobsDelta>, bool) {
        (std::mem::take(&mut self.deltas), self.truncated)
    }
}

/// The capture record's probabilities: present only when the gateway requested
/// them, so a caller-requested answer (already in the response frames) and every
/// record without injection serialize exactly as before.
#[derive(Clone, Debug, Default, Deserialize, Serialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub(crate) struct Captured {
    /// Always true: the caller never asked for these probabilities.
    pub logprobs_injected: bool,
    /// A bound (the attempt's or the collector's) stopped retaining them.
    pub truncated: bool,
    pub content: Vec<TokenLogprob>,
    pub refusal: Vec<TokenLogprob>,
    /// Retained-size estimate charged against the collector's budgets.
    #[serde(skip)]
    pub bytes: usize,
}

impl Captured {
    pub(crate) fn injected() -> Self {
        Self {
            logprobs_injected: true,
            ..Self::default()
        }
    }

    pub(crate) fn heap_bytes(&self) -> usize {
        std::mem::size_of::<Self>() + self.bytes
    }

    /// Append deltas while `allowed(extra)` admits their size; otherwise mark truncated.
    pub(crate) fn extend(
        &mut self,
        deltas: Vec<ChoiceLogprobsDelta>,
        truncated: bool,
        mut allowed: impl FnMut(usize) -> bool,
    ) -> usize {
        let mut added = 0usize;
        self.truncated |= truncated;
        for delta in deltas {
            if self.truncated {
                break;
            }
            let bytes = delta.retained_bytes();
            if !allowed(added.saturating_add(bytes)) {
                self.truncated = true;
                break;
            }
            added += bytes;
            if let Some(logprobs) = delta.logprobs {
                self.content.extend(logprobs.content.unwrap_or_default());
                self.refusal.extend(logprobs.refusal.unwrap_or_default());
            }
        }
        self.bytes += added;
        added
    }
}

#[cfg(test)]
#[path = "logprobs_test.rs"]
mod tests;
