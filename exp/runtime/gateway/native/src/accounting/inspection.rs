//! Ordered output policies and deterministic streaming, without interpreter I/O.

use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, Mutex, OnceLock};
use std::time::{Duration, Instant};

use serde::Deserialize;
use serde_json::{json, Value};

use crate::errors::PublicError;
use crate::guardrails::detector::{Detector, DetectorSpec};

#[derive(Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct Check {
    #[serde(deserialize_with = "crate::lossless_json::deserialize")]
    pub(super) policy: Value,
    #[serde(deserialize_with = "crate::lossless_json::deserialize")]
    pub(super) check: Value,
    #[serde(deserialize_with = "crate::lossless_json::deserialize")]
    pub(super) adapter: Value,
}

pub(super) struct Inspection {
    checks: Vec<(Check, Option<Arc<Detector>>)>,
    deadline: Instant,
}

pub(super) fn decision(action: &str, check: Option<&str>, class: &str, message: &str) -> Value {
    let mut details = json!({"action": action});
    if let Some(id) = check {
        details["check_id"] = json!(id);
    }
    json!({"action": action, "failure": {"failure_class": class, "safe_message": message,
        "safe_details": details, "retryable_same_deployment": false, "failover_eligible": false}})
}
pub(super) fn failed(action: &str, check: Option<&str>) -> Value {
    decision(
        action,
        check,
        "guardrail",
        if action == "block" {
            "The request was blocked by a gateway guardrail."
        } else {
            "A gateway guardrail could not complete this request."
        },
    )
}
pub(super) fn uncertain(check: &Check) -> Option<Value> {
    (check.policy["protected"] == true).then(|| {
        decision(
            "error",
            check.check["check_id"].as_str(),
            "unavailable",
            "Content inspection is unavailable. Retry later.",
        )
    })
}
pub(super) fn cached_detector(value: &Value) -> Option<Arc<Detector>> {
    static CACHE: OnceLock<Mutex<DetectorCache>> = OnceLock::new();
    let key = value.to_string();
    if let Some(detector) = CACHE.get_or_init(Mutex::default).lock().ok()?.get(&key) {
        return Some(Arc::new(detector.as_ref().clone()));
    }
    // Expensive compilation never holds the shared cache lock.
    let spec: DetectorSpec = crate::lossless_json::from_value(value.clone()).ok()?;
    let detector = Arc::new(Detector::compile(&spec).ok()?);
    CACHE.get()?.lock().ok()?.insert(key, detector.clone());
    // The cache retains immutable compiled programs. Request-local clones own
    // their bounded mutable search workspace, which is freed with the check.
    Some(Arc::new(detector.as_ref().clone()))
}

#[derive(Default)]
struct DetectorCache {
    entries: VecDeque<(String, Arc<Detector>)>,
    bytes: usize,
}
impl DetectorCache {
    const MAX_BYTES: usize = 16 * 1024 * 1024;
    const MAX_ENTRIES: usize = 64;
    fn get(&mut self, key: &str) -> Option<Arc<Detector>> {
        let index = self.entries.iter().position(|(k, _)| k == key)?;
        let entry = self.entries.remove(index)?;
        let detector = entry.1.clone();
        self.entries.push_back(entry);
        Some(detector)
    }
    fn insert(&mut self, key: String, detector: Arc<Detector>) {
        if self.get(&key).is_some() {
            return;
        }
        let bytes = key.len() + detector.cache_bytes();
        if bytes > Self::MAX_BYTES {
            return;
        }
        while self.bytes + bytes > Self::MAX_BYTES || self.entries.len() >= Self::MAX_ENTRIES {
            let Some((old_key, old_detector)) = self.entries.pop_front() else {
                break;
            };
            self.bytes -= old_key.len() + old_detector.cache_bytes();
        }
        self.bytes += bytes;
        self.entries.push_back((key, detector));
    }
}

impl Inspection {
    pub fn new(checks: &[Check], deadline: Instant) -> Self {
        Self {
            checks: checks
                .iter()
                .map(|c| {
                    (
                        c.clone(),
                        if c.adapter["kind"] == "regex" {
                            cached_detector(&c.adapter["specification"])
                        } else {
                            None
                        },
                    )
                })
                .collect(),
            deadline,
        }
    }

    pub fn output(&self, data: &Value) -> Result<Value, PublicError> {
        let original = completion(data)?;
        let mut current = original.clone();
        let mut approved = Vec::new();
        for (index, _) in self.checks.iter().enumerate() {
            match self.run(index, &current) {
                Ok(next) => {
                    current = next;
                    approved.push((index, current.clone()));
                }
                Err(decision) => return Ok(decision),
            }
        }
        // Each earlier check must still accept the exact final subject. Replaying
        // the chain wholesale could conceal a later rewrite undoing a prior one.
        for (index, prior) in approved {
            if prior == current {
                continue;
            }
            match self.run(index, &current) {
                Err(decision) => return Ok(decision),
                Ok(value) if value != current => {
                    return Ok(failed(
                        "block",
                        self.checks[index].0.check["check_id"].as_str(),
                    ))
                }
                _ => (),
            }
        }
        Ok(if current == original {
            json!({"action": "allow"})
        } else {
            json!({"action": "modify", "replacement_text": current["text"]})
        })
    }

    fn run(&self, index: usize, subject: &Value) -> Result<Value, Value> {
        let (c, detector) = &self.checks[index];
        let id = c.check["check_id"].as_str();
        let max = c.policy["max_response_bytes"].as_u64().unwrap_or(0);
        if serde_json::to_vec(subject).map_or(true, |v| v.len() as u64 > max) {
            return Err(failed("error", None));
        }
        let started = Instant::now();
        let budget = Duration::from_millis(c.check["timeout_ms"].as_u64().unwrap_or(0))
            .min(self.deadline.saturating_duration_since(started));
        let inspected = if budget.is_zero() {
            Err(())
        } else {
            self.inspect(c, detector.as_deref(), subject, budget)
        };
        let inspected = if started.elapsed() > budget {
            Err(())
        } else {
            inspected
        };
        let verdict = match inspected {
            Ok(v) => v,
            Err(_) => return uncertain(c).map_or_else(|| Ok(subject.clone()), Err),
        };
        if verdict["coverage_error"] == true {
            return Err(decision("error", id, "unsupported_capability",
                "Content inspection does not support this complete request. Use a supported context or modality."));
        }
        if verdict["flagged"] != true {
            return Ok(subject.clone());
        }
        match c.check["action"].as_str() {
            Some("allow") => Ok(subject.clone()),
            Some("modify") => {
                if subject["tool_calls"]
                    .as_array()
                    .is_none_or(|a| !a.is_empty())
                {
                    return Err(failed("block", id));
                }
                let Some(text) = verdict["replacement_text"].as_str() else {
                    return Err(failed("error", id));
                };
                Ok(json!({"text": text, "refusal": false, "tool_calls": [], "context": []}))
            }
            Some("block") => Err(failed("block", id)),
            _ => Err(failed("error", id)),
        }
    }

    fn inspect(
        &self,
        c: &Check,
        detector: Option<&Detector>,
        subject: &Value,
        budget: Duration,
    ) -> Result<Value, ()> {
        match c.adapter["kind"].as_str() {
            Some("coverage_error") => Ok(json!({"coverage_error": true})),
            Some("regex") => {
                let detector = detector.ok_or(())?;
                let text = subject["text"].as_str().ok_or(())?;
                let rewritten = detector.redact(text).map_err(|_| ())?;
                let mut flagged = rewritten.is_some();
                for context in subject["context"].as_array().ok_or(())? {
                    flagged |= detector
                        .matches(context.as_str().ok_or(())?)
                        .map_err(|_| ())?;
                }
                for call in subject["tool_calls"].as_array().ok_or(())? {
                    flagged |= detector
                        .matches(call["arguments"].as_str().ok_or(())?)
                        .map_err(|_| ())?;
                }
                let replacement = (flagged && c.check["action"] == "modify")
                    .then(|| rewritten.unwrap_or_else(|| text.to_string()));
                Ok(json!({"flagged": flagged, "replacement_text": replacement}))
            }
            Some("keyword") => {
                let mut parts = vec![subject["text"].as_str().ok_or(())?];
                for v in subject["context"].as_array().ok_or(())? {
                    parts.push(v.as_str().ok_or(())?);
                }
                for v in subject["tool_calls"].as_array().ok_or(())? {
                    parts.push(v["arguments"].as_str().ok_or(())?);
                }
                let text = fold(&parts.join("\n"));
                let flagged = c.adapter["needles"]
                    .as_array()
                    .ok_or(())?
                    .iter()
                    .any(|n| n.as_str().is_some_and(|n| text.contains(n)));
                Ok(json!({"flagged": flagged}))
            }
            Some("http_json") => {
                let payload = json!({"action": c.check["action"], "capability": c.check["capability"],
                    "check_id": c.check["check_id"], "stage": "output", "completion": subject});
                let verdict = super::inspection_http::inspect(&c.adapter, payload, budget)?;
                validate_verdict(verdict, c.check["action"].as_str().ok_or(())?)
            }
            _ => Err(()),
        }
    }

    pub fn segment(&self, data: &Value) -> Result<Value, PublicError> {
        let Some((c, Some(detector))) = self.checks.first().filter(|_| self.checks.len() == 1)
        else {
            return Ok(failed("error", None));
        };
        if c.check["action"] != "modify"
            || c.adapter["specification"]["patterns"]
                .as_array()
                .is_none_or(|p| !p.is_empty())
        {
            return Ok(failed("error", None));
        }
        let pending = data["pending"].as_str().ok_or_else(PublicError::internal)?;
        let settled = data["settled_bytes"]
            .as_u64()
            .ok_or_else(PublicError::internal)?;
        if settled.saturating_add(pending.len() as u64)
            > c.policy["max_response_bytes"].as_u64().unwrap_or(0)
        {
            return Ok(failed("error", None));
        }
        let start = Instant::now();
        let budget = Duration::from_millis(c.check["timeout_ms"].as_u64().unwrap_or(0))
            .min(self.deadline.saturating_duration_since(start));
        if budget.is_zero() {
            return Ok(failed("error", None));
        }
        let boundary = if data["final"] == true {
            pending.len()
        } else {
            boundary(pending, &c.adapter)?
        };
        let rewritten = detector.redact(&pending[..boundary]);
        if start.elapsed() > budget || rewritten.is_err() {
            return Ok(failed("error", c.check["check_id"].as_str()));
        }
        let rewritten = rewritten.unwrap();
        Ok(
            json!({"action": "allow", "release": rewritten.as_deref().unwrap_or(&pending[..boundary]),
            "pending": &pending[boundary..], "flagged": rewritten.is_some()}),
        )
    }
}

fn completion(data: &Value) -> Result<Value, PublicError> {
    let context = data.get("context").cloned().unwrap_or_else(|| json!([]));
    if context
        .as_array()
        .is_none_or(|a| a.iter().any(|v| !v.is_string()))
    {
        return Err(PublicError::internal());
    }
    let mut calls = Vec::new();
    if let Some(raw) = data["tool_calls"].as_array() {
        for call in raw {
            let id = call["call_id"]
                .as_str()
                .filter(|s| !s.is_empty())
                .or_else(|| call["id"].as_str());
            let args = call["arguments"]
                .as_str()
                .filter(|s| !s.is_empty())
                .or_else(|| call["raw_arguments"].as_str())
                .unwrap_or("");
            if let (Some(id), Some(name)) = (id, call["name"].as_str()) {
                if id.is_empty()
                    || id.chars().count() > 65_536
                    || name.is_empty()
                    || name.chars().count() > 256
                    || args.chars().count() > 4_000_000
                {
                    return Err(PublicError::internal());
                }
                calls.push(json!({"call_id": id, "name": name, "arguments": args}));
            }
        }
    }
    Ok(
        json!({"text": data["text"].as_str().unwrap_or(""), "refusal": data["refusal"].as_bool().unwrap_or(false), "tool_calls": calls, "context": context}),
    )
}

pub(super) fn validate_verdict(mut v: Value, action: &str) -> Result<Value, ()> {
    if v.as_object().is_none_or(|m| {
        m.keys().any(|k| {
            !matches!(
                k.as_str(),
                "flagged" | "replacement_text" | "replacement_messages"
            )
        })
    }) {
        return Err(());
    }
    let flagged = match &v["flagged"] {
        Value::Bool(b) => *b,
        Value::Number(n) if n.as_u64().is_some_and(|n| n <= 1) => n.as_u64() == Some(1),
        Value::String(s) => match s.to_lowercase().as_str() {
            "1" | "true" | "t" | "on" | "yes" | "y" => true,
            "0" | "false" | "f" | "off" | "no" | "n" => false,
            _ => return Err(()),
        },
        _ => return Err(()),
    };
    let has_text = !v["replacement_text"].is_null();
    let has_messages = !v["replacement_messages"].is_null();
    if has_messages
        || (has_text && !v["replacement_text"].is_string())
        || (has_text && (!flagged || action != "modify"))
        || (flagged && action == "modify" && !has_text)
    {
        return Err(());
    }
    v["flagged"] = json!(flagged);
    Ok(v)
}

pub(super) fn fold(text: &str) -> String {
    static FOLD: OnceLock<HashMap<char, String>> = OnceLock::new();
    let map = FOLD.get_or_init(|| {
        crate::lossless_json::from_str(include_str!("../generated/casefold.json"))
            .expect("generated Unicode casefold table")
    });
    let mut out = String::with_capacity(text.len());
    for c in text.chars() {
        if let Some(folded) = map.get(&c) {
            out.push_str(folded);
        } else {
            out.push(c);
        }
    }
    out
}

fn boundary(text: &str, adapter: &Value) -> Result<usize, PublicError> {
    let window = adapter["stream_window"]
        .as_u64()
        .filter(|v| *v > 0)
        .ok_or_else(PublicError::internal)? as usize;
    let chars: Vec<(usize, char)> = text.char_indices().collect();
    let floor = chars.len().saturating_sub(window);
    let mut boundary = text.len();
    for family in adapter["specification"]["builtin_patterns"]
        .as_array()
        .ok_or_else(PublicError::internal)?
    {
        let alphabet = match family.as_str() {
            Some("email") => ".!#$%&'*+/=?^_`{|}~-@",
            Some("credit_card") => " -",
            Some("api_key") => "_-",
            _ => return Err(PublicError::internal()),
        };
        let mut i = chars.len();
        while i > floor
            && (chars[i - 1].1.is_ascii_digit()
                || (family != "credit_card" && chars[i - 1].1.is_ascii_alphabetic())
                || alphabet.contains(chars[i - 1].1))
        {
            i -= 1;
        }
        let at = if i == floor && floor > 0 {
            0
        } else {
            chars.get(i).map_or(text.len(), |v| v.0)
        };
        boundary = boundary.min(at);
    }
    Ok(boundary)
}

#[cfg(test)]
mod cache_memory_tests {
    use super::*;

    #[test]
    fn detector_cache_evicts_by_program_memory_and_preserves_recent_entries() {
        let spec = DetectorSpec {
            patterns: vec!["a".into(); 16],
            builtin_patterns: vec![],
            replacement: "x".into(),
        };
        let detector = Arc::new(Detector::compile(&spec).unwrap());
        let mut cache = DetectorCache::default();
        for key in ["a", "b", "c"] {
            cache.insert(key.into(), detector.clone());
        }
        assert!(cache.get("a").is_some());
        cache.insert("d".into(), detector);
        assert!(cache.get("b").is_none());
        assert!(cache.get("a").is_some());
        assert!(cache.bytes <= DetectorCache::MAX_BYTES);
        assert!(cache.entries.len() <= DetectorCache::MAX_ENTRIES);
        let invalid = DetectorSpec {
            patterns: vec![],
            builtin_patterns: vec!["email".into(); 4],
            replacement: "x".into(),
        };
        assert!(Detector::compile(&invalid).is_err());
    }
}
