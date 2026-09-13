//! Native execution of a deterministic-only output guardrail chain.
//!
//! The control plane resolves the per-identity policy once at admission and
//! hands the ordered chain over on the admission wire whenever every check
//! binds a deterministic detector. Rust then inspects and redacts the
//! buffered completion in place, so a guarded request pays no python
//! callback, no GIL acquisition, and no JSON round trip of the completion.
//!
//! A chain with any non-deterministic adapter is not planned here: it keeps
//! crossing the python boundary unchanged.
//!
//! This module never logs completions, matches, or replacements.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};

use serde::Deserialize;
use serde_json::{json, Value};

use crate::errors::{Failure, FailureClass};
use crate::events::Event;
use crate::guardrails::detector::Detector;

/// Compiled deterministic detectors, keyed by policy `adapter_id`.
pub type DetectorMap = HashMap<String, Arc<Detector>>;

/// One resolved deterministic check in an output chain.
///
/// `check_id` and `capability` are the authored content-free identities the
/// decision stream names. `timeout_ms` is the authored inspection budget,
/// which bounds this check just as it bounds the python adapter.
#[derive(Debug, Clone, Deserialize)]
pub struct PlanCheck {
    pub action: String,
    pub adapter_id: String,
    #[serde(default)]
    pub check_id: String,
    #[serde(default)]
    pub capability: String,
    pub timeout_ms: u64,
}

/// The resolved deterministic output chain for one admitted request.
#[derive(Debug, Clone, Deserialize)]
pub struct OutputPlan {
    #[serde(default)]
    pub protected: bool,
    pub max_response_bytes: usize,
    #[serde(default)]
    pub policy_id: String,
    #[serde(default)]
    pub organization_id: String,
    #[serde(default)]
    pub identity_id: String,
    pub checks: Vec<PlanCheck>,
}

/// The buffered completion projection an output check inspects.
struct Completion {
    text: String,
    refusal: bool,
    tool_calls: Vec<ToolCall>,
}

/// One completed tool invocation presented to an output check.
struct ToolCall {
    call_id: String,
    name: String,
    arguments: String,
}

/// Build one fail-closed guardrail failure without completion content.
fn error_failure() -> Failure {
    Failure::new(
        FailureClass::Guardrail,
        "A gateway guardrail could not complete this request.",
    )
}

/// Build the blocked-by-policy failure without completion content.
fn block_failure() -> Failure {
    Failure::new(
        FailureClass::Guardrail,
        "The request was blocked by a gateway guardrail.",
    )
}

/// Project collected events into the inspected completion.
fn projection(events: &[Event]) -> Completion {
    let mut completion = Completion {
        text: String::new(),
        refusal: false,
        tool_calls: Vec::new(),
    };
    for event in events {
        match event {
            Event::TextDelta(delta) | Event::ProviderTextDelta { delta, .. } => {
                completion.text.push_str(delta);
            }
            Event::RefusalDelta(_) | Event::ProviderRefusalDelta { .. } => {
                completion.refusal = true;
            }
            Event::ToolCallCompleted { call, .. } => completion.tool_calls.push(ToolCall {
                call_id: call.call_id.clone(),
                name: call.name.clone(),
                arguments: call.raw_arguments.clone(),
            }),
            _ => {}
        }
    }
    completion
}

/// The UTF-8 size of the canonical classifier subject.
///
/// This mirrors `GuardrailCompletion.content_bytes`: deterministic JSON with
/// sorted keys, no insignificant whitespace, and no ASCII escaping, so the
/// native bound admits and rejects exactly what the python bound does.
fn content_bytes(completion: &Completion) -> usize {
    let calls: Vec<Value> = completion
        .tool_calls
        .iter()
        .map(|call| {
            json!({
                "arguments": call.arguments,
                "call_id": call.call_id,
                "name": call.name,
            })
        })
        .collect();
    let subject = json!({
        "refusal": completion.refusal,
        "text": completion.text,
        "tool_calls": calls,
    });
    crate::encode::compact_json(&subject).len()
}

/// Emit one content-free decision line, the native mirror of the python
/// engine's `guardrail decision` record. It carries identities and latency
/// only, never completion text, matches, or replacements.
fn record(plan: &OutputPlan, check: Option<&PlanCheck>, action: &str, elapsed: Duration) {
    let line = json!({
        "event": "guardrail_decision",
        "policy_id": plan.policy_id,
        "organization_id": plan.organization_id,
        "identity_id": plan.identity_id,
        "check_id": check.map(|entry| entry.check_id.as_str()),
        "capability": check.map(|entry| entry.capability.as_str()),
        "action": action,
        "latency_ms": elapsed.as_secs_f64() * 1000.0,
    });
    eprintln!("exp-gateway-native: {line}");
}

/// Apply the fail-closed rule for one uncertain check.
///
/// Returns `Ok(())` when a non-protected identity skips the check and
/// continues the remaining chain.
fn uncertain(plan: &OutputPlan, check: &PlanCheck) -> Result<(), Failure> {
    record(plan, Some(check), "error", Duration::ZERO);
    if plan.protected {
        return Err(error_failure());
    }
    Ok(())
}

/// Run one deterministic output chain over the buffered events.
///
/// `deadline` is the request-wide deadline already in force on this route.
/// An expired deadline is an uncertain check, exactly as in the python
/// chain.
pub fn enforce(
    plan: &OutputPlan,
    detectors: &DetectorMap,
    events: Vec<Event>,
    deadline: Instant,
) -> Result<Vec<Event>, Failure> {
    let completion = projection(&events);
    if content_bytes(&completion) > plan.max_response_bytes {
        record(plan, None, "error", Duration::ZERO);
        return Err(error_failure());
    }
    let mut text = completion.text;
    let mut rewritten = false;
    for check in &plan.checks {
        // The check runs under the tighter of its authored timeout and the
        // remaining request deadline, exactly as the python chain does.
        let budget = Duration::from_millis(check.timeout_ms)
            .min(deadline.saturating_duration_since(Instant::now()));
        if budget == Duration::ZERO {
            uncertain(plan, check)?;
            continue;
        }
        let Some(detector) = detectors.get(&check.adapter_id) else {
            uncertain(plan, check)?;
            continue;
        };
        let started = Instant::now();
        let redacted = match detector.redact(&text) {
            Ok(value) => value,
            Err(_) => {
                uncertain(plan, check)?;
                continue;
            }
        };
        let mut flagged = redacted.is_some();
        let mut limited = false;
        for call in &completion.tool_calls {
            match detector.matches(&call.arguments) {
                Ok(found) => flagged |= found,
                Err(_) => limited = true,
            }
        }
        let elapsed = started.elapsed();
        // A scan that overran its budget is uncertain: an inspection the
        // python chain would have abandoned never returns a verdict here.
        if limited || elapsed > budget {
            uncertain(plan, check)?;
            continue;
        }
        if !flagged {
            record(plan, Some(check), "allow", elapsed);
            continue;
        }
        record(plan, Some(check), &check.action, elapsed);
        match check.action.as_str() {
            "allow" => {}
            "modify" => {
                // Tool-call arguments are inspected but never rewritten, so a
                // completion that carries one cannot be modified.
                if !completion.tool_calls.is_empty() {
                    return Err(block_failure());
                }
                if let Some(value) = redacted {
                    text = value;
                    rewritten = true;
                }
            }
            "block" => return Err(block_failure()),
            _ => return Err(error_failure()),
        }
    }
    if !rewritten {
        return Ok(events);
    }
    Ok(crate::guardrails::apply_text_replacement(&events, &text))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::events::CompletedToolCall;
    use crate::guardrails::detector::DetectorSpec;

    /// Build one detector map with a single email rule under `adapter_id`.
    fn detectors(adapter_id: &str, replacement: &str) -> DetectorMap {
        let detector = Detector::compile(&DetectorSpec {
            patterns: vec![],
            builtin_patterns: vec!["email".to_string()],
            replacement: replacement.to_string(),
        })
        .expect("the rule compiles");
        HashMap::from([(adapter_id.to_string(), Arc::new(detector))])
    }

    /// Build one single-check plan with the given action.
    fn plan(action: &str, protected: bool) -> OutputPlan {
        OutputPlan {
            protected,
            max_response_bytes: 1_048_576,
            policy_id: "policy".to_string(),
            organization_id: "org".to_string(),
            identity_id: "identity".to_string(),
            checks: vec![PlanCheck {
                action: action.to_string(),
                adapter_id: "pii".to_string(),
                check_id: "pii-out".to_string(),
                capability: "pii".to_string(),
                timeout_ms: 5_000,
            }],
        }
    }

    /// One future deadline that never expires during a unit test.
    fn deadline() -> Instant {
        Instant::now() + Duration::from_secs(30)
    }

    /// Build one completed tool call event.
    fn tool_event(arguments: &str) -> Event {
        Event::ToolCallCompleted {
            index: 0,
            call: CompletedToolCall {
                namespace: None,
                caller: None,
                call_id: "call-1".to_string(),
                name: "lookup".to_string(),
                provider_item_id: None,
                provider_status: None,
                raw_arguments: arguments.to_string(),
                custom: false,
            },
        }
    }

    #[test]
    fn a_clean_completion_is_returned_unchanged() {
        let events = vec![Event::TextDelta("all clear".to_string()), Event::Completed];
        let result = enforce(
            &plan("modify", true),
            &detectors("pii", "[REDACTED]"),
            events,
            deadline(),
        )
        .expect("the chain allows");
        assert!(matches!(result[0], Event::TextDelta(ref text) if text == "all clear"));
    }

    #[test]
    fn a_modify_check_redacts_the_buffered_text() {
        let events = vec![
            Event::TextDelta("mail ada@example.com".to_string()),
            Event::Completed,
        ];
        let result = enforce(
            &plan("modify", true),
            &detectors("pii", "[REDACTED]"),
            events,
            deadline(),
        )
        .expect("the chain modifies");
        assert!(matches!(result[0], Event::TextDelta(ref text) if text == "mail [REDACTED]"));
    }

    #[test]
    fn a_block_check_fails_the_request() {
        let events = vec![Event::TextDelta("ada@example.com".to_string())];
        let failure = enforce(
            &plan("block", true),
            &detectors("pii", "[REDACTED]"),
            events,
            deadline(),
        )
        .expect_err("the chain blocks");
        assert_eq!(failure.failure_class, FailureClass::Guardrail);
    }

    #[test]
    fn a_tool_call_match_is_blocked_rather_than_rewritten() {
        let events = vec![
            Event::TextDelta("see attachment".to_string()),
            tool_event("{\"to\":\"ada@example.com\"}"),
        ];
        let failure = enforce(
            &plan("modify", true),
            &detectors("pii", "[REDACTED]"),
            events,
            deadline(),
        )
        .expect_err("a tool completion cannot be rewritten");
        assert_eq!(
            failure.safe_message,
            "The request was blocked by a gateway guardrail."
        );
    }

    #[test]
    fn an_oversized_completion_fails_closed() {
        let events = vec![Event::TextDelta("hello there".to_string())];
        let mut small = plan("modify", false);
        small.max_response_bytes = 8;
        let failure = enforce(&small, &detectors("pii", "[REDACTED]"), events, deadline())
            .expect_err("the subject exceeds the policy bound");
        assert_eq!(
            failure.safe_message,
            "A gateway guardrail could not complete this request."
        );
    }

    #[test]
    fn a_missing_adapter_fails_closed_only_when_protected() {
        let events = vec![Event::TextDelta("ada@example.com".to_string())];
        let empty: DetectorMap = HashMap::new();
        assert!(enforce(&plan("modify", true), &empty, events.clone(), deadline()).is_err());
        let skipped = enforce(&plan("modify", false), &empty, events, deadline())
            .expect("a non protected identity skips the check");
        assert!(matches!(skipped[0], Event::TextDelta(ref text) if text == "ada@example.com"));
    }

    #[test]
    fn an_exhausted_check_timeout_is_an_uncertain_check() {
        let events = vec![Event::TextDelta("ada@example.com".to_string())];
        let mut expired = plan("modify", true);
        expired.checks[0].timeout_ms = 0;
        assert!(enforce(
            &expired,
            &detectors("pii", "[REDACTED]"),
            events,
            deadline(),
        )
        .is_err());
    }

    #[test]
    fn an_expired_deadline_is_an_uncertain_check() {
        let events = vec![Event::TextDelta("ada@example.com".to_string())];
        let expired = Instant::now() - Duration::from_secs(1);
        assert!(enforce(
            &plan("modify", true),
            &detectors("pii", "[REDACTED]"),
            events,
            expired,
        )
        .is_err());
    }
}
