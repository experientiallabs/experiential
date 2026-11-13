//! Ordered input reinspection after a tool round, separate from initial billing approval.

use std::time::{Duration, Instant};

use serde_json::{json, Value};
use sha2::{Digest, Sha256};

use super::{
    input, input_rewrite, input_window,
    inspection::{self, Check},
    search_subject::Subject,
};

fn coverage(check: Option<&str>) -> Value {
    inspection::decision("error", check, "unsupported_capability",
        "Content inspection does not support this complete request. Use a supported context or modality.")
}

pub(super) fn prepare(check: &Check, subject: &Subject) -> Result<Check, ()> {
    let mut c = check.clone();
    match c.adapter["kind"].as_str() {
        Some("keyword" | "regex") => {
            c.adapter["subjects"] = json!(subject.subjects()?);
            if c.adapter["kind"] == "regex" && c.check["action"] == "modify" {
                c.adapter["immutable_context"] = json!(subject.context()?);
                c.adapter["redaction_messages"] = subject.request["messages"].clone();
            }
        }
        Some("http_json") => {
            c.adapter["input_template"] = json!(true);
            c.adapter["payload"] = json!({"action": c.check["action"], "capability": c.check["capability"],
                "check_id": c.check["check_id"], "stage": c.check["stage"], "request": subject.visible});
        }
        Some("decision_model") => {
            let state = input_window::window(&subject.request, &c.adapter["window"])?;
            if state.is_empty() {
                c.adapter = json!({"kind": "allow"});
            } else {
                let digest = c.adapter["configuration_digest"].as_str().ok_or(())?;
                c.adapter["cache_key"] = json!(format!(
                    "{:x}",
                    Sha256::digest(format!("{digest}\n{state}").as_bytes())
                ));
                c.adapter["subject_sha256"] =
                    json!(format!("{:x}", Sha256::digest(state.as_bytes())));
                c.adapter["payload"]["state"] = json!(state);
            }
        }
        _ => (),
    }
    Ok(c)
}

pub(super) fn read_only(checks: &[Check], subject: &Subject) -> Vec<Check> {
    checks
        .iter()
        .map(|check| {
            if check.adapter["input_limit"]
                .as_u64()
                .is_some_and(|limit| subject.size().map_or(true, |size| size as u64 > limit))
            {
                let mut unavailable = check.clone();
                unavailable.adapter = json!({"kind": "coverage_error", "oversized": true});
                unavailable
            } else {
                prepare(check, subject).unwrap_or_else(|_| {
                    let mut unavailable = check.clone();
                    unavailable.adapter = json!({"kind": "unavailable"});
                    unavailable
                })
            }
        })
        .collect()
}

async fn one(
    check: &Check,
    subject: &Subject,
    deadline: Instant,
    observed: &mut Value,
    classification: &mut Value,
) -> Result<Subject, Value> {
    let id = check.check["check_id"].as_str();
    let unavailable = |failure: Value, observed: &mut Value| {
        *observed = failure.clone();
        if check.policy["input_fail_open"] == true {
            Ok(subject.clone())
        } else {
            Err(failure)
        }
    };
    if check.adapter["input_limit"]
        .as_u64()
        .is_some_and(|limit| subject.size().map_or(true, |size| size as u64 > limit))
    {
        return unavailable(coverage(None), observed);
    }
    let started = Instant::now();
    let budget = Duration::from_millis(check.check["timeout_ms"].as_u64().unwrap_or(0))
        .min(deadline.saturating_duration_since(started));
    if budget.is_zero() {
        *observed = inspection::decision(
            "error",
            id,
            "unavailable",
            "Content inspection is unavailable. Retry later.",
        );
        return inspection::uncertain(check).map_or_else(
            || Ok(subject.clone()),
            |failure| unavailable(failure, observed),
        );
    }
    let check_for_input = prepare(check, subject).ok();
    let result = match &check_for_input {
        Some(prepared) => input::inspect(prepared, budget).await,
        None => Err(()),
    };
    let result = result.and_then(|mut verdict| {
        if started.elapsed() > budget {
            return Err(());
        }
        if check.adapter["kind"] == "http_json" && !verdict["replacement_messages"].is_null() {
            verdict["replacement_messages"] = json!(input_rewrite::hosted(
                subject,
                &verdict["replacement_messages"]
            )?);
        }
        if started.elapsed() > budget {
            return Err(());
        }
        Ok(verdict)
    });
    if let Some(check_for_input) =
        check_for_input.filter(|check| check.adapter["kind"] == "decision_model")
    {
        *classification = json!({"classifier_kind":"decision_model",
            "subject_sha256":check_for_input.adapter["subject_sha256"]});
        if let Ok(verdict) = &result {
            classification["scores"] = verdict["scores"].clone();
            classification["cached"] = verdict["cached"].clone();
        }
    }
    let verdict = match result {
        Err(_) => {
            *observed = inspection::decision(
                "error",
                id,
                "unavailable",
                "Content inspection is unavailable. Retry later.",
            );
            return inspection::uncertain(check).map_or_else(
                || Ok(subject.clone()),
                |failure| unavailable(failure, observed),
            );
        }
        Ok(v) if v["coverage_error"] == true => return unavailable(coverage(id), observed),
        Ok(v) => v,
    };
    if verdict["flagged"] != true || check.check["action"] == "allow" {
        return Ok(subject.clone());
    }
    let action = check.check["action"].as_str().unwrap_or("error");
    if action != "modify" {
        return Err(inspection::failed(action, id));
    }
    let replacement = verdict["replacement_messages"]
        .as_array()
        .cloned()
        .and_then(|m| input_rewrite::restore(subject, m).ok());
    let Some(replacement) = replacement else {
        return unavailable(inspection::failed("error", id), observed);
    };
    let mut result = subject.clone();
    result.set_messages(replacement);
    Ok(result)
}

async fn recorded(
    check: &Check,
    subject: &Subject,
    deadline: Instant,
    reporting: Option<&super::input_reports::Factory>,
) -> Result<Subject, Value> {
    let started = Instant::now();
    let mut observed = Value::Null;
    let mut classification = Value::Null;
    let result = one(check, subject, deadline, &mut observed, &mut classification).await;
    if let Some(reporting) = reporting {
        let mut decision = if !observed.is_null() {
            observed
        } else {
            match &result {
                Ok(next) => {
                    json!({"action":if next.request == subject.request {"allow"} else {"modify"}})
                }
                Err(failure) => failure.clone(),
            }
        };
        if let Some(fields) = classification.as_object() {
            for (key, value) in fields {
                decision[key] = value.clone();
            }
        }
        reporting.decision(check, &decision, started);
    }
    result
}

pub(super) async fn enforce(
    checks: &[Check],
    subject: &Subject,
    deadline: Instant,
    reporting: Option<&super::input_reports::Factory>,
) -> Result<Subject, Value> {
    let mut current = subject.clone();
    let mut approved = Vec::new();
    for (i, check) in checks.iter().enumerate() {
        current = recorded(check, &current, deadline, reporting).await?;
        approved.push((i, current.request.clone()));
    }
    for (i, prior) in approved {
        if prior != current.request
            && recorded(&checks[i], &current, deadline, reporting)
                .await?
                .request
                != current.request
        {
            return Err(inspection::failed(
                "block",
                checks[i].check["check_id"].as_str(),
            ));
        }
    }
    Ok(current)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn prepare_failure_uses_the_same_uncertainty_rule_for_both_execution_modes() {
        let subject: Subject = crate::lossless_json::from_value(json!({
            "request": {"messages": [{"role": "user", "content": "fixture"}]},
            "visible": {}, "message_default": {}, "call_default": {},
            "message_keys": [], "call_keys": [], "tool_keys": [], "definitions": {}
        }))
        .unwrap();
        for protected in [false, true] {
            let check: Check = crate::lossless_json::from_value(json!({
                "policy": {"protected": protected, "mode": "enforce"},
                "check": {"check_id": "check", "action": "block", "timeout_ms": 1000},
                "adapter": {"kind": "decision_model", "window": {}}
            }))
            .unwrap();
            assert!(prepare(&check, &subject).is_err());
            let mut observed = Value::Null;
            let mut classification = Value::Null;
            let result = one(
                &check,
                &subject,
                Instant::now() + Duration::from_secs(1),
                &mut observed,
                &mut classification,
            )
            .await;
            assert_eq!(result.is_err(), protected);
            assert_eq!(observed["failure"]["failure_class"], "unavailable");
            assert!(
                classification.is_null(),
                "preparation is not a classifier attempt"
            );
            let parallel = input::InputGate::start(read_only(&[check], &subject), 1.0)
                .unwrap()
                .wait();
            assert_eq!(parallel["action"] == "error", protected);
            if let Err(error) = result {
                assert_eq!(error["failure"]["failure_class"], "unavailable");
            }
        }
    }

    #[test]
    fn modifying_followups_fail_open_on_missing_coverage_but_keep_known_denials() {
        let runtime = super::super::inspection_http::runtime().unwrap();
        let case: Value = crate::lossless_json::from_str(
            include_str!("../../tests/generated/input_programs.jsonl")
                .lines()
                .next()
                .unwrap(),
        )
        .unwrap();
        let subject: Subject = crate::lossless_json::from_value(case["subject"].clone()).unwrap();
        let mut check: Check = crate::lossless_json::from_value(case["checks"][0].clone()).unwrap();
        check.policy["input_fail_open"] = json!(true);
        check.policy["protected"] = json!(true);
        check.check["action"] = json!("modify");
        for adapter in [
            json!({"kind": "unavailable"}),
            json!({"kind": "coverage_error"}),
            json!({"kind": "allow", "input_limit": 0}),
            json!({"kind": "decision_model"}),
        ] {
            check.adapter = adapter;
            let actual = runtime
                .block_on(one(
                    &check,
                    &subject,
                    Instant::now() + Duration::from_secs(1),
                    &mut Value::Null,
                ))
                .unwrap();
            assert_eq!(actual.request, subject.request);
        }
        let denied_case: Value = crate::lossless_json::from_str(
            include_str!("../../tests/generated/input_programs.jsonl")
                .lines()
                .nth(4)
                .unwrap(),
        )
        .unwrap();
        let subject: Subject =
            crate::lossless_json::from_value(denied_case["subject"].clone()).unwrap();
        let mut check: Check =
            crate::lossless_json::from_value(denied_case["checks"][0].clone()).unwrap();
        check.policy["input_fail_open"] = json!(true);
        let denied = runtime
            .block_on(one(
                &check,
                &subject,
                Instant::now() + Duration::from_secs(1),
                &mut Value::Null,
            ))
            .err()
            .expect("an explicit detected violation still denies the request");
        assert_eq!(denied["action"], "block");
    }

    #[test]
    fn evolving_input_programs_match_python_including_final_revalidation() {
        let runtime = super::super::inspection_http::runtime().unwrap();
        for (i, line) in include_str!("../../tests/generated/input_programs.jsonl")
            .lines()
            .enumerate()
        {
            let case: Value = crate::lossless_json::from_str(line).unwrap();
            let subject = crate::lossless_json::from_value(case["subject"].clone()).unwrap();
            let checks: Vec<Check> =
                crate::lossless_json::from_value(case["checks"].clone()).unwrap();
            let actual = match runtime.block_on(enforce(
                &checks,
                &subject,
                Instant::now() + Duration::from_secs(30),
                None,
            )) {
                Ok(s) => json!({"request": s.request}),
                Err(v) => json!({"failure": v["failure"]}),
            };
            assert_eq!(
                actual["request"], case["expected"]["request"],
                "case {i} request"
            );
            for key in [
                "failure_class",
                "safe_message",
                "safe_details",
                "retryable_same_deployment",
                "failover_eligible",
            ] {
                assert_eq!(
                    actual["failure"][key], case["expected"]["failure"][key],
                    "case {i} failure {key}"
                );
            }
            assert_ne!(
                actual["failure"]["safe_details"]["input_guardrail_denied"], true,
                "the raw classifier verdict does not carry settlement policy"
            );
        }
    }
}
