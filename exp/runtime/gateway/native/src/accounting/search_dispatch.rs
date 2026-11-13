//! Request-owned tool execution and the next physical dispatch on the same rung.

use serde::Deserialize;
use serde_json::{json, Value};
use sha2::{Digest, Sha256};

use crate::errors::PublicError;

use super::{
    dispatch::{Dispatch, State},
    input_program,
    inspection::Check,
    search_counter::Counter,
    search_round::{self, Search},
    search_subject::{items, Subject},
    search_wire::Plan,
};

#[derive(Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct Configuration {
    search: search_round::Configuration,
    pub plans: Vec<Plan>,
    counter: Counter,
    subject: Subject,
    checks: Vec<Check>,
    #[serde(default)]
    observations: Vec<Check>,
}

pub(super) struct Session {
    plans: Vec<Plan>,
    counter: Counter,
    checks: Vec<Check>,
    observations: Vec<Check>,
    search: Search,
    subject: Subject,
}

impl Session {
    pub fn new(config: Configuration) -> Result<Self, PublicError> {
        super::tokenizer::warm()?;
        let search = Search::new(config.search)?;
        Ok(Self {
            plans: config.plans,
            counter: config.counter,
            checks: config.checks,
            observations: config.observations,
            search,
            subject: config.subject,
        })
    }
}

impl Dispatch {
    pub(super) fn search_round(
        &self,
        input: &Value,
        state: &mut State,
    ) -> Result<Value, PublicError> {
        let depth = input["route_depth"]
            .as_u64()
            .map(|d| d as usize)
            .ok_or_else(PublicError::internal)?;
        if !state
            .active
            .as_ref()
            .and_then(|id| state.attempts.get(id))
            .is_some_and(|a| a.depth == depth && a.terminal.is_none())
        {
            return Err(PublicError::internal());
        }
        let session = state.search.as_mut().ok_or_else(PublicError::internal)?;
        let plan = session.plans.get(depth).ok_or_else(PublicError::internal)?;
        let runtime = super::inspection_http::runtime().map_err(|_| PublicError::internal())?;
        // The strict wait ran without holding dispatch. Recheck after acquiring it
        // so cancellation or a new denial cannot start a paid search.
        let approved = self.resources.status();
        if approved["action"] != "allow" && approved["action"] != "pending" {
            if approved["failure"].is_object() {
                return Ok(json!({"inspection_failure": approved["failure"], "rounds": []}));
            }
            return Err(PublicError::internal());
        }
        if let Some(inspection) = &self.inspection {
            let decision = inspection.output(
                &json!({"tool_calls": input["calls"], "text": "", "context": [], "refusal": false}),
            )?;
            if decision["action"] != "allow" {
                return Ok(json!({"inspection_failure": decision["failure"], "rounds": []}));
            }
        }
        // Preserve an executed search for billing even when its next input is denied.
        let rounds = session.search.perform(&input["calls"])?;
        let tools = session.search.tools();
        let frame = session
            .search
            .frames
            .last()
            .ok_or_else(PublicError::internal)?;
        let mut subject = session.subject.clone();
        subject.append(&tools, frame)?;
        if !session.observations.is_empty() {
            let checks = input_program::read_only(&session.observations, &subject);
            if let Ok(gate) = super::input::InputGate::start(
                checks,
                self.deadline
                    .saturating_duration_since(std::time::Instant::now())
                    .as_secs_f64(),
            ) {
                gate.bind_reporter(self.resources.reporting.as_ref().and_then(|r| r.reporter()));
            }
        }
        let inspected = if session.checks.is_empty() {
            Ok(subject)
        } else if session
            .checks
            .iter()
            .all(|check| check.check["action"] != "modify")
        {
            let checks = input_program::read_only(&session.checks, &subject);
            let gate = super::input::InputGate::start(
                checks,
                self.deadline
                    .saturating_duration_since(std::time::Instant::now())
                    .as_secs_f64(),
            )
            .map_err(|_| PublicError::internal())?;
            gate.bind_reporter(self.resources.reporting.as_ref().and_then(|r| r.reporter()));
            if session
                .checks
                .iter()
                .all(|check| check.policy["input_execution"] == "parallel")
            {
                // The prior search turn settles under its original decision. Activate
                // this inspection only when its successor attempt is reserved.
                self.resources.stage_reinspection(gate)?;
                Ok(subject)
            } else {
                let decision = gate.wait();
                if decision["action"] == "allow" {
                    Ok(subject)
                } else {
                    Err(decision)
                }
            }
        } else {
            runtime.block_on(input_program::enforce(
                &session.checks,
                &subject,
                self.deadline,
                self.resources.reporting.as_ref(),
            ))
        };
        let inspected = match inspected {
            Ok(subject) => subject,
            Err(decision) => {
                self.resources
                    .deny_reinspection(decision["failure"].clone())?;
                return Ok(json!({"inspection_failure": decision["failure"], "rounds": rounds}));
            }
        };
        let wire = plan.render_subject(&tools, &inspected.request)?;
        let (counted, reserved) = session
            .counter
            .count_messages(&tools, items(&inspected.request["messages"]))?;
        if let Some(recovery) = &self.config.recovery {
            state.recovery_key = inspected.recovery_prefix()?.map(|prefix| {
                let mut key = recovery.key.clone();
                key[3] = prefix;
                key
            });
        }
        if let Some(binding) = state
            .features
            .signing
            .get_mut(depth)
            .and_then(Option::as_mut)
        {
            let body = wire["upstream_body"]
                .as_str()
                .ok_or_else(PublicError::internal)?;
            if wire["url"] != binding.url {
                return Err(PublicError::internal());
            }
            binding.body_sha256 = format!("{:x}", Sha256::digest(body.as_bytes()));
        }
        if let Some(authority) = state
            .features
            .reasoning
            .get_mut(depth)
            .and_then(Option::as_mut)
        {
            authority.history_sha256 = history(&inspected)?;
        }
        state.input_tokens = reserved;
        state.counted_input_tokens = counted;
        session.subject = inspected;
        Ok(json!({"wire": wire, "rounds": rounds, "exhausted": session.search.exhausted()}))
    }
}

fn history(subject: &Subject) -> Result<String, PublicError> {
    let mut messages = items(&subject.visible["messages"]).to_vec();
    for (visible, observed) in messages.iter_mut().zip(items(&subject.request["messages"])) {
        if visible["content"].is_null() {
            visible["content"] = json!("");
        }
        let mut blocks = items(&observed["provider_reasoning"]).to_vec();
        for block in &mut blocks {
            if let Some(object) = block.as_object_mut() {
                for key in ["output_index", "status", "carrier_size_bytes"] {
                    object.remove(key);
                }
            }
        }
        visible["provider_reasoning"] = json!(blocks);
    }
    Ok(format!(
        "{:x}",
        Sha256::digest(super::canonical::bytes(&json!(messages), false)?)
    ))
}
