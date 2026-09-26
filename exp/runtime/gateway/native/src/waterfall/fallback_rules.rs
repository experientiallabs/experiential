//! Per-rung conditional failover ("fallback rules"): the data-plane half of
//! `failover_only_on`.
//!
//! A rung authored with `failover_only_on` is FAILOVER-ONLY: it is never the
//! first dial and it serves as a successor only when the failure the ladder
//! is walking from spells one of its tokens (a failover-eligible failure
//! class by wire name, `refusal` for any provider refusal, or
//! `refusal:<reason>` for one bounded category). A rung with no set is
//! unrestricted and behaves exactly as before. The reference case is a
//! customer's own OpenAI key enrolled in a trusted-access program: house
//! rungs take every request, and only a house rung's `refusal:cyber_policy`
//! sends the request on to the customer's key.
//!
//! The control plane chooses the depth (`native_fallback_rules.py` mirrors
//! these rules and records `fallback_reason = failover_only_on:<token>`);
//! this module decides whether a successor is possible at all before the
//! bridge is asked, whether refusal deltas are worth withholding for a rule
//! rung downstream, and whether the depth the control plane answered honors
//! the rules, failing the request closed when it does not.

use super::wire::DeploymentWire;
use crate::errors::{Failure, FailureClass, RefusalReason};

/// The closed token vocabulary, verbatim the python `FAILOVER_TOKENS`
/// (`exp/common/models/failover_tokens.py`, whose test pins this list). A
/// token outside it never matches, so a set can only promise failovers the
/// waterfall performs: the caller's own errors and the gateway's budget
/// refusals are not tokens.
pub(crate) const FAILOVER_TOKENS: &[&str] = &[
    "throttled",
    "timeout",
    "transport",
    "provider_internal",
    "provider_quota",
    "provider_authentication",
    "provider_not_found",
    "unavailable",
    "empty_completion",
    "malformed_response",
    "guardrail",
    "refusal",
    "refusal:cyber_policy",
    "refusal:cbrn",
    "refusal:content_policy",
    "refusal:recitation",
    "refusal:data_inspection",
    "refusal:unspecified",
];

/// The bare refusal token: matches a provider refusal of any reason.
const REFUSAL_TOKEN: &str = "refusal";

/// The token one failure spells: its class wire name, or `refusal:<reason>`
/// (an unnamed refusal spells `refusal:unspecified`, so a rung that opted
/// into one category never takes a refusal the provider filed under none).
pub(crate) fn failure_token(failure: &Failure) -> String {
    match failure.failure_class {
        FailureClass::Refusal => format!(
            "refusal:{}",
            failure
                .refusal_reason
                .unwrap_or(RefusalReason::Unspecified)
                .as_str()
        ),
        class => class.as_str().to_string(),
    }
}

/// How one rung stands toward one failure.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum RungEligibility {
    /// No set authored: the rung follows the route policy as always.
    Unrestricted,
    /// The rung's set names this failure: eligible whatever the policy says.
    Matched,
    /// The rung's set does not name this failure: never dialed for it.
    Excluded,
}

/// Judge `wire` as a successor for `failure`.
pub(crate) fn rung_eligibility(wire: &DeploymentWire, failure: &Failure) -> RungEligibility {
    let Some(tokens) = wire.failover_only_on.as_deref() else {
        return RungEligibility::Unrestricted;
    };
    let token = failure_token(failure);
    let known = FAILOVER_TOKENS.contains(&token.as_str());
    let named = tokens.iter().any(|candidate| candidate == &token);
    let any_refusal = failure.failure_class == FailureClass::Refusal
        && tokens.iter().any(|candidate| candidate == REFUSAL_TOKEN);
    if known && (named || any_refusal) {
        RungEligibility::Matched
    } else {
        RungEligibility::Excluded
    }
}

/// Whether some rung after `depth` may take `failure`: an unrestricted rung
/// when `unrestricted_advance` (the route policy's own verdict on the
/// failure) holds, or any rung whose set names the failure. A route with no
/// sets reduces to `depth + 1 < route.len() && unrestricted_advance`.
pub(crate) fn successor_available(
    route: &[DeploymentWire],
    depth: usize,
    failure: &Failure,
    unrestricted_advance: bool,
) -> bool {
    route
        .iter()
        .skip(depth + 1)
        .any(|wire| match rung_eligibility(wire, failure) {
            RungEligibility::Unrestricted => unrestricted_advance,
            RungEligibility::Matched => true,
            RungEligibility::Excluded => false,
        })
}

/// Whether the rung at `depth` should withhold refusal DELTAS for a rule rung
/// downstream even though the alias revision did not opt into refusal
/// failover. Refusal text carries no bounded reason, so only a later rung
/// that accepts an unnamed refusal (`refusal` or `refusal:unspecified`) can
/// ever take it; a rung restricted to `refusal:cyber_policy` leaves the
/// stream's refusal text flowing to the caller exactly as today.
pub(crate) fn refusal_deltas_withheld_for(route: &[DeploymentWire], depth: usize) -> bool {
    let unnamed = Failure::new(FailureClass::Refusal, "provider refused the request");
    successor_available(route, depth, &unnamed, false)
}

/// Whether the depth the control plane reserved honors the rules: a first
/// dial (`current_depth` is `None`) or a change of depth may land on a
/// restricted rung only when the failure being failed over from matches its
/// set; a redial of the same depth is always the rung's own business. A
/// violation is a wire-contract failure the waterfall answers internal.
pub(crate) fn dial_admitted(
    wire: &DeploymentWire,
    current_depth: Option<usize>,
    depth: usize,
    last_failure: Option<&Failure>,
) -> bool {
    if wire.failover_only_on.is_none() || current_depth == Some(depth) {
        return true;
    }
    match (current_depth, last_failure) {
        (Some(_), Some(failure)) => rung_eligibility(wire, failure) == RungEligibility::Matched,
        _ => false,
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use serde_json::Value;

    use super::*;

    fn rung(tokens: Option<&[&str]>) -> DeploymentWire {
        DeploymentWire {
            native_tool_translation: Default::default(),
            provider: "openai".to_string(),
            deployment_id: "d".to_string(),
            exact_model_id: "fixture-model".into(),
            dialect: "openai_compatible".to_string(),
            url: "https://provider.test".to_string(),
            headers: HashMap::new(),
            model_id: String::new(),
            billing_customer_managed: tokens.is_some(),
            timeout_seconds: 60.0,
            upstream_payload: Value::Null,
            upstream_body: None,
            fireworks_reasoning_route_sha256: None,
            hunyuan_reasoning_route_sha256: None,
            reasoning_output_exposed: false,
            stop_sequences: Vec::new(),
            serialize_tool_calls: false,
            image_output: false,
            idempotency_key: "op".to_string(),
            time_to_first_byte_base_seconds: None,
            time_to_first_byte_seconds_per_million_input_tokens: None,
            time_to_first_token_base_seconds: None,
            throttle_redial_budget: 0,
            throttle_redial: None,
            failover_only_on: tokens.map(|set| set.iter().map(|t| t.to_string()).collect()),
            zdr_constrained: false,
        }
    }

    fn cyber() -> Failure {
        Failure::refusal(RefusalReason::CyberPolicy)
    }

    fn throttled() -> Failure {
        Failure::new(FailureClass::Throttled, "throttled").with_retry(false, true)
    }

    #[test]
    fn every_token_is_a_wire_name_the_waterfall_can_fail_over_from() {
        let classes: Vec<&str> = FAILOVER_TOKENS
            .iter()
            .copied()
            .filter(|token| !token.starts_with("refusal"))
            .collect();
        for class in [
            FailureClass::Throttled,
            FailureClass::Timeout,
            FailureClass::Transport,
            FailureClass::ProviderInternal,
            FailureClass::ProviderQuota,
            FailureClass::ProviderAuthentication,
            FailureClass::ProviderNotFound,
            FailureClass::Unavailable,
            FailureClass::EmptyCompletion,
            FailureClass::MalformedResponse,
            FailureClass::Guardrail,
        ] {
            assert!(classes.contains(&class.as_str()), "{class:?}");
        }
        assert_eq!(classes.len(), 11);
        for reason in [
            RefusalReason::CyberPolicy,
            RefusalReason::Cbrn,
            RefusalReason::ContentPolicy,
            RefusalReason::Recitation,
            RefusalReason::DataInspection,
            RefusalReason::Unspecified,
        ] {
            let token = format!("refusal:{}", reason.as_str());
            assert!(FAILOVER_TOKENS.contains(&token.as_str()), "{token}");
        }
        assert!(FAILOVER_TOKENS.contains(&"refusal"));
        assert_eq!(FAILOVER_TOKENS.len(), 18);
        // Never tokens: the caller's own errors and the gateway's budget verdict.
        for class in [
            FailureClass::InvalidRequest,
            FailureClass::QuotaExceeded,
            FailureClass::Internal,
        ] {
            assert!(!FAILOVER_TOKENS.contains(&class.as_str()));
        }
    }

    #[test]
    fn failure_tokens_spell_the_class_or_the_refusal_reason() {
        assert_eq!(failure_token(&throttled()), "throttled");
        assert_eq!(failure_token(&cyber()), "refusal:cyber_policy");
        let unnamed = Failure::new(FailureClass::Refusal, "provider refused the request");
        assert_eq!(failure_token(&unnamed), "refusal:unspecified");
    }

    #[test]
    fn a_rule_rung_matches_its_tokens_and_the_bare_refusal_matches_any_reason() {
        let exact = rung(Some(&["refusal:cyber_policy"]));
        assert_eq!(rung_eligibility(&exact, &cyber()), RungEligibility::Matched);
        assert_eq!(
            rung_eligibility(&exact, &Failure::refusal(RefusalReason::Cbrn)),
            RungEligibility::Excluded
        );
        assert_eq!(
            rung_eligibility(&exact, &throttled()),
            RungEligibility::Excluded
        );
        let any = rung(Some(&["refusal", "throttled"]));
        assert_eq!(rung_eligibility(&any, &cyber()), RungEligibility::Matched);
        assert_eq!(
            rung_eligibility(&any, &throttled()),
            RungEligibility::Matched
        );
        assert_eq!(
            rung_eligibility(&any, &Failure::new(FailureClass::Timeout, "t")),
            RungEligibility::Excluded
        );
        assert_eq!(
            rung_eligibility(&rung(None), &throttled()),
            RungEligibility::Unrestricted
        );
        // An authored token outside the vocabulary never matches anything.
        let bogus = rung(Some(&["invalid_request"]));
        let invalid = Failure::new(FailureClass::InvalidRequest, "bad");
        assert_eq!(
            rung_eligibility(&bogus, &invalid),
            RungEligibility::Excluded
        );
        // An empty set is a rung that is never dialed at all.
        assert_eq!(
            rung_eligibility(&rung(Some(&[])), &cyber()),
            RungEligibility::Excluded
        );
    }

    #[test]
    fn a_successor_is_available_only_where_the_rules_or_the_policy_admit_it() {
        let route = [
            rung(None),
            rung(Some(&["refusal:cyber_policy"])),
            rung(None),
        ];
        // A refusal the policy would not advance still reaches the rule rung.
        assert!(successor_available(&route, 0, &cyber(), false));
        // A throttle skips the rule rung but the policy admits the plain rung.
        assert!(successor_available(&route, 0, &throttled(), true));
        // With nothing unrestricted left and no match, the ladder is over.
        let tail = [rung(None), rung(Some(&["refusal:cyber_policy"]))];
        assert!(!successor_available(&tail, 0, &throttled(), true));
        assert!(successor_available(&tail, 0, &cyber(), false));
        // An unrestricted route reduces to the historical bound.
        let plain = [rung(None), rung(None)];
        assert!(successor_available(&plain, 0, &throttled(), true));
        assert!(!successor_available(&plain, 0, &throttled(), false));
        assert!(!successor_available(&plain, 1, &throttled(), true));
        // Every rung restricted: nothing after depth 0 takes a throttle.
        let all = [rung(Some(&["refusal"])), rung(Some(&["refusal"]))];
        assert!(!successor_available(&all, 0, &throttled(), true));
    }

    #[test]
    fn refusal_deltas_are_withheld_only_for_a_rung_taking_unnamed_refusals() {
        let cyber_only = [rung(None), rung(Some(&["refusal:cyber_policy"]))];
        assert!(!refusal_deltas_withheld_for(&cyber_only, 0));
        let any = [rung(None), rung(Some(&["refusal"]))];
        assert!(refusal_deltas_withheld_for(&any, 0));
        let unnamed = [rung(None), rung(Some(&["refusal:unspecified"]))];
        assert!(refusal_deltas_withheld_for(&unnamed, 0));
        // The last rung has nothing downstream to withhold for.
        assert!(!refusal_deltas_withheld_for(&any, 1));
    }

    #[test]
    fn a_reserved_depth_must_honor_the_rules() {
        let plain = rung(None);
        let rule = rung(Some(&["refusal:cyber_policy"]));
        // First dial: unrestricted only.
        assert!(dial_admitted(&plain, None, 0, None));
        assert!(!dial_admitted(&rule, None, 0, None));
        // Successor: the failure must match the rung's set.
        assert!(dial_admitted(&rule, Some(0), 1, Some(&cyber())));
        assert!(!dial_admitted(&rule, Some(0), 1, Some(&throttled())));
        assert!(dial_admitted(&plain, Some(0), 1, Some(&throttled())));
        // A same-depth redial on a rule rung is that rung's own business.
        assert!(dial_admitted(&rule, Some(1), 1, Some(&throttled())));
    }
}
