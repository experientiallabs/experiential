//! One physical provider generation dispatch, returning before any successor reservation.

use super::{
    billed_empty_completion, commit, customer_owned, dispatch_headers, empty_completion_failure,
    fallback_rules, first_byte_allowance, first_token_allowance, is_semantic, open_phase_bound,
    settle_output_less, unreported_empty_completion, AttemptEnd, CommittedAttempt, DeploymentWire,
    WaterfallContext, MAXIMUM_WITHHELD_REFUSAL_BYTES, MAXIMUM_WITHHELD_REFUSAL_EVENTS,
};
use crate::dialects::Dialect;
use crate::errors::{Failure, FailureClass};
use crate::events::{Event, Usage};
use crate::rate_limit_headers::harvest_rate_limit_headers;
use crate::relay::{ended_without_terminal, remaining, track_event, UpstreamRelay};
use crate::replay_repair::AttemptRepair;
use crate::settlement::AttemptGuard;
use crate::tool_search::withheld_overflow_failure;
use crate::upstream::open_stream;
use serde_json::Value;
use std::time::{Duration, Instant};

#[path = "google_cache.rs"]
mod google_cache;

pub(super) async fn run_attempt(
    ctx: &WaterfallContext<'_>,
    guard: &mut AttemptGuard,
    wire: &DeploymentWire,
    depth: usize,
    repaired: &mut Option<Value>,
    reactive_repair: bool,
) -> AttemptEnd {
    let Some(dialect) = Dialect::from_str(&wire.dialect) else {
        // Admission validated every dialect; reaching here is wire drift.
        return AttemptEnd::Ladder {
            failure: Failure::new(
                FailureClass::Internal,
                "gateway engine does not support the resolved provider dialect",
            ),
            refusal_eligible: false,
            exhaustion_flush: Vec::new(),
            usage: None,
            tool_names: Vec::new(),
            opened: false,
            encrypted_reasoning_stripped: false,
        };
    };
    // Body-signing dialects sign immediately before every physical attempt
    // so neither queue time nor a spent prior attempt can age the signature
    // (a same-deployment redial or a failover advance both call `run_attempt`
    // again, so each always gets a fresh signature); signing failures are
    // neither same-deployment-retryable nor failover-eligible, matching the
    // python executor's hard stop on an authentication failure.
    let headers = match dispatch_headers(ctx.bridge, ctx.request_id, wire).await {
        Ok(headers) => headers,
        Err(_) => {
            return AttemptEnd::Ladder {
                failure: Failure::new(
                    FailureClass::ProviderAuthentication,
                    "provider dispatch signing failed",
                ),
                refusal_eligible: false,
                exhaustion_flush: Vec::new(),
                usage: None,
                tool_names: Vec::new(),
                opened: false,
                encrypted_reasoning_stripped: false,
            };
        }
    };
    let cached_wire =
        match google_cache::prepare(ctx, wire, reactive_repair || repaired.is_some()).await {
            Ok(wire) => wire,
            Err(failure) => {
                return AttemptEnd::Ladder {
                    failure,
                    refusal_eligible: false,
                    exhaustion_flush: Vec::new(),
                    usage: None,
                    tool_names: Vec::new(),
                    opened: false,
                    encrypted_reasoning_stripped: false,
                };
            }
        };
    let wire = cached_wire.as_ref().unwrap_or(wire);
    // The connection's raw timeout paces each BODY chunk read, exactly like
    // the python streaming path. The open (request/response-header) phase is
    // bounded by the fail-fast time-to-first-byte window alone (fresh per
    // attempt) and the request deadline: a dead lane that never answers is
    // abandoned in seconds, and a deployment whose authored first-byte
    // allowance exceeds the per-chunk timeout (a 120 s header hold on a
    // reasoning lane that thinks before its first header) is honored rather
    // than silently cut at the per-chunk 60 s. Before 0.3.74 the open bound
    // also took the per-chunk timeout, so every authored allowance above it
    // was a no-op: 251 of 251 header timeouts on lanes carrying 90 s and
    // 120 s allowances cut at exactly 60 s (production, 2026-09-16).
    let phase_timeout = Duration::from_secs_f64(wire.timeout_seconds.max(0.001));
    let first_byte_allowance_for = || {
        first_byte_allowance(
            wire,
            ctx.time_to_first_byte,
            ctx.time_to_first_byte_slope_seconds_per_million_input_tokens,
            ctx.approximate_input_tokens,
        )
    };
    let first_byte_deadline = Instant::now() + first_byte_allowance_for();
    // The first-token bound the relay enforces once the headers are in: its
    // own base (thinking models on a chat wire stream nothing for a minute
    // and more), the same input slope, absolute from the same dial.
    let first_token_allowance_for = || {
        first_token_allowance(
            wire,
            ctx.time_to_first_token,
            ctx.time_to_first_byte_slope_seconds_per_million_input_tokens,
            ctx.approximate_input_tokens,
        )
    };
    let first_token_deadline = Instant::now() + first_token_allowance_for();
    // What is already known repairs the first dial: the payload this request
    // stripped on an earlier dial of the rung, else the payloads this worker
    // remembers the caller's provider refusing.
    let mut repair = AttemptRepair::begin(wire, ctx.caller_scope, repaired, ctx.request_id);
    if reactive_repair {
        repair.mark_reactive_successor();
    }
    // A repaired payload returns to the outer loop for its own reservation.
    // The refused dial's usage belongs to this attempt alone.
    {
        let observation = guard.begin_dial_observation();
        let open_bound = open_phase_bound(remaining(ctx.deadline), remaining(first_byte_deadline));
        guard.mark_dispatched();
        let response = match open_stream(
            ctx.http,
            &wire.url,
            &headers,
            &wire.idempotency_key,
            repair.payload(),
            repair.raw_body(),
            open_bound,
            dialect,
        )
        .await
        {
            Ok(response) => response,
            Err(failure) => {
                if repair.repair_after(&failure) {
                    return AttemptEnd::Repair {
                        failure,
                        usage: None,
                        tool_names: Vec::new(),
                    };
                }
                return AttemptEnd::Ladder {
                    failure: customer_owned(failure, wire),
                    refusal_eligible: false,
                    exhaustion_flush: Vec::new(),
                    usage: None,
                    tool_names: Vec::new(),
                    opened: false,
                    encrypted_reasoning_stripped: false,
                };
            }
        };
        repair.dial_opened();
        let encrypted_reasoning_stripped = repair.stripped();
        guard.mark_opened();
        // The opened response's allowlisted rate-limit headers settle with this
        // attempt whatever its terminal outcome; a failed OPEN instead carries
        // them on its failure (attached in `open_stream`).
        guard.record_rate_limit_headers(harvest_rate_limit_headers(response.headers()));
        let mut relay = match wire
            .fireworks_reasoning_route_sha256
            .clone()
            .or_else(|| wire.hunyuan_reasoning_route_sha256.clone())
        {
            Some(route_sha256) => UpstreamRelay::new_with_reasoning_content_route(
                response,
                dialect,
                first_token_deadline,
                Some(route_sha256),
            ),
            None => UpstreamRelay::new(response, dialect, first_token_deadline),
        };
        if wire.image_output {
            relay.allow_image_output();
        }
        relay.set_observation(observation);
        relay.set_stop_sequences(wire.stop_sequences.iter().cloned());
        relay.set_probability_output(ctx.chat_logprobs, &wire.upstream_payload);
        relay.set_serialize_tool_calls(wire.serialize_tool_calls);
        relay.set_native_tool_translation(wire.native_tool_translation.clone());
        relay.set_tool_search_tool_name(ctx.tool_search.map(|search| search.tool_name.clone()));
        if !wire.model_id.is_empty() {
            relay.set_request_words([wire.model_id.clone()]);
        }
        if wire.billing_customer_managed {
            // Applied to every failure the relay yields, before or after commit,
            // so a committed stream's late credential error is the customer's too.
            relay.set_customer_managed_provider(Some(wire.provider.clone()));
        }
        // Refusal deltas are withheld when the alias revision opted into
        // refusal failover, or when a failover-only rung downstream accepts
        // an unnamed refusal (`fallback_rules`); the ladder decision below
        // still distinguishes the two, so the policy alone never advances a
        // refusal onto an unrestricted rung it did not opt into.
        let refusal_failover = ctx.policy.refusal_failover
            || fallback_rules::refusal_deltas_withheld_for(ctx.route, depth);
        // Per dial: tracked facts belong to the dial that produced them; a
        // refused dial's billed usage travels through the relay above.
        let mut usage: Option<Usage> = None;
        let mut tool_names: Vec<String> = Vec::new();
        let mut withheld: Vec<Event> = Vec::new();
        let mut withheld_bytes = 0usize;
        let mut private_reasoning = commit::PrivateReasoning::default();
        loop {
            let event = match relay
                .next_event(ctx.deadline, phase_timeout, guard.started)
                .await
            {
                Ok(Some(event)) => event,
                Ok(None) => {
                    return AttemptEnd::Ladder {
                        failure: ended_without_terminal(),
                        refusal_eligible: false,
                        exhaustion_flush: Vec::new(),
                        usage: relay.usage_before_failure(usage),
                        tool_names,
                        opened: true,
                        encrypted_reasoning_stripped,
                    }
                }
                Err(failure) => {
                    return AttemptEnd::Ladder {
                        failure,
                        refusal_eligible: false,
                        exhaustion_flush: Vec::new(),
                        usage: relay.usage_before_failure(usage),
                        tool_names,
                        opened: true,
                        encrypted_reasoning_stripped,
                    }
                }
            };
            track_event(&event, &mut usage, &mut tool_names);
            guard.record_first_token(relay.first_token_at());
            if private_reasoning.withhold(&event, wire.reasoning_output_exposed) {
                relay.private_progress();
                continue;
            }
            if matches!(event, Event::GeminiThoughtPart(_))
                || crate::logprobs::withhold_before_commit(&event, refusal_failover)
            {
                let event_bytes = crate::relay::event_retained_bytes(&event);
                if withheld_bytes.saturating_add(event_bytes) > MAXIMUM_WITHHELD_REFUSAL_BYTES
                    || withheld.len() + 1 > MAXIMUM_WITHHELD_REFUSAL_EVENTS
                {
                    let visible_refusal = withheld.iter().any(crate::logprobs::is_refusal_text)
                        || crate::logprobs::is_refusal_text(&event);
                    if !withheld.iter().any(crate::logprobs::is_refusal)
                        && !crate::logprobs::is_refusal(&event)
                    {
                        return AttemptEnd::Ladder {
                            failure: Failure::new(
                                FailureClass::MalformedResponse,
                                crate::dialects::OUTPUT_OVERFLOW_MESSAGE,
                            )
                            .with_retry(false, true),
                            refusal_eligible: false,
                            exhaustion_flush: Vec::new(),
                            usage,
                            tool_names,
                            opened: true,
                            encrypted_reasoning_stripped,
                        };
                    }
                    let prefix = private_reasoning.prefix(&mut withheld, event, usage.as_ref());
                    let tool_search_dropped_after_output = relay.withheld_search_call_seen();
                    relay.commit();
                    return AttemptEnd::Committed(Box::new(CommittedAttempt {
                        depth,
                        prefix,
                        relay,
                        usage,
                        tool_names,
                        visible_refusal,
                        encrypted_reasoning_stripped,
                        tool_search_rounds: Vec::new(),
                        tool_search_dropped_after_output,
                    }));
                }
                withheld_bytes += event_bytes;
                withheld.push(event);
                continue;
            }
            if is_semantic(&event)
                || (private_reasoning.completes(&event) && !relay.withheld_search_call_seen())
            {
                // Outward output freezes this deployment. A private-only
                // successful terminal retains its existing encoding and seal
                // contract; a private-only failure remains failover-safe.
                let visible_refusal = withheld.iter().any(crate::logprobs::is_refusal_text)
                    || crate::logprobs::is_refusal_text(&event);
                let prefix = private_reasoning.prefix(&mut withheld, event, usage.as_ref());
                // A search call withheld in the same turn is dropped: the
                // rung is frozen on this output, and the caller is told.
                let tool_search_dropped_after_output = relay.withheld_search_call_seen();
                relay.commit();
                return AttemptEnd::Committed(Box::new(CommittedAttempt {
                    depth,
                    prefix,
                    relay,
                    usage,
                    tool_names,
                    visible_refusal,
                    encrypted_reasoning_stripped,
                    tool_search_rounds: Vec::new(),
                    tool_search_dropped_after_output,
                }));
            }
            if !event.is_terminal() {
                // Usage stays tracked; other precommit bookkeeping stays private.
                continue;
            }
            match &event {
                Event::Failed(failure) => {
                    usage = relay.usage_before_failure(usage);
                    if !withheld.iter().any(crate::logprobs::is_refusal)
                        && repair.repair_after(failure)
                    {
                        return AttemptEnd::Repair {
                            failure: failure.clone(),
                            usage,
                            tool_names,
                        };
                    }
                    let typed_refusal = failure.failure_class == FailureClass::Refusal;
                    let exhaustion_flush = if withheld.iter().any(crate::logprobs::is_refusal_text)
                        && !typed_refusal
                    {
                        let mut flush = std::mem::take(&mut withheld);
                        flush.push(event.clone());
                        flush
                    } else {
                        withheld.clear();
                        Vec::new()
                    };
                    return AttemptEnd::Ladder {
                        failure: failure.clone(),
                        refusal_eligible: typed_refusal && ctx.policy.refusal_failover,
                        exhaustion_flush,
                        usage,
                        tool_names,
                        opened: true,
                        encrypted_reasoning_stripped,
                    };
                }
                _ => {
                    if withheld.iter().any(crate::logprobs::is_refusal) {
                        // A refusal-only stream that terminated successfully is
                        // a provider refusal: withhold the output and advance,
                        // matching the python executor's converted terminal.
                        withheld.clear();
                        return AttemptEnd::Ladder {
                            failure: Failure::new(
                                FailureClass::Refusal,
                                "provider refused the request",
                            ),
                            refusal_eligible: ctx.policy.refusal_failover,
                            exhaustion_flush: Vec::new(),
                            usage,
                            tool_names,
                            opened: true,
                            encrypted_reasoning_stripped,
                        };
                    }
                    if matches!(event, Event::Completed | Event::StoppedAtSequence(_))
                        && relay.withheld_search_overflowed()
                    {
                        // The model flooded the gateway's search tool past the
                        // per-dial bound: the gateway's own limit, so the dial
                        // fails closed instead of running an oversized round.
                        return AttemptEnd::Ladder {
                            failure: withheld_overflow_failure(),
                            refusal_eligible: false,
                            exhaustion_flush: Vec::new(),
                            usage,
                            tool_names,
                            opened: true,
                            encrypted_reasoning_stripped,
                        };
                    }
                    if matches!(event, Event::Completed | Event::StoppedAtSequence(_))
                        && relay.withheld_search_call_count() > 0
                    {
                        // The turn's only output was the gateway's search
                        // tool: nothing reached the caller and nothing
                        // committed, so the waterfall runs the search and
                        // dials this depth again with the extended context.
                        return AttemptEnd::ToolSearchRound {
                            calls: relay.take_withheld_search_calls(),
                            usage,
                            tool_names,
                        };
                    }
                    if billed_empty_completion(&event, usage.as_ref()) {
                        // A `stop` that billed output tokens yet carried no
                        // semantic event is the provider's fault, not an answer
                        // (a reasoning-only turn on a rung whose reasoning the
                        // gateway strips): it takes the ladder like any other
                        // pre-commit failure instead of settling an empty success.
                        return AttemptEnd::Ladder {
                            failure: empty_completion_failure(wire),
                            refusal_eligible: false,
                            exhaustion_flush: Vec::new(),
                            usage,
                            tool_names,
                            opened: true,
                            encrypted_reasoning_stripped,
                        };
                    }
                    if unreported_empty_completion(&event, usage.as_ref()) {
                        if ctx.output_token_cap.is_some() {
                            // A `stop` with no output and no usage report on a
                            // capped request: the only benign reading is a
                            // budget the provider's hidden reasoning exhausted
                            // before the first visible token, mislabelled as a
                            // plain stop (Meta muse-spark under a small
                            // `max_tokens`, 2026-09-15). Answer `Incomplete` so
                            // the caller sees `length` and raises the cap,
                            // instead of an empty completed answer.
                            return settle_output_less(
                                ctx,
                                guard,
                                Event::Incomplete,
                                withheld,
                                usage,
                                tool_names,
                                depth,
                                encrypted_reasoning_stripped,
                            )
                            .await;
                        }
                        // Uncapped, nothing sent, nothing accounted: the provider
                        // delivered nothing at all. Nothing was committed outward,
                        // so the ladder is safe, exactly like the billed twin.
                        return AttemptEnd::Ladder {
                            failure: empty_completion_failure(wire),
                            refusal_eligible: false,
                            exhaustion_flush: Vec::new(),
                            usage,
                            tool_names,
                            opened: true,
                            encrypted_reasoning_stripped,
                        };
                    }
                    // A successful terminal with no semantic output and nothing
                    // billed for it (a budget exhausted before the first delta,
                    // a zero-token stop): retain the output-less continuation
                    // while the attempt is still in flight, settle, then answer
                    // with the tracked usage ahead of the terminal so the
                    // encoders keep the client-visible token accounting.
                    return settle_output_less(
                        ctx,
                        guard,
                        event,
                        withheld,
                        usage,
                        tool_names,
                        depth,
                        encrypted_reasoning_stripped,
                    )
                    .await;
                }
            }
        }
    }
}
