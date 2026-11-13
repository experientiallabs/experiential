"""Serialize initial admission facts for interpreter-independent request accounting."""

from __future__ import annotations

import json
import time
from typing import assert_never

from exp.common.core.artifacts import JsonObject, JsonValue
from exp.common.models.catalog import BillingSource
from exp.common.models.gateway_catalog import ExactModelDeployment
from exp.runtime.gateway.attempt_tokens import counted_input_tokens, worst_case_output_tokens
from exp.runtime.gateway.audio_contracts import SpeechRequest, TranscriptionRequest
from exp.runtime.gateway.cache_write import requests_hour_cache
from exp.runtime.gateway.class_throttles import is_self_hosted_rung
from exp.runtime.gateway.contracts import GatewayRequest
from exp.runtime.gateway.decisions_contracts import DecisionRequest
from exp.runtime.gateway.embeddings_contracts import EmbeddingsRequest
from exp.runtime.gateway.images_contracts import ImagesRequest, image_variant
from exp.runtime.gateway.lane_saturation import lane_saturated_failure
from exp.runtime.gateway.native_execution import InflightRequest, deployment_health_key
from exp.runtime.gateway.native_fallback_rules import route_fallback_rules
from exp.runtime.gateway.native_feature_registration import feature_registration
from exp.runtime.gateway.native_recovery_state import recovery_registration
from exp.runtime.gateway.native_rung_policy import keeps_cache_placement, shed_keeps_pin
from exp.runtime.gateway.video.contracts import VideoRequest


def registration_json(entry: InflightRequest, *, bind_sticky: bool) -> str:
    """Transfer immutable admission facts; physical prices and state stay Rust-owned.

    Args:
        entry: Fully admitted authority, route and bounded request.
        bind_sticky: Whether ordinary affinity is owned by this process.

    Returns:
        Strict native registration JSON, including private native feature authority.
    """
    authorization = entry.authorization
    snapshot = entry.route.snapshot
    now = time.monotonic()
    chat = isinstance(entry.request, GatewayRequest)
    payload: JsonObject = {
        "request_id": str(authorization.request_id),
        "organization_id": str(authorization.organization_id),
        "remaining_seconds": max(0.0, entry.deadline_monotonic - now),
        "surface": authorization.surface.value,
        "lanes": [
            _lane(entry, depth, deployment)
            for depth, deployment in enumerate(entry.route.deployments)
        ],
        "rules": [
            None if rules is None else list[JsonValue](sorted(rules))
            for rules in route_fallback_rules(entry.route)
        ],
        "attempt_policy": entry.attempt_policy.model_dump(mode="json"),
        "refusal_failover": authorization.refusal_failover,
        "selected_route": entry.route.resolved_route_id is not None,
        "priority": authorization.priority_admission,
        "weight": authorization.fair_share_weight,
        "fast": authorization.fast,
        "class": authorization.priority_class,
        "class_cap": authorization.self_hosted_class_concurrency,
        "pace": authorization.output_tokens_per_second,
        "pace_saturation": authorization.output_pace_saturation,
        "input_tokens": entry.reserved_input_tokens(),
        "counted_input_tokens": counted_input_tokens(entry.request),
        "output_ceiling": entry.request.maximum_output_tokens if chat else None,
        "service_tier": entry.request.service_tier if chat else None,
        "redial_budgets": list(
            entry.throttle_redial_budgets or (0,) * len(entry.route.deployments)
        ),
        "fingerprint": None
        if entry.affinity_fingerprint is None
        else entry.affinity_fingerprint.hex(),
        "bind_sticky": bind_sticky
        and not snapshot.model_stages
        and snapshot.failover_mode == "maximize_cache_affinity",
        "sticky_preferred": entry.sticky_preferred,
        "caching_off": authorization.prompt_caching_disabled and chat,
        "recovery_scoped": entry.recovery_scoped,
        "verified_warm_deployment": entry.verified_warm_deployment_id,
        "verified_warm_seconds": max(0.0, entry.verified_warm_until_monotonic - now),
        "recovery_reason": entry.recovery_reason,
        "fallback_reason": entry.route.fallback_reason,
        "paid_prework": not entry.no_paid_prework,
        "web_search_requests": entry.web_search_requests,
        "capacity_message": lane_saturated_failure(authorization).safe_message,
        "caller_narrowing": list[JsonValue](entry.route.caller_narrowing),
        "features": feature_registration(entry),
        "recovery": None if bind_sticky else recovery_registration(entry),
    }
    # Feature compilation may tokenize or construct provider templates. Its
    # admission time must not renew the serving deadline or warm-route lease.
    now = time.monotonic()
    payload["remaining_seconds"] = max(0.0, entry.deadline_monotonic - now)
    payload["verified_warm_seconds"] = max(0.0, entry.verified_warm_until_monotonic - now)
    return json.dumps(payload, separators=(",", ":"))


def _lane(entry: InflightRequest, depth: int, deployment: ExactModelDeployment) -> JsonObject:
    """Describe one certified physical lane without selecting a price card."""
    stage = entry.route.snapshot.stage_for_depth(depth)
    policy = deployment.gateway.dispatch
    output = (
        entry.reserved_output_tokens_by_depth[depth]
        if entry.reserved_output_tokens_by_depth
        else worst_case_output_tokens(entry.request, deployment)
    )
    dispatch: JsonObject = {"self_hosted": is_self_hosted_rung(deployment)}
    if policy is not None:
        dispatch.update(
            {
                "concurrency_bound": policy.concurrency_bound,
                "requests_per_minute": policy.requests_per_minute,
                "tokens_per_minute": policy.tokens_per_minute,
                "cache_priority_alpha": policy.cache_priority_alpha,
                "fresh_session_spill_fraction": policy.fresh_session_spill_fraction,
                "refusing": policy.saturation == "refuse",
                "priority_overflow_paying": policy.priority_overflow_paying,
                "priority_overflow_pro": policy.priority_overflow_pro,
                "priority_tier_attested": policy.priority_tier_attested
                and deployment.billing_source is BillingSource.HOST_MANAGED,
                "fast_reserve_fraction": policy.fast_reserve_fraction,
                "sticky_spill_seconds": policy.sticky_spill_seconds,
                "fleet": None if policy.fleet is None else policy.fleet.model_dump(mode="json"),
            }
        )
    explicit = (
        isinstance(entry.request, GatewayRequest)
        and entry.request.maximum_output_tokens is not None
    )
    return {
        "key": list(deployment_health_key(entry.authorization, deployment)),
        "pool_id": str(stage.pool_id),
        "stage": {
            "lead": entry.route.snapshot.deployment_ids.index(stage.deployment_ids[0]),
            "failover_mode": stage.failover_mode,
            "throttle_cache_threshold": stage.throttle_cache_threshold,
            "throttle_redial": stage.throttle_redial is not None,
        },
        "policy": dispatch,
        "prices": deployment.gateway.prices.model_dump(mode="json"),
        "ceiling": _ceiling(entry, deployment, output),
        "output_tokens": output,
        "output_ceiling": output if explicit else None,
        "customer_managed": deployment.billing_source is BillingSource.CUSTOMER_MANAGED,
        "forwards_tier": depth < len(entry.tier_forwarded_by_depth)
        and entry.tier_forwarded_by_depth[depth],
        "route_reason": entry.route.attempt_route_reason(deployment),
        "keeps_pin": shed_keeps_pin(entry.route, depth),
        "cache_placed": keeps_cache_placement(entry.route, deployment),
    }


def _ceiling(entry: InflightRequest, deployment: ExactModelDeployment, output: int) -> JsonObject:
    """Freeze quantity bounds; Rust prices them at the dispatch-time schedule."""
    request = entry.request
    input_tokens = entry.reserved_input_tokens()
    match request:
        case GatewayRequest() | DecisionRequest():
            capabilities = deployment.gateway.capabilities
            return {
                "kind": "tokens",
                "input": input_tokens,
                "output": output,
                "cached": capabilities.reports_cached_input_tokens,
                "writes": capabilities.reports_cache_creation_input_tokens,
                "hour": requests_hour_cache(request),
                "twelve": isinstance(request, GatewayRequest) and request.prompt_cache_ttl == "12h",
                "reasoning": capabilities.reports_reasoning_tokens,
            }
        case EmbeddingsRequest():
            return {"kind": "embeddings", "input": input_tokens}
        case ImagesRequest():
            return {
                "kind": "images",
                "input": input_tokens,
                "count": request.n,
                "variant": image_variant(request),
            }
        case SpeechRequest() | TranscriptionRequest():
            return {
                "kind": "audio",
                "input": input_tokens,
                "output": None
                if deployment.capabilities is None
                else deployment.capabilities.maximum_output_tokens,
                "unit": "character" if isinstance(request, SpeechRequest) else "audio_second",
                "quantity": request.input_characters
                if isinstance(request, SpeechRequest)
                else request.maximum_audio_seconds,
            }
        case VideoRequest():
            # Deferred jobs own their durable transaction separately from this callback path.
            raise ValueError("Deferred video jobs require their own native transaction owner")
        case _:
            assert_never(request)
