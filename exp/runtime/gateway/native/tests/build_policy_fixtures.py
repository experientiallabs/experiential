"""Record deterministic routing-policy traces from the Python reference owners."""

from __future__ import annotations

import json
import random
from dataclasses import asdict
from pathlib import Path
from typing import TypedDict

from pydantic import TypeAdapter

from exp.common.core.artifacts import JsonObject
from exp.common.models.dispatch_policy import GatewayThrottleRedialPolicy
from exp.common.models.gateway_catalog import FailoverMode
from exp.runtime.gateway.contracts import GatewayFailure, GatewayFailureClass, GatewayHeaderTimeout
from exp.runtime.gateway.health import DeploymentHealthRegistry
from exp.runtime.gateway.lane_ejection import LaneEjectionPolicy
from exp.runtime.gateway.native_execution import next_route_candidate, throttle_disposition
from exp.runtime.gateway.rung_admission import RungLoadRegistry, RungShed
from exp.runtime.gateway.sticky_affinity import StickySpillRegistry


class _Common(TypedDict):
    organization_id: str
    weight: int
    bound: int | None
    fair_share: bool
    requests_per_minute: int | None
    tokens_per_minute: int | None
    cache_priority_alpha: float | None
    reserved_tokens: int
    fast_reserve: int | None


class _Reserve(_Common):
    warm_session: bool
    fresh_spill_fraction: float | None
    force: bool
    hard_bound: bool
    overflow_ceiling: float | None
    rate_retry: bool
    class_cap: tuple[int, int] | None
    fast: bool
    fast_reserve_hard: bool
    failover_cap: int | None


class _Policy(TypedDict):
    maximum_total_attempts: int
    maximum_same_deployment_attempts: int
    physical_route_cap: int | None


class _Stage(TypedDict):
    lead: int
    failover_mode: FailoverMode
    throttle_cache_threshold: float | None
    throttle_redial: bool


_JUDGED_DEPTHS: tuple[tuple[int, ...], ...] = ((), (0,), (1,), (2,), (0, 2), (0, 1, 2))
"""Depth sets a reservation may claim; each is judged on its own."""


def _account(failure: GatewayFailure | None) -> JsonObject | None:
    """Reduce a recorded account failure to the facts a refusal names.

    Args:
        failure: The account failure holding the judged circuits open, if any.

    Returns:
        Its class, message and ownership, or ``None``.
    """
    if failure is None:
        return None
    return {
        "failure_class": failure.failure_class.value,
        "safe_message": failure.safe_message,
        "customer_owned": failure.customer_owned,
    }


def health_trace(
    *,
    seed: int = 96334,
    steps: int = 700,
    jumps: tuple[float, ...] = (0, 0, 0.01, 1, 5, 30, 60, 21600),
    classes: tuple[GatewayFailureClass, ...] = tuple(GatewayFailureClass),
    waits: tuple[int | None, ...] = (None, 1, 30, 7200, 100000),
    resets: tuple[float, ...] = (0.0, 5.0, 3600.0, 864000.0),
) -> list[JsonObject]:
    """Probe overlapping circuit, quota and physical-lane windows across revisions.

    Args:
        seed: Random seed of the operation sequence.
        steps: Number of recorded operations.
        jumps: Clock advances drawn before each operation.
        classes: Failure classes a ``failed`` operation draws from.
        waits: Provider-stated retry waits a ``failed`` operation draws from.
        resets: Plan-window resets an ``exhausted`` operation draws from.

    Returns:
        One recorded Python reference observation per operation.
    """
    now = 100.0
    registry = DeploymentHealthRegistry(clock=lambda: now, lane_ejection=LaneEjectionPolicy())
    keys = [
        ("revision-a", "lane-a", "connection-a"),
        ("revision-b", "lane-a", "connection-a"),
        ("revision-a", "lane-b", "connection-b"),
    ]
    rng = random.Random(seed)
    trace = []
    for _ in range(steps):
        now += rng.choice(jumps)
        key = rng.randrange(len(keys))
        op = rng.choice(
            [
                "claim",
                "claim_last_resort",
                "claim_forced",
                "claim_throttle_redial",
                "failed",
                "failed",
                "succeeded",
                "dispatch_opened",
                "exhausted",
                "release_probe",
            ]
        )
        argument = None
        if op == "failed":
            failure = GatewayFailure(
                failure_class=rng.choice(classes),
                safe_message=f"fixture {len(trace)}",
                retry_after_seconds=rng.choice(waits),
                header_timeout=rng.choice(
                    [None, GatewayHeaderTimeout.ALLOWANCE, GatewayHeaderTimeout.BUDGET]
                ),
                customer_owned=rng.random() < 0.5,
                tenant_origin=rng.random() < 0.25,
            )
            argument = failure.model_dump(mode="json")
            result = registry.failed(keys[key], failure)
        elif op == "exhausted":
            argument = rng.choice(resets)
            result = registry.exhausted(keys[key], argument)
        else:
            result = getattr(registry, op)(keys[key])
        trace.append(
            {
                "now": now,
                "key": key,
                "op": op,
                "argument": argument,
                "result": result,
                "ejected": [registry.lane_ejected(k) for k in keys],
                "suppressed": [registry.suppressed(k) for k in keys],
                "order": registry.order(tuple(keys), [0, 1, 2]),
                "remaining": registry.throttled_remaining_seconds(tuple(keys)),
                "circuit": [
                    _account(registry.circuit_failure(tuple(keys), depths))
                    for depths in _JUDGED_DEPTHS
                ],
                "windows": [
                    registry.throttled_remaining_seconds(tuple(keys), depths)
                    for depths in _JUDGED_DEPTHS
                ],
            }
        )
    return trace


def loads_trace() -> list[JsonObject]:
    """Exercise real weighted deficits, rate calibration, fast shares and ticket cleanup."""
    now = 100.0
    registry = RungLoadRegistry(clock=lambda: now)
    keys = [("lane-a", "connection-a"), ("lane-b", "connection-b")]
    rng = random.Random(89022)
    tickets = {}
    trace = []
    for index in range(900):
        now += rng.choice([0, 0, 0, 0.01, 0.1, 1, 5, 11, 30, 60, 3601])
        key = rng.randrange(len(keys))
        org = f"org-{rng.randrange(5)}"
        op = rng.choice(
            [
                "reserve",
                "reserve",
                "reserve",
                "throttle",
                "sample",
                "release",
                "retract",
                "restamp",
                "bind",
                "release_attempt",
                "can_admit",
            ]
        )
        arguments: JsonObject = {}
        result = None
        if op in {"reserve", "can_admit"}:
            bound = rng.choice([None, 1, 5, 8, 10])
            common: _Common = dict(
                organization_id=org,
                weight=rng.choice([1, 1, 2, 3, 8]),
                bound=bound,
                fair_share=rng.choice([True, False]),
                requests_per_minute=rng.choice([None, 1, 8, 30]),
                tokens_per_minute=rng.choice([None, 500, 1000]),
                cache_priority_alpha=rng.choice([None, 0.0, 0.25, 1.0]),
                reserved_tokens=rng.choice([0, 10, 700, 2500]),
                fast_reserve=None if bound is None else rng.choice([None, bound // 2]),
            )
            if op == "reserve":
                reservation: _Reserve = dict(
                    **common,
                    warm_session=rng.choice([True, False]),
                    fresh_spill_fraction=rng.choice([None, 0.5, 0.7]),
                    force=rng.choice([False, False, True]),
                    hard_bound=rng.choice([False, True]),
                    overflow_ceiling=rng.choice([None, 4.0, 7.5, 12.0]),
                    rate_retry=rng.choice([False, True]),
                    class_cap=rng.choice([None, (0, 1), (1, 2)]),
                    fast=rng.choice([False, False, True]),
                    fast_reserve_hard=rng.choice([False, True]),
                    failover_cap=rng.choice([None, 2]),
                )
                cap = reservation["class_cap"]
                arguments = TypeAdapter(JsonObject).validate_python(
                    {**reservation, "class_cap": None if cap is None else list(cap)}
                )
                answer = registry.reserve(keys[key], **reservation)
                if isinstance(answer, RungShed):
                    result = {"shed": asdict(answer)}
                else:
                    tickets[index] = answer
                    result = {"ticket": index}
            else:
                arguments = TypeAdapter(JsonObject).validate_python(common)
                result = registry.can_admit(keys[key], **common)
        elif op == "throttle":
            registry.record_throttle(keys[key])
        elif op == "sample":
            input_tokens = rng.choice([0, 100, 1000])
            cached_tokens = rng.choice([0, 50, 1200])
            arguments = dict(
                organization_id=org, input_tokens=input_tokens, cached_tokens=cached_tokens
            )
            registry.record_settle(
                keys[key], org, input_tokens=input_tokens, cached_tokens=cached_tokens
            )
        else:
            ticket = rng.choice(list(tickets)) if tickets else -1
            arguments = {"ticket": ticket}
            held = tickets.get(ticket, "absent")
            if op == "release":
                registry.release_ticket(held)
            elif op == "restamp":
                registry._restamp_window(held, now)
            elif op == "bind":
                registry.bind(held, f"attempt-{ticket}")
            elif op == "release_attempt":
                registry.release_attempt(f"attempt-{ticket}")
            else:
                # Exercise the exact retract branch without a network-backed fleet gate.
                entry = registry._ticket_windows.pop(held, None)
                lane = registry._tickets.get(held, (None, ""))[0]
                rung = registry._rungs.get(lane)
                if entry is not None and rung is not None and entry in rung.window:
                    rung.window.remove(entry)
                    rung.window_requests -= 1
                    rung.window_tokens -= entry[1]
                registry.release_ticket(held)
        trace.append(
            {
                "now": now,
                "key": key,
                "op": op,
                "arguments": arguments,
                "result": result,
                "inflight": [
                    [
                        registry.inflight(k),
                        *[registry.inflight(k, f"org-{n}") for n in range(5)],
                        registry.fast_inflight(k),
                        registry.failover_inflight(k),
                    ]
                    for k in keys
                ],
                "cached": [
                    [registry.cached_fraction(k, f"org-{n}") for n in range(5)] for k in keys
                ],
                "ceilings": registry.learned_ceilings(),
            }
        )
    return trace


def selection_cases() -> list[JsonObject]:
    """Freeze retry decisions with arbitrary authored rules and previously skipped rungs."""
    rng = random.Random(739482)
    cases = []
    now = 100.0
    keys = tuple(("revision", f"lane-{n}", "connection") for n in range(4))
    for _ in range(500):
        health = DeploymentHealthRegistry(clock=lambda: now, lane_ejection=LaneEjectionPolicy())
        prior = []
        for depth in range(4):
            failure = GatewayFailure(
                failure_class=rng.choice(list(GatewayFailureClass)), safe_message="fixture"
            )
            health.failed(keys[depth], failure)
            prior.append(failure.model_dump(mode="json"))
        failure = GatewayFailure(
            failure_class=rng.choice(list(GatewayFailureClass)),
            safe_message="fixture",
            retryable_same_deployment=rng.choice([True, False]),
            failover_eligible=rng.choice([True, False]),
        )
        current = rng.randrange(4)
        physical = [rng.randrange(4) for _ in range(4)]
        ordinary = [rng.randrange(count + 1) for count in physical]
        policy: _Policy = dict(
            maximum_total_attempts=rng.randrange(1, 9),
            maximum_same_deployment_attempts=rng.randrange(1, 5),
            physical_route_cap=rng.choice([None, 1, 2, 3, 4]),
        )
        stage: _Stage = dict(
            lead=0,
            failover_mode=rng.choice(
                ["maximize_availability", "maximize_cache", "maximize_cache_affinity"]
            ),
            throttle_cache_threshold=rng.choice([None, 0.0, 0.5, 1.0]),
            throttle_redial=rng.choice([True, False]),
        )
        rules = [
            rng.choice(
                [
                    None,
                    ["throttled"],
                    ["refusal"],
                    ["provider_internal", "transport"],
                    ["refusal:cyber_policy"],
                    ["guardrail"],
                ]
            )
            for _ in range(4)
        ]
        fraction = rng.choice([0.0, 0.5, 0.9, 1.0])
        refusal = rng.choice([True, False])
        backoff = rng.choice([True, False])
        budget = rng.randrange(4)
        result = next_route_candidate(
            health=health,
            keys=keys,
            failure=failure,
            current_depth=current,
            attempt_counts=ordinary,
            total_attempts=sum(physical),
            refusal_failover=refusal,
            failover_mode=stage["failover_mode"],
            throttle_cache_threshold=stage["throttle_cache_threshold"],
            cached_fraction=fraction,
            throttle_redial=GatewayThrottleRedialPolicy(
                max_attempts=3, base_delay_ms=500, max_delay_ms=8000
            )
            if stage["throttle_redial"]
            else None,
            throttle_backoff=backoff,
            throttle_redial_budget=budget,
            fallback_rules=tuple(None if rule is None else frozenset(rule) for rule in rules),
            physical_attempt_counts=physical,
            **policy,
        )
        disposition = throttle_disposition(
            failure,
            throttle_cache_threshold=stage["throttle_cache_threshold"],
            cached_fraction=fraction,
        )
        if stage["throttle_redial"] and failure.failure_class is GatewayFailureClass.THROTTLED:
            disposition = (
                None
                if result is None
                else "throttle_backoff"
                if result == current
                else "throttle_failover_cold"
            )
        cases.append(
            dict(
                prior=prior,
                failure=failure.model_dump(mode="json"),
                current=current,
                physical_counts=physical,
                ordinary_counts=ordinary,
                total=sum(physical),
                policy=policy,
                stage=stage,
                rules=rules,
                cached_fraction=fraction,
                refusal_failover=refusal,
                throttle_backoff=backoff,
                redial_budget=budget,
                result=result,
                disposition=disposition,
            )
        )
    return TypeAdapter(list[JsonObject]).validate_python(cases)


def sticky_trace() -> list[JsonObject]:
    """Expiry, refresh, maximum age and bounded LRU order use the original owner."""
    rng = random.Random(93512)
    now = 100.0
    sticky = StickySpillRegistry(maximum_bindings=4, clock=lambda: now)
    cases = []
    for _ in range(300):
        now += rng.choice([0, 1, 5, 10, 30])
        fingerprint = str(rng.randrange(7))
        deployment = str(rng.randrange(3))
        ttl = rng.choice([0, 10, 30, 90])
        op = rng.choice(["bind", "bind", "clear"])
        if op == "bind":
            sticky.bind(fingerprint.encode(), deployment, ttl_seconds=ttl)
        else:
            sticky.clear(fingerprint.encode())
        bindings = [sticky.bound_deployment(str(n).encode()) for n in range(7)]
        cases.append(
            dict(
                now=now,
                fingerprint=fingerprint,
                deployment=deployment,
                ttl=ttl,
                op=op,
                bindings=bindings,
                size=sticky.size(),
            )
        )
    return TypeAdapter(list[JsonObject]).validate_python(cases)


if __name__ == "__main__":
    destination = Path(__file__).parent / "generated"
    destination.mkdir(exist_ok=True)
    for name, cases in [
        ("health", health_trace()),
        # Short clock steps over account, throttle and transport failures keep
        # several circuits open at once, so account answers span many depths.
        (
            "circuits",
            health_trace(
                seed=40213,
                steps=900,
                jumps=(0, 0, 0.01, 1, 5, 12, 31),
                classes=(
                    GatewayFailureClass.PROVIDER_AUTHENTICATION,
                    GatewayFailureClass.PROVIDER_NOT_FOUND,
                    GatewayFailureClass.PROVIDER_QUOTA,
                    GatewayFailureClass.THROTTLED,
                    GatewayFailureClass.TRANSPORT,
                    GatewayFailureClass.REFUSAL,
                ),
                waits=(None, 1, 30),
                resets=(0.0, 5.0, 40.0),
            ),
        ),
        ("loads", loads_trace()),
        ("selection", selection_cases()),
        ("sticky", sticky_trace()),
    ]:
        (destination / f"{name}.jsonl").write_text(
            "\n".join(json.dumps(case, sort_keys=True) for case in cases) + "\n"
        )
