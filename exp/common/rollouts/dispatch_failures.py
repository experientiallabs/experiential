"""Shared classification of ambiguous-spend and retryable dispatch failures.

A simulated cell can fail after a provider request left the process but before its priced
response arrived, so the exact spend of that dispatch is unknown. These helpers give every
spend reconciler one canonical way to recognize such evidence, distinguish a reservation
estimate from a proven bound, and decide whether the failed dispatch belongs to the retryable class
that resume may re-execute under fresh budget. A completed response whose pricing is
unavailable or whose tool JSON is truncated may also qualify when an uncapped request ledger
authorized the retry.
"""

from __future__ import annotations

import math

from exp.common.core.artifacts import FailureCode, StructuredFailure

UNKNOWN_DISPATCH_RESERVED_COST_KEY = "unknown_dispatch_reserved_cost_usd"
"""Failure-detail key holding the retained reservation estimate for an unknown dispatch."""

UNKNOWN_DISPATCH_IS_UPPER_BOUND_KEY = "unknown_dispatch_reserved_cost_is_upper_bound"
"""Failure-detail key stating whether the reservation bounds the unknown liability."""

_RETRYABLE_DISPATCH_PHASES = frozenset({"candidate_or_world_model", "world_model_protocol"})
"""Persisted failure phases whose retryable provider failures resume may re-execute.

Candidate or world-model transport dispatches and stochastic world-model protocol
outputs both fail for reasons that say nothing deterministic about the episode, so a
fresh sample of the same cell is meaningful evidence.
"""


def unknown_spend_failure(failure: StructuredFailure | None) -> bool:
    """Return whether a persisted failure left its dispatched provider spend unknown.

    Args:
        failure: Structured failure retained by a rollout artifact, or ``None``.

    Returns:
        ``True`` when a provider or environment dispatch has no priced outcome, or when a
        stale paid-cell claim makes the cell's spend permanently ambiguous.
    """
    if failure is None:
        return False
    return (
        failure.details.get("provider_dispatch_unknown_spend") is True
        or failure.details.get("environment_dispatch_unknown_spend") is True
        or failure.details.get("phase") == "paid_cell_stale_lease"
    )


def unknown_dispatch_reserved_cost_usd(failure: StructuredFailure | None) -> float | None:
    """Return the retained estimate, without asserting that it bounds the unknown charge.

    Args:
        failure: Structured failure retained by a rollout artifact, or ``None``.

    Returns:
        The nonnegative finite reserved amount persisted with the failure, or ``None`` when
        the value is absent or unusable.
    """
    if failure is None:
        return None
    value = failure.details.get(UNKNOWN_DISPATCH_RESERVED_COST_KEY)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    amount = float(value)
    if not math.isfinite(amount) or amount < 0:
        return None
    return amount


def unknown_dispatch_reservation_is_upper_bound(failure: StructuredFailure | None) -> bool:
    """Distinguish bounded dispatch reservations from unpriceable paid outcomes.

    An explicit marker must be a true boolean. Without that marker, existing transport and
    whole-cell reservations retain their bounded meaning, but saved pricing or truncation
    classifications never establish a bound, even when they retain a numeric estimate.

    Args:
        failure: Immutable failure evidence whose reservation is being reconciled.

    Returns:
        Whether the marker and failure semantics permit using a separately validated
        numeric reservation as a cost bound.
    """
    if failure is None:
        return False
    if failure.exception_type in {
        "ProviderPricingUnavailableError",
        "ProviderTruncatedResponseError",
    } or failure.details.get("retry_classification") in (
        "unpriceable_completed_response",
        "truncated_completed_response",
    ):
        return False
    return failure.details.get(UNKNOWN_DISPATCH_IS_UPPER_BOUND_KEY, True) is True


def retryable_dispatch_failure(failure: StructuredFailure | None) -> bool:
    """Return whether a persisted provider dispatch failure is stochastically retryable.

    Candidate or world-model transport, explicitly uncapped pricing or truncation, and
    world-model protocol output failures qualify only when persisted as retryable.
    Budget, validation, and stale-lease failures never qualify.

    Args:
        failure: Structured failure retained by a rollout artifact, or ``None``.

    Returns:
        ``True`` when resume may deliberately re-execute the cell as a new attempt.
    """
    if failure is None:
        return False
    if failure.code != FailureCode.PROVIDER:
        return False
    if failure.details.get("phase") not in _RETRYABLE_DISPATCH_PHASES:
        return False
    return failure.retryable
