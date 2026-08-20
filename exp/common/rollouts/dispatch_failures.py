"""Shared classification of ambiguous-spend and retryable dispatch failures.

A simulated cell can fail after a provider request left the process but before its priced
response arrived, so the exact spend of that dispatch is unknown. These helpers give every
spend reconciler one canonical way to recognize such evidence, read its persisted worst-case
reservation, and decide whether the failed dispatch belongs to the transport-retryable class
that resume may re-execute under fresh budget.
"""

from __future__ import annotations

import math

from exp.common.core.artifacts import FailureCode, StructuredFailure

UNKNOWN_DISPATCH_RESERVED_COST_KEY = "unknown_dispatch_reserved_cost_usd"
"""Failure-detail key holding the conservative worst-case charge for an unknown dispatch."""

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
    """Return the persisted worst-case charge for one unknown-spend dispatch failure.

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


def retryable_dispatch_failure(failure: StructuredFailure | None) -> bool:
    """Return whether a persisted provider dispatch failure is stochastically retryable.

    Only candidate or world-model transport dispatch failures and world-model protocol
    output failures qualify, and only when the persisted failure is marked retryable.
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
