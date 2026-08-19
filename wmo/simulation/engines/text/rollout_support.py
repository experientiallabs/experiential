"""Small evidence helpers shared by the text-simulation orchestration path."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime
from typing import cast

from wmo.common.core.artifacts import (
    FailureAttribution,
    FailureCode,
    StructuredFailure,
)
from wmo.common.models import NumericMeasurement, OperationEconomics
from wmo.common.rollouts import (
    RolloutArtifact,
    RolloutEventKind,
    RolloutSpan,
    unknown_dispatch_reserved_cost_usd,
    unknown_spend_failure,
)
from wmo.runtime.agents import AgentEpisode
from wmo.simulation.engines.text.environment import TextOnlyToolUseError
from wmo.simulation.engines.text.redaction import redact_span


def combine_spans(
    agent_events: Sequence[RolloutSpan],
    candidate_spans: Sequence[RolloutSpan],
    world_model_spans: Sequence[RolloutSpan],
    redacted_field_names: frozenset[str],
) -> tuple[RolloutSpan, ...]:
    """Merge redacted evidence while keeping recorder-owned candidate calls canonical.

    Args:
        agent_events: Customer agent lifecycle evidence from every loop iteration.
        candidate_spans: Recorder-owned candidate calls.
        world_model_spans: Recorder-owned world-model calls.
        redacted_field_names: Keys that must never persist in evidence.

    Returns:
        Time-ordered spans with unique agent span IDs across repeated agent episodes.
    """
    copied_agent_events: list[RolloutSpan] = []
    for episode_index, event in enumerate(agent_events):
        if event.kind == RolloutEventKind.AGENT_MODEL_CALL:
            continue
        copied = redact_span(event, redacted_field_names)
        copied_agent_events.append(
            copied.model_copy(update={"span_id": f"agent-{episode_index}-{copied.span_id}"})
        )
    combined = (*copied_agent_events, *candidate_spans, *world_model_spans)
    return tuple(sorted(combined, key=lambda span: (span.started_at, span.ended_at, span.span_id)))


def failure_span(
    started_at: datetime,
    ended_at: datetime,
    failure: StructuredFailure,
) -> RolloutSpan:
    """Build a minimum lifecycle span for a failure that preceded any model call."""
    return RolloutSpan(
        span_id="lifecycle-failure",
        kind=RolloutEventKind.LIFECYCLE,
        started_at=started_at,
        ended_at=ended_at,
        payload={"phase": failure.details.get("phase", "simulation")},
        failure=failure,
    )


def internal_failure(phase: str, exception: Exception) -> StructuredFailure:
    """Normalize a local orchestration exception without retaining arbitrary exception text."""
    return StructuredFailure(
        code=FailureCode.INTERNAL,
        message=f"{phase} failed with {type(exception).__name__}",
        exception_type=type(exception).__name__,
        attribution=FailureAttribution.AGENT,
        details={"phase": phase},
    )


def normalize_text_tool_failure(episode: AgentEpisode) -> StructuredFailure | None:
    """Translate a tool attempt into the text mode's explicit unsupported-cell evidence.

    Args:
        episode: Customer-agent episode that may have ended at the text-only tool boundary.

    Returns:
        The original failure, a normalized unsupported failure, or ``None``.
    """
    failure = episode.failure
    if failure is None or failure.exception_type != TextOnlyToolUseError.__name__:
        return failure
    return StructuredFailure(
        code=FailureCode.UNSUPPORTED,
        message="text world-model simulation cannot execute a customer-agent tool call",
        exception_type=failure.exception_type,
        attribution=FailureAttribution.TOOL,
        details={"phase": "agent_tool_call"},
    )


def known_total_spend(
    rollouts: Sequence[RolloutArtifact],
) -> float | None:
    """Return total conservative provider spend, or ``None`` if any episode is unpriced.

    Args:
        rollouts: Completed text-simulation rollout artifacts to total.

    Returns:
        The known conservative provider spend, or ``None`` when any billed call is not priced
        and has no persisted worst-case reservation.
    """
    values = tuple(rollout_spend(rollout) for rollout in rollouts)
    if any(value is None for value in values):
        return None
    return sum(cast(float, value) for value in values)


def rollout_spend(
    rollout: RolloutArtifact,
) -> float | None:
    """Return reconciled provider cost without treating unknown dispatches as free.

    An unknown-spend failure is charged its persisted worst-case reservation instead of
    poisoning the whole total, so one ambiguous dispatch stays conservatively inside the
    ceiling while every other completed cell remains priced exactly.

    Args:
        rollout: Completed text-simulation rollout whose recorded calls are inspected.

    Returns:
        Observed candidate and world-model cost plus conservative retrieval and reservation
        charges, or ``None`` when any dispatched operation has unpriceable spend.
    """
    reserved: float | None = None
    if unknown_spend_failure(rollout.failure):
        reserved = unknown_dispatch_reserved_cost_usd(rollout.failure)
        if reserved is None:
            return None
    roles = (
        (rollout.candidate_economics, RolloutEventKind.AGENT_MODEL_CALL),
        (rollout.world_model_economics, RolloutEventKind.SIMULATOR_WORLD_MODEL_CALL),
    )
    total = reserved if reserved is not None else 0.0
    for economics, span_kind in roles:
        made_call = any(span.kind == span_kind for span in rollout.spans)
        if not made_call:
            continue
        if economics is None or economics.cost_usd is None:
            return None
        total += economics.cost_usd.value
    retrieval = rollout.retrieval_economics
    if retrieval is not None and retrieval.cost_usd is not None:
        total += retrieval.cost_usd.value
    return total


def elapsed_seconds(started_at: float, ended_at: float) -> float:
    """Return a nonnegative orchestration duration from an injected monotonic clock."""
    return max(0.0, ended_at - started_at)


def orchestration_economics(duration_seconds: float) -> OperationEconomics:
    """Record simulator-owned elapsed time without attributing it to the candidate model."""
    return OperationEconomics(
        latency_seconds=NumericMeasurement(value=duration_seconds, provenance="observed")
    )
