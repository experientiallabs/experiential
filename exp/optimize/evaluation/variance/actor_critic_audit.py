"""Audit collection, training-eligibility reasons and point ranking of one actor-critic run.

Pure functions over the run's saved draws and measured outcomes; no call is made here.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from exp.common.core.artifacts import JsonObject
from exp.optimize.evaluation.variance.rewards import PointOutcome

if TYPE_CHECKING:
    from exp.optimize.evaluation.variance.actor_critic import CriticDraw, JudgeDraw


def audit_flags(
    judges: tuple[JudgeDraw, ...], critics: tuple[CriticDraw, ...]
) -> tuple[JsonObject, ...]:
    """Collect every audit flag with the role and draw that raised it."""
    flags: list[JsonObject] = []
    for draw in judges:
        if draw.selection is not None:
            flags.extend(
                {"role": "judge", "draw": draw.draw, **flag.model_dump(mode="json")}
                for flag in draw.selection.audit
            )
    for draw in critics:
        if draw.assessment is not None:
            flags.extend(
                {"role": "critic", "draw": draw.draw, **flag.model_dump(mode="json")}
                for flag in draw.assessment.audit
            )
    return tuple(flags)


def ineligible_reasons(
    audit: tuple[JsonObject, ...], critics: tuple[CriticDraw, ...]
) -> tuple[str, ...]:
    """Explain why a run's records may not be trained on yet.

    A screening run never qualifies on its own: no point has been shown to be a decision until
    competing actions from the same checkpoint have been run under a frozen policy and budget.

    Args:
        audit: Every audit flag the run raised.
        critics: Every critic draw.

    Returns:
        The reasons, most fundamental first.
    """
    reasons = [
        "no chosen point has been tested with competing actions from the same checkpoint",
    ]
    if audit:
        reasons.append(f"{len(audit)} audit findings have no recorded disposition")
    unscored = [draw.draw for draw in critics if draw.score is None or draw.score.reward is None]
    if unscored:
        reasons.append(f"critic draws {unscored} have no scored estimate")
    return tuple(reasons)


def rank_points(outcomes: tuple[PointOutcome, ...], drops: dict[str, float]) -> tuple[int, ...]:
    """Order points by uncertainty of the measured rate, then by the drop that follows them."""

    def key(outcome: PointOutcome) -> tuple[float, float, int]:
        """Most informative first; unscored points last."""
        rate = outcome.rate
        uncertainty = -1.0 if rate is None else rate * (1.0 - rate)
        return (-uncertainty, -drops.get(outcome.point_id, 0.0), outcome.step)

    return tuple(outcome.step for outcome in sorted(outcomes, key=key))
