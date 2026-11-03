"""The privileged evidence packet judges and critics read about one group of attempts.

The agent under test never sees this packet. Judges and critics see all of it: every attempt in the
group with its grade, the group's success rate, the task verifier, and, for the one trajectory
whose value is being estimated, every boundary at which a continuation could start, each with the
conversation the actor had seen by then. Every locator an answer may cite is listed in the packet's
inventory, so an answer that cites anything else is refused rather than trusted.

Point ``k`` of the picked trajectory is the boundary before its ``k``-th tool turn: point 0 is the
study's starting checkpoint and point ``k`` (k >= 1) is the checkpoint saved after turn ``k``.
A point with no remaining actor calls cannot be continued and is not offered.
"""

from __future__ import annotations

from collections.abc import Callable

from pydantic import Field

from exp.common.core.artifacts import ArtifactInput, ContractModel, JsonObject
from exp.optimize.evaluation.continuations.study import ContinuationStudy, known_outcomes
from exp.simulation.continuations import ContinuationCheckpoint


class EstimationPool(ContractModel):
    """The continuable boundaries of the picked trajectory.

    Attributes:
        branch: Index of the picked attempt within the group.
        points: Checkpoint reference per offered point, keyed by point number.
    """

    branch: int = Field(ge=0)
    points: dict[int, ArtifactInput]


class EvidencePacket(ContractModel):
    """Everything a judge or critic is shown, plus the locators it may cite.

    Attributes:
        pool: The offered estimation points.
        payload: The rendered packet sent to the role.
        refs: Every locator an answer may cite.
    """

    pool: EstimationPool
    payload: JsonObject
    refs: frozenset[str]


def group_rate(study: ContinuationStudy) -> tuple[int, int]:
    """Return ``(passes, graded)`` over the group's assessed attempts.

    Args:
        study: The group of attempts.

    Returns:
        Passing attempts and attempts with a known grade.
    """
    labels = known_outcomes(branch.assessment for branch in study.branches)
    return sum(labels), len(labels)


def candidate_branches(study: ContinuationStudy) -> tuple[int, ...]:
    """Return completed attempts with at least one saved tool turn after the start.

    Whether enough of their points remain continuable is checked against the saved checkpoints
    when the packet is rendered.

    Args:
        study: The group of attempts.

    Returns:
        Indexes of attempts a trajectory may be picked from.
    """
    return tuple(
        branch.index
        for branch in study.branches
        if branch.status == "completed" and branch.checkpoints
    )


def evidence_packet(
    study: ContinuationStudy,
    branch: int,
    *,
    maximum_model_calls: int,
    checkpoint: Callable[[ArtifactInput], ContinuationCheckpoint],
) -> EvidencePacket:
    """Render the privileged packet for one picked attempt of a group.

    Args:
        study: The group of attempts, all from one starting checkpoint.
        branch: Index of the picked attempt.
        maximum_model_calls: The actor's call allowance.
        checkpoint: Loader for saved continuation checkpoints.

    Returns:
        The packet, its estimation pool and its citation inventory.

    Raises:
        ValueError: The picked attempt is absent, incomplete or offers fewer than two points.
    """
    rows = {row.index: row for row in study.branches}
    picked = rows.get(branch)
    if picked is None or picked.status != "completed":
        raise ValueError(f"group attempt {branch} is absent or did not complete")
    boundaries = (study.spec.checkpoint, *picked.checkpoints)
    points = {
        turn: reference
        for turn, reference in enumerate(boundaries)
        if checkpoint(reference).actor.calls < maximum_model_calls
    }
    if len(points) < 2:
        raise ValueError(f"group attempt {branch} offers fewer than two continuable points")
    passes, graded = group_rate(study)
    refs = {"group", "verifier"}
    attempts: list[JsonObject] = []
    for row in study.branches:
        refs.add(f"attempt:{row.index}")
        attempts.append(
            {
                "attempt": row.index,
                "status": row.status,
                "grade": None if row.assessment is None else row.assessment.model_dump(mode="json"),
                "outcome": row.outcome,
            }
        )
    offered: list[JsonObject] = []
    for turn, reference in points.items():
        refs.add(f"point:{turn}")
        offered.append(
            {
                "point": turn,
                "actor_calls_spent": checkpoint(reference).actor.calls,
                "public_prefix": [
                    message.model_dump(mode="json")
                    for message in checkpoint(reference).actor.messages
                ],
            }
        )
    payload: JsonObject = {
        "group": {
            "attempts_planned": len(study.branches),
            "attempts_graded": graded,
            "passes": passes,
            "success_rate": None if graded == 0 else passes / graded,
        },
        "verifier": study.spec.verifier.model_dump(mode="json"),
        "attempts": attempts,
        "picked_attempt": branch,
        "points": offered,
    }
    return EvidencePacket(
        pool=EstimationPool(branch=branch, points=points), payload=payload, refs=frozenset(refs)
    )
