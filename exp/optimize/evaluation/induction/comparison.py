"""Paired replay of recorded cases through two or more executable world arms.

This measures what induced code changes: per-step exact replies and per-episode wall time of,
for example, induced programs with the grounded predictor against the predictor alone. Cases
are fidelity-replay :class:`ReplayCase` values (so a held-out panel selected and reserved by
the audit owner can be replayed unchanged); each arm opens a fresh world per case, applies the
public prefix without scoring it, then executes every measured recorded call in order. Replies
are compared with the fidelity owner's strict JSON equality and broken down by its reply kinds;
uncertainty is reported over lineages with repeats averaged first. World or provider failures
are raised, never scored, so an interrupted comparison stays incomplete.

Reports contain measurements only, with no approval, gate or threshold.
"""

import json
import time
from collections.abc import Callable, Mapping, Sequence

from pydantic import Field

from exp.common.core.artifacts import ContractModel, JsonObject, JsonValue, strict_equal
from exp.common.evaluations.uncertainty import unit_difference_summary, unit_mean_summary
from exp.common.models import ToolCall
from exp.optimize.evaluation.fidelity.replay import (
    REPLY_KINDS,
    ReplayCase,
    ReplyKindCount,
    reply_kind,
)
from exp.runtime.environments.interface import EnvironmentRuntime

type ArmFactory = Callable[[ReplayCase], EnvironmentRuntime]


class ArmReplay(ContractModel):
    """One arm's measurements over every case.

    Attributes:
        arm: Arm name.
        steps: Measured recorded calls.
        exact: Calls whose reply equalled the recorded reply exactly.
        reply_kinds: Exact and measured counts per recorded-reply kind.
        induced_steps: Measured calls answered by induced code rather than a predictor.
        episode_seconds: Wall time of each case's episode, prefix included, in case order.
        lineage_exact: Exact rate averaged within each lineage, then across lineages.
    """

    arm: str
    steps: int = Field(ge=0)
    exact: int = Field(ge=0)
    reply_kinds: tuple[ReplyKindCount, ...]
    induced_steps: int = Field(ge=0)
    episode_seconds: tuple[float, ...]
    lineage_exact: JsonObject


class WorldReplayComparison(ContractModel):
    """Paired measurements of every arm on the same cases.

    Attributes:
        cases: Cases replayed.
        arms: Per-arm measurements, in the order given.
        exact_difference: First arm minus second arm in per-lineage exact rate, paired by lineage.
    """

    cases: int = Field(ge=0)
    arms: tuple[ArmReplay, ...]
    exact_difference: JsonObject | None


def compare_world_replay(
    cases: Sequence[ReplayCase],
    arms: Mapping[str, ArmFactory],
    *,
    clock: Callable[[], float] = time.perf_counter,
) -> WorldReplayComparison:
    """Replay every case through every arm and return paired measurements.

    Args:
        cases: Recorded cases with public prefixes and measured steps.
        arms: Arm name to a factory opening that arm's world for one case.
        clock: Monotonic seconds, injectable for tests.

    Returns:
        Measurements per arm and the first-minus-second paired exact difference.

    Raises:
        ValueError: Fewer than one arm is given.
    """
    if not arms:
        raise ValueError("compare at least one world arm")
    measured: list[ArmReplay] = []
    outcomes: list[dict[str, list[float]]] = []
    for name, factory in arms.items():
        kinds = {kind: [0, 0] for kind in REPLY_KINDS}
        steps = exact = induced = 0
        seconds: list[float] = []
        by_lineage: dict[str, list[float]] = {}
        for case in cases:
            started = clock()
            with factory(case).open(case.task) as world:
                for index, row in enumerate(_prefix(case)):
                    world.execute(
                        ToolCall(name=row[0], call_id=f"prefix-{index}", arguments=row[1])
                    )
                for index, step in enumerate(case.steps):
                    observation = world.execute(
                        ToolCall(name=step.name, call_id=f"step-{index}", arguments=step.arguments)
                    )
                    matched = strict_equal(_decoded(observation.content), step.reply)
                    kind = kinds[reply_kind(step.reply)]
                    kind[0] += matched
                    kind[1] += 1
                    steps += 1
                    exact += matched
                    induced += observation.metadata.get("transition") == "induced"
                    by_lineage.setdefault(case.task.lineage_group_id, []).append(float(matched))
            seconds.append(clock() - started)
        outcomes.append(by_lineage)
        measured.append(
            ArmReplay(
                arm=name,
                steps=steps,
                exact=exact,
                reply_kinds=tuple(
                    ReplyKindCount(kind=kind, exact=counts[0], steps=counts[1])
                    for kind, counts in kinds.items()
                ),
                induced_steps=induced,
                episode_seconds=tuple(seconds),
                lineage_exact=unit_mean_summary(by_lineage),
            )
        )
    return WorldReplayComparison(
        cases=len(cases),
        arms=tuple(measured),
        exact_difference=unit_difference_summary(outcomes[0], outcomes[1])
        if len(outcomes) > 1
        else None,
    )


def _prefix(case: ReplayCase) -> list[tuple[str, JsonObject]]:
    """Return the public prefix calls a case's world must apply before measured steps."""
    rows = case.task.initial_context.get("public_prefix", [])
    calls = []
    for row in rows if isinstance(rows, list) else []:
        action = row.get("action") if isinstance(row, dict) else None
        if isinstance(action, dict) and isinstance(action.get("name"), str):
            arguments = action.get("arguments")
            calls.append((str(action["name"]), arguments if isinstance(arguments, dict) else {}))
    return calls


def _decoded(content: str) -> JsonValue:
    """Decode a reply from JSON text when it is JSON, matching how recorded replies are read."""
    try:
        return json.loads(content)
    except ValueError:
        return content
