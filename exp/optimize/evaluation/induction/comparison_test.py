"""Paired held-out replay of induced+predictor against the predictor alone, scripted."""

from itertools import count

import pytest

from exp.common.core.artifacts import JsonObject, JsonValue
from exp.optimize.evaluation.fidelity.replay import RecordedExchange, ReplayCase
from exp.optimize.evaluation.induction.comparison import compare_world_replay
from exp.runtime.environments.interface import EnvironmentRuntime
from exp.runtime.environments.worlds import ExecutableWorldRuntime
from exp.simulation.induction.backend import InducedWorldFactory
from exp.simulation.induction.sandbox import ProgramSandbox
from exp.simulation.induction.tests.fixtures_test import (
    RECORD_FIT_CALLS,
    RECORD_PROGRAMS,
    RECORD_TOOLS,
    grounded_fallback,
    programs,
    record_episode,
    record_manifest,
    task,
)
from exp.simulation.induction.validation import validate_programs

CALLS: tuple[tuple[str, JsonObject], ...] = (
    ("count_open", {}),
    ("open_ticket", {"title": "laptop"}),
    ("get_ticket", {"id": 2}),
    ("close_ticket", {"id": 2}),
    ("count_open", {}),
)


def _case(name: str, lineage: str) -> ReplayCase:
    """Record one native episode as a replay case with a one-call public prefix."""
    native = record_episode(name, CALLS)
    public = task(RECORD_TOOLS, task_id=name).model_copy(
        update={
            "lineage_group_id": lineage,
            "initial_context": {
                "public_prefix": [
                    {"action": {"name": "count_open", "arguments": {}}, "reply": {"open": 1}}
                ]
            },
        }
    )
    return ReplayCase(
        task=public,
        trace_id=name,
        prefix=1,
        steps=tuple(
            RecordedExchange(name=step.tool, arguments=step.arguments, reply=step.reply)
            for step in native.steps[1:]
        ),
    )


def test_induced_and_predicted_arms_are_measured_on_the_same_cases() -> None:
    """Induced code reproduces every reply; the scripted predictor misses some, and is timed."""
    initial = record_episode("start", CALLS[:1]).initial_state
    cases = (_case("case-a", "lineage-a"), _case("case-b", "lineage-b"))

    def predictor(case: ReplayCase) -> list[tuple[str, JsonValue, list[JsonObject]]]:
        """Script one plausible but imperfect prediction per call, prefix included."""
        del case
        return [
            ("prefix-0", {"open": 1}, []),
            ("step-0", {"id": 2, "status": "open"}, []),
            ("step-1", {"id": 2, "title": "laptop", "status": "new"}, []),
            ("step-2", {"id": 2, "status": "closed"}, []),
            ("step-3", {"open": 2}, []),
        ]

    with ProgramSandbox() as sandbox:
        coverage = validate_programs(
            programs(RECORD_PROGRAMS),
            (record_episode("fit", RECORD_FIT_CALLS),),
            record_manifest(),
            sandbox,
        ).coverage

        def induced(case: ReplayCase) -> EnvironmentRuntime:
            """Open induced programs over a fresh scripted predictor."""
            fallback, _, _ = grounded_fallback(
                record_manifest(), case.task, initial, predictor(case)
            )
            return ExecutableWorldRuntime(
                record_manifest(),
                InducedWorldFactory(coverage, fallback, sandbox),
                owner="simulated",
            )

        def predicted(case: ReplayCase) -> EnvironmentRuntime:
            """Open the scripted predictor alone."""
            fallback, _, _ = grounded_fallback(
                record_manifest(), case.task, initial, predictor(case)
            )
            return ExecutableWorldRuntime(record_manifest(), fallback, owner="simulated")

        ticks = count()
        comparison = compare_world_replay(
            cases, {"induced": induced, "predicted": predicted}, clock=lambda: float(next(ticks))
        )
    arms = {arm.arm: arm for arm in comparison.arms}
    assert (arms["induced"].exact, arms["induced"].steps) == (8, 8)
    assert arms["induced"].induced_steps == 8
    assert (arms["predicted"].exact, arms["predicted"].steps) == (4, 8)
    assert arms["predicted"].induced_steps == 0
    assert arms["induced"].episode_seconds == (1.0, 1.0)
    assert arms["induced"].lineage_exact["estimate"] == 1.0
    assert comparison.exact_difference is not None
    assert comparison.exact_difference["estimate"] == pytest.approx(0.5)
    assert comparison.exact_difference["independent_units"] == 2
    kinds = {row.kind: row for row in arms["predicted"].reply_kinds}
    assert sum(row.steps for row in kinds.values()) == 8


def test_at_least_one_arm_is_required() -> None:
    """An empty comparison is refused."""
    with pytest.raises(ValueError, match="at least one"):
        compare_world_replay((), {})
