"""Miss retest: replay the learned traces the simulator got wrong, on the final memory.

Learning predicts each trace before learning it (test-then-train), so its misses are the turns
the simulator could not yet reproduce. This retest replays every learned trace with at least
one inexact reply against the final memory checkpoint, frozen, and compares each turn before
and after learning: fixed, still wrong, or regressed.

The retest reads what memory learned from the trace itself, so it checks retention (did the
lessons of a miss stick), not generalization; test-then-train fidelity is the generalization
measure. It runs through the shared trace-replay path and the existing fidelity owner.
"""

from collections.abc import Callable
from datetime import datetime

from exp.common.core.artifacts import ArtifactInput, ContractModel
from exp.common.evaluations.reconstruction import DualReport
from exp.common.project import ProjectStore
from exp.optimize.evaluation.fidelity import FidelityMemory, PreparedFidelity
from exp.optimize.evaluation.held_out_fidelity import PANEL_JUDGE_EFFORT
from exp.optimize.evaluation.trace_learning import learned_runs, learning_plan
from exp.optimize.evaluation.trace_replay import (
    DEADLINE,
    ReplayCase,
    ReplayResult,
    register_replay,
    replay_spec,
    saved_results,
)
from exp.simulation.world_model.experience_memory import memory_head

LABEL = "retest"


class RetestPanel(ContractModel):
    """Registered retest runs of learned traces that had a miss, with their learning results.

    Attributes:
        world_model: Hybrid simulator the memory was learned on.
        memory_version: Final memory checkpoint every retest reads, frozen.
        learned: Number of traces learned so far.
        cases: Learned traces with at least one inexact first-measurement reply, in trace order.
        before: Each case's first saved test-then-train measurement, in the same order.
        prepared: One saved fidelity registration per case, in the same order.
        retired: Earlier attempts that could not complete and were replaced.
    """

    world_model: ArtifactInput
    memory_version: int
    learned: int
    cases: tuple[ReplayCase, ...]
    before: tuple[ReplayResult, ...]
    prepared: tuple[PreparedFidelity, ...]
    retired: tuple[str, ...] = ()


class TurnChanges(ContractModel):
    """Per-turn comparison of learning (before) and retest (after) for the same traces.

    Attributes:
        missed: Turns inexact before learning.
        fixed: Of those, turns exact on retest.
        regressed: Turns exact before learning and inexact on retest.
        before_exact: Exact turns before learning.
        after_exact: Exact turns on retest.
        steps: Compared turns.
    """

    missed: int
    fixed: int
    regressed: int
    before_exact: int
    after_exact: int
    steps: int


def register_miss_retest(
    project: ProjectStore,
    count: int,
    *,
    code_revision: str,
    clock: Callable[[], datetime],
    retry_incomplete: bool = False,
) -> RetestPanel:
    """Register retests of up to ``count`` missed traces without provider calls or credentials.

    Args:
        project: Project with a completed build and learning pass.
        count: Maximum traces, the earliest misses in trace order first.
        code_revision: Exact producing implementation identity.
        clock: Aware clock for deadlines and attempt retirement.
        retry_incomplete: Replace interrupted runs that already bought calls, buying them again.

    Returns:
        The panel; unchanged completed runs are their saved registrations.

    Raises:
        ValueError: Nothing learned yet, or no learned trace had a miss.
    """
    # The retest belongs to the build: it reads the build's all-traces memory, never the
    # audit's fit-only memory, even when the audit learned more recently.
    plan = learning_plan(project, scope="all")
    head = memory_head(project.artifacts, plan.lineage)
    runs = learned_runs(project, plan)
    if head is None or not runs:
        raise ValueError(
            "nothing has been learned yet; run exp build PROJECT with a native decision model "
            "for memory recall in the catalog"
        )
    missed = [(case, result) for case, _, result in runs if result.exact < result.steps][:count]
    if not missed:
        raise ValueError(
            f"every one of the {len(runs)} learned traces was reproduced exactly before it was "
            "learned; there is no miss to retest"
        )
    memory = FidelityMemory(
        lineage=plan.lineage, mode="frozen", checkpoint=head.version, withhold_own=False
    )
    now = clock()
    prepared: list[PreparedFidelity] = []
    retired: list[str] = []
    for case, _ in missed:
        spec = replay_spec(
            project,
            plan.world_model,
            case,
            deadline=now + DEADLINE,
            label=LABEL,
            memory=memory,
            retrieval=plan.recall,
            judge_effort=PANEL_JUDGE_EFFORT,
        )
        registration, replaced = register_replay(
            project,
            spec,
            label=LABEL,
            trace=case.trace_id,
            code_revision=code_revision,
            now=now,
            retry_incomplete=retry_incomplete,
        )
        prepared.append(registration)
        retired.extend(replaced)
    return RetestPanel(
        world_model=plan.world_model,
        memory_version=head.version,
        learned=len(runs),
        cases=tuple(case for case, _ in missed),
        before=tuple(result for _, result in missed),
        prepared=tuple(prepared),
        retired=tuple(retired),
    )


def saved_retest_results(project: ProjectStore, panel: RetestPanel) -> tuple[ReplayResult, ...]:
    """Results of every retest run that has a saved report, without provider calls.

    Args:
        project: Project holding the saved reports.
        panel: Registered panel.

    Returns:
        One result per completed run, in panel order.
    """
    return saved_results(project, panel.cases, panel.prepared)


def turn_changes(panel: RetestPanel, after: tuple[ReplayResult, ...]) -> TurnChanges:
    """Compare each completed retest with its learning run, turn by turn.

    Args:
        panel: Registered panel with each case's learning result.
        after: Completed retest results.

    Returns:
        Fixed, regressed and exact counts over the completed traces' turns.
    """
    before = {result.trace_id: result for result in panel.before}
    missed = fixed = regressed = before_exact = after_exact = steps = 0
    for result in after:
        old, new = exact_turns(before[result.trace_id].report), exact_turns(result.report)
        for was, now in zip(old, new, strict=True):
            steps += 1
            before_exact += was
            after_exact += now
            missed += not was
            fixed += not was and now
            regressed += was and not now
    return TurnChanges(
        missed=missed,
        fixed=fixed,
        regressed=regressed,
        before_exact=before_exact,
        after_exact=after_exact,
        steps=steps,
    )


def exact_turns(report: DualReport) -> list[bool]:
    """Whether each planned turn's simulated reply equals the recorded reply exactly.

    Args:
        report: Saved dual report of one replay.

    Returns:
        One flag per planned turn; a turn without a mechanical row is not exact.
    """
    rows = report.mechanical.get("steps")
    flags = [
        isinstance(row, dict) and row.get("trajectory_reply_exact") is True
        for row in (rows if isinstance(rows, list) else [])
    ]
    return (flags + [False] * report.planned_steps)[: report.planned_steps]
