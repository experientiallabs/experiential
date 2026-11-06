"""A tag monitor halt blocks the next front-door phase until it is acknowledged.

The real front door (build, families, rules and registered starts over the bootstrap fixture)
runs with the ``tag-monitor-rules`` slot (checkpoint 3) after the preparation. Its scripted
gatherer hands it two admitted rules whose compiled tags never fire, so the statistics halt. The
run stops after the monitor slot with the halt's reason; no pipeline runs. A rerun replays the
saved report and stops again. Once a person acknowledges the halt, the rerun continues into every
start's pipeline.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from exp.optimize.evaluation.variance.frontdoor import (
    FrontDoorApproval,
    FrontDoorDependencies,
    plan_front_door,
    run_front_door,
)
from exp.optimize.evaluation.variance.frontdoor_executors import SlotContext
from exp.optimize.evaluation.variance.frontdoor_states import all_statecharts
from exp.optimize.evaluation.variance.frontdoor_test import front_door
from exp.optimize.evaluation.variance.tag_monitor import acknowledge_halt, load_report
from exp.optimize.evaluation.variance.tag_monitor_gather import Gathered
from exp.optimize.evaluation.variance.tag_monitor_slot import TagMonitorExecutor
from exp.optimize.evaluation.variance.tag_monitor_stats import RuleEvidence, RuleInputs
from exp.simulation.rules.admission import AdmissionVerdict, RuleTagMatrix, TrajectoryTags


def dead_rules(context: SlotContext) -> Gathered | None:
    """Two admitted rules whose compiled questions never fire on any trajectory."""
    if "preparation" not in context.results:
        return None
    rules = tuple(
        RuleEvidence(
            verdict=AdmissionVerdict(
                rule_id=rule,
                admitted=True,
                status="supported",
                supporting=("t1", "t2"),
                supporting_lineages=("L1", "L2"),
                reason="fixture",
            ),
            matrix=RuleTagMatrix(
                rule_id=rule,
                question_ids=(f"{rule}.applies.q",),
                rows=tuple(
                    TrajectoryTags(trace_id=t, lineage_id=f"L{t}", values=(0.1,))
                    for t in ("t1", "t2", "t3")
                ),
            ),
        )
        for rule in ("rule.a", "rule.b")
    )
    return RuleInputs(rules=rules), (context.results["preparation"],), "rules-fixture"


def test_a_monitor_halt_blocks_the_pipelines_until_acknowledged(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Stopped with the halt's reason, stopped again on rerun, continued once acknowledged."""
    run = front_door(tmp_path, monkeypatch)
    project = run.project
    spec = run.spec.model_copy(update={"slots": {"tag-monitor-rules": {}}})
    deps = FrontDoorDependencies(
        builder=run.builder,
        bootstrap=run.dependencies.bootstrap,
        executors=run.dependencies.executors,
        slots={"tag-monitor-rules": TagMonitorExecutor("rules", gather=dead_rules)},
    )

    def consent() -> FrontDoorApproval:
        """Consent to the current plan."""
        plan = plan_front_door(project, spec, code_revision="test", dependencies=deps)
        return FrontDoorApproval(plan_sha256=plan.digest, ceiling_usd=10.0, unpriced_attempts=1_000)

    plan = plan_front_door(project, spec, code_revision="test", dependencies=deps)
    stages = [row.stage for row in plan.stages]
    assert stages.index("slot tag-monitor-rules") == stages.index("bootstrap-preparation") + 1
    first = run_front_door(
        project, spec, code_revision="test", approval=consent(), dependencies=deps
    )
    assert first.stopped is not None and "checkpoint 3 (rules) halted" in first.stopped
    assert "rules_never_fire" in first.stopped and "acknowledge" in first.stopped.lower()
    assert first.preparation is not None and first.pipelines == ()
    assert not run.stages["error-mining"].runs
    report = load_report(project, first.slots["tag-monitor-rules"])
    (halt,) = report.halts
    assert report.number == 3 and report.state == "halted"

    replanned = plan_front_door(project, spec, code_revision="test", dependencies=deps)
    (monitor,) = [row for row in replanned.stages if row.stage == "slot tag-monitor-rules"]
    assert monitor.status == "done" and any(line.startswith("HALT") for line in monitor.lines)
    second = run_front_door(
        project, spec, code_revision="test", approval=consent(), dependencies=deps
    )
    assert second.stopped == first.stopped and not run.stages["error-mining"].runs

    acknowledge_halt(project, halt.halt_id, by="raghav", note="fixture rules; proceed")
    third = run_front_door(
        project, spec, code_revision="test", approval=consent(), dependencies=deps
    )
    assert third.stopped is None and third.complete
    assert len(run.stages["error-mining"].runs) == len(third.bindings) == 2


def test_every_monitor_machine_is_in_the_statecharts() -> None:
    """The checkpoint, halt and proposal machines render with the pipeline's other machines."""
    charts = all_statecharts()
    for name in ("tag-monitor-checkpoint", "tag-monitor-halt", "tag-monitor-proposal"):
        assert charts[name].startswith("stateDiagram-v2")
    assert "sampled --> review_skipped: review_skipped" in charts["tag-monitor-checkpoint"]
    assert "raised --> acknowledged: acknowledged" in charts["tag-monitor-halt"]
