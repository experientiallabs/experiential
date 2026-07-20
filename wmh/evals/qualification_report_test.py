"""Tests for identity-safe benchmark qualification reports."""

from __future__ import annotations

import hashlib

import pytest

from wmh.agents import default_agent
from wmh.evals.harbor.config import HarborEnvironmentBackend
from wmh.evals.harbor.paired_runner import (
    HarborExecutionPlan,
    PrequalifiedHarborRoster,
    QualifiedHarborTask,
)
from wmh.evals.qualification_report import BenchmarkQualificationReport
from wmh.evals.study_provenance import HarnessOptimizationCodeProvenance


def _digest(value: str) -> str:
    return "sha256:" + hashlib.sha256(value.encode()).hexdigest()


def _provenance() -> HarnessOptimizationCodeProvenance:
    return HarnessOptimizationCodeProvenance(
        baseline_source_commit="5f0a5b056be7eedea41591dad8b25f836a243cc6",
        launch_orchestration_commit="1" * 40,
    )


def _roster(plan: HarborExecutionPlan) -> PrequalifiedHarborRoster:
    return PrequalifiedHarborRoster(
        execution_plan_digest=plan.digest,
        tasks=tuple(
            QualifiedHarborTask(
                task_id=task_id,
                dataset_id="private-source",
                content_digest=_digest(f"content:{task_id}"),
                task_key=_digest(f"key:{task_id}"),
                task_environment_digest=_digest(f"environment:{task_id}"),
                environment_backend=HarborEnvironmentBackend.LOCAL,
            )
            for task_id in ("confirmation-secret", "discovery-secret")
        ),
    )


def test_report_binds_roster_and_code_without_disclosing_task_identities() -> None:
    plan = HarborExecutionPlan.freeze(
        reference_harness=default_agent("baseline"),
        reward_key="reward",
    )
    roster = _roster(plan)
    provenance = _provenance()

    report = BenchmarkQualificationReport.capture(
        code_provenance=provenance,
        execution_plan=plan,
        roster=roster,
    )

    report.validate_roster(
        code_provenance=provenance,
        execution_plan=plan,
        roster=roster,
    )
    published = report.model_dump_json()
    assert report.code_provenance == provenance
    assert report.qualified_roster_digest == roster.digest
    assert report.qualified_task_count == 2
    assert "discovery-secret" not in published
    assert "confirmation-secret" not in published
    assert "private-source" not in published
    assert "content:" not in published


def test_report_rejects_a_different_roster_with_the_same_task_count() -> None:
    plan = HarborExecutionPlan.freeze(
        reference_harness=default_agent("baseline"),
        reward_key="reward",
    )
    roster = _roster(plan)
    provenance = _provenance()
    report = BenchmarkQualificationReport.capture(
        code_provenance=provenance,
        execution_plan=plan,
        roster=roster,
    )
    tasks = list(roster.tasks)
    tasks[1] = tasks[1].model_copy(
        update={"task_environment_digest": _digest("different-environment")}
    )
    drifted = PrequalifiedHarborRoster(
        execution_plan_digest=plan.digest,
        tasks=tuple(tasks),
    )

    with pytest.raises(ValueError, match="exact qualified roster"):
        report.validate_roster(
            code_provenance=provenance,
            execution_plan=plan,
            roster=drifted,
        )


def test_report_rejects_a_different_launch_commit() -> None:
    plan = HarborExecutionPlan.freeze(
        reference_harness=default_agent("baseline"),
        reward_key="reward",
    )
    roster = _roster(plan)
    provenance = _provenance()
    report = BenchmarkQualificationReport.capture(
        code_provenance=provenance,
        execution_plan=plan,
        roster=roster,
    )
    changed = provenance.model_copy(update={"launch_orchestration_commit": "2" * 40})

    with pytest.raises(ValueError, match="exact qualified roster"):
        report.validate_roster(
            code_provenance=changed,
            execution_plan=plan,
            roster=roster,
        )
