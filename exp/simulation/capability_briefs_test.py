"""Capability hypotheses retain uncertainty, tool requirements and immutable task intent."""

from datetime import UTC, datetime
from pathlib import Path

import pytest

from exp.common.project import ProjectStore
from exp.common.tasks import TaskCase
from exp.simulation.capability_briefs import (
    CapabilityAssumption,
    CapabilityBrief,
    RequiredInformation,
    load_capability_brief,
    persist_capability_brief,
)
from exp.simulation.evaluation.tasks import EvaluationTask, TaskVerifier, persist_evaluation_task
from exp.simulation.evaluation.tasks_test import specification


def brief() -> CapabilityBrief:
    """Specify a frontier reconciliation need with an unresolved concurrency assumption."""
    return CapabilityBrief(
        category="frontier",
        novelty="new_situation",
        user="operations analyst",
        outcome="Reconcile only authorized rows",
        capability="reconciliation under concurrent updates",
        gap="Recorded requests never interleave a new update",
        required_tools=(),
        required_information=(
            RequiredInformation(fact="authorization", obtain_via="public request context"),
        ),
        allowed_changes=("authorized rows",),
        preserved=("all unrelated sheets",),
        unavailable_behavior="Ask for missing authorization",
        success_evidence=("authorized rows match and all other data is preserved",),
        assumptions=(
            CapabilityAssumption(
                statement="Concurrent edits serialize per workbook", authority="speculation"
            ),
        ),
    )


def test_unknown_behavior_remains_explicit_and_revision_cannot_replace_brief(
    tmp_path: Path,
) -> None:
    """A task can be investigated, but actor success cannot resolve its unsupported assumptions."""
    project = ProjectStore(tmp_path, "case")
    original = brief()
    reference = persist_capability_brief(
        project.artifacts, original, created_at=datetime.now(UTC), code_revision="test"
    )
    assert load_capability_brief(project.artifacts, reference) == original
    assert original.blockers() == (
        "unresolved assumption: Concurrent edits serialize per workbook",
    )
    source = specification(project)
    task = EvaluationTask(
        intent=original.outcome,
        task=TaskCase(
            task_id="task",
            lineage_group_id="family",
            origin="authored",
            partition="fit",
            instruction="Reconcile the supplied workbook.",
            workload_weight=1,
        ),
        world_model=source,
        checkpoint=source,
        verifier=TaskVerifier(description="fixture", outcome_schema={}),
        challenges=(),
        capability_brief=reference,
    )
    parent = persist_evaluation_task(
        project.artifacts, task, created_at=datetime.now(UTC), code_revision="test"
    )
    with pytest.raises(ValueError, match="capability brief"):
        persist_evaluation_task(
            project.artifacts,
            task.model_copy(update={"parent": parent, "capability_brief": None}),
            created_at=datetime.now(UTC),
            code_revision="test",
        )
    with pytest.raises(ValueError, match="unavailable tools"):
        original.model_copy(update={"required_tools": ("invented_tool",)}).require_task(
            task.task, task.intent
        )


def test_a_source_label_without_evidence_is_not_an_observation() -> None:
    """No source-free assertion can turn an unobserved capability into grounded behavior."""
    with pytest.raises(ValueError, match="immutable sources"):
        CapabilityAssumption(
            statement="Concurrent edits serialize", authority="recorded_observation"
        )
