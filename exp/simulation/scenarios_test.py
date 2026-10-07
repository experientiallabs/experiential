"""Authored scenario selection over a completed build's unchanged trace evidence."""

import math
from pathlib import Path

import pytest

from exp.common.tasks import load_task_set
from exp.optimize.router.automatic.service_test import _completed_project
from exp.simulation.mining.bindings import load_task_set_lineage_bindings
from exp.simulation.scenarios import (
    AuthoredScenario,
    authored_task,
    select_authored_scenarios,
)


def _scenario(scenario_id: str, **fields: str) -> AuthoredScenario:
    """Return one plain authored scenario.

    Args:
        scenario_id: Caller identity.
        **fields: Optional field overrides.

    Returns:
        Authored scenario with a deterministic instruction.
    """
    return AuthoredScenario.model_validate(
        {"scenario_id": scenario_id, "instruction": f"Handle {scenario_id}.", **fields}
    )


def test_selection_adds_authored_tasks_and_keeps_build_evidence(tmp_path: Path) -> None:
    """The selected set appends authored tasks, verifies its lineage, and rebinds the review."""
    project, _, _ = _completed_project(tmp_path)
    original = project.load_project().build
    assert original is not None
    mined = load_task_set(project.artifacts, original.task_set.artifact_id).tasks

    selected = select_authored_scenarios(
        project, (_scenario("refund", reference_answer="Refunded."), _scenario("lookup"))
    )
    build = project.load_project().build
    assert build is not None and build.task_set == selected
    assert build.model_copy(update={"task_set": original.task_set}) == original
    tasks = load_task_set(project.artifacts, selected.artifact_id).tasks
    assert tasks[: len(mined)] == mined
    assert [task.origin for task in tasks[len(mined) :]] == ["authored", "authored"]
    assert tasks[len(mined)].reference_answer == "Refunded."
    assert all(task.source_trace_ids == () for task in tasks[len(mined) :])
    bindings = load_task_set_lineage_bindings(project.artifacts, selected.artifact_id)
    assert bindings.task_set_id == selected.artifact_id
    review = project.read_review()
    assert isinstance(review, dict)
    assert review["build_review"]["task_set"] == selected.model_dump(mode="json")

    assert (
        select_authored_scenarios(
            project, (_scenario("refund", reference_answer="Refunded."), _scenario("lookup"))
        )
        == selected
    )
    replaced = select_authored_scenarios(project, (_scenario("other"),))
    replaced_tasks = load_task_set(project.artifacts, replaced.artifact_id).tasks
    assert len(replaced_tasks) == len(mined) + 1
    weight = math.fsum(task.workload_weight for task in mined) / len(mined)
    assert replaced_tasks[-1] == authored_task(_scenario("other"), workload_weight=weight)
    assert select_authored_scenarios(project, ()) == original.task_set
    restored = project.load_project().build
    assert restored == original


def test_selection_rejects_duplicate_ids_and_tools() -> None:
    """A caller's scenario and tool identities must be unambiguous."""
    with pytest.raises(ValueError, match="unique names"):
        AuthoredScenario.model_validate(
            {
                "scenario_id": "lookup",
                "instruction": "Look up order 7.",
                "tools": [
                    {"name": "lookup", "description": "Find.", "input_schema": {}},
                    {"name": "lookup", "description": "Find again.", "input_schema": {}},
                ],
            }
        )


def test_selection_rejects_duplicate_scenario_ids(tmp_path: Path) -> None:
    """Repeating a scenario identity fails before any artifact is written."""
    project, _, _ = _completed_project(tmp_path)
    before = project.load_project()
    with pytest.raises(ValueError, match="must be unique"):
        select_authored_scenarios(project, (_scenario("refund"), _scenario("refund")))
    assert project.load_project() == before


def test_authored_task_identity_binds_every_field() -> None:
    """Changing any authored field changes the task identity; the lineage follows the ID."""
    base = authored_task(_scenario("refund"), workload_weight=1.0)
    changed = authored_task(_scenario("refund", reference_answer="Refunded."), workload_weight=1.0)
    assert base.task_id != changed.task_id
    assert base.lineage_group_id == changed.lineage_group_id
    assert base.partition == "fit" and base.origin == "authored"
