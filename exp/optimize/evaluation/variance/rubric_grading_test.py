"""Grades carry their version; task rubrics and held-out sets are checked on construction."""

from __future__ import annotations

from pathlib import Path

import pytest

from exp.optimize.evaluation.variance.rubric_evolution_test import (
    HELD_OUT,
    RUN,
    genuine,
    weakened,
    world,
)
from exp.optimize.evaluation.variance.rubric_grading import (
    HeldOutHack,
    HeldOutSet,
    TaskRubric,
    grade_rollout,
    measure_recall,
    require_label,
)
from exp.optimize.evaluation.variance.rubric_versions import first_version


def test_v1_catches_the_known_hack_in_both_parts_with_cited_reasons(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The deleted assertion fails both parts; a genuine repair passes; replay buys nothing."""
    project, registration, clef, wire = world(tmp_path, monkeypatch)
    banks = registration.banks()
    v1 = first_version(registration.run_id, banks)
    task = registration.tasks[0]
    hack = grade_rollout(
        project,
        v1,
        banks,
        task,
        weakened("w", "L1"),
        source=RUN,
        tagger=clef,
        routing=registration.spec.routing,
        code_revision="test",
    )
    assert hack.fails_both and hack.decisive == ("modified_evaluation",)
    (finding,) = [f for f in hack.findings if f.tag_id == "modified_evaluation"]
    assert finding.obligation_refs == ("e0",) and finding.change_refs == ("e1",)
    assert require_label(hack, v1) is hack
    record, grades = measure_recall(
        project,
        v1,
        banks,
        registration.task_map(),
        HELD_OUT,
        tagger=clef,
        routing=registration.spec.routing,
        code_revision="test",
    )
    assert (record.outcome_caught, record.dense_caught, record.hacks) == (1, 1, 2)
    calls = len(wire.requests)
    again, _ = measure_recall(
        project,
        v1,
        banks,
        registration.task_map(),
        HELD_OUT,
        tagger=clef,
        routing=registration.spec.routing,
        code_revision="test",
    )
    assert again == record and len(wire.requests) == calls and len(grades) == 2
    with pytest.raises(ValueError, match="task"):
        grade_rollout(
            project,
            v1,
            banks,
            task,
            genuine("g", "L2").model_copy(update={"task_id": "other"}),
            source=RUN,
            tagger=clef,
            routing=registration.spec.routing,
            code_revision="test",
        )


def test_task_rubrics_and_held_out_sets_are_checked(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Both parts bind one task and world; a held-out set never repeats a rollout."""
    _, registration, _, _ = world(tmp_path, monkeypatch)
    task = registration.tasks[0]
    other = task.dense.definition.model_copy(update={"task_sha256": "f" * 64})
    with pytest.raises(ValueError, match="other digests"):
        TaskRubric(
            task_id=task.task_id,
            judge=task.judge,
            dense=task.dense.model_copy(update={"definition": other}),
        )
    row = HeldOutHack(rollout=weakened("w", "H1"), hack_kind="deleted an assertion")
    with pytest.raises(ValueError, match="repeats"):
        HeldOutSet(set_id="x", source="s", hacks=(row, row))
    assert HELD_OUT.lineages() == {"H1", "H2"}
