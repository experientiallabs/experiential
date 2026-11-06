"""Versions rebind both rubric parts, pin their banks, and are recorded in package provenance."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from exp.optimize.evaluation.continuations.job_test import job_fixture
from exp.optimize.evaluation.executable.task import ExecutableTask, MiningProvenance
from exp.optimize.evaluation.variance.dense_vocabulary import LOOK_ALIKES
from exp.optimize.evaluation.variance.rubric_evolution_test import world
from exp.optimize.evaluation.variance.rubric_items_test import item
from exp.optimize.evaluation.variance.rubric_versions import (
    RecallRecord,
    bind_version,
    first_version,
    rubric_provenance,
    version_banks,
    versioned_dense,
    versioned_judge,
)


def test_a_version_appends_its_criteria_and_dense_tags_and_changes_both_identities(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The base rubric is kept; the item is a required criterion and a decisive cited penalty."""
    project, registration, _, _ = world(tmp_path, monkeypatch)
    banks = registration.banks()
    task = registration.tasks[0]
    v1 = first_version(registration.run_id, banks)
    assert versioned_judge(project, v1, task.judge, code_revision="test") == task.judge
    v2 = v1.model_copy(update={"version": 2, "parent": v1.identity, "items": (item("fixture"),)})
    judge = versioned_judge(project, v2, task.judge, code_revision="test")
    assert judge.definition.identity != task.judge.definition.identity
    stored = project.artifacts.read(judge.definition.rubric.artifact_id)
    content = json.loads(stored.payloads["rubric.json"])
    criteria = content["body"]["criteria"]
    assert [c["criterion_id"] for c in criteria] == [
        "tests_pass",
        "kept_expectations",
        "evo.fixture",
    ]
    assert criteria[-1]["role"] == "required_correctness"
    assert content["rubric_version"] == {"version": 2, "sha256": v2.identity}
    dense, step = versioned_dense(v2, banks, task.dense)
    tag = dense.definition.tag("rubric.fixture")
    assert tag.severity == "decisive" and tag.obligation_required
    assert tag.acceptable_when == LOOK_ALIKES and "rubric.fixture" in step.question_ids()
    assert dense.definition.identity != task.dense.definition.identity
    changed = banks[0].model_copy(update={"version": 2})
    with pytest.raises(ValueError, match="changed"):
        version_banks(v2, [changed, banks[1]])
    bookkeeping = v2.model_copy(update={"items": (item("fixture", quiet=1),)})
    assert bookkeeping.identity == v2.identity
    retired = v2.model_copy(
        update={"items": (item("fixture").model_copy(update={"status": "retired"}),)}
    )
    assert retired.identity != v2.identity


def test_a_package_records_its_rubric_version_in_mining_provenance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Binding a version changes the task identity and keeps provenance complete."""
    _, job, _ = job_fixture(tmp_path / "job", monkeypatch)
    project, registration, _, _ = world(tmp_path / "world", monkeypatch)
    banks = registration.banks()
    task_rubric = registration.tasks[0]
    recall = RecallRecord(
        held_out_sha256=registration.held_out.identity, hacks=2, outcome_caught=2, dense_caught=2
    )
    v1 = first_version(registration.run_id, banks)
    v2 = v1.model_copy(
        update={"version": 2, "epoch": 2, "parent": v1.identity, "items": (item("fixture"),)}
    ).model_copy(update={"recall": recall})
    provenance = MiningProvenance(
        task_id="t", step=3, tier="glm-5.3", tier_pass_rates={"glm-5.3": 0.4}
    )
    assert "rubric" not in provenance.model_dump(mode="json")
    task = ExecutableTask(
        intent="Repair parse_date.",
        job=job,
        judge=task_rubric.judge,
        dense=task_rubric.dense,
        provenance=provenance,
    )
    judge = versioned_judge(project, v2, task_rubric.judge, code_revision="test")
    dense, _ = versioned_dense(v2, banks, task_rubric.dense)
    bound = bind_version(task, v2, judge, dense)
    assert bound.provenance is not None and bound.provenance.rubric == rubric_provenance(v2)
    record = bound.provenance.rubric
    assert record is not None
    assert (record.version, record.epoch, record.parent_sha256) == (2, 2, v1.identity)
    assert record.items == (item("fixture").identity,) and record.outcome_recall == 1.0
    assert bound.profile() != task.profile() and bound.judge == judge
    MiningProvenance.model_validate(bound.provenance.model_dump(mode="json"))
    with pytest.raises(ValueError, match="mining provenance"):
        bind_version(task.model_copy(update={"provenance": None}), v2, judge, dense)


def test_recall_lowering_is_read_per_part_on_one_held_out_set() -> None:
    """Either part dropping counts; different held-out sets are refused."""
    base = RecallRecord(held_out_sha256="a" * 64, hacks=4, outcome_caught=2, dense_caught=3)
    same = base.model_copy(update={"outcome_caught": 3})
    lower = base.model_copy(update={"dense_caught": 2})
    assert base.lowered_by(same) == () and base.lowered_by(lower) == ("dense",)
    with pytest.raises(ValueError, match="different held-out"):
        base.lowered_by(base.model_copy(update={"held_out_sha256": "b" * 64}))
