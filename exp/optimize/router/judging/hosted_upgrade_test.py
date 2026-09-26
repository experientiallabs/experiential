"""A default hosted judge does not prevent subsequent rubric authoring."""

from pathlib import Path
from unittest.mock import patch

from exp.common.core.artifacts import ArtifactInput
from exp.common.project.project import ProjectHostedJudgeEvidence
from exp.optimize.router.judging.artifacts import write_production_rollout
from exp.optimize.router.judging.service import (
    prepare_manual_judge_calibration,
    prepare_manual_judge_setup,
    write_lineage_split,
)
from exp.optimize.router.judging.service_test import _TIME, _built_store, _catalog, _setup


def test_authoring_after_default_judge_retains_the_original_build(tmp_path: Path) -> None:
    """Adding judge provenance is not a change to the captured task or completed build."""
    store = _built_store(tmp_path)
    original = store.load_project()
    configured = original.model_copy(
        update={
            "hosted_judge": ProjectHostedJudgeEvidence(
                setup=ArtifactInput(artifact_id="default-setup", sha256="a" * 64),
                calibration=ArtifactInput(artifact_id="default-calibration", sha256="b" * 64),
            ),
        }
    )
    with patch.object(store, "load_project", return_value=configured):
        plan = prepare_manual_judge_setup(
            store,
            _catalog(),
            created_at=_TIME,
            code_revision="test",
        )
    assert original.build is not None
    assert plan.build.task_set == original.build.task_set


def test_new_judge_package_reuses_the_frozen_lineage_recipe(tmp_path: Path) -> None:
    """A package update preserves unchanged splits and their producer provenance."""
    store = _built_store(tmp_path)
    setup = _setup(store)
    plan = prepare_manual_judge_calibration(store, sample_size=1)
    inputs = (
        write_production_rollout(store, setup, plan.tasks[0], plan.traces[0], _TIME, "original"),
    )
    old = write_lineage_split(store, setup, plan, inputs, _TIME, "original")
    new = write_lineage_split(store, setup, plan, inputs, _TIME, "upgraded")
    assert old == new
    assert store.artifacts.read(old.split_id).manifest.code_revision == "original"
