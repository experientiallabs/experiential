"""Saved variance-build artifacts are content-addressed, adopted when identical and pinned."""

from pathlib import Path

import pytest

from exp.common.core.artifacts import ArtifactInput
from exp.common.project import ProjectConfig, ProjectStore
from exp.optimize.evaluation.variance.build_records import (
    EXPOSURE_TYPE,
    MATRIX_TYPE,
    load_exposure_plan,
    load_tag_matrix,
    save_artifact,
)
from exp.optimize.evaluation.variance.exposure import ExposurePlan


def _project(root: Path) -> ProjectStore:
    """Initialize an empty project."""
    project = ProjectStore(root, "records")
    project.initialize(ProjectConfig(project_id="records"))
    return project


def _plan(seed: int) -> ExposurePlan:
    """A two-lineage exposure plan."""
    return ExposurePlan(
        bindings=ArtifactInput(artifact_id="bindings", sha256="0" * 64),
        seed=seed,
        probe_fraction=0.5,
        roles={"a": "world_learning", "b": "mining_probes"},
        traces={"t-a": "a", "t-b": "b"},
    )


def test_an_identical_save_adopts_the_saved_artifact(tmp_path: Path) -> None:
    """Saving the same content twice returns one reference and loads it back."""
    project = _project(tmp_path)
    files = {"exposure.json": _plan(0)}

    first = save_artifact(project.artifacts, EXPOSURE_TYPE, files, (), "test")

    assert save_artifact(project.artifacts, EXPOSURE_TYPE, files, (), "test") == first
    assert load_exposure_plan(project, first) == _plan(0)
    other = save_artifact(project.artifacts, EXPOSURE_TYPE, {"exposure.json": _plan(1)}, (), "t")
    assert other != first
    assert len(project.artifacts.list_ids(artifact_type=EXPOSURE_TYPE)) == 2


def test_a_reader_refuses_another_artifact_type(tmp_path: Path) -> None:
    """A reference is read only as the artifact type it was saved as."""
    project = _project(tmp_path)
    saved = save_artifact(project.artifacts, EXPOSURE_TYPE, {"exposure.json": _plan(0)}, (), "t")

    with pytest.raises(ValueError, match=f"pinned {MATRIX_TYPE}"):
        load_tag_matrix(project, saved)
