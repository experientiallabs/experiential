"""Tests for safe project and immutable artifact path construction."""

from __future__ import annotations

from pathlib import Path

import pytest

from wmo.common.project import ProjectPathError, ProjectPaths


def test_project_paths_reject_traversal_artifact_directories(tmp_path: Path) -> None:
    """Artifact path helpers cannot escape the assigned project directory."""
    paths = ProjectPaths(root=tmp_path / ".wmo", project_id="support-project")

    with pytest.raises(ProjectPathError):
        paths.artifact_directory("../outside")


def test_project_paths_keep_runtime_state_outside_immutable_artifacts(tmp_path: Path) -> None:
    """The mutable journal has one project-local path that cannot overlap artifacts."""
    paths = ProjectPaths(root=tmp_path / ".wmo", project_id="support-project")

    assert paths.runtime_directory == tmp_path / ".wmo/projects/support-project/runtime"
    assert paths.runtime_journal == paths.runtime_directory / "interactions.jsonl"
    assert not paths.runtime_journal.is_relative_to(paths.artifacts_directory)
