"""Tests for safe project and immutable artifact path construction."""

from __future__ import annotations

from pathlib import Path

import pytest

from exp.common.project import ProjectPathError, ProjectPaths


@pytest.mark.parametrize("project_id", ["../outside", "/outside", "a/b", "a\\b"])
def test_project_paths_reject_unsafe_project_identity(tmp_path: Path, project_id: str) -> None:
    """A project namespace cannot escape its filesystem or SQLite ownership boundary."""
    with pytest.raises(ProjectPathError):
        ProjectPaths(root=tmp_path, project_id=project_id)


def test_project_paths_keep_runtime_state_outside_immutable_artifacts(tmp_path: Path) -> None:
    """The mutable journal has one project-local path that cannot overlap artifacts."""
    paths = ProjectPaths(root=tmp_path / ".exp", project_id="support-project")

    assert paths.runtime_directory == tmp_path / ".exp/projects/support-project/runtime"
    assert paths.runtime_journal == paths.runtime_directory / "interactions.jsonl"
