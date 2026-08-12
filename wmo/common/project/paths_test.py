"""Tests for safe project and immutable artifact path construction."""

from __future__ import annotations

from pathlib import Path

import pytest

from wmo.common.project import ProjectPathError, ProjectPaths


def test_project_paths_reject_traversal_and_absolute_artifact_file_paths(tmp_path: Path) -> None:
    """Artifact path helpers cannot escape the assigned project directory."""
    paths = ProjectPaths(root=tmp_path / ".wmo", project_id="support-project")

    with pytest.raises(ProjectPathError):
        paths.artifact_directory("../outside")
    with pytest.raises(ProjectPathError):
        paths.artifact_file("artifact-1", "../outside.json")
    with pytest.raises(ProjectPathError):
        paths.artifact_file("artifact-1", "/outside.json")


def test_project_paths_reject_a_symlink_escape(tmp_path: Path) -> None:
    """Resolved artifact data paths must remain below their artifact directory."""
    paths = ProjectPaths(root=tmp_path / ".wmo", project_id="support-project")
    artifact_directory = paths.artifact_directory("artifact-1")
    artifact_directory.mkdir(parents=True)
    outside = tmp_path / "outside"
    outside.mkdir()
    (artifact_directory / "link").symlink_to(outside, target_is_directory=True)

    with pytest.raises(ProjectPathError, match="escapes"):
        paths.artifact_file("artifact-1", "link/data.json")
