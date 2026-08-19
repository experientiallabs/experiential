"""Safe paths for the small project-local `.wmo` layout."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from wmo.common.core.artifacts import validate_artifact_id


class ProjectPathError(ValueError):
    """A caller supplied an unsafe project, artifact, or artifact-file path."""


def validate_local_id(value: str, *, label: str) -> str:
    """Validate an ID that becomes one local filesystem path component.

    Args:
        value: Proposed project or artifact identifier.
        label: Human-readable identifier category for an error message.

    Returns:
        The validated identifier.

    Raises:
        ProjectPathError: The identifier is not a single canonical local path component.
    """
    try:
        return validate_artifact_id(value)
    except ValueError as exc:
        raise ProjectPathError(f"invalid {label} {value!r}: use a lowercase stable ID") from exc


@dataclass(frozen=True)
class ProjectPaths:
    """Resolves the canonical local layout for one named WMO project."""

    root: Path
    project_id: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "root", Path(self.root))
        validate_local_id(self.project_id, label="project ID")

    @property
    def projects_directory(self) -> Path:
        """Return the root directory that contains named project directories."""
        return self.root / "projects"

    @property
    def project_directory(self) -> Path:
        """Return this project's directory under `.wmo/projects/`."""
        return self.projects_directory / self.project_id

    @property
    def project_toml(self) -> Path:
        """Return the project configuration file path."""
        return self.project_directory / "project.toml"

    @property
    def review_json(self) -> Path:
        """Return the sole mutable review-draft file path."""
        return self.project_directory / "review.json"

    @property
    def runtime_directory(self) -> Path:
        """Return the mutable runtime-state directory outside immutable artifacts."""
        return self.project_directory / "runtime"

    @property
    def runtime_journal(self) -> Path:
        """Return the append-only routed-interaction journal path."""
        return self.runtime_directory / "interactions.jsonl"

    @property
    def artifacts_directory(self) -> Path:
        """Return the directory that contains completed immutable artifacts."""
        return self.project_directory / "artifacts"

    def artifact_directory(self, artifact_id: str) -> Path:
        """Return the safe directory reserved for one immutable artifact.

        Args:
            artifact_id: Stable local artifact identifier.

        Returns:
            The path beneath this project's artifact directory.
        """
        return self.artifacts_directory / validate_local_id(artifact_id, label="artifact ID")
