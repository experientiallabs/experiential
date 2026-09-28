"""Safe paths for the small project-local `.exp` layout."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from exp.common.core.artifacts import validate_artifact_id


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
    """Resolves the canonical local layout for one named EXP project."""

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
        """Return this project's directory under `.exp/projects/`."""
        return self.projects_directory / self.project_id

    @property
    def runtime_directory(self) -> Path:
        """Return the mutable runtime-state directory outside immutable artifacts."""
        return self.project_directory / "runtime"

    @property
    def runtime_journal(self) -> Path:
        """Return the append-only routed-interaction journal path."""
        return self.runtime_directory / "interactions.jsonl"
