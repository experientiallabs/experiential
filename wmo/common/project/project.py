"""Project configuration contract and TOML loading for the new local layout."""

from __future__ import annotations

import tomllib
from pathlib import Path

import tomli_w
from pydantic import Field, model_validator

from wmo.common.core.artifacts import (
    ArtifactId,
    ArtifactInput,
    ContractModel,
    SecretBoundaryError,
    assert_secret_free,
)
from wmo.common.core.files import write_text_atomic


class ProjectConfigError(ValueError):
    """A project configuration file was absent, malformed, or violated its local contract."""


class AgentConfiguration(ContractModel):
    """Optional explicit factory for a customer-provided injectable agent runtime."""

    factory: str = Field(min_length=1, max_length=512)


class ProjectModelConfiguration(ContractModel):
    """Project-selected aliases isolated from later shared catalog role changes."""

    world_model: ArtifactId
    judge: ArtifactId
    embedder: ArtifactId
    candidates: tuple[ArtifactId, ...] = ()


class ProjectRetrievalConfiguration(ContractModel):
    """Mutable retrieval controls applied to future immutable build artifacts."""

    top_k: int = Field(default=5, gt=0)


class ProjectBudgetConfiguration(ContractModel):
    """Finite spend limits applied to future project workflow calls."""

    maximum_build_cost_usd: float = Field(default=5.0, gt=0)


class ProjectBuildArtifacts(ContractModel):
    """Exact immutable outputs selected as the project's current completed build."""

    trace_dataset: ArtifactInput
    task_set: ArtifactInput
    serving_rag: ArtifactInput
    fit_rag: ArtifactInput
    world_model: ArtifactInput

    @model_validator(mode="after")
    def _require_distinct_artifacts(self) -> ProjectBuildArtifacts:
        """Reject a completed build whose semantic outputs reuse one artifact ID.

        Returns:
            The validated completed-build pointers.

        Raises:
            ValueError: Two distinct build outputs name the same immutable artifact.
        """
        artifact_ids = tuple(
            item.artifact_id
            for item in (
                self.trace_dataset,
                self.task_set,
                self.serving_rag,
                self.fit_rag,
                self.world_model,
            )
        )
        if len(set(artifact_ids)) != len(artifact_ids):
            raise ValueError("project build pointers must name distinct immutable artifacts")
        return self


class ProjectConfig(ContractModel):
    """Project-local configuration that names no provider credentials or secret references."""

    schema_version: int = Field(default=2, ge=1)
    project_id: ArtifactId
    trace_source: str | None = Field(default=None, max_length=64)
    models: ProjectModelConfiguration | None = None
    retrieval: ProjectRetrievalConfiguration = Field(default_factory=ProjectRetrievalConfiguration)
    budgets: ProjectBudgetConfiguration = Field(default_factory=ProjectBudgetConfiguration)
    build: ProjectBuildArtifacts | None = None
    agent: AgentConfiguration | None = None
    model_optimization_config: ArtifactInput | None = None
    redacted_field_names: tuple[str, ...] = ()


def load_project_config(path: Path) -> ProjectConfig:
    """Load and validate one project TOML file.

    Args:
        path: Path to ``project.toml``.

    Returns:
        The immutable typed configuration.

    Raises:
        ProjectConfigError: The file is missing, malformed, or does not satisfy the contract.
    """
    try:
        with path.open("rb") as handle:
            raw_config = tomllib.load(handle)
    except FileNotFoundError as exc:
        raise ProjectConfigError(f"project configuration does not exist: {path}") from exc
    except tomllib.TOMLDecodeError as exc:
        raise ProjectConfigError(f"project configuration is invalid TOML: {path}") from exc
    try:
        config = ProjectConfig.model_validate(raw_config)
        assert_secret_free(config)
    except (SecretBoundaryError, ValueError) as exc:
        raise ProjectConfigError(f"project configuration is invalid: {exc}") from exc
    return config


def write_project_config(path: Path, config: ProjectConfig) -> None:
    """Atomically materialize a typed project configuration.

    Args:
        path: Destination ``project.toml`` path.
        config: Validated project configuration.

    Raises:
        ProjectConfigError: The configuration violates the no-secret project boundary.
    """
    try:
        assert_secret_free(config)
    except SecretBoundaryError as exc:
        raise ProjectConfigError(f"project configuration is invalid: {exc}") from exc
    payload = tomli_w.dumps(config.model_dump(mode="json", exclude_none=True))
    write_text_atomic(path, payload)
