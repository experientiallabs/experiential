"""Project configuration contract and TOML loading for the new local layout."""

from __future__ import annotations

import tomllib
from pathlib import Path, PurePosixPath, PureWindowsPath
from urllib.parse import urlsplit

import tomli_w
from pydantic import Field, field_validator, model_validator

from wmo.common.core.artifacts import (
    ArtifactId,
    ArtifactInput,
    ContractModel,
    SecretBoundaryError,
    assert_secret_free,
)
from wmo.common.core.files import write_text_atomic


def _exclude_absent(value: object) -> bool:
    """Return whether an optional compatibility field should be omitted from serialization.

    Args:
        value: Field value being serialized.

    Returns:
        True only for an absent optional value.
    """
    return value is None


class ProjectConfigError(ValueError):
    """A project configuration file was absent, malformed, or violated its local contract."""


def require_durable_source_id(source_id: str) -> str:
    """Require a stable acquisition label rather than a worker-local path.

    Args:
        source_id: Caller-owned source label to persist in immutable provenance.

    Returns:
        The unchanged durable source label.

    Raises:
        ValueError: The label is blank, padded, or path-shaped without a URI scheme.
    """
    if not source_id or source_id != source_id.strip():
        raise ValueError(
            "source_id must be a nonblank durable acquisition label without surrounding spaces"
        )
    windows_path = PureWindowsPath(source_id)
    uri_scheme = urlsplit(source_id).scheme
    if (
        PurePosixPath(source_id).is_absolute()
        or windows_path.is_absolute()
        or bool(windows_path.drive)
        or source_id.casefold().startswith("file:")
        or "\\" in source_id
        or ("/" in source_id and not uri_scheme)
    ):
        raise ValueError(
            "source_id must be a durable acquisition label, not a worker-local filesystem path"
        )
    return source_id


class AgentConfiguration(ContractModel):
    """Optional explicit factory for a customer-provided injectable agent runtime."""

    factory: str = Field(min_length=1, max_length=512)
    code_revision: str | None = Field(
        default=None,
        min_length=1,
        max_length=256,
        exclude_if=_exclude_absent,
    )


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


class ProjectTracePreparationSettings(ContractModel):
    """Provider-free settings fixed before canonical trace preparation starts."""

    source_kind: str = Field(min_length=1, max_length=64)
    fit_task_budget: int = Field(default=50, ge=0)
    held_out_task_budget: int = Field(default=20, ge=0)
    descriptor_dimensions: int = Field(default=64, ge=8)

    @field_validator("source_kind")
    @classmethod
    def _normalize_source_kind(cls, value: str) -> str:
        """Return one canonical declared source name."""
        normalized = value.strip().casefold()
        if not normalized:
            raise ValueError("trace source kind must not be blank")
        return normalized


class ProjectProviderFreeStage(ContractModel):
    """Exact immutable trace and task pointers selected before provider-backed work."""

    schema_version: int = 1
    trace_dataset: ArtifactInput
    task_set: ArtifactInput

    @model_validator(mode="after")
    def _require_distinct_artifacts(self) -> ProjectProviderFreeStage:
        """Reject a stage that reuses one artifact for both semantic outputs."""
        if self.trace_dataset.artifact_id == self.task_set.artifact_id:
            raise ValueError("provider-free stage trace and task artifacts must be distinct")
        return self


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
    trace_preparation: ProjectTracePreparationSettings | None = None
    provider_free_stage: ProjectProviderFreeStage | None = None
    models: ProjectModelConfiguration | None = None
    retrieval: ProjectRetrievalConfiguration | None = Field(
        default_factory=ProjectRetrievalConfiguration
    )
    budgets: ProjectBudgetConfiguration | None = Field(default_factory=ProjectBudgetConfiguration)
    build: ProjectBuildArtifacts | None = None
    agent: AgentConfiguration | None = None
    model_optimization_config: ArtifactInput | None = None
    redacted_field_names: tuple[str, ...] = ()

    @model_validator(mode="before")
    @classmethod
    def _keep_provider_free_bootstrap_minimal(cls, value: object) -> object:
        """Omit late retrieval and spend setup only for trace-first Project configuration."""
        if not isinstance(value, dict) or value.get("trace_preparation") is None:
            return value
        updated = dict(value)
        updated.setdefault("retrieval", None)
        updated.setdefault("budgets", None)
        return updated

    @model_validator(mode="after")
    def _require_trace_preparation_for_provider_free_stage(self) -> ProjectConfig:
        """Require Project-owned preparation settings before selecting provider-free evidence."""
        if self.provider_free_stage is not None and self.trace_preparation is None:
            raise ValueError("provider-free stage requires Project trace preparation settings")
        return self


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
