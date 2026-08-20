"""Project configuration contract and TOML loading for the new local layout."""

from __future__ import annotations

import tomllib
from decimal import Decimal
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Literal
from urllib.parse import urlsplit

import tomli_w
from pydantic import Field, field_validator, model_validator

from exp.common.core.artifacts import (
    ArtifactId,
    ArtifactInput,
    ContractModel,
    SecretBoundaryError,
    assert_secret_free,
)
from exp.common.core.files import write_text_atomic
from exp.common.core.money import exact_usd


def _exclude_absent(value: object) -> bool:
    """Return whether an optional absent field should be omitted from serialization.

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
    incumbent: ArtifactId | None = None

    @model_validator(mode="after")
    def _require_coherent_router_roles(self) -> ProjectModelConfiguration:
        """Require unique candidates and an incumbent drawn from that candidate set."""
        if len(set(self.candidates)) != len(self.candidates):
            raise ValueError("project candidate aliases must not repeat")
        if self.incumbent is not None and self.incumbent not in self.candidates:
            raise ValueError("project incumbent must also be a selected candidate")
        return self


class ProjectSystemConfiguration(ContractModel):
    """Bounded built-in system supported by the hosted Project workflow."""

    kind: Literal["builtin_chat"] = "builtin_chat"
    system_prompt: str = Field(min_length=1, max_length=20_000)
    maximum_model_calls: int = Field(default=8, ge=1, le=64)

    @field_validator("system_prompt", mode="before")
    @classmethod
    def _normalize_system_prompt(cls, value: object) -> str:
        """Trim the required prompt and reject non-text or whitespace-only values."""
        if not isinstance(value, str):
            raise ValueError("built-in chat system_prompt must be text")
        normalized = value.strip()
        if not normalized:
            raise ValueError("built-in chat system_prompt must not be blank")
        return normalized


class ProjectRetrievalConfiguration(ContractModel):
    """Mutable retrieval controls applied to future immutable build artifacts."""

    top_k: int = Field(default=5, gt=0)


class ProjectBudgetConfiguration(ContractModel):
    """Finite spend limits applied to future project workflow calls."""

    maximum_build_cost_usd: Decimal = Field(default=Decimal("5.000000"), gt=0)
    maximum_provider_cost_usd: Decimal | None = Field(default=None, gt=0)

    @field_validator(
        "maximum_build_cost_usd",
        "maximum_provider_cost_usd",
        mode="before",
    )
    @classmethod
    def _require_exact_budget(cls, value: object) -> Decimal | None:
        """Return one canonical numeric(20,6) budget value or reject it."""
        return None if value is None else exact_usd(value)


class ProjectHostedSetup(ContractModel):
    """One late secret-free setup mutation for a prepared hosted Project."""

    system: ProjectSystemConfiguration
    models: ProjectModelConfiguration
    model_catalog: ArtifactInput
    retrieval: ProjectRetrievalConfiguration
    budgets: ProjectBudgetConfiguration

    @model_validator(mode="after")
    def _require_complete_hosted_roles(self) -> ProjectHostedSetup:
        """Require the first hosted workflow's baseline and candidate shape."""
        if len(self.models.candidates) < 2:
            raise ValueError("hosted Project setup requires at least two router candidates")
        if self.models.incumbent is None:
            raise ValueError("hosted Project setup requires a baseline incumbent")
        if self.budgets.maximum_provider_cost_usd is None:
            raise ValueError("hosted Project setup requires one finite provider-spend ceiling")
        return self


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


class ProjectHostedJudgeEvidence(ContractModel):
    """Machine-only judge setup and calibration selected for hosted routing."""

    setup: ArtifactInput
    calibration: ArtifactInput
    status: Literal["provisional"] = "provisional"

    @model_validator(mode="after")
    def _require_distinct_evidence(self) -> ProjectHostedJudgeEvidence:
        """Require setup and calibration to remain separate immutable artifacts."""
        if self.setup.artifact_id == self.calibration.artifact_id:
            raise ValueError("hosted judge setup and calibration artifacts must be distinct")
        return self


class ProjectRouterPolicyArtifacts(ContractModel):
    """Exact frozen fit-only policy selection completed before held-out reporting."""

    policy_lock: ArtifactInput
    policy: ArtifactInput
    spend_ledger: ArtifactInput

    @model_validator(mode="after")
    def _require_distinct_policy_artifacts(self) -> ProjectRouterPolicyArtifacts:
        """Require policy, lock, and spend evidence to use distinct artifact identities."""
        values = (self.policy_lock, self.policy, self.spend_ledger)
        if len({item.artifact_id for item in values}) != len(values):
            raise ValueError("router policy selection artifacts must be distinct")
        return self


class ProjectRouterReportArtifacts(ContractModel):
    """Exact held-out report and final spend selection after the policy lock."""

    report: ArtifactInput
    spend_ledger: ArtifactInput

    @model_validator(mode="after")
    def _require_distinct_report_artifacts(self) -> ProjectRouterReportArtifacts:
        """Require report and final spend evidence to use distinct artifact identities."""
        if self.report.artifact_id == self.spend_ledger.artifact_id:
            raise ValueError("router report and final spend artifacts must be distinct")
        return self


class ProjectConfig(ContractModel):
    """Project-local configuration that names no provider credentials or secret references."""

    schema_version: int = Field(default=4, ge=1)
    project_id: ArtifactId
    trace_source: str | None = Field(default=None, max_length=64)
    trace_preparation: ProjectTracePreparationSettings | None = None
    provider_free_stage: ProjectProviderFreeStage | None = None
    system: ProjectSystemConfiguration | None = None
    models: ProjectModelConfiguration | None = None
    model_catalog: ArtifactInput | None = None
    retrieval: ProjectRetrievalConfiguration | None = Field(
        default_factory=ProjectRetrievalConfiguration
    )
    budgets: ProjectBudgetConfiguration | None = Field(default_factory=ProjectBudgetConfiguration)
    build: ProjectBuildArtifacts | None = None
    build_spend_ledger: ArtifactInput | None = None
    hosted_judge: ProjectHostedJudgeEvidence | None = None
    router_policy: ProjectRouterPolicyArtifacts | None = None
    router_report: ProjectRouterReportArtifacts | None = None
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
        hosted_setup = (
            self.system,
            self.models,
            self.model_catalog,
            self.retrieval,
            self.budgets,
        )
        if self.system is not None and any(value is None for value in hosted_setup):
            raise ValueError("hosted system selection requires one complete late setup")
        if self.build_spend_ledger is not None and self.system is None:
            raise ValueError("hosted build spend evidence requires a bound hosted setup")
        if self.build_spend_ledger is not None and self.build is None:
            raise ValueError("build spend evidence requires a completed build")
        if self.hosted_judge is not None and self.build is None:
            raise ValueError("hosted judge evidence requires a completed build")
        if self.router_policy is not None and self.hosted_judge is None:
            raise ValueError("router policy selection requires hosted judge evidence")
        if self.router_report is not None and self.router_policy is None:
            raise ValueError("router report selection requires a frozen router policy")
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
