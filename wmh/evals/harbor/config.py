"""Typed construction of Harbor 0.18 jobs for WMH benchmark runs."""

from __future__ import annotations

from enum import StrEnum
from pathlib import Path
from typing import Self

import harbor
from harbor.models.environment_type import EnvironmentType
from harbor.models.job.config import DatasetConfig, JobConfig, RetryConfig
from harbor.models.trial.config import AgentConfig, EnvironmentConfig
from pydantic import BaseModel, Field, field_validator, model_validator

from wmh.evals.harbor.docker_environment import REAPING_DOCKER_ENVIRONMENT_IMPORT_PATH
from wmh.evals.harbor.e2b_environment import EXACT_E2B_ENVIRONMENT_IMPORT_PATH
from wmh.tracking.budget import BudgetAccountBinding

SUPPORTED_HARBOR_VERSION = "0.18.0"


class HarborEnvironmentBackend(StrEnum):
    """Where Harbor runs tasks, with isolated local Docker as the default."""

    LOCAL = "local"
    E2B = "e2b"


class HarborJobSpec(BaseModel):
    """Stable task-matrix inputs used to construct one exact Harbor job configuration."""

    job_name: str = Field(min_length=1)
    jobs_dir: Path
    datasets: list[DatasetConfig] = Field(min_length=1)
    n_attempts: int = Field(default=1, ge=1)
    n_concurrent_trials: int = Field(default=1, ge=1)
    agent_n_concurrent: int | None = Field(default=None, ge=1)
    environment_backend: HarborEnvironmentBackend = HarborEnvironmentBackend.LOCAL
    max_retries: int = Field(
        default=0,
        ge=0,
        description="Unsupported above zero until WMH records every Harbor retry attempt",
    )
    retry_exceptions: set[str] = Field(
        default_factory=set,
        description="Reserved for a future audited retry-attempt ledger",
    )
    artifact_paths: list[str] = Field(default_factory=list)

    @field_validator("retry_exceptions")
    @classmethod
    def _reject_blank_exception_names(cls, value: set[str]) -> set[str]:
        if any(not name.strip() for name in value):
            raise ValueError("retry_exceptions cannot contain blank exception names")
        return value

    @model_validator(mode="after")
    def _validate_concurrency_and_retries(self) -> Self:
        if (
            self.agent_n_concurrent is not None
            and self.agent_n_concurrent > self.n_concurrent_trials
        ):
            raise ValueError("agent_n_concurrent cannot exceed n_concurrent_trials")
        if self.max_retries > 0:
            raise ValueError(
                "max_retries greater than zero is unsupported until WMH has an atomic "
                "Harbor attempt ledger"
            )
        if self.retry_exceptions:
            raise ValueError("retry_exceptions are unsupported while Harbor retries are disabled")
        return self


def build_harbor_job_config(
    spec: HarborJobSpec,
    *,
    agent: AgentConfig,
    task_resource_budget_bindings: tuple[BudgetAccountBinding, ...] = (),
) -> JobConfig:
    """Translate a stable WMH job spec to Harbor 0.18's programmatic ``JobConfig``.

    Docker is the explicit local task environment. Selecting E2B constructs an E2B environment
    directly, so missing E2B credentials or extras fail in Harbor instead of falling back locally.

    Raises:
        RuntimeError: If the imported Harbor version differs from the version this adapter targets.
    """
    _require_supported_harbor_version()
    spec = HarborJobSpec.model_validate(spec.model_dump())
    local_environment = spec.environment_backend is HarborEnvironmentBackend.LOCAL
    if local_environment and task_resource_budget_bindings:
        raise ValueError("local Harbor task environments cannot consume a resource meter")
    frozen_bindings = tuple(
        BudgetAccountBinding.model_validate(binding.model_dump())
        for binding in task_resource_budget_bindings
    )
    environment_type = EnvironmentType.DOCKER if local_environment else EnvironmentType.E2B
    if agent.n_concurrent != spec.agent_n_concurrent:
        raise ValueError("agent n_concurrent must already match HarborJobSpec.agent_n_concurrent")
    retry = RetryConfig(max_retries=0, include_exceptions=None)
    config = JobConfig(
        job_name=spec.job_name,
        jobs_dir=spec.jobs_dir,
        n_attempts=spec.n_attempts,
        n_concurrent_trials=spec.n_concurrent_trials,
        datasets=[dataset.model_copy(deep=True) for dataset in spec.datasets],
        agents=[agent.model_copy(deep=True)],
        environment=EnvironmentConfig(
            type=environment_type,
            import_path=(
                REAPING_DOCKER_ENVIRONMENT_IMPORT_PATH
                if local_environment
                else EXACT_E2B_ENVIRONMENT_IMPORT_PATH
            ),
            force_build=False,
            delete=True,
            mounts=None,
            extra_docker_compose=[],
            env={},
            kwargs=(
                {}
                if not frozen_bindings
                else {
                    "resource_budget_bindings": [
                        binding.model_dump(mode="json") for binding in frozen_bindings
                    ]
                }
            ),
            extra_allowed_hosts=[],
        ),
        retry=retry,
        artifacts=list(spec.artifact_paths),
    )
    validate_controlled_harbor_environment(config.environment, expected_type=environment_type)
    return config


def validate_controlled_harbor_environment(
    environment: EnvironmentConfig,
    *,
    expected_type: EnvironmentType | None = None,
) -> None:
    """Require WMH's credential-bearing Harbor process to control every host-facing input.

    Task-authored environment definitions are checked separately. This boundary covers the
    run-level Harbor environment configuration, which otherwise supports arbitrary imports,
    host mounts, Compose overlays, host environment interpolation, and backend kwargs.
    """
    if environment.type not in {EnvironmentType.DOCKER, EnvironmentType.E2B}:
        raise ValueError("Harbor environment type must be Docker or E2B")
    if expected_type is not None and environment.type is not expected_type:
        raise ValueError("Harbor environment type differs from the frozen job backend")
    expected_import_path = (
        REAPING_DOCKER_ENVIRONMENT_IMPORT_PATH
        if environment.type is EnvironmentType.DOCKER
        else EXACT_E2B_ENVIRONMENT_IMPORT_PATH
    )
    if environment.import_path != expected_import_path:
        raise ValueError(
            "Harbor environment import_path must name the trusted WMH adapter for its backend"
        )
    if environment.force_build:
        raise ValueError("Harbor environment force_build must remain disabled")
    if not environment.delete:
        raise ValueError("Harbor environment cleanup must remain enabled")
    if environment.mounts:
        raise ValueError("Harbor environment host mounts are unsupported")
    if environment.extra_docker_compose:
        raise ValueError("Harbor environment Compose overlays are unsupported")
    if environment.env:
        raise ValueError("Harbor environment host variables are unsupported")
    if environment.kwargs:
        if environment.type is not EnvironmentType.E2B or set(environment.kwargs) != {
            "resource_budget_bindings"
        }:
            raise ValueError("Harbor environment backend kwargs are unsupported")
        raw_bindings = environment.kwargs["resource_budget_bindings"]
        if not isinstance(raw_bindings, list) or not raw_bindings:
            raise ValueError("Harbor task resource budget bindings must be a nonempty list")
        bindings = [BudgetAccountBinding.model_validate(value) for value in raw_bindings]
        if len({(item.policy_digest, item.meter_id) for item in bindings}) != len(bindings):
            raise ValueError("Harbor task resource budget bindings must be unique")
        if len({item.policy_digest for item in bindings}) != 1:
            raise ValueError("Harbor task resource budgets must share one policy")
    if environment.extra_allowed_hosts:
        raise ValueError("Harbor run-level extra allowed hosts are unsupported")


def _require_supported_harbor_version() -> None:
    actual = harbor.__version__
    if actual != SUPPORTED_HARBOR_VERSION:
        raise RuntimeError(
            f"WMH Harbor evaluation requires harbor=={SUPPORTED_HARBOR_VERSION}, found {actual}; "
            "install the exact supported Harbor evaluation dependency"
        )
