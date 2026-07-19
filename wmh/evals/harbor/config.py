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


def build_harbor_job_config(spec: HarborJobSpec, *, agent: AgentConfig) -> JobConfig:
    """Translate a stable WMH job spec to Harbor 0.18's programmatic ``JobConfig``.

    Docker is the explicit local task environment. Selecting E2B constructs an E2B environment
    directly, so missing E2B credentials or extras fail in Harbor instead of falling back locally.

    Raises:
        RuntimeError: If the imported Harbor version differs from the version this adapter targets.
    """
    _require_supported_harbor_version()
    spec = HarborJobSpec.model_validate(spec.model_dump())
    environment_type = {
        HarborEnvironmentBackend.LOCAL: EnvironmentType.DOCKER,
        HarborEnvironmentBackend.E2B: EnvironmentType.E2B,
    }[spec.environment_backend]
    if agent.n_concurrent != spec.agent_n_concurrent:
        raise ValueError("agent n_concurrent must already match HarborJobSpec.agent_n_concurrent")
    retry = RetryConfig(max_retries=0, include_exceptions=None)
    return JobConfig(
        job_name=spec.job_name,
        jobs_dir=spec.jobs_dir,
        n_attempts=spec.n_attempts,
        n_concurrent_trials=spec.n_concurrent_trials,
        datasets=[dataset.model_copy(deep=True) for dataset in spec.datasets],
        agents=[agent.model_copy(deep=True)],
        environment=EnvironmentConfig(type=environment_type, delete=True),
        retry=retry,
        artifacts=list(spec.artifact_paths),
    )


def _require_supported_harbor_version() -> None:
    actual = harbor.__version__
    if actual != SUPPORTED_HARBOR_VERSION:
        raise RuntimeError(
            f"WMH Harbor evaluation requires harbor=={SUPPORTED_HARBOR_VERSION}, found {actual}; "
            "install the exact supported Harbor evaluation dependency"
        )
