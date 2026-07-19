"""Cycle-free immutable evidence types shared by Harbor qualification and scoring."""

from __future__ import annotations

from typing import Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from wmh.core.text import validate_durable_text
from wmh.evals.harbor.config import HarborEnvironmentBackend
from wmh.evals.harbor.e2b_environment import ExactE2BBuildSpec
from wmh.tracking.budget import TimedResourceClass, TimedResourceRole

_DIGEST_PATTERN = r"^sha256:[0-9a-f]{64}$"


class QualifiedE2BBuildIdentity(BaseModel):
    """Exact prequalified E2B build identity required again before scored launch."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    build_config_digest: str = Field(pattern=_DIGEST_PATTERN)
    build_record_digest: str = Field(pattern=_DIGEST_PATTERN)
    environment_id: str = Field(min_length=1, max_length=512)
    build_context_digest: str = Field(pattern=_DIGEST_PATTERN)
    docker_image: str | None = Field(default=None, min_length=1, max_length=2_048)
    cpu_count: int = Field(ge=1)
    memory_mb: int = Field(ge=1)
    template_id: str = Field(pattern=r"^[A-Za-z0-9_.-]{1,512}$")
    build_id: str = Field(pattern=r"^[A-Za-z0-9_.-]{1,512}$")

    @model_validator(mode="after")
    def _bind_build_spec(self) -> Self:
        spec = ExactE2BBuildSpec(
            environment_id=self.environment_id,
            build_context_digest=self.build_context_digest,
            docker_image=self.docker_image,
            cpu_count=self.cpu_count,
            memory_mb=self.memory_mb,
        )
        if self.build_config_digest != spec.digest:
            raise ValueError("qualified E2B build config digest is inconsistent")
        if self.template_id == self.build_id:
            raise ValueError("qualified E2B template and build identities must differ")
        return self


class QualifiedHarborTask(BaseModel):
    """Pre-run immutable identities for one qualified Harbor task environment."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    task_id: str = Field(min_length=1, max_length=512)
    dataset_id: str = Field(min_length=1, max_length=512)
    content_digest: str = Field(pattern=_DIGEST_PATTERN)
    task_key: str = Field(pattern=_DIGEST_PATTERN)
    task_environment_digest: str = Field(pattern=_DIGEST_PATTERN)
    environment_backend: HarborEnvironmentBackend
    requested_storage_mb: int | None = Field(default=None, ge=1)
    observed_storage_mb: int | None = Field(default=None, ge=1)
    e2b_launch_config_digest: str | None = Field(default=None, pattern=_DIGEST_PATTERN)
    e2b_build_config_digest: str | None = Field(default=None, pattern=_DIGEST_PATTERN)
    e2b_build_record_digest: str | None = Field(default=None, pattern=_DIGEST_PATTERN)
    task_resource_class_digest: str | None = Field(default=None, pattern=_DIGEST_PATTERN)
    e2b_build_identity: QualifiedE2BBuildIdentity | None = None
    task_resource_class: TimedResourceClass | None = None

    @field_validator("task_id", "dataset_id")
    @classmethod
    def _require_canonical_task_id(cls, value: str) -> str:
        if value != value.strip():
            raise ValueError("qualified task and dataset IDs cannot have surrounding whitespace")
        validate_durable_text(value, field="qualified Harbor task or dataset id")
        return value

    @model_validator(mode="after")
    def _require_backend_qualification(self) -> Self:
        e2b_fields = (
            self.e2b_launch_config_digest,
            self.e2b_build_config_digest,
            self.e2b_build_record_digest,
            self.task_resource_class_digest,
            self.e2b_build_identity,
            self.task_resource_class,
        )
        if self.environment_backend is HarborEnvironmentBackend.LOCAL:
            if any(value is not None for value in e2b_fields):
                raise ValueError("local task qualification cannot carry E2B build identities")
            if self.observed_storage_mb is not None:
                raise ValueError("local task qualification cannot carry E2B storage metrics")
        elif any(value is None for value in e2b_fields):
            raise ValueError(
                "E2B task qualification requires exact build and resource class identities"
            )
        else:
            assert self.e2b_build_identity is not None
            assert self.task_resource_class is not None
            if (
                self.e2b_build_config_digest != self.e2b_build_identity.build_config_digest
                or self.e2b_build_record_digest != self.e2b_build_identity.build_record_digest
                or self.task_resource_class_digest != self.task_resource_class.digest
            ):
                raise ValueError("E2B task qualification identities are inconsistent")
            if self.task_resource_class.role is not TimedResourceRole.TASK_ENVIRONMENT:
                raise ValueError("E2B task qualification names the wrong resource role")
            if (
                self.task_resource_class.cpu_count != self.e2b_build_identity.cpu_count
                or self.task_resource_class.memory_mb != self.e2b_build_identity.memory_mb
            ):
                raise ValueError("E2B task build and launch resource identities differ")
            if self.requested_storage_mb is None:
                if self.observed_storage_mb is not None:
                    raise ValueError("unrequested E2B storage cannot have observed capacity")
            elif (
                self.observed_storage_mb is None
                or self.observed_storage_mb < self.requested_storage_mb
            ):
                raise ValueError("E2B observed storage is below the requested minimum")
        return self
