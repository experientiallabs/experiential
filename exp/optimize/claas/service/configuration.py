"""Finite learning-run limits and inspectable completion receipts."""

from __future__ import annotations

from typing import Literal

from pydantic import Field, model_validator

from exp.common.core.artifacts import ContractModel
from exp.optimize.claas.service.contracts import RunMode
from exp.optimize.claas.training_contracts import TrainingCheckpoint, TrainingResult


class RunConfiguration(ContractModel):
    """Bound resident lifetime, queue retention, and update admission.

    Attributes:
        mode: Resident generation service by default, or one finite imported-data burst.
        training_admission: Automatic readiness-based updates by default. Explicit
            admission accepts only complete caller-selected batches in resident mode.
        maximum_updates: Positive optimizer-update ceiling, defaulting to 100.
        maximum_run_seconds: Finite lifetime in seconds, defaulting to one hour.
        minimum_ready_examples: Automatic batch readiness threshold, defaulting to four.
            Explicit admission ignores this threshold.
        maximum_buffer_records: Retained evidence count ceiling, defaulting to 10,000.
        maximum_buffer_bytes: Total retained evidence ceiling, defaulting to 256 MiB.
        maximum_record_bytes: Individual evidence ceiling, defaulting to 2 MiB.
        maximum_update_receipt_bytes: Per-update retained receipt ceiling, defaulting
            to 8 MiB and bounded above by 64 MiB.
        cleanup_timeout_seconds: Positive cleanup deadline, defaulting to two minutes.
    """

    mode: RunMode = "run"
    training_admission: Literal["automatic", "explicit"] = "automatic"
    maximum_updates: int = Field(default=100, strict=True, ge=1, le=1_000_000)
    maximum_run_seconds: float = Field(default=3600, gt=0, le=86400, allow_inf_nan=False)
    minimum_ready_examples: int = Field(default=4, strict=True, ge=1, le=4096)
    maximum_buffer_records: int = Field(default=10_000, strict=True, ge=1, le=1_000_000)
    maximum_buffer_bytes: int = Field(default=268_435_456, strict=True, ge=1024)
    maximum_record_bytes: int = Field(default=2_097_152, strict=True, ge=1024)
    maximum_update_receipt_bytes: int = Field(
        default=8_388_608, strict=True, ge=1024, le=67_108_864
    )
    cleanup_timeout_seconds: float = Field(default=120, gt=0, le=3600, allow_inf_nan=False)

    @model_validator(mode="after")
    def _validate_admission(self) -> RunConfiguration:
        """Keep explicit submission available only while a resident service accepts requests."""
        if self.training_admission == "explicit" and self.mode != "run":
            raise ValueError("explicit training admission requires mode='run'")
        return self


class BatchStatus(ContractModel):
    """Durable acceptance or completion of one exact response batch.

    Attributes:
        batch_id: Caller-owned immutable batch identity.
        batch_sha256: Digest of the complete token-exact leased training payload.
        expected_policy_revision: Policy bound to the selected responses.
        response_ids: Complete ordered selection, without silently dropped responses.
        state: Pending until a checkpoint is durably acknowledged; pending does not
            promise that a failed or stopped host is still executing.
        result: Completed native checkpoint receipt, absent before acknowledgement.
    """

    batch_id: str
    batch_sha256: str
    expected_policy_revision: str
    response_ids: tuple[str, ...]
    state: Literal["pending", "completed"]
    result: TrainingResult | None = None


class BufferStatus(ContractModel):
    """All retained records, including explicitly rejected and consumed evidence."""

    pending_feedback: int = 0
    ready: int = 0
    inflight: int = 0
    consumed: int = 0
    rejected: int = 0
    retained_bytes: int = 0
    rejection_reasons: dict[str, int] = Field(default_factory=dict)


class RunStatus(ContractModel):
    """Current run state without hiding queued work when compute stops."""

    mode: RunMode
    state: Literal["created", "starting", "running", "closing", "closed", "failed"]
    updates: int
    policy_revision: str
    buffer: BufferStatus
    stop_reason: str | None = None
    failure_type: str | None = None
    cleanup_failure_type: str | None = None


class RunReport(ContractModel):
    """Terminal receipt and the last atomically acknowledged checkpoint."""

    status: RunStatus
    checkpoint: TrainingCheckpoint | None = None
