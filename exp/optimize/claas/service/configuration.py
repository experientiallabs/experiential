"""Finite learning-run limits and inspectable completion receipts."""

from typing import Literal

from pydantic import Field

from exp.common.core.artifacts import ContractModel
from exp.optimize.claas.service.contracts import RunMode
from exp.optimize.claas.training_contracts import TrainingCheckpoint


class RunConfiguration(ContractModel):
    """Bound resident lifetime, queue retention, and automatic update admission."""

    mode: RunMode = "run"
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
