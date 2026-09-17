"""Finite capacity and ownership for a resident single-GPU veRL learning run."""

from pathlib import Path
from typing import Literal

from pydantic import Field, model_validator

from exp.common.core.artifacts import ContractModel


class ResidentVerlSettings(ContractModel):
    """Choose durable storage and bounded capacity without allocating compute.

    Native operations are joined on cancellation, so their timeouts are soft.
    Local process hosting or a remote container lifetime supplies the hard bound.
    """

    checkpoint_root: Path
    lineage_id: str = Field(default="main", min_length=1, max_length=512)
    decoder: Literal["text", "hermes", "qwen35"] = "qwen35"
    startup_timeout_seconds: float = Field(default=600, gt=0, le=3600, allow_inf_nan=False)
    operation_timeout_seconds: float = Field(default=1800, gt=0, le=86400, allow_inf_nan=False)
    rollout_gpu_memory_utilization: float = Field(default=0.5, gt=0, lt=1, allow_inf_nan=False)
    maximum_output_tokens: int = Field(default=2048, strict=True, ge=1, le=131072)

    @model_validator(mode="after")
    def _absolute_root(self) -> "ResidentVerlSettings":
        """Reject ambiguous checkpoint destinations before runtime initialization."""
        if not self.checkpoint_root.is_absolute():
            raise ValueError("checkpoint_root must be an absolute durable directory")
        return self
