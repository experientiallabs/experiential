"""Frozen versus learning memory policy, including bounded post-trace consolidation."""

from typing import Literal

from pydantic import Field, model_validator

from exp.common.core.artifacts import ContractModel
from exp.simulation.world_model.dreaming import DreamPolicy


class FidelityMemory(ContractModel):
    """Cross-trajectory memory: learn from this trajectory, or evaluate a frozen checkpoint.

    Attributes:
        lineage: Project-local versioned memory sequence.
        mode: ``learn`` pins the current head and publishes the next version after a
            completed trajectory; ``frozen`` reads an exact checkpoint and never publishes.
        checkpoint: Exact frozen version; absent in learning mode.
        reserved: Frozen evaluation that declares this task lineage reserved: it is refused if
            the lineage already consumed the task lineage, and later learning of it is refused.
            Without it a frozen run is a diagnostic and relies on same-trajectory withholding.
        hide_task_lineage: Learning only: memory from earlier trajectories of this task lineage
            is hidden from the run, so every prediction is made on a task memory has not seen
            (test-then-train), the same exclusion recorded-evidence retrieval applies.
        withhold_own: False only for an unreserved frozen retest that reads what memory learned
            from this very trajectory: a retention check, never a generalization measure.
        overview_lines: Line budget of the memory overview delivered with every prediction.
        maximum_compressions: Compression calls a learning run may buy after publication.
        dreaming: Optional consolidation before this trajectory commits its memory version.
    """

    lineage: str = Field(pattern=r"^[a-z][a-z0-9-]{0,63}$")
    mode: Literal["learn", "frozen"]
    checkpoint: int | None = Field(default=None, ge=1, strict=True)
    reserved: bool = Field(default=False, strict=True, exclude_if=lambda value: not value)
    hide_task_lineage: bool = Field(default=False, strict=True, exclude_if=lambda value: not value)
    withhold_own: bool = Field(default=True, strict=True, exclude_if=lambda value: value)
    overview_lines: int = Field(default=32, ge=1, strict=True)
    maximum_compressions: int = Field(default=16, ge=0, strict=True)

    dreaming: DreamPolicy | None = Field(default=None, exclude_if=lambda value: value is None)

    @model_validator(mode="after")
    def checkpoint_mode(self) -> "FidelityMemory":
        """Reserved evaluation names its checkpoint; learning always follows the head."""
        if (self.mode == "frozen") != (self.checkpoint is not None):
            raise ValueError("frozen memory requires a checkpoint; learning uses the head")
        if self.reserved and self.mode != "frozen":
            raise ValueError("only frozen evaluation can declare a reserved task lineage")
        if self.hide_task_lineage and self.mode != "learn":
            raise ValueError("hiding the task lineage applies only while learning")
        if not self.withhold_own and (self.mode != "frozen" or self.reserved):
            raise ValueError("only an unreserved frozen retest may read its own trajectory")
        if self.dreaming is not None and self.mode != "learn":
            raise ValueError("dreaming runs only after learning, never during frozen evaluation")
        return self
