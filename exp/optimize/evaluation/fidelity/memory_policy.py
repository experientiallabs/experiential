"""Frozen versus learning memory policy, including bounded post-trace consolidation."""

import json
from typing import Literal

from pydantic import Field, model_validator

from exp.common.core.artifacts import ArtifactInput, ContractModel
from exp.common.project import ArtifactStore, artifact_input
from exp.simulation.world_model.dreaming import DreamPolicy
from exp.simulation.world_model.experience_memory import MemoryVersion, MemoryView

LESSON_SET_TYPE = "learning-lesson-set"


class LessonSelection(ContractModel):
    """The screened lessons of one frozen memory checkpoint, as a frozen run reads them.

    Attributes:
        lineage: Memory lineage whose misses were screened.
        version: Memory version screened; a frozen run must read exactly this checkpoint.
        lessons: Research record IDs screened in, the only records and hypothesis lines the
            run delivers. Every other miss stays source evidence only.
    """

    lineage: str = Field(pattern=r"^[a-z][a-z0-9-]{0,63}$")
    version: int = Field(ge=1, strict=True)
    lessons: tuple[str, ...]


def load_lesson_selection(store: ArtifactStore, reference: ArtifactInput) -> LessonSelection:
    """Read a saved lesson set, verifying its manifest and type.

    Args:
        store: Project artifact owner.
        reference: Exact lesson-set reference named by the memory policy.

    Returns:
        The screened selection.

    Raises:
        ValueError: The reference is stale or is not a lesson set.
    """
    stored = store.read(reference.artifact_id)
    if artifact_input(stored.manifest) != reference or stored.manifest.artifact_type != (
        LESSON_SET_TYPE
    ):
        raise ValueError("memory lessons must name an exact saved lesson set; screen again")
    return LessonSelection.model_validate(
        json.loads(store.read_bytes(reference.artifact_id, "selection.json"))
    )


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
        lessons: Frozen only: a saved lesson set of this checkpoint. The run then reads only
            its screened-in records and their hypothesis lines; original observations and
            every screened-out miss stay source evidence and are never delivered.
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
    lessons: ArtifactInput | None = Field(default=None, exclude_if=lambda value: value is None)

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
        if self.lessons is not None and self.mode != "frozen":
            raise ValueError("a lesson set restricts only a frozen checkpoint, never learning")
        return self


def screened_lessons(
    store: ArtifactStore, memory: FidelityMemory, version: MemoryVersion | None
) -> frozenset[str] | None:
    """Screened-in record IDs a frozen run is restricted to, or None to read all memory.

    Args:
        store: Project artifact owner.
        memory: The run's memory policy.
        version: Checkpoint the run reads.

    Returns:
        The lesson record IDs, or None without a lesson set.

    Raises:
        ValueError: The lesson set screens another checkpoint than the one read.
    """
    if memory.lessons is None:
        return None
    selection = load_lesson_selection(store, memory.lessons)
    if version is None or (selection.lineage, selection.version) != (
        memory.lineage,
        version.version,
    ):
        raise ValueError(
            "the lesson set screens a different memory checkpoint than this run reads; "
            "screen the checkpoint the run reads"
        )
    return frozenset(selection.lessons)


def lesson_view(view: MemoryView, lessons: frozenset[str]) -> MemoryView:
    """Keep only screened-in records; observations and screened-out misses are not served.

    Args:
        view: The run's eligible memory view.
        lessons: Screened-in research record IDs.

    Returns:
        The same view restricted to lessons; withheld and retired identities are unchanged.
    """
    return view.model_copy(
        update={
            "records": {k: v for k, v in view.records.items() if k in lessons},
            "sources": {k: v for k, v in view.sources.items() if k in lessons},
        }
    )
