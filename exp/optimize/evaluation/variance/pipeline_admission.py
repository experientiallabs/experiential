"""Native paid-stage scope and committed-owner completion hooks."""

from contextlib import AbstractContextManager, nullcontext
from typing import TYPE_CHECKING, Protocol

from exp.common.core.artifacts import ArtifactInput

if TYPE_CHECKING:
    from exp.optimize.evaluation.variance.pipeline_spec import StageContext


class PipelineStageAdmission(Protocol):
    """Admit an exact native stage before its owner can dispatch."""

    def __call__(self, context: "StageContext", /) -> AbstractContextManager[object] | None:
        """Return the original-ledger paid scope or refuse before execution."""

    def completed(self, context: "StageContext", owner: ArtifactInput, /) -> None:
        """Verify the native committed owner, including saved-owner reconciliation."""


def stage_scope(context: "StageContext") -> AbstractContextManager[object]:
    """Enter the exact native caller's optional stage admission scope."""
    admission = context.dependencies.stage_admission
    if admission is None:
        return nullcontext()
    scope = admission(context)
    return nullcontext() if scope is None else scope


def complete_stage(context: "StageContext", owner: ArtifactInput) -> None:
    """Publish a completion only after the pipeline owner committed its real artifact."""
    admission = context.dependencies.stage_admission
    if admission is not None:
        admission.completed(context, owner)
