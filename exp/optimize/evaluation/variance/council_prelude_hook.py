"""Native prelude admission around real TraceML, rubric and behavioral owners."""

from __future__ import annotations

from contextlib import AbstractContextManager, nullcontext
from typing import TYPE_CHECKING, Literal, Protocol

from exp.common.core.artifacts import ArtifactInput, ContractModel
from exp.optimize.evaluation.continuations.job import ContinuationJob
from exp.optimize.evaluation.variance.judged import EpisodeJudge

if TYPE_CHECKING:
    from exp.optimize.evaluation.variance.actor_critic import ActorCriticSpec
    from exp.optimize.evaluation.variance.pipeline_spec import StageContext


class CouncilPreludeCheckpoint(ContractModel):
    """Exact pending owner before an ActorCriticSpec with its authored references exists."""

    phase: Literal["traceml_group", "traceml_labels", "mining_rubric", "behavior_reference"]
    actor_alias: str | None = None
    job: ContinuationJob
    group: ArtifactInput | None = None
    judge: EpisodeJudge | None = None
    completed_labels: tuple[ArtifactInput, ...] = ()
    completed_error_points: tuple[ArtifactInput, ...] = ()
    completed_rubric: ArtifactInput | None = None


class CouncilPreludeAdmission(Protocol):
    """Native preparation-qualified authority; no fake rubric or free prelude inference."""

    def __call__(
        self, context: StageContext, checkpoint: CouncilPreludeCheckpoint, /
    ) -> AbstractContextManager[object] | None:
        """Return exact native priced scope or raise before its clients/owners dispatch."""


class CouncilStudyRegistration(Protocol):
    """Qualify all real authored study registrations before any point-selector wave."""

    def __call__(self, context: StageContext, studies: tuple[ActorCriticSpec, ...], /) -> None:
        """Register exact generated references under the existing qualified prelude policy."""


def prelude_phase(
    context: StageContext, checkpoint: CouncilPreludeCheckpoint
) -> AbstractContextManager[object]:
    """Wrap the actual owner; legacy callers without conditional authority stay unchanged."""
    admission = context.dependencies.council_prelude_admission
    if admission is None:
        return nullcontext()
    scope = admission(context, checkpoint)
    return nullcontext() if scope is None else scope
