"""Native prelude admission around real TraceML, rubric and behavioral owners."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import AbstractContextManager, contextmanager, nullcontext
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

    def completed(
        self,
        context: StageContext,
        checkpoint: CouncilPreludeCheckpoint,
        owners: tuple[ArtifactInput, ...],
        /,
    ) -> None:
        """Verify exact completed native owners while their paid envelope remains active."""


class CouncilStudyRegistration(Protocol):
    """Qualify all real authored study registrations before any point-selector wave."""

    def __call__(self, context: StageContext, studies: tuple[ActorCriticSpec, ...], /) -> None:
        """Register exact generated references under the existing qualified prelude policy."""


@contextmanager
def prelude_phase(
    context: StageContext, checkpoint: CouncilPreludeCheckpoint
) -> Iterator[Callable[[tuple[ArtifactInput, ...]], None]]:
    """Wrap the real owner and expose a completion sink only to its returned artifacts."""
    admission = context.dependencies.council_prelude_admission
    scope = None if admission is None else admission(context, checkpoint)

    def completed(owners: tuple[ArtifactInput, ...]) -> None:
        """Forward actual immutable owner references to the native admission owner."""
        if admission is not None:
            admission.completed(context, checkpoint, owners)

    with nullcontext() if scope is None else scope:
        yield completed
