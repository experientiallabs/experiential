"""Immutable council phase checkpoints and native continuation inventory.

These hooks are an admission seam, not an authority or a cost estimate. A caller must price
the native jobs/role draws under its actual ledger before returning from admission.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from contextlib import AbstractContextManager, nullcontext
from typing import TYPE_CHECKING, Literal, Protocol

from exp.common.core.artifacts import ArtifactInput, ContractModel
from exp.optimize.evaluation.continuations.job import ContinuationJob
from exp.optimize.evaluation.variance.council_parallel import council_draws

if TYPE_CHECKING:
    from exp.optimize.evaluation.variance.actor_critic import ActorCriticSpec, PointResult

CouncilPhase = Literal[
    "actors", "initial_judging", "selectors", "critics", "continuations", "settled"
]


class CouncilCheckpoint(ContractModel):
    """One immutable native wave and the full mandatory logical work still outstanding.

    Attributes:
        run_identity: Native actor-critic owner, stable across cache replay.
        phase: The wave whose admission is requested; settled dispatches nothing.
        group: The committed initial-study result, when available.
        initial_judged: Exact committed LLM judging of that group, when available.
        selected: Every distinct selected boundary, without dropping mandatory points.
        completed: Continuation boundaries whose owners returned.
        wave: Boundaries in this bounded continuation wave.
        jobs: Exact native study jobs for this wave; roles are referenced by draw index.
        draws: Native selector/critic indices for this wave.
        initial_judgments_remaining: Original group labels required before point selection.
        actors_remaining: Mandatory original group attempts if not yet committed.
        selectors_remaining: Mandatory selector draws not yet settled.
        critics_remaining: Mandatory critic draws not yet settled.
        suffixes_remaining: Mandatory continuations across all known uncompleted points.
        minimum_suffixes_remaining: At least one chosen point is required for a successful study.
        point_capacity: Maximum possible distinct points before selection is known.
        failures: Returned point failures, never relabeled as successful samples.
    """

    run_identity: str
    phase: CouncilPhase
    group: ArtifactInput | None = None
    initial_judged: ArtifactInput | None = None
    selected: tuple[int, ...] = ()
    completed: tuple[int, ...] = ()
    wave: tuple[int, ...] = ()
    jobs: tuple[ContinuationJob, ...] = ()
    draws: tuple[int, ...] = ()
    initial_judgments_remaining: int = 0
    actors_remaining: int = 0
    selectors_remaining: int = 0
    critics_remaining: int = 0
    suffixes_remaining: int = 0
    minimum_suffixes_remaining: int = 0
    point_capacity: int = 0
    failures: tuple[str, ...] = ()


class CouncilAdmission(Protocol):
    """Native authority callback; refusal must raise before this wave dispatches."""

    def __call__(
        self, spec: ActorCriticSpec, checkpoint: CouncilCheckpoint, /
    ) -> AbstractContextManager[object] | None:
        """Check native costs/future work and optionally return its genuine paid-wave scope."""


def admit_phase(
    admission: CouncilAdmission | None,
    spec: ActorCriticSpec,
    identity: str,
    phase: CouncilPhase,
    *,
    group: ArtifactInput | None = None,
    selected: tuple[int, ...] = (),
    initial_judged: ArtifactInput | None = None,
) -> AbstractContextManager[object]:
    """Expose exact native inventory and enter the admitted scope around physical work."""
    if admission is None:
        return nullcontext()
    context = admission(
        spec,
        CouncilCheckpoint(
            run_identity=identity,
            phase=phase,
            group=group,
            initial_judged=initial_judged,
            selected=selected,
            jobs=(spec.initial,) if phase == "actors" else (),
            draws=tuple(range(len(spec.judges if phase == "selectors" else spec.critics)))
            if phase in ("selectors", "critics")
            else (),
            initial_judgments_remaining=spec.initial.study.repeats
            if phase in ("actors", "initial_judging")
            else 0,
            actors_remaining=spec.initial.study.repeats if phase == "actors" else 0,
            selectors_remaining=len(spec.judges)
            if phase in ("actors", "initial_judging", "selectors")
            else 0,
            critics_remaining=len(spec.critics),
            suffixes_remaining=len(selected) * spec.continuations,
            minimum_suffixes_remaining=(
                len(selected) * spec.continuations if selected else spec.continuations
            ),
            point_capacity=len(spec.judges) * spec.points,
        ),
    )
    return nullcontext() if context is None else context


def phase_draws[T](
    admission: CouncilAdmission | None,
    spec: ActorCriticSpec,
    identity: str,
    phase: CouncilPhase,
    calls: Sequence[Callable[[], T]],
    *,
    group: ArtifactInput,
    selected: tuple[int, ...] = (),
    initial_judged: ArtifactInput | None = None,
) -> tuple[T, ...]:
    """Run an independent role wave entirely within its native admitted scope."""
    with admit_phase(
        admission,
        spec,
        identity,
        phase,
        group=group,
        selected=selected,
        initial_judged=initial_judged,
    ):
        return council_draws(calls, spec.concurrency)


def point_job(spec: ActorCriticSpec, checkpoint: ArtifactInput) -> ContinuationJob:
    """Return the continuation job that starts at one point."""
    return spec.initial.model_copy(
        update={
            "maximum_physical_attempts": spec.continuation_physical_attempts,
            "study": spec.initial.study.model_copy(
                update={
                    "checkpoint": checkpoint,
                    "repeats": spec.continuations,
                    "critic_prediction": None,
                    "predicted_success": None,
                    "selection_plan": None,
                }
            ),
        }
    )


def continuation_waves(
    admission: CouncilAdmission | None,
    spec: ActorCriticSpec,
    identity: str,
    group: ArtifactInput,
    selected: tuple[int, ...],
    checkpoints: dict[int, ArtifactInput],
    call: Callable[[int], PointResult],
    initial_judged: ArtifactInput | None = None,
) -> tuple[PointResult, ...]:
    """Admit bounded waves, then expose settled owners before any subsequent wave starts."""
    results: list[PointResult] = []
    completed: list[int] = []
    for offset in range(0, len(selected), spec.concurrency):
        wave = selected[offset : offset + spec.concurrency]
        jobs = tuple(point_job(spec, checkpoints[point]) for point in wave)

        def emit(
            phase: CouncilPhase,
            wave: tuple[int, ...] = wave,
            jobs: tuple[ContinuationJob, ...] = jobs,
        ) -> AbstractContextManager[object]:
            """Publish the current native inventory and return its real scoped dispatch context."""
            if admission is not None:
                context = admission(
                    spec,
                    CouncilCheckpoint(
                        run_identity=identity,
                        phase=phase,
                        group=group,
                        initial_judged=initial_judged,
                        selected=selected,
                        completed=tuple(completed),
                        wave=wave,
                        jobs=jobs,
                        suffixes_remaining=(len(selected) - len(completed)) * spec.continuations,
                        minimum_suffixes_remaining=(len(selected) - len(completed))
                        * spec.continuations,
                        point_capacity=len(selected),
                        failures=tuple(row.failure for row in results if row.failure),
                    ),
                )
                return nullcontext() if context is None else context
            return nullcontext()

        calls = tuple((lambda point=point: call(point)) for point in wave)
        with emit("continuations"):
            results.extend(council_draws(calls, spec.concurrency))
        completed.extend(wave)
        with emit("settled"):
            pass
    return tuple(results)
