"""Persisted, admitted family review draws and their credential-free cost inventory."""

from datetime import UTC, datetime

from exp.common.core.artifacts import ArtifactEnvelope, ContractModel, sha256_json
from exp.common.models import ModelRequest, ModelResponse
from exp.common.project import ProjectStore
from exp.common.state_machine import TransitionJournal
from exp.optimize.evaluation.executable.costs import evaluation_costs
from exp.optimize.evaluation.executable.model_costs import CostComponent
from exp.optimize.evaluation.factory.authoring import author_response
from exp.optimize.evaluation.variance.build_records import VarianceBuildPlan
from exp.optimize.evaluation.variance.pipeline_spec import BuildSettings
from exp.simulation.mining.environment_review import CompatibilityReviewer
from exp.simulation.mining.review_states import (
    DRAW_MACHINE,
    DrawState,
    TransientReviewFailure,
    is_transient,
)


def review_costs(
    project: ProjectStore, settings: BuildSettings, plan: VarianceBuildPlan
) -> tuple[CostComponent, ...]:
    """Price every bounded compatibility review draw through the registered author owner."""
    if plan.saved or not plan.review_calls:
        return ()
    if settings.reviewer is None:
        raise ValueError("environment families need build.reviewer settings before spend planning")
    return tuple(
        row.scaled(plan.review_calls) for row in evaluation_costs(project, settings.reviewer)
    )


def family_reviewer(
    project: ProjectStore, build: BuildSettings, *, code_revision: str
) -> CompatibilityReviewer:
    """Bind one reviewer whose calls use exact saved author-response replay and admission.

    Each draw identity walks :data:`DRAW_MACHINE`. A completed response is saved and replays
    (``answered``). A transport-level failure (:func:`is_transient`: a retryable or transport
    error, or an attempt budget exhausted on them) saves a ``<identity>-transient-<n>`` receipt,
    moves the draw to ``transient_failed`` and raises :class:`TransientReviewFailure`, so it is
    never read as a review. The next run moves it back to ``pending`` and draws again under the
    fresh attempt identity ``<identity>-retry-<n>``, whose own intent, response and attempt
    ledger are separate, so the abandoned attempt's intent never blocks it.
    """
    settings = build.reviewer
    if settings is None:
        raise ValueError("environment families need build.reviewer settings before execution")
    store = project.artifacts
    journal = TransitionJournal(store.project_directory / "environment-review" / "draws.jsonl")

    def draw(request: ModelRequest, *, identity: str) -> ModelResponse:
        """Buy or replay one exact compatibility request under the enclosing build lock."""
        failed = 0
        while store.exists(f"{identity}-transient-{failed + 1}"):
            failed += 1
        state = DRAW_MACHINE.move(identity, None, DrawState.PENDING.value)
        for _ in range(failed):
            state = DRAW_MACHINE.move(identity, state, "failed_transiently")
            state = DRAW_MACHINE.move(identity, state, "retried")
        attempt = identity if failed == 0 else f"{identity}-retry-{failed}"
        try:
            _, response = author_response(
                project,
                settings,
                request,
                identity=attempt,
                sources=(),
                code_revision=code_revision,
            )
        except Exception as error:
            reason = is_transient(error)
            if reason is None:
                raise
            receipt = TransientReceipt(attempt=attempt, reason=reason, error=type(error).__name__)
            store.write_json(
                artifact_id=f"{identity}-transient-{failed + 1}",
                artifact_type=TRANSIENT_TYPE,
                envelope=ArtifactEnvelope(
                    schema_version=1, created_at=datetime.now(UTC), code_revision=code_revision
                ),
                files={"receipt.json": receipt},
            )
            DRAW_MACHINE.move(identity, state, "failed_transiently", journal)
            raise TransientReviewFailure(identity, reason) from error
        DRAW_MACHINE.move(identity, state, "answered", journal)
        return response

    return CompatibilityReviewer(
        draw=draw, model=settings.model, configuration_sha256=sha256_json(settings)
    )


TRANSIENT_TYPE = "environment-review-transient"
"""Artifact type of one transport-level draw failure receipt (``receipt.json``)."""


class TransientReceipt(ContractModel):
    """One draw attempt that failed at the transport level; a failed attempt, never a review.

    Attributes:
        attempt: The attempt identity that failed.
        reason: Its retry classification.
        error: The failure's type name.
    """

    attempt: str
    reason: str
    error: str
