"""Persisted, admitted family review draws and their credential-free cost inventory."""

from collections.abc import Mapping

from exp.common.core.artifacts import ArtifactInput, sha256_json
from exp.common.models import ModelRequest, ModelResponse
from exp.common.project import ProjectStore
from exp.optimize.evaluation.executable.costs import evaluation_costs
from exp.optimize.evaluation.executable.model_costs import CostComponent
from exp.optimize.evaluation.factory.authoring import author_response
from exp.optimize.evaluation.variance.build_records import VarianceBuildPlan
from exp.optimize.evaluation.variance.pipeline_spec import BuildSettings
from exp.simulation.mining.environment_review import CompatibilityReviewer


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
    project: ProjectStore,
    build: BuildSettings,
    *,
    code_revision: str,
    author_retries: Mapping[str, ArtifactInput] | None = None,
    execution_window: ArtifactInput | None = None,
) -> CompatibilityReviewer:
    """Bind admitted review calls, forwarding retry grants only for named logical identities.

    Retry references are operational authority and do not change the semantic configuration.
    """
    settings = build.reviewer
    if settings is None:
        raise ValueError("environment families need build.reviewer settings before execution")

    def draw(request: ModelRequest, *, identity: str) -> ModelResponse:
        """Buy or replay one exact compatibility request under the enclosing build lock."""
        _, response = author_response(
            project,
            settings,
            request,
            identity=identity,
            sources=(),
            code_revision=code_revision,
            **({"execution_window": execution_window} if execution_window is not None else {}),
            **(
                {"retry_authorization": author_retries[identity]}
                if author_retries and identity in author_retries
                else {}
            ),
        )
        return response

    return CompatibilityReviewer(
        draw=draw, model=settings.model, configuration_sha256=sha256_json(settings)
    )
