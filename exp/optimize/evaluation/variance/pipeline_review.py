"""Persisted, admitted family review draws and their credential-free cost inventory."""

from exp.common.core.artifacts import sha256_json
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
    project: ProjectStore, build: BuildSettings, *, code_revision: str
) -> CompatibilityReviewer:
    """Bind one reviewer whose calls use exact saved author-response replay and admission."""
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
        )
        return response

    return CompatibilityReviewer(
        draw=draw, model=settings.model, configuration_sha256=sha256_json(settings)
    )
