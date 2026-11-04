"""The variance-build pipeline executor, including bounded compatibility review."""

from exp.common.core.artifacts import JsonObject
from exp.common.project import ProjectStore, artifact_input
from exp.optimize.evaluation.variance.build import (
    plan_variance_build,
    run_variance_build,
    variance_build_costs,
)
from exp.optimize.evaluation.variance.pipeline_errors import (
    build_lines,
    cluster_namer,
    naming_costs,
)
from exp.optimize.evaluation.variance.pipeline_review import family_reviewer, review_costs
from exp.optimize.evaluation.variance.pipeline_spec import (
    BuildOutputs,
    CostLine,
    DecisionVarianceSpec,
    StageContext,
    StageRun,
    cost_lines,
)


class _BuildStage:
    """The ``variance-build`` stage."""

    def keys(self, project: ProjectStore, spec: DecisionVarianceSpec) -> JsonObject:
        """The build reads only its spec and the project's completed build."""
        del project, spec
        return {}

    def plan(
        self, project: ProjectStore, spec: DecisionVarianceSpec, code_revision: str
    ) -> tuple[tuple[CostLine, ...], tuple[str, ...]]:
        """Price every decision call of the variance build from its own plan."""
        plan = plan_variance_build(
            project,
            spec.build.spec,
            code_revision=code_revision,
            reviewer_model=None if spec.build.reviewer is None else spec.build.reviewer.model,
        )
        roles = ", ".join(f"{role} {count.traces}" for role, count in sorted(plan.roles.items()))
        lines = (
            f"Variance build: traces by role {roles}; exposure fixed before tagging; "
            f"{plan.decision_calls} decision calls; final_evaluation is never tagged.",
            *build_lines(project, spec, plan),
        )
        if plan.saved:
            return (), (*lines, "The variance build report is saved; it replays.")
        costs = variance_build_costs(project, plan, spec.build.spec.decision_alias)
        return cost_lines(
            (*costs, *naming_costs(project, spec, plan), *review_costs(project, spec.build, plan))
        ), lines

    def run(self, context: StageContext) -> StageRun:
        """Run or replay the variance build."""
        project, spec = context.project, context.spec.build.spec
        reviewer = context.dependencies.build_reviewer
        if spec.environment_families is not None:
            settings = context.spec.build.reviewer
            assert settings is not None
            if reviewer is None:
                reviewer = family_reviewer(
                    project, context.spec.build, code_revision=context.code_revision
                )
            elif reviewer.model != settings.model:
                raise ValueError("the injected family reviewer must match build.reviewer.model")
        namer = context.dependencies.build_namer
        if namer is None and spec.cluster_naming is not None:
            plan = plan_variance_build(
                project,
                spec,
                code_revision=context.code_revision,
                reviewer_model=None if reviewer is None else reviewer.model,
            )
            namer = None if plan.saved else cluster_namer(context, plan)
        report = run_variance_build(
            project,
            spec,
            code_revision=context.code_revision,
            tagger=context.dependencies.build_tagger,
            constructor=context.dependencies.build_constructor,
            namer=namer,
            reviewer=reviewer,
        )
        saved = artifact_input(project.artifacts.read(report.identity + "-result").manifest)
        outputs = BuildOutputs(
            report=saved,
            exposure=report.exposure,
            world_model=report.world_model,
            tag_retrieval=report.tag_retrieval,
            trace_dataset=report.build.trace_dataset,
            rules_traces=report.rules_stage_traces,
            index=report.index,
            retrieval_scope=report.retrieval_scope,
            selected_probes=report.selected_probes,
            probe_ranking=report.probe_ranking,
        )
        summary: JsonObject = {
            "exposure": report.exposure.artifact_id,
            "world_model": report.world_model.artifact_id,
            "rules_traces": len(report.rules_stage_traces),
            "probes": len(report.probes),
            "selected_probes": list(report.selected_probes),
            "cluster_admitted": None
            if report.admission is None
            else len(report.admission.admitted),
        }
        refs = (saved, report.exposure, report.world_model, report.tag_retrieval, report.index)
        return StageRun(outputs, refs, summary)


BUILD_STAGE = _BuildStage()
"""The owner-bound variance-build executor."""
