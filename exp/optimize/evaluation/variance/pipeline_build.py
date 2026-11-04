"""The variance-build pipeline executor, including bounded compatibility review."""

from exp.common.core.artifacts import JsonObject, sha256_json
from exp.common.project import ProjectStore, artifact_input
from exp.optimize.evaluation.executable.grounded import GroundedRuntimeDefinition
from exp.optimize.evaluation.variance.bootstrap_stage import bootstrapped_registration
from exp.optimize.evaluation.variance.build import (
    plan_variance_build,
    run_variance_build,
    variance_build_costs,
)
from exp.optimize.evaluation.variance.build_records import REPORT_TYPE, VarianceBuildReport
from exp.optimize.evaluation.variance.pipeline_errors import (
    build_lines,
    cluster_namer,
    naming_costs,
)
from exp.optimize.evaluation.variance.pipeline_review import family_reviewer, review_costs
from exp.optimize.evaluation.variance.pipeline_spec import (
    BuildOutputs,
    BuildSettings,
    CostLine,
    DecisionVarianceSpec,
    StageContext,
    StageRun,
    cost_lines,
)


class _BuildStage:
    """The ``variance-build`` stage."""

    def keys(self, project: ProjectStore, spec: DecisionVarianceSpec) -> JsonObject:
        """Bind the selected bootstrap task as well as the shared family build."""
        del project
        binding = spec.mining.bootstrap
        return {} if binding is None else {"bootstrap": sha256_json(binding)}

    def plan(
        self, project: ProjectStore, spec: DecisionVarianceSpec, code_revision: str
    ) -> tuple[tuple[CostLine, ...], tuple[str, ...]]:
        """Price every decision call of the variance build from its own plan."""
        saved = _saved_report(project, spec.build)
        if saved is not None:
            _outputs(StageContext(project, spec, "planning", code_revision), saved)
            return (), ("Reuses the pinned completed family build and its frozen review evidence.",)
        plan = plan_variance_build(
            project,
            spec.build.spec,
            code_revision=code_revision,
            reviewer_model=None if spec.build.reviewer is None else spec.build.reviewer.model,
            reviewer_configuration_sha256=(
                None if spec.build.reviewer is None else sha256_json(spec.build.reviewer)
            ),
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
        saved = _saved_report(project, context.spec.build)
        if saved is not None:
            return _outputs(context, saved)
        reviewer = context.dependencies.build_reviewer
        if spec.environment_families is not None:
            settings = context.spec.build.reviewer
            assert settings is not None
            if reviewer is None:
                reviewer = family_reviewer(
                    project, context.spec.build, code_revision=context.code_revision
                )
            elif reviewer.model != settings.model or reviewer.configuration_sha256 != sha256_json(
                settings
            ):
                raise ValueError("the injected family reviewer must match build.reviewer settings")
        namer = context.dependencies.build_namer
        if namer is None and spec.cluster_naming is not None:
            plan = plan_variance_build(
                project,
                spec,
                code_revision=context.code_revision,
                reviewer_model=None if reviewer is None else reviewer.model,
                reviewer_configuration_sha256=(
                    None if reviewer is None else reviewer.configuration_sha256
                ),
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
        return _outputs(context, report)


def _saved_report(project: ProjectStore, settings: BuildSettings) -> VarianceBuildReport | None:
    """Verify explicit frozen build provenance before planning or reusing its family inputs."""
    reference = settings.saved_report
    if reference is None:
        return None
    manifest = project.artifacts.read(reference.artifact_id).manifest
    if artifact_input(manifest) != reference or manifest.artifact_type != REPORT_TYPE:
        raise ValueError("build.saved_report must pin an immutable variance-build report")
    report = VarianceBuildReport.model_validate_json(
        project.artifacts.read_bytes(reference.artifact_id, "report.json")
    )
    if report.spec != settings.spec or reference.artifact_id != report.identity + "-result":
        raise ValueError(
            "build.saved_report differs from build.spec; select the exact frozen build"
        )
    return report


def _outputs(context: StageContext, report: VarianceBuildReport) -> StageRun:
    """Return exact saved family inputs after checking the bootstrap's world binding."""
    project = context.project
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
    binding = context.spec.mining.bootstrap
    if binding is not None:
        registration = bootstrapped_registration(project, binding)
        scoping = report.environment_families
        if scoping is None:
            raise ValueError("a bootstrapped pipeline needs its selected environment-family build")
        scope = scoping.scope(registration.family_id)
        definition = GroundedRuntimeDefinition.model_validate(registration.job.configuration)
        if definition.world_model != scope.world_model:
            raise ValueError("the bootstrap world differs from its saved family scope")
        outputs = outputs.model_copy(
            update={
                "world_model": scope.world_model,
                "tag_retrieval": scope.tag_retrieval,
                "rules_traces": scope.rules_traces,
                "index": scope.index,
                "retrieval_scope": scope.retrieval_scope,
                "selected_probes": (registration.task_id,),
            }
        )
    summary: JsonObject = {
        "exposure": report.exposure.artifact_id,
        "world_model": outputs.world_model.artifact_id,
        "rules_traces": len(outputs.rules_traces),
        "probes": len(report.probes),
        "selected_probes": list(report.selected_probes),
        "cluster_admitted": None if report.admission is None else len(report.admission.admitted),
    }
    refs = (saved, report.exposure, outputs.world_model, outputs.tag_retrieval, outputs.index)
    return StageRun(outputs, refs, summary)


BUILD_STAGE = _BuildStage()
"""The owner-bound variance-build executor."""
