"""Use one pinned prepared family's real index, rules corpus and native query catalog."""

from exp.common.core.artifacts import ArtifactInput
from exp.common.models import ModelCatalog, load_model_catalog
from exp.common.project import ProjectStore, artifact_input
from exp.optimize.evaluation.continuations.job import ContinuationJob
from exp.optimize.evaluation.executable.catalog_binding import load_runtime_catalog
from exp.optimize.evaluation.executable.grounded import GroundedRuntimeDefinition
from exp.optimize.evaluation.variance.bootstrap_contracts import BootstrapBinding
from exp.optimize.evaluation.variance.bootstrap_stage import bootstrapped_registration
from exp.optimize.evaluation.variance.build_records import load_exposure_plan
from exp.optimize.evaluation.variance.family_simulators import load_family_simulator
from exp.optimize.evaluation.variance.pipeline_spec import BuildOutputs, RulesOutputs, StageContext
from exp.simulation.continuations import load_continuation_checkpoint
from exp.simulation.retrieval.store import load_rag_index
from exp.simulation.world_model.runtime import load_grounded_world_model_artifact


def native_world_catalog(project: ProjectStore, runtime: GroundedRuntimeDefinition) -> ModelCatalog:
    """Resolve the same verified private/global catalog as the actual frozen world factory."""
    return (
        load_model_catalog(project.model_catalog_path)
        if runtime.catalog_binding is None
        else load_runtime_catalog(project, runtime.catalog_binding)
    )


def mining_family_build(context: StageContext, build: BuildOutputs) -> BuildOutputs:
    """Return an explicit in-memory view; never copy, relabel or mutate a completed build."""
    reference = context.spec.mining.family_simulator
    if reference is None:
        return build
    binding = context.spec.mining.bootstrap
    if binding is None:
        raise ValueError("family retrieval requires its native task-start binding")
    job = RulesOutputs.model_validate(context.upstream["rules"].outputs).job
    return prepared_family_build(
        context.project, build, family_ref=reference, binding=binding, job=job
    )


def prepared_family_build(
    project: ProjectStore,
    build: BuildOutputs,
    *,
    family_ref: ArtifactInput,
    binding: BootstrapBinding,
    job: ContinuationJob,
) -> BuildOutputs:
    """Use the same exact family boundary before rules-stage handoff and after it."""
    family = load_family_simulator(project.artifacts, family_ref)
    registration = bootstrapped_registration(project, binding)
    if job != registration.job:
        raise ValueError("mining retrieval differs from its exact prepared family/runtime")
    runtime = GroundedRuntimeDefinition.model_validate(job.configuration)
    excluded = {"task", "manifest", "initial_state", "probe_paths", "facts"}
    if (
        family.runtime is None
        or family.family_id != registration.family_id
        or runtime.model_dump(exclude=excluded) != family.runtime.model_dump(exclude=excluded)
        or family.world_model != runtime.world_model
        or family.rule_book != runtime.rule_book
        or family.rule_memory != runtime.rule_memory
        or family.rules_report != registration.rules_report
    ):
        raise ValueError("mining retrieval differs from its exact prepared family/runtime")
    world = load_grounded_world_model_artifact(project.artifacts, family.world_model)
    loaded = load_rag_index(project.artifacts, family.index.artifact_id)
    if (
        artifact_input(loaded.manifest) != family.index
        or world.serving_rag != family.index
        or loaded.index.corpus_scope != family.corpus
        or loaded.index.included_partitions != ("fit",)
    ):
        raise ValueError("family retrieval has another index, scope or evaluation exposure")
    exposure = load_exposure_plan(project, build.exposure)
    exposure.require(family.members, "world_learning")
    learning = family.rules_traces or family.members
    exposure.require(learning, "world_learning")
    exposure.require(tuple(row.trace_id for row in loaded.transitions), "world_learning")
    start = load_continuation_checkpoint(project.artifacts, job.study.checkpoint)
    exposure.require_lineage(start.task.lineage_group_id, "mining_probes")
    if any(
        row.trace_id in start.task.source_trace_ids or row.lineage_id == start.task.lineage_group_id
        for row in loaded.transitions
    ):
        raise ValueError("mining probe entered the family's learning/retrieval corpus")
    return build.model_copy(
        update={
            "world_model": family.world_model,
            "index": family.index,
            "retrieval_scope": family.corpus,
            "tag_retrieval": family.tag_retrieval,
            "rules_traces": learning,
        }
    )
