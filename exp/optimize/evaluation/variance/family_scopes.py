"""Scope a variance build per environment family instead of once for the whole build.

When a variance build registers ``environment_families``, its saved tag clustering is reviewed
into environment families (:mod:`exp.simulation.mining.environment_families`) by an injected,
admitted reviewer over world-learning traces only. Each compatible family, and each uncertain
group the spec explicitly admits, then gets its own simulator inputs keyed by its family
identity:

- a scoped, tagged fit-only index over its members that are in the build's tagged retrieval
  scope, reusing the build's saved vectors exactly as the build-wide index does;
- the build's world model rebound to that index; when the build constructed its world (or the
  build's world carries knowledge), the family's world is reconstructed from its own members
  only, so no knowledge from another family or an excluded group survives;
- the world-learning traces its one rule memory is learned from (the rules stage runs once per
  family over exactly these);
- its tasks: world-learning tasks whose source traces are members, and mining-probe tasks placed
  by the fitted mixture's posterior over their source trace's tags, into the one admitted family
  of that cluster whose reconciled contracts cover the task's tools (otherwise unplaced); tool
  names are compared by canonical key (:mod:`exp.simulation.rules.tool_names`), so ``Write``
  is covered by a ``write_file`` contract. Placed
  tasks are saved into the scoping's ``environment-families`` record, so a family loaded from
  it lists exactly its scope's tasks;
- a tag retrieval binding over its placed probe tasks, under the build's treatment policy.

Uncertain groups that are not admitted are reported and left out of training. Held-out traces
never enter: the clustering is over world learning, and a final-evaluation trace is refused.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from datetime import UTC, datetime, timedelta

from exp.common.core.artifacts import ArtifactInput, JsonObject, sha256_json
from exp.common.project import ProjectStore, artifact_input
from exp.common.tasks import TaskCase
from exp.common.traces import Trace
from exp.optimize.evaluation.variance.build_records import (
    BINDING_TYPE,
    ConstructionSettings,
    FamilyScope,
    FamilyScoping,
    ProbePlacement,
    VarianceBuildReport,
    VarianceBuildSpec,
    load_exposure_plan,
    load_tag_matrix,
    load_tag_retrieval_binding,
    save_artifact,
)
from exp.optimize.evaluation.variance.exposure import ExposurePlan
from exp.optimize.evaluation.variance.saved_vectors import saved_vector_embedder
from exp.optimize.evaluation.variance.tag_retrieval import query_task_tags
from exp.simulation.mining.environment_families import (
    EnvironmentFamily,
    FamilyTask,
    admitted_families,
    environment_families,
    family_task,
    persist_environment_families,
    with_placed,
)
from exp.simulation.mining.environment_review import CompatibilityReviewer
from exp.simulation.mining.error_tags import load_error_tags
from exp.simulation.mining.tag_clusters import TagClustering, assign_clusters, load_tag_clustering
from exp.simulation.mining.tagging import TagMatrix
from exp.simulation.retrieval.build import persist_trace_rag
from exp.simulation.retrieval.build_inputs import load_completed_build_rag_lineage_bindings
from exp.simulation.retrieval.embedding import RAGEmbeddingCache
from exp.simulation.retrieval.scope import CorpusScope
from exp.simulation.retrieval.store import load_rag_index
from exp.simulation.retrieval.tags import TagRetrievalBinding, TaskQueryTags
from exp.simulation.rules.tool_names import ToolNames
from exp.simulation.world_model.construction import construct_world_model
from exp.simulation.world_model.runtime import (
    load_grounded_world_model_artifact,
    persist_scoped_world_model,
)

Constructor = Callable[[JsonObject, str], JsonObject]
"""Admitted world-model constructor, as :func:`construct_world_model` takes it."""


def require_family_review(spec: VarianceBuildSpec, reviewer: CompatibilityReviewer | None) -> None:
    """Refuse, before any call, a family registration without clustering or a reviewer.

    Raises:
        ValueError: Families are requested without clustering or an admitted reviewer.
    """
    if spec.environment_families is not None and (reviewer is None or spec.clustering is None):
        raise ValueError(
            "environment families make paid review calls over clusters; pass an admitted "
            "reviewer and keep clustering on, or leave environment_families unset"
        )


def scope_families(
    project: ProjectStore,
    report: VarianceBuildReport,
    *,
    traces: Mapping[str, Trace],
    tasks: Sequence[TaskCase],
    reviewer: CompatibilityReviewer | None,
    code_revision: str,
    constructor: Constructor | None = None,
) -> VarianceBuildReport:
    """Review the build's clusters into families and give each its own simulator inputs.

    Args:
        project: Project holding the build and receiving every artifact.
        report: The build-wide report, with its clustering saved.
        traces: Build traces by identity; only world-learning members are read.
        tasks: Build tasks; held-out ones are never attached.
        reviewer: Admitted compatibility reviewer.
        code_revision: Producing source identity.
        constructor: Admitted world-model constructor. When the build constructed its world,
            or the build's world carries knowledge from traces outside a family, each family's
            world is reconstructed from its own members only; without a constructor that is
            refused.

    Returns:
        The report with ``environment_families`` set, or unchanged when none were requested.

    Raises:
        ValueError: Families are requested without a clustering or a reviewer, a family's
            index needs a vector the build's checkpoint lacks, or a family's world would keep
            knowledge from outside the family and no constructor is given.
    """
    spec = report.spec.environment_families
    if spec is None:
        return report
    require_family_review(report.spec, reviewer)
    if report.clustering is None or reviewer is None:
        raise ValueError("environment families need the build's saved clustering and a reviewer")
    store, build = project.artifacts, report.build
    plan = load_exposure_plan(project, report.exposure)
    clustering = load_tag_clustering(store, report.clustering)
    learning = {t for t in plan.traces if plan.role_of(t) == "world_learning"}
    held_out = {t for t in plan.traces if plan.role_of(t) == "final_evaluation"}
    record, reviewed = environment_families(
        store,
        clustering,
        reviewer,
        traces=traces,
        eligible=learning,
        held_out=held_out,
        tasks=[task for task in tasks if _role(plan, task) == "world_learning"],
        spec=spec.review,
        code_revision=code_revision,
        inputs=(report.clustering,),
    )
    admitted = admitted_families(record, spec.admit_uncertain, traces=traces)
    matrices = [load_tag_matrix(project, row.matrix) for row in report.tagging]
    if report.errors is not None:
        errors = load_error_tags(store, report.errors).matrix
        matrices.append(errors.model_copy(update={"view": "errors"}))
    placements, placed = _place(plan, tasks, clustering, matrices, admitted)
    record = with_placed(record, placed)
    saved = persist_environment_families(
        store, record, code_revision=code_revision, inputs=(reviewed,)
    )
    admitted = admitted_families(record, spec.admit_uncertain, traces=traces)
    task_matrix = next(m for m in matrices if m.view == "task")
    policy = load_tag_retrieval_binding(project, report.tag_retrieval).policy
    serving = load_rag_index(store, build.serving_rag.artifact_id).index
    fit = load_rag_index(store, build.fit_rag.artifact_id)
    tags = load_rag_index(store, report.index.artifact_id).index.trace_tags
    embedder = saved_vector_embedder(serving)
    lineage = load_completed_build_rag_lineage_bindings(store, build)
    base = load_grounded_world_model_artifact(store, build.world_model)
    rebuild = report.constructed is not None or base.knowledge is not None
    in_scope = set(report.retrieval_scope.ids)
    scopes: list[FamilyScope] = []
    left_out: dict[str, str] = {}
    for family in admitted:
        ids = sorted(set(family.members) & in_scope)
        if not ids:
            left_out[family.family_id] = "no member is in the build's tagged retrieval scope"
            continue
        scope = CorpusScope.declare(
            name="family-" + family.family_id[-20:],
            kind="trace",
            ids=ids,
            source_plan=sha256_json(plan),
        )
        persisted = persist_trace_rag(
            store,
            (build.trace_dataset,),
            lineage,
            created_at=fit.index.created_at,
            code_revision=code_revision,
            embedder=embedder,
            default_top_k=serving.default_top_k,
            included_partitions=frozenset({"fit"}),
            embedding_cache=RAGEmbeddingCache(
                embedder, maximum_chunk_bytes=serving.embedding_chunk_bytes
            ),
            corpus_scope=scope,
            trace_tags=tags,
        )
        index = artifact_input(persisted.manifest)
        source_world, construction = build.world_model, None
        if rebuild:
            construction = CorpusScope.declare(
                name="family-construction-" + family.family_id[-20:],
                kind="trace",
                ids=family.members,
                source_plan=sha256_json(plan),
            )
            source_world = _construct(
                project, report, construction, traces, constructor, family.family_id, code_revision
            )
        world = persist_scoped_world_model(
            store, source_world, index, retrieval_scope=scope, construction_scope=construction
        )
        queries: dict[str, TaskQueryTags] = {}
        recorded = persisted.index.trace_tags
        for task in family.tasks:
            row = next((p for p in placements if p.task_id == task.task_id), None)
            if recorded is not None and row is not None and row.trace_id is not None:
                queries[task.task_id] = query_task_tags(
                    task_matrix, row.trace_id, index_tags=recorded, plan=plan
                )
        binding = None
        if queries:
            content = {"binding.json": TagRetrievalBinding(policy=policy, tasks=queries)}
            binding = save_artifact(store, BINDING_TYPE, content, (index,), code_revision)
        scopes.append(
            FamilyScope(
                family_id=family.family_id,
                status=family.status,
                cluster_id=family.cluster_id,
                path=family.path,
                members=family.members,
                retrieval_scope=scope,
                index=index,
                world_model=world,
                tag_retrieval=binding,
                rules_traces=family.members,
                tools=tuple(sorted(family.tool_contracts)),
                tasks=family.task_ids(),
            )
        )
    admitted_ids = set(spec.admit_uncertain)
    scoping = FamilyScoping(
        families=saved,
        scopes=tuple(scopes),
        uncertain=tuple(g.group_id for g in record.uncertain if g.group_id not in admitted_ids),
        admitted_uncertain=tuple(sorted(admitted_ids)),
        left_out=left_out,
        unresolved=dict(record.unresolved),
        placements=placements,
    )
    return report.model_copy(update={"environment_families": scoping})


def _role(plan: ExposurePlan, task: TaskCase) -> str | None:
    """Return a task's data role, or None when its lineage is not in the plan."""
    return plan.roles.get(task.lineage_group_id)


def _place(
    plan: ExposurePlan,
    tasks: Sequence[TaskCase],
    clustering: TagClustering,
    matrices: Sequence[TagMatrix],
    families: Sequence[EnvironmentFamily],
) -> tuple[tuple[ProbePlacement, ...], dict[str, list[FamilyTask]]]:
    """Place each mining-probe task in one family by its source trace's cluster posterior."""
    rows: list[ProbePlacement] = []
    placed: dict[str, list[FamilyTask]] = {}
    for task in tasks:
        if _role(plan, task) != "mining_probes":
            continue
        where = assign_clusters(clustering, matrices, task.source_trace_ids)
        trace = next((t for t in task.source_trace_ids if t in where), None)
        if trace is None:
            rows.append(ProbePlacement(task_id=task.task_id, reason="no tagged source trace"))
            continue
        cluster, posterior = where[trace]
        tools = sorted({tool.name for tool in task.tools})
        candidates = [
            f
            for f in families
            if f.cluster_id == cluster and ToolNames(f.tool_contracts).covers(tools)
        ]
        found = ProbePlacement(
            task_id=task.task_id, trace_id=trace, cluster_id=cluster, posterior=posterior, reason=""
        )
        if len(candidates) != 1:
            why = (
                f"cluster {cluster} has no admitted family"
                if not any(f.cluster_id == cluster for f in families)
                else f"{len(candidates)} admitted families of cluster {cluster} have contracts "
                f"covering the task's tools {tools}"
            )
            rows.append(found.model_copy(update={"reason": why}))
            continue
        family = candidates[0]
        placed.setdefault(family.family_id, []).append(family_task(task, "mining_probes"))
        why = f"posterior {posterior:.3f} in cluster {cluster}; family {family.path}"
        rows.append(found.model_copy(update={"family_id": family.family_id, "reason": why}))
    return tuple(rows), placed


def _construct(
    project: ProjectStore,
    report: VarianceBuildReport,
    scope: CorpusScope,
    traces: Mapping[str, Trace],
    constructor: Constructor | None,
    family_id: str,
    code_revision: str,
) -> ArtifactInput:
    """Reconstruct the world model from exactly one family's members.

    Raises:
        ValueError: No constructor is given.
    """
    if constructor is None:
        raise ValueError(
            f"family {family_id} would keep world knowledge constructed from traces outside "
            "it; pass an admitted constructor so each family's world is built from its own "
            "members"
        )
    store, build = project.artifacts, report.build
    settings = report.spec.construction or ConstructionSettings()
    world = load_grounded_world_model_artifact(store, build.world_model)
    built = construct_world_model(
        tuple(traces[trace] for trace in scope.ids),
        store=store,
        serving_rag=build.serving_rag,
        complete=constructor,
        model_alias=world.model_alias,
        model=world.model,
        run_id=f"{report.identity}-construction-{family_id}",
        clock=lambda: datetime.now(UTC),
        deadline=datetime.now(UTC) + timedelta(minutes=settings.deadline_minutes),
        code_revision=code_revision,
        top_k=world.top_k,
        construction_guidance=settings.construction_guidance,
        prediction_guidance=settings.prediction_guidance,
        construction_scope=scope,
        lineage_bindings=load_completed_build_rag_lineage_bindings(store, build),
    )
    return artifact_input(built.simulator.manifest)
