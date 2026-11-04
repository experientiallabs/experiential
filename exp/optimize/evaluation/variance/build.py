"""Turn a completed ``exp build`` into the decision-variance pipeline's simulator.

One run is a frozen, saved, replayable stage over the project's selected build:

1. **Exposure.** The build's lineage bindings are split into roles by
   :func:`~exp.optimize.evaluation.variance.exposure.plan_exposure` before any tag is bought,
   and the plan is saved as a ``variance-exposure-plan`` artifact.
2. **Tags.** A decision model answers a named local question bank over the ``task`` view of
   every world-learning and mining-probe trace; final-evaluation traces are never rendered or
   sent. An optional second bank answers the ``observations`` view of world-learning traces
   only, as annotation. Decisions are bought through the shared decision cache, and every
   matrix is saved as a ``variance-tag-matrix`` artifact naming its bank, revision and view. A
   trace whose tagging fails is recorded with its error and is never redrawn.
3. **Scopes.** Optional tag rules narrow the construction and retrieval corpora inside world
   learning; by default both hold every world-learning trace. Traces whose tags the index would
   record but could not be bought are left out of retrieval.
4. **Index.** The scoped, tagged fit-only index is built with the build serving index's
   embedder identity, chunking and ``default_top_k``. Vectors come from the build's embedding
   checkpoint: a semantic embedder is bound to a client that refuses every call, so a missing
   vector fails the stage instead of being bought again.
5. **World model.** The build's world model (or, when requested, one constructed over the
   construction scope with an injected admitted constructor) is rebound to the scoped index.
6. **Bindings.** Every mining-probe task with tagged source evidence receives task-view query
   tags under the spec's tag policy, saved as a ``tag-retrieval-binding``; a control binding
   with weight zero is saved beside it for cosine-only baselines.
7. **Report.** Exposure counts, tagging coverage and failures, every artifact and the inputs the
   rules stage and actor-critic runs need are saved as a ``variance-build-report`` artifact.

The run identity binds the spec, the producing code, the selected build, the decision model and
the exact question banks, and a repeated run returns the saved report without any call.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

from exp.common.core.artifacts import (
    ArtifactEnvelope,
    ArtifactInput,
    JsonObject,
    sha256_json,
    stable_id,
)
from exp.common.core.locks import CoordinationLock
from exp.common.models import Embedding, ModelSnapshot, load_model_catalog
from exp.common.project import (
    ArtifactStore,
    ProjectBuildArtifacts,
    ProjectStore,
    artifact_input,
)
from exp.common.tasks import TaskCase
from exp.common.tasks.store import load_task_set
from exp.common.traces import Trace, load_trace_dataset
from exp.optimize.evaluation.executable.model_costs import CostComponent, decision_cost
from exp.optimize.evaluation.variance.build_records import (
    BINDING_TYPE,
    EXPOSURE_TYPE,
    MATRIX_TYPE,
    REPORT_TYPE,
    ConstructionSettings,
    ProbeBinding,
    RoleCount,
    ScopeRule,
    TagFailure,
    TagMatrixRecord,
    VarianceBuildPlan,
    VarianceBuildReport,
    VarianceBuildSpec,
    ViewPlan,
    ViewTagging,
    save_artifact,
)
from exp.optimize.evaluation.variance.exposure import (
    ExposurePlan,
    Role,
    plan_exposure,
    world_learning_scope,
)
from exp.optimize.evaluation.variance.tag_retrieval import index_tag_view, query_task_tags
from exp.runtime.gateway.decisions_contracts import MAX_DECISION_QUESTIONS
from exp.runtime.models.providers.admitted_transport import AdmittedJsonTransport
from exp.runtime.models.providers.transport import (
    HttpxJsonTransport,
    ProviderTransportError,
    RetryPolicy,
)
from exp.runtime.models.registry import RuntimeModelCatalog
from exp.runtime.sessions.dispatch_ledger import DispatchLedger
from exp.simulation.mining.bindings import load_task_set_lineage_bindings
from exp.simulation.mining.decisions import DecisionsTraceTagger
from exp.simulation.mining.questions import QuestionBank, resolve_question_bank
from exp.simulation.mining.tag_selection import (
    TagSelection,
    select_traces,
)
from exp.simulation.mining.tagging import (
    DecisionCache,
    TagMatrix,
    TraceTaggingError,
    distinct_units,
    question_chunks,
)
from exp.simulation.mining.views import Unit, observations_view, task_view
from exp.simulation.retrieval.build import persist_trace_rag
from exp.simulation.retrieval.build_inputs import load_completed_build_rag_lineage_bindings
from exp.simulation.retrieval.contracts import RAGIndex
from exp.simulation.retrieval.embedding import (
    RAGEmbedderBinding,
    RAGEmbeddingCache,
    default_rag_embedder,
)
from exp.simulation.retrieval.embedding_checkpoint import RAGEmbeddingCheckpoint
from exp.simulation.retrieval.embedding_inputs import plan_rag_embedding_inputs
from exp.simulation.retrieval.scope import CorpusScope
from exp.simulation.retrieval.store import load_rag_index
from exp.simulation.retrieval.tags import (
    RAGIndexTags,
    TagRetrievalBinding,
    TagRetrievalPolicy,
    TagView,
    TaskQueryTags,
)
from exp.simulation.world_model.construction import construct_world_model
from exp.simulation.world_model.runtime import (
    load_grounded_world_model_artifact,
    persist_scoped_world_model,
)

Constructor = Callable[[JsonObject, str], JsonObject]
"""Admitted world-model constructor, as :func:`construct_world_model` takes it."""

_ROLES: tuple[Role, ...] = ("world_learning", "mining_probes", "final_evaluation")
_TASK_ROLES: tuple[Role, ...] = ("world_learning", "mining_probes")


@dataclass(frozen=True)
class _Prepared:
    """Everything a run reads, resolved and validated before any call.

    Attributes:
        build: Selected completed build.
        plan: Exposure plan, not yet saved.
        traces: Build traces by identity.
        tasks: Build tasks in task-set order.
        banks: Selected bank per tagged view.
        units: Rendered units per tagged view, in trace order.
        summary: The credential-free summary.
    """

    build: ProjectBuildArtifacts
    plan: ExposurePlan
    traces: Mapping[str, Trace]
    tasks: tuple[TaskCase, ...]
    banks: Mapping[TagView, QuestionBank]
    units: Mapping[TagView, tuple[Unit, ...]]
    summary: VarianceBuildPlan


def plan_variance_build(
    project: ProjectStore,
    spec: VarianceBuildSpec,
    *,
    code_revision: str,
    decision_model: ModelSnapshot | None = None,
) -> VarianceBuildPlan:
    """Summarize a variance build without credentials, provider calls or publication.

    Args:
        project: Project with a completed build.
        spec: The registration.
        code_revision: Producing source identity.
        decision_model: Identity of an injected tagger; the catalog alias's when omitted.

    Returns:
        Traces per role, decisions per view, embedding reuse and the run identity.

    Raises:
        ValueError: There is no completed build, a bank or alias is missing, or a rule names a
            view that is not tagged.
    """
    return _prepare(project, spec, code_revision, decision_model).summary


def variance_build_costs(
    project: ProjectStore, plan: VarianceBuildPlan, alias: str
) -> tuple[CostComponent, ...]:
    """Price every decision call at the retry ceiling from the catalog, or leave it unpriced.

    Args:
        project: Project whose catalog prices the decision alias.
        plan: The summary whose calls are priced.
        alias: The decision alias.

    Returns:
        One decision component; no embedding component, since vectors are reused.
    """
    if plan.saved or plan.decision_attempts == 0:
        return ()
    return (decision_cost(project, alias, plan.decision_attempts),)


def run_variance_build(
    project: ProjectStore,
    spec: VarianceBuildSpec,
    *,
    code_revision: str,
    tagger: DecisionsTraceTagger | None = None,
    constructor: Constructor | None = None,
) -> VarianceBuildReport:
    """Run, or replay, the variance build over the project's completed build.

    Args:
        project: Project with a completed ``exp build``.
        spec: The registration.
        code_revision: Producing source identity.
        tagger: Decision tagger with a project decision cache; built from the spec's catalog
            alias over the admitted transport when omitted.
        constructor: Admitted world-model constructor, required when the spec requests
            construction and unused otherwise.

    Returns:
        The report, also saved as a ``variance-build-report`` artifact.

    Raises:
        ValueError: There is no completed build, a bank, alias or constructor is missing, a rule
            selects nothing, every trace of a view failed tagging, or the build's embedding
            checkpoint lacks a vector the scoped index needs.
        FileLockTimeout: Another process is running the same variance build.
    """
    spec = VarianceBuildSpec.model_validate_json(spec.model_dump_json())
    if spec.construction is not None and constructor is None:
        raise ValueError(
            "construction makes paid reconstruction calls; pass an admitted constructor or "
            "leave construction off to bind the build's world model"
        )
    if tagger is not None and tagger.cache is None:
        raise ValueError("a variance build tagger needs a project decision cache to replay")
    model = None if tagger is None else ModelSnapshot.model_validate(tagger.model_identity)
    prepared = _prepare(project, spec, code_revision, model)
    identity, store = prepared.summary.identity, project.artifacts
    with CoordinationLock(store.project_directory / (identity + ".lock"), timeout=0):
        result_id = identity + "-result"
        if store.exists(result_id):
            return VarianceBuildReport.model_validate_json(
                store.read_bytes(result_id, "report.json")
            )
        build = prepared.build
        exposure = save_artifact(
            store, EXPOSURE_TYPE, {"exposure.json": prepared.plan}, (build.task_set,), code_revision
        )
        active = tagger or _admitted_tagger(project, spec, prepared.summary, code_revision)
        tagging, matrices = _tag(store, prepared, active, code_revision)
        report = _bind(
            project,
            spec,
            prepared,
            exposure,
            tagging,
            matrices,
            code_revision=code_revision,
            constructor=constructor,
        )
        dependencies = {
            exposure,
            report.index,
            report.world_model,
            report.tag_retrieval,
            report.control_retrieval,
            *(row.matrix for row in tagging),
        }
        store.write_json(
            artifact_id=result_id,
            artifact_type=REPORT_TYPE,
            envelope=ArtifactEnvelope(
                schema_version=1,
                created_at=datetime.now(UTC),
                code_revision=code_revision,
                inputs=tuple(sorted(dependencies, key=lambda ref: ref.artifact_id)),
            ),
            files={"report.json": report},
        )
        return report


def _prepare(
    project: ProjectStore,
    spec: VarianceBuildSpec,
    code_revision: str,
    model: ModelSnapshot | None,
) -> _Prepared:
    """Load the build, freeze exposure, render units and compute the summary.

    Args:
        project: Project with a completed build.
        spec: The registration.
        code_revision: Producing source identity.
        model: Identity of an injected tagger, or None for the catalog alias's.

    Returns:
        The prepared run.

    Raises:
        ValueError: There is no completed build, a bank or alias is missing, or a rule names an
            untagged view.
    """
    build = project.load_project().build
    if build is None:
        raise ValueError("a variance build needs a completed `exp build` of this project")
    store = project.artifacts
    rows = load_task_set_lineage_bindings(store, build.task_set.artifact_id).bindings
    plan = plan_exposure(build.task_set, rows, seed=spec.seed, probe_fraction=spec.probe_fraction)
    if model is None:
        catalog = load_model_catalog(project.model_catalog_path)
        if spec.decision_alias not in catalog.models:
            raise ValueError(
                f"decision alias {spec.decision_alias!r} is not in the model catalog; configure "
                "a decision model on a typesafe connection"
            )
        model = RuntimeModelCatalog(catalog, environment={}).snapshot(spec.decision_alias)[0]
    banks: dict[TagView, QuestionBank] = {
        "task": resolve_question_bank(spec.task_bank).families(*spec.task_families)
    }
    if spec.observations_bank is not None:
        banks["observations"] = resolve_question_bank(spec.observations_bank).families(
            *spec.observations_families
        )
    for rule in (spec.construction_rule, spec.retrieval_rule):
        if rule is not None and rule.view not in banks:
            raise ValueError(
                f"tag rule {rule.rule.name!r} reads the {rule.view} view, which is not tagged; "
                "name an observations_bank or write the rule over the task view"
            )
    traces = {
        trace.trace_id: trace
        for trace in load_trace_dataset(store, build.trace_dataset.artifact_id).traces
    }
    tasks = load_task_set(store, build.task_set.artifact_id).tasks
    ordered = sorted(trace for trace in traces if trace in plan.traces)
    units: dict[TagView, tuple[Unit, ...]] = {
        "task": tuple(
            task_view(traces[trace]) for trace in ordered if plan.role_of(trace) in _TASK_ROLES
        )
    }
    if "observations" in banks:
        units["observations"] = tuple(
            observations_view(traces[trace])
            for trace in ordered
            if plan.role_of(trace) == "world_learning"
        )
    views = tuple(_view_plan(view, banks[view], units[view], plan) for view in sorted(banks))
    calls = sum(view.decision_calls for view in views)
    serving = load_rag_index(store, build.serving_rag.artifact_id).index
    total, saved_inputs = _embedding_reuse(project, build, plan, serving)
    identity = stable_id(
        "variance-build",
        {
            "spec": spec.identity(code_revision),
            "build": build.model_dump(mode="json"),
            "decision_model": model.model_dump(mode="json"),
            "banks": {view: sha256_json(bank) for view, bank in sorted(banks.items())},
        },
    )
    summary = VarianceBuildPlan(
        identity=identity,
        saved=store.exists(identity + "-result"),
        decision_model=model,
        roles=_role_counts(plan, tasks),
        views=views,
        decision_calls=calls,
        decision_attempts=calls * RetryPolicy().maximum_attempts,
        embedder=serving.embedder,
        embedding_inputs=total,
        embedding_inputs_saved=saved_inputs,
        construction=spec.construction is not None,
    )
    return _Prepared(build, plan, traces, tasks, banks, units, summary)


def _view_plan(
    view: TagView, bank: QuestionBank, units: tuple[Unit, ...], plan: ExposurePlan
) -> ViewPlan:
    """Count the decisions one view may buy.

    Args:
        view: Rendered view.
        bank: Selected bank.
        units: Rendered units of the view.
        plan: Exposure plan naming each unit's role.

    Returns:
        The view's plan.
    """
    per_unit = len(question_chunks(bank.questions, MAX_DECISION_QUESTIONS))
    distinct = len(distinct_units(units)) if units else 0
    return ViewPlan(
        view=view,
        bank_id=bank.bank_id,
        bank_version=bank.version,
        bank_sha256=sha256_json(bank),
        questions=len(bank.questions),
        roles=tuple(
            role for role in _ROLES if any(plan.role_of(u.trace_id) == role for u in units)
        ),
        traces=len(units),
        distinct_units=distinct,
        calls_per_unit=per_unit,
        decision_calls=distinct * per_unit,
    )


def _role_counts(plan: ExposurePlan, tasks: Sequence[TaskCase]) -> dict[Role, RoleCount]:
    """Count lineages, traces and tasks per role.

    Args:
        plan: Exposure plan.
        tasks: Build tasks.

    Returns:
        Counts for every role.
    """
    return {
        role: RoleCount(
            lineages=sum(1 for value in plan.roles.values() if value == role),
            traces=sum(1 for lineage in plan.traces.values() if plan.roles[lineage] == role),
            tasks=sum(1 for task in tasks if plan.roles.get(task.lineage_group_id) == role),
        )
        for role in _ROLES
    }


def _embedding_reuse(
    project: ProjectStore, build: ProjectBuildArtifacts, plan: ExposurePlan, serving: RAGIndex
) -> tuple[int, int]:
    """Count the world-learning embedding inputs and how many the build's checkpoint holds.

    Args:
        project: Project owning the checkpoint.
        build: Completed build whose fit index holds every fit transition.
        plan: Exposure plan.
        serving: The build's serving index envelope.

    Returns:
        Distinct inputs of every world-learning transition, and those already saved.
    """
    fit = load_rag_index(project.artifacts, build.fit_rag.artifact_id)
    keys = tuple(
        row.key_text
        for row in fit.transitions
        if row.trace_id in plan.traces and plan.role_of(row.trace_id) == "world_learning"
    )
    if not keys:
        return 0, 0
    texts = plan_rag_embedding_inputs(keys, maximum_chunk_bytes=serving.embedding_chunk_bytes).texts
    checkpoint = RAGEmbeddingCheckpoint(
        project.paths, serving.embedder, maximum_chunk_bytes=serving.embedding_chunk_bytes
    )
    return len(texts), len(checkpoint.load(texts))


def _admitted_tagger(
    project: ProjectStore, spec: VarianceBuildSpec, summary: VarianceBuildPlan, code_revision: str
) -> DecisionsTraceTagger:
    """Resolve the decision alias and post its decisions through the admitted transport.

    Every physical attempt is claimed in this run's dispatch ledger and, under a consented
    operation ceiling, reserves its cost priced as the decision alias.

    Args:
        project: Project owning the catalog, cache and receipts.
        spec: The registration.
        summary: The plan whose identity names the ledger and whose attempts bound it.
        code_revision: Producing source identity, recorded on attempt receipts.

    Returns:
        A tagger over the project's decision cache.

    Raises:
        TraceTaggingError: The alias is not a native decision alias.
        ValueError: Its resolved identity differs from the summarized one.
    """
    catalog = load_model_catalog(project.model_catalog_path)
    resolved = RuntimeModelCatalog(catalog).resolve(spec.decision_alias)
    if resolved.snapshot != summary.decision_model:
        raise ValueError(f"decision alias {spec.decision_alias!r} changed since it was planned")
    identity = summary.identity
    transport = AdmittedJsonTransport(
        HttpxJsonTransport(),
        ledger=DispatchLedger(
            project.artifacts.project_directory / (identity + "-physical.sqlite"),
            max(1, summary.decision_attempts),
            sha256_json(identity),
        ),
        artifacts=project.artifacts,
        code_revision=code_revision,
        deadline=datetime.now(UTC) + timedelta(hours=spec.tagging_deadline_hours),
    ).priced_as(spec.decision_alias)
    return DecisionsTraceTagger.for_alias(
        resolved,
        transport=transport,
        cache=DecisionCache(project.paths),
        maximum_concurrency=spec.concurrency,
    )


def _tag(
    store: ArtifactStore,
    prepared: _Prepared,
    tagger: DecisionsTraceTagger,
    code_revision: str,
) -> tuple[tuple[ViewTagging, ...], dict[TagView, TagMatrix]]:
    """Tag every view, save its matrix and record coverage and failures.

    Args:
        store: Artifact store receiving the matrices.
        prepared: The prepared run.
        tagger: The decision tagger.
        code_revision: Producing source identity.

    Returns:
        Coverage per view and the matrices by view.

    Raises:
        TraceTaggingError: Every unit of a view failed.
    """
    rows: list[ViewTagging] = []
    matrices: dict[TagView, TagMatrix] = {}
    model = prepared.summary.decision_model
    for view_plan in prepared.summary.views:
        view, bank = view_plan.view, prepared.banks[view_plan.view]
        matrix, failed = _tag_units(tagger, prepared.units[view], bank)
        failures = tuple(
            TagFailure(trace_id=trace, role=prepared.plan.role_of(trace), view=view, error=error)
            for trace, error in failed
        )
        record = TagMatrixRecord(
            bank_id=bank.bank_id,
            bank_version=bank.version,
            bank_sha256=view_plan.bank_sha256,
            view=view,
            roles=view_plan.roles,
            decision_model=model,
            matrix_sha256=sha256_json(matrix),
            traces=len(matrix.rows),
        )
        saved = save_artifact(
            store,
            MATRIX_TYPE,
            {"record.json": record, "matrix.json": matrix},
            (prepared.build.trace_dataset,),
            code_revision,
        )
        tagged: dict[Role, int] = {role: 0 for role in view_plan.roles}
        for row in matrix.rows:
            tagged[prepared.plan.role_of(row.trace_id)] += 1
        rows.append(ViewTagging(plan=view_plan, matrix=saved, tagged=tagged, failures=failures))
        matrices[view] = matrix
    return tuple(rows), matrices


def _tag_units(
    tagger: DecisionsTraceTagger, units: tuple[Unit, ...], bank: QuestionBank
) -> tuple[TagMatrix, tuple[tuple[str, str], ...]]:
    """Tag units in one pass; when it fails, find each failing trace and keep the rest.

    A failed pass is retried unit by unit. Committed decisions replay from the cache and a
    decision whose outcome is unknown is refused by the cache, so finding the failing traces
    buys nothing the first pass did not already attempt or would not have bought. A spend or
    admission refusal is not a tagging failure and stops the run.

    Args:
        tagger: The decision tagger, with a cache.
        units: Rendered units of one view.
        bank: Questions to ask.

    Returns:
        The matrix of every unit that was answered, and each failed trace with its error.

    Raises:
        TraceTaggingError: There are no units, or every unit failed.
    """
    if not units:
        raise TraceTaggingError(f"no trace to tag with bank {bank.bank_id}; check the exposure")
    try:
        return tagger.tag(units, bank), ()
    except (TraceTaggingError, ProviderTransportError):
        pass
    kept: list[Unit] = []
    failed: list[tuple[str, str]] = []
    for unit in units:
        try:
            tagger.tag([unit], bank)
        except (TraceTaggingError, ProviderTransportError) as exc:
            failed.append((unit.trace_id, str(exc)))
        else:
            kept.append(unit)
    if not kept:
        raise TraceTaggingError(
            f"every {units[0].view} unit failed tagging with bank {bank.bank_id}, for example "
            f"trace {failed[0][0]}: {failed[0][1]}"
        )
    return tagger.tag(kept, bank), tuple(failed)


def _bind(
    project: ProjectStore,
    spec: VarianceBuildSpec,
    prepared: _Prepared,
    exposure: ArtifactInput,
    tagging: tuple[ViewTagging, ...],
    matrices: Mapping[TagView, TagMatrix],
    *,
    code_revision: str,
    constructor: Constructor | None,
) -> VarianceBuildReport:
    """Declare scopes, build the index, bind the world model and the probe bindings.

    Args:
        project: Project receiving every artifact.
        spec: The registration.
        prepared: The prepared run.
        exposure: Saved exposure plan.
        tagging: Coverage per view.
        matrices: Matrices by view.
        code_revision: Producing source identity.
        constructor: Admitted constructor when construction is requested.

    Returns:
        The report.

    Raises:
        ValueError: A rule selects nothing, or the checkpoint lacks a needed vector.
    """
    store, plan, build = project.artifacts, prepared.plan, prepared.build
    construction_selection = _select(spec.construction_rule, matrices, prepared, plan)
    retrieval_selection = _select(spec.retrieval_rule, matrices, prepared, plan)
    construction_scope = world_learning_scope(
        plan, name="world-learning-construction", selection=construction_selection
    )
    retrieval_scope = world_learning_scope(
        plan, name="world-learning-retrieval", selection=retrieval_selection
    )
    untagged = {
        failure.trace_id
        for row in tagging
        for failure in row.failures
        if failure.role == "world_learning"
    }
    if untagged & set(retrieval_scope.ids):
        kept = tuple(trace for trace in retrieval_scope.ids if trace not in untagged)
        if not kept:
            raise ValueError("every retrieval trace failed tagging; no scoped index can be built")
        retrieval_scope = CorpusScope.declare(
            name=retrieval_scope.name, kind="trace", ids=kept, source_plan=sha256_json(plan)
        )
    views = [
        index_tag_view(
            matrices[view],
            bank=prepared.banks[view],
            plan=plan,
            view=view,
            threshold=spec.threshold,
        )
        for view in sorted(matrices)
    ]
    index_tags = RAGIndexTags(views=tuple(views))
    fit = load_rag_index(store, build.fit_rag.artifact_id)
    serving = load_rag_index(store, build.serving_rag.artifact_id).index
    embedder = _saved_vector_embedder(serving)
    persisted = persist_trace_rag(
        store,
        (build.trace_dataset,),
        load_completed_build_rag_lineage_bindings(store, build),
        created_at=fit.index.created_at,
        code_revision=code_revision,
        embedder=embedder,
        default_top_k=serving.default_top_k,
        included_partitions=frozenset({"fit"}),
        embedding_cache=RAGEmbeddingCache(
            embedder, maximum_chunk_bytes=serving.embedding_chunk_bytes
        ),
        corpus_scope=retrieval_scope,
        trace_tags=index_tags,
    )
    index = artifact_input(persisted.manifest)
    constructed = None
    source_world = build.world_model
    if spec.construction is not None and constructor is not None:
        constructed = _construct(
            project, spec.construction, prepared, construction_scope, constructor, code_revision
        )
        source_world = constructed
    world = persist_scoped_world_model(
        store,
        source_world,
        index,
        retrieval_scope=retrieval_scope,
        construction_scope=construction_scope if constructed is not None else None,
    )
    recorded = persisted.index.trace_tags
    assert recorded is not None
    probes, queries = _probe_tags(prepared, matrices["task"], recorded)
    task_bank = prepared.banks["task"]
    treatment = TagRetrievalPolicy(
        bank_id=task_bank.bank_id,
        bank_version=task_bank.version,
        threshold=spec.threshold,
        weight=spec.tag_weight,
        mode=spec.tag_mode,
        families=spec.tag_families,
    )
    control = treatment.model_copy(update={"weight": 0.0, "mode": "boost", "families": ()})
    bindings = tuple(
        save_artifact(
            store,
            BINDING_TYPE,
            {"binding.json": TagRetrievalBinding(policy=policy, tasks=queries)},
            (index,),
            code_revision,
        )
        for policy in (treatment, control)
    )
    rules_traces = construction_scope.ids
    return VarianceBuildReport(
        spec=spec,
        identity=prepared.summary.identity,
        build=build,
        plan=prepared.summary,
        exposure=exposure,
        tagging=tagging,
        construction_scope=construction_scope,
        retrieval_scope=retrieval_scope,
        construction_selection={}
        if construction_selection is None
        else construction_selection.uncertain,
        retrieval_selection={} if retrieval_selection is None else retrieval_selection.uncertain,
        index=index,
        index_transitions=persisted.index.transition_count,
        constructed=constructed,
        world_model=world,
        tag_retrieval=bindings[0],
        control_retrieval=bindings[1],
        probes=probes,
        rules_stage_traces=rules_traces,
        next_steps=_next_steps(exposure, world, bindings, len(rules_traces), len(queries)),
    )


def _select(
    rule: ScopeRule | None,
    matrices: Mapping[TagView, TagMatrix],
    prepared: _Prepared,
    plan: ExposurePlan,
) -> TagSelection | None:
    """Apply one optional rule to the world-learning rows of its view's matrix.

    Args:
        rule: The rule, or None for the whole role.
        matrices: Matrices by view.
        prepared: The prepared run, holding each view's bank.
        plan: Exposure plan.

    Returns:
        The selection inside world learning, or None.
    """
    if rule is None:
        return None
    matrix = matrices[rule.view]
    rows = tuple(row for row in matrix.rows if plan.role_of(row.trace_id) == "world_learning")
    inside = matrix.model_copy(update={"rows": rows})
    selection = select_traces(inside, rule.rule, bank=prepared.banks[rule.view])
    return plan.restrict(selection, "world_learning")


def _saved_vector_embedder(serving: RAGIndex) -> RAGEmbedderBinding:
    """Bind the build's embedder identity so the scoped index reuses its saved vectors.

    The local hashing embedder is free and deterministic, so it is bound as itself. A semantic
    embedder is bound to a client that refuses every call: the scoped index holds a subset of
    the build's fit transitions, whose vectors the build saved in the project's embedding
    checkpoint under the same identity and chunking.

    Args:
        serving: The build's serving index envelope.

    Returns:
        The binding.
    """
    hashing = default_rag_embedder(serving.embedding_dimension)
    if hashing.snapshot == serving.embedder:
        return hashing
    return RAGEmbedderBinding(
        client=_SavedVectorsOnly(serving.embedder.model_id),
        snapshot=serving.embedder,
        maximum_input_tokens=serving.embedding_chunk_bytes,
    )


class _SavedVectorsOnly:
    """An embedding client for vectors the build already saved; any call is refused."""

    def __init__(self, model_id: str) -> None:
        """Name the embedder whose vectors must already be saved."""
        self._model_id = model_id

    def embed(self, texts: Sequence[str]) -> tuple[Embedding, ...]:
        """Refuse: the build's embedding checkpoint must already hold every input.

        Raises:
            ValueError: Always, naming how many inputs were missing.
        """
        raise ValueError(
            f"{len(texts)} embedding inputs for {self._model_id} are not in the build's "
            "embedding checkpoint; rerun `exp build` for this project so the scoped index can "
            "reuse its vectors"
        )


def _construct(
    project: ProjectStore,
    settings: ConstructionSettings,
    prepared: _Prepared,
    scope: CorpusScope,
    constructor: Constructor,
    code_revision: str,
) -> ArtifactInput:
    """Reconstruct the world model from exactly the construction scope's traces.

    Args:
        project: Project receiving the constructed world model.
        settings: Construction settings.
        prepared: The prepared run.
        scope: Construction scope.
        constructor: Admitted constructor.
        code_revision: Producing source identity.

    Returns:
        The constructed world model over the build's serving index, recording the scope.
    """
    store, build = project.artifacts, prepared.build
    world = load_grounded_world_model_artifact(store, build.world_model)
    started = datetime.now(UTC)
    built = construct_world_model(
        tuple(prepared.traces[trace] for trace in scope.ids),
        store=store,
        serving_rag=build.serving_rag,
        complete=constructor,
        model_alias=world.model_alias,
        model=world.model,
        run_id=prepared.summary.identity + "-construction",
        clock=lambda: datetime.now(UTC),
        deadline=started + timedelta(minutes=settings.deadline_minutes),
        code_revision=code_revision,
        top_k=world.top_k,
        construction_guidance=settings.construction_guidance,
        prediction_guidance=settings.prediction_guidance,
        construction_scope=scope,
        lineage_bindings=load_completed_build_rag_lineage_bindings(store, build),
    )
    return artifact_input(built.simulator.manifest)


def _probe_tags(
    prepared: _Prepared, matrix: TagMatrix, index_tags: RAGIndexTags
) -> tuple[tuple[ProbeBinding, ...], dict[str, TaskQueryTags]]:
    """Read every mining-probe task's query tags from its first tagged source trace.

    Args:
        prepared: The prepared run.
        matrix: Task-view matrix.
        index_tags: The scoped index's tag record.

    Returns:
        One row per probe task in task order, and query tags keyed by task.
    """
    plan = prepared.plan
    tagged = {row.trace_id for row in matrix.rows}
    rows: list[ProbeBinding] = []
    queries: dict[str, TaskQueryTags] = {}
    for task in prepared.tasks:
        if plan.roles.get(task.lineage_group_id) != "mining_probes":
            continue
        trace = next((item for item in task.source_trace_ids if item in tagged), None)
        fired = 0
        if trace is not None:
            query = query_task_tags(matrix, trace, index_tags=index_tags, plan=plan)
            queries[task.task_id] = query
            fired = len(query.fired)
        rows.append(
            ProbeBinding(
                task_id=task.task_id, lineage_id=task.lineage_group_id, trace_id=trace, fired=fired
            )
        )
    return tuple(rows), queries


def _next_steps(
    exposure: ArtifactInput,
    world: ArtifactInput,
    bindings: tuple[ArtifactInput, ...],
    rules_traces: int,
    probes: int,
) -> tuple[str, ...]:
    """Describe how later stages consume this build's artifacts.

    Args:
        exposure: Saved exposure plan.
        world: Scoped world model.
        bindings: Treatment and control bindings.
        rules_traces: Number of traces for the rules stage.
        probes: Number of probe tasks with query tags.

    Returns:
        One line per consumer.
    """
    return (
        f"rules-stage: exposure {exposure.artifact_id} (load_exposure_plan), traces "
        f"rules_stage_traces ({rules_traces} world-learning traces)",
        f"actor-critic: one spec per mining-probe task, with the same exposure plan; the "
        f"simulator is world model {world.artifact_id}",
        f"tag retrieval: load_grounded_world_model({world.artifact_id}, tag_retrieval="
        f"load_tag_retrieval_binding({bindings[0].artifact_id})) for {probes} probe tasks; "
        f"control {bindings[1].artifact_id} ranks by cosine similarity alone",
    )
