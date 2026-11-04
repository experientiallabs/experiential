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
   The error bank (``error_bank``) is asked over the observations view of every world-learning
   and mining-probe trace and saved as an ``error-tags`` record.
3. **Scopes.** The world-learning tag space (every tagged view plus the error tags) is clustered
   with a data-chosen cluster count and saved as ``tag-clusters`` (named by an injected model
   when ``cluster_naming`` is set); cluster admission then chooses which world-learning traces
   enter the construction and retrieval corpora. An explicit tag rule overrides admission for
   its corpus; with neither, a corpus holds every world-learning trace. The report records why
   each trace was admitted. Traces whose tags the index would record but could not be bought
   are left out of retrieval.
4. **Index.** The scoped, tagged fit-only index is built with the build serving index's
   embedder identity, chunking and ``default_top_k``. Vectors come from the build's embedding
   checkpoint: a semantic embedder is bound to a client that refuses every call, so a missing
   vector fails the stage instead of being bought again.
5. **World model.** The build's world model (or, when requested, one constructed over the
   construction scope with an injected admitted constructor) is rebound to the scoped index.
6. **Bindings.** Every mining-probe task with tagged source evidence receives task-view query
   tags under the spec's tag policy, saved as a ``tag-retrieval-binding``; a control binding
   with weight zero is saved beside it for cosine-only baselines. Probe tasks are ranked by
   their source traces' error tags and the most error-laden are selected for error mining.
7. **Report.** Exposure counts, tagging coverage and failures, every artifact and the inputs the
   rules stage and actor-critic runs need are saved as a ``variance-build-report`` artifact.

The run identity binds the spec, the producing code, the selected build, the decision model and
the exact question banks, and a repeated run returns the saved report without any call.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

from exp.common.core.artifacts import (
    ArtifactEnvelope,
    ArtifactInput,
    JsonObject,
    Sha256,
    sha256_json,
    stable_id,
)
from exp.common.core.locks import CoordinationLock
from exp.common.models import ModelSnapshot, load_model_catalog
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
    ScopeRule,
    TagFailure,
    TagMatrixRecord,
    VarianceBuildPlan,
    VarianceBuildReport,
    VarianceBuildSpec,
    ViewTagging,
    corpus_scope,
    next_steps,
    probe_selection,
    role_counts,
    save_artifact,
    trace_admissions,
    view_plan,
)
from exp.optimize.evaluation.variance.exposure import (
    ExposurePlan,
    Role,
    plan_exposure,
)
from exp.optimize.evaluation.variance.family_scopes import require_family_review, scope_families
from exp.optimize.evaluation.variance.saved_vectors import saved_vector_embedder
from exp.optimize.evaluation.variance.tag_retrieval import index_tag_view, query_task_tags
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
from exp.simulation.mining.dispatch import dispatch_tagging
from exp.simulation.mining.environment_review import CompatibilityReviewer
from exp.simulation.mining.error_tags import (
    ErrorTags,
    bank_for_view,
    error_bank,
    persist_error_tags,
    tag_errors,
)
from exp.simulation.mining.questions import QuestionBank, resolve_question_bank
from exp.simulation.mining.tag_cluster_names import ClusterNamer, named_clusters
from exp.simulation.mining.tag_clusters import (
    ClusterAdmission,
    admit_by_cluster,
    cluster_tags,
    persist_tag_clustering,
)
from exp.simulation.mining.tag_selection import (
    TagSelection,
    select_traces,
)
from exp.simulation.mining.tagging import (
    DecisionCache,
    TagMatrix,
    TagRow,
    TraceTaggingError,
    distinct_units,
    tagged_view,
)
from exp.simulation.mining.views import Unit, observations_view, task_view
from exp.simulation.retrieval.build import persist_trace_rag
from exp.simulation.retrieval.build_inputs import load_completed_build_rag_lineage_bindings
from exp.simulation.retrieval.contracts import RAGIndex
from exp.simulation.retrieval.embedding import (
    RAGEmbeddingCache,
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
        errors: Error bank and its observations units, when error tags are requested.
    """

    build: ProjectBuildArtifacts
    plan: ExposurePlan
    traces: Mapping[str, Trace]
    tasks: tuple[TaskCase, ...]
    banks: Mapping[TagView, QuestionBank]
    units: Mapping[TagView, tuple[Unit, ...]]
    summary: VarianceBuildPlan
    errors: tuple[QuestionBank, tuple[Unit, ...]] | None


def plan_variance_build(
    project: ProjectStore,
    spec: VarianceBuildSpec,
    *,
    code_revision: str,
    decision_model: ModelSnapshot | None = None,
    reviewer_model: ModelSnapshot | None = None,
    reviewer_configuration_sha256: Sha256 | None = None,
) -> VarianceBuildPlan:
    """Summarize a variance build without credentials, provider calls or publication.

    Args:
        project: Project with a completed build.
        spec: The registration.
        code_revision: Producing source identity.
        decision_model: Identity of an injected tagger; the catalog alias's when omitted.
        reviewer_model: Identity of the compatibility reviewer, part of the run identity when
            the spec requests environment families.
        reviewer_configuration_sha256: Frozen caller settings beyond the reviewer model.

    Returns:
        Traces per role, decisions per view, embedding reuse and the run identity.

    Raises:
        ValueError: There is no completed build, a bank or alias is missing, or a rule names a
            view that is not tagged.
    """
    return _prepare(
        project, spec, code_revision, decision_model, reviewer_model, reviewer_configuration_sha256
    ).summary


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
    namer: ClusterNamer | None = None,
    reviewer: CompatibilityReviewer | None = None,
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
        namer: Admitted cluster-naming model, required when the spec requests naming.
        reviewer: Admitted compatibility reviewer, required when the spec requests environment
            families; each family is then scoped by ``scope_families``.

    Returns:
        The report, also saved as a ``variance-build-report`` artifact.

    Raises:
        ValueError: There is no completed build, a bank, alias, constructor or namer is
            missing, a rule or cluster admission selects nothing, every trace of a view failed
            tagging, or the build's embedding checkpoint lacks a vector the scoped index needs.
        FileLockTimeout: Another process is running the same variance build.
    """
    spec = VarianceBuildSpec.model_validate_json(spec.model_dump_json())
    if spec.construction is not None and constructor is None:
        raise ValueError(
            "construction makes paid reconstruction calls; pass an admitted constructor or "
            "leave construction off to bind the build's world model"
        )
    if spec.cluster_naming is not None and (namer is None or spec.clustering is None):
        raise ValueError(
            "cluster naming makes paid calls over clusters; pass an admitted namer and keep "
            "clustering on, or leave cluster_naming unset"
        )
    if tagger is not None and tagger.cache is None:
        raise ValueError("a variance build tagger needs a project decision cache to replay")
    model = None if tagger is None else ModelSnapshot.model_validate(tagger.model_identity)
    reviewed = None if reviewer is None else reviewer.model
    prepared = _prepare(
        project,
        spec,
        code_revision,
        model,
        reviewed,
        None if reviewer is None else reviewer.configuration_sha256,
    )
    identity, store = prepared.summary.identity, project.artifacts
    with CoordinationLock(store.project_directory / (identity + ".lock"), timeout=0):
        result_id = identity + "-result"
        if store.exists(result_id):
            return VarianceBuildReport.model_validate_json(
                store.read_bytes(result_id, "report.json")
            )
        require_family_review(spec, reviewer)
        build = prepared.build
        exposure = save_artifact(
            store, EXPOSURE_TYPE, {"exposure.json": prepared.plan}, (build.task_set,), code_revision
        )
        active = tagger or _admitted_tagger(project, spec, prepared.summary, code_revision)
        tagging, matrices = _tag(store, prepared, active, code_revision)
        errors = _tag_errors(store, prepared, active, code_revision)
        clusters = _cluster(store, spec, prepared, matrices, errors, namer, code_revision)
        report = _bind(
            project,
            spec,
            prepared,
            exposure,
            tagging,
            matrices,
            code_revision=code_revision,
            constructor=constructor,
            admission=None if clusters is None else clusters[2],
        )
        ranking, selected, failures = probe_selection(
            prepared.plan, prepared.tasks, None if errors is None else errors[0], spec.probe_limit
        )
        report = report.model_copy(
            update={
                "errors": None if errors is None else errors[1],
                "error_failures": failures,
                "clustering": None if clusters is None else clusters[0],
                "cluster_names": None if clusters is None else clusters[1],
                "admission": None if clusters is None else clusters[2],
                "probe_ranking": ranking,
                "selected_probes": selected,
            }
        )
        report = scope_families(
            project,
            report,
            traces=prepared.traces,
            tasks=prepared.tasks,
            reviewer=reviewer,
            code_revision=code_revision,
            constructor=constructor,
        )
        families = report.environment_families
        dependencies = {
            exposure,
            report.index,
            report.world_model,
            report.tag_retrieval,
            report.control_retrieval,
            *(row.matrix for row in tagging),
            *(ref for ref in (report.errors, report.clustering, report.cluster_names) if ref),
            *(() if families is None else families.references()),
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
    reviewer: ModelSnapshot | None = None,
    reviewer_configuration: Sha256 | None = None,
) -> _Prepared:
    """Load the build, freeze exposure, render units and compute the summary.

    Args:
        project: Project with a completed build.
        spec: The registration.
        code_revision: Producing source identity.
        model: Identity of an injected tagger, or None for the catalog alias's.
        reviewer: Identity of the compatibility reviewer, keyed when families are requested.
        reviewer_configuration: Frozen caller settings beyond the reviewer model.

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
    units_of = {trace for trace in ordered if plan.role_of(trace) in _TASK_ROLES}
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
    views = tuple(view_plan(view, banks[view], units[view], plan) for view in sorted(banks))
    errors = None
    if spec.error_bank is not None:
        asked = bank_for_view(
            error_bank(spec.error_bank, families=spec.error_families), "observations"
        )
        errors = (asked, tuple(observations_view(traces[t]) for t in ordered if t in units_of))
    error_plan = None if errors is None else view_plan("observations", *errors, plan)
    calls = sum(
        view.decision_calls for view in (*views, *(() if error_plan is None else (error_plan,)))
    )
    serving = load_rag_index(store, build.serving_rag.artifact_id).index
    total, saved_inputs = _embedding_reuse(project, build, plan, serving)
    identity = stable_id(
        "variance-build",
        {
            "spec": spec.identity(code_revision),
            "build": build.model_dump(mode="json"),
            "decision_model": model.model_dump(mode="json"),
            **(
                {}
                if spec.environment_families is None or reviewer is None
                else {
                    "reviewer": reviewer.model_dump(mode="json"),
                    "reviewer_configuration": reviewer_configuration,
                }
            ),
            "banks": {view: sha256_json(bank) for view, bank in sorted(banks.items())},
            "error_bank": None if errors is None else sha256_json(errors[0]),
        },
    )
    summary_roles = role_counts(plan, tasks)
    summary = VarianceBuildPlan(
        identity=identity,
        saved=store.exists(identity + "-result"),
        decision_model=model,
        roles=summary_roles,
        views=views,
        decision_calls=calls,
        decision_attempts=calls * RetryPolicy().maximum_attempts,
        embedder=serving.embedder,
        embedding_inputs=total,
        embedding_inputs_saved=saved_inputs,
        construction=spec.construction is not None,
        errors=error_plan,
        naming_calls=0
        if spec.cluster_naming is None or spec.clustering is None
        else spec.cluster_naming.calls(spec.clustering.max_clusters),
        review_calls=0
        if spec.environment_families is None
        else spec.environment_families.calls(summary_roles["world_learning"].traces),
    )
    return _Prepared(build, plan, traces, tasks, banks, units, summary, errors)


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
    for planned in prepared.summary.views:
        view, bank = planned.view, prepared.banks[planned.view]
        matrix, failed = _tag_units(tagger, prepared.units[view], bank)
        failures = tuple(
            TagFailure(trace_id=trace, role=prepared.plan.role_of(trace), view=view, error=error)
            for trace, error in failed
        )
        record = TagMatrixRecord(
            bank_id=bank.bank_id,
            bank_version=bank.version,
            bank_sha256=planned.bank_sha256,
            view=view,
            roles=planned.roles,
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
        tagged: dict[Role, int] = {role: 0 for role in planned.roles}
        for row in matrix.rows:
            tagged[prepared.plan.role_of(row.trace_id)] += 1
        rows.append(ViewTagging(plan=planned, matrix=saved, tagged=tagged, failures=failures))
        matrices[view] = matrix
    return tuple(rows), matrices


def _tag_units(
    tagger: DecisionsTraceTagger, units: tuple[Unit, ...], bank: QuestionBank
) -> tuple[TagMatrix, tuple[tuple[str, str], ...]]:
    """Tag independent units, keeping input failures apart from fatal refusals.

    Args:
        tagger: Decision client with its concurrency bound.
        units: Rendered traces of one view.
        bank: Questions to ask.

    Returns:
        Tagged rows and recoverable failures; fatal refusals propagate unchanged.
    """
    view = tagged_view(units)
    decided = distinct_units(units)

    def answer(unit: Unit) -> TagRow | str:
        """Return one row or a recoverable failure, preserving fatal refusals."""
        try:
            return tagger.tag([unit], bank).rows[0]
        except (TraceTaggingError, ProviderTransportError) as exc:
            return str(exc)

    outcomes = dispatch_tagging(decided, answer, maximum_concurrency=tagger.maximum_concurrency)
    by_hash = {unit.unit_hash: result for unit, result in zip(decided, outcomes, strict=True)}
    rows: dict[str, TagRow] = {}
    failed: list[tuple[str, str]] = []
    for unit in units:
        result = by_hash[unit.unit_hash]
        if isinstance(result, str):
            failed.append((unit.trace_id, result))
        else:
            rows.setdefault(unit.trace_id, result.model_copy(update={"trace_id": unit.trace_id}))
    if not rows:
        raise TraceTaggingError(
            f"every {units[0].view} unit failed tagging with bank {bank.bank_id}, for example "
            f"trace {failed[0][0]}: {failed[0][1]}"
        )
    return TagMatrix(
        bank_id=bank.bank_id,
        bank_version=bank.version,
        question_ids=bank.question_ids(),
        view=view,
        rows=tuple(rows.values()),
    ), tuple(failed)


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
    admission: ClusterAdmission | None,
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
        admission: Cluster admission narrowing every corpus without an explicit rule.

    Returns:
        The report.

    Raises:
        ValueError: A rule selects nothing, or the checkpoint lacks a needed vector.
    """
    store, plan, build = project.artifacts, prepared.plan, prepared.build
    construction_selection = _select(spec.construction_rule, matrices, prepared, plan)
    retrieval_selection = _select(spec.retrieval_rule, matrices, prepared, plan)
    construction_scope = corpus_scope(plan, "construction", construction_selection, admission)
    retrieval_scope = corpus_scope(plan, "retrieval", retrieval_selection, admission)
    admissions = trace_admissions(
        (construction_scope, spec.construction_rule),
        (retrieval_scope, spec.retrieval_rule),
        admission,
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
    embedder = saved_vector_embedder(serving)
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
        admissions=admissions,
        next_steps=next_steps(exposure, world, bindings, len(rules_traces), len(queries)),
    )


def _tag_errors(
    store: ArtifactStore,
    prepared: _Prepared,
    tagger: DecisionsTraceTagger,
    code_revision: str,
) -> tuple[ErrorTags, ArtifactInput] | None:
    """Tag world-learning and mining-probe observations with the error bank and save them.

    Returns:
        The record and its artifact, or None when no error bank is named.
    """
    if prepared.errors is None:
        return None
    bank, units = prepared.errors
    errors = tag_errors(
        units, tagger=tagger, bank=bank, unit_concurrency=tagger.maximum_concurrency
    )
    return errors, persist_error_tags(
        store, errors, code_revision=code_revision, inputs=(prepared.build.trace_dataset,)
    )


def _cluster(
    store: ArtifactStore,
    spec: VarianceBuildSpec,
    prepared: _Prepared,
    matrices: Mapping[TagView, TagMatrix],
    errors: tuple[ErrorTags, ArtifactInput] | None,
    namer: ClusterNamer | None,
    code_revision: str,
) -> tuple[ArtifactInput, ArtifactInput | None, ClusterAdmission] | None:
    """Cluster the world-learning tag space, name it when asked and admit traces by cluster.

    Returns:
        The saved clustering, the saved names (or None) and the admission, or None when
        clustering is off.
    """
    if spec.clustering is None:
        return None
    plan = prepared.plan
    joined = [matrices[view] for view in sorted(matrices)]
    texts = {f"{v}:{q.question_id}": q.text for v, b in prepared.banks.items() for q in b.questions}
    if errors is not None:
        joined.append(errors[0].matrix.model_copy(update={"view": "errors"}))
        bank = prepared.errors[0] if prepared.errors else None
        texts |= {f"errors:{q.question_id}": q.text for q in (bank.questions if bank else ())}
    learning = {t for t in plan.traces if plan.role_of(t) == "world_learning"}
    clustering = cluster_tags(joined, spec.clustering, trace_ids=learning & set(prepared.traces))
    inputs = () if errors is None else (errors[1],)
    saved = persist_tag_clustering(store, clustering, code_revision=code_revision, inputs=inputs)
    names = reference = None
    if spec.cluster_naming is not None and namer is not None:
        exemplars = {
            trace: observations_view(prepared.traces[trace]).text
            for cluster in clustering.clusters
            for trace in cluster.exemplars
        }
        names, reference = named_clusters(
            store,
            clustering,
            namer,
            questions=texts,
            exemplar_texts=exemplars,
            spec=spec.cluster_naming,
            code_revision=code_revision,
            inputs=(saved,),
        )
    held_out = {t for t in plan.traces if plan.role_of(t) == "final_evaluation"}
    admission = admit_by_cluster(
        clustering,
        spec.cluster_admission,
        eligible=learning,
        held_out=held_out,
        names=None if names is None else names.mapping(),
    )
    return saved, reference, admission


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
