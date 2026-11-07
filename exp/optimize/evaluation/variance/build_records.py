"""Contracts and saved artifacts of the variance build.

A variance build (:mod:`exp.optimize.evaluation.variance.build`) saves its exposure plan, every
tag matrix, both tag retrieval bindings and its report as content-addressed artifacts. This module
holds their contracts, the pure builders of their records (view plans, role counts, the
world-learning corpora from a rule or cluster admission with why each trace was admitted, and the
error-ranked probe selection) and the readers later stages use to load them by the references a
report names.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from typing import Literal

from pydantic import BaseModel, Field

from exp.common.core.artifacts import (
    ArtifactEnvelope,
    ArtifactInput,
    ContractModel,
    sha256_json,
    stable_id,
)
from exp.common.models import ModelSnapshot
from exp.common.project import (
    ArtifactAlreadyExistsError,
    ArtifactStore,
    ProjectBuildArtifacts,
    ProjectStore,
    artifact_input,
)
from exp.common.tasks import TaskCase
from exp.optimize.evaluation.variance.exposure import ExposurePlan, Role, world_learning_scope
from exp.runtime.gateway.decisions_contracts import MAX_DECISION_QUESTIONS
from exp.simulation.mining.decisions import DEFAULT_DECISION_ALIAS
from exp.simulation.mining.dispatch import DEFAULT_TAGGING_CONCURRENCY
from exp.simulation.mining.environment_families import EnvironmentFamilySpec, FamilyStatus
from exp.simulation.mining.error_tags import (
    ERROR_BANK_ID,
    ErrorTags,
    TaskErrorRank,
    rank_error_tasks,
)
from exp.simulation.mining.questions import QuestionBank
from exp.simulation.mining.tag_cluster_names import ClusterNamingSpec
from exp.simulation.mining.tag_clusters import (
    ClusterAdmission,
    ClusterAdmissionSpec,
    ClusteringSpec,
)
from exp.simulation.mining.tag_selection import DEFAULT_FIRE_THRESHOLD, TagRule, TagSelection
from exp.simulation.mining.tagging import TagMatrix, distinct_units, question_chunks
from exp.simulation.mining.views import Unit
from exp.simulation.retrieval.scope import CorpusScope
from exp.simulation.retrieval.tags import TagRetrievalBinding, TagRetrievalMode, TagView

DEFAULT_TASK_BANK = "capability-shape-task-v1"
"""Local question bank asked about the task view unless the spec names another."""

EXPOSURE_TYPE = "variance-exposure-plan"
"""Artifact type of a saved :class:`ExposurePlan` (``exposure.json``)."""

MATRIX_TYPE = "variance-tag-matrix"
"""Artifact type of a saved tag matrix (``record.json`` and ``matrix.json``)."""

BINDING_TYPE = "tag-retrieval-binding"
"""Artifact type of a saved :class:`TagRetrievalBinding` (``binding.json``)."""

REPORT_TYPE = "variance-build-report"
"""Artifact type of a saved :class:`VarianceBuildReport` (``report.json``)."""

_ROLES: tuple[Role, ...] = ("world_learning", "mining_probes", "final_evaluation")


Corpus = Literal["construction", "retrieval"]
"""A world-learning corpus a trace may be admitted to."""


class ScopeRule(ContractModel):
    """A tag rule narrowing one world-learning corpus, read from one view's matrix.

    Attributes:
        view: The matrix the rule reads; ``observations`` needs an observations bank.
        rule: Which fired propositions keep a world-learning trace.
    """

    view: TagView = "task"
    rule: TagRule


class ConstructionSettings(ContractModel):
    """Optional paid reconstruction of the world model over the construction scope.

    Attributes:
        deadline_minutes: Wall-clock allowance of the constructor run.
        construction_guidance: Instructions under the constructor's frozen output contract.
        prediction_guidance: Predictor instructions bound into the constructed artifact.
    """

    deadline_minutes: float = Field(default=60.0, gt=0.0)
    construction_guidance: str = ""
    prediction_guidance: str = ""


class VarianceBuildSpec(ContractModel):
    """One registered variance build over the project's completed build.

    Attributes:
        decision_alias: Catalog alias of the decision model, on a typesafe connection.
        task_bank: Local question bank asked about the task view.
        task_families: Families of the task bank to ask; all when empty.
        observations_bank: Optional bank asked about world-learning observations views.
        observations_families: Families of the observations bank to ask; all when empty.
        seed: Seed of the fit-lineage split.
        probe_fraction: Share of fit lineages reserved as mining probes.
        error_bank: Local error bank asked over the observations view of every world-learning
            and mining-probe trace (never final evaluation), or None to skip error tags. Error
            tags rank the mining probes and join the clustered tag space.
        error_families: Families of the error bank to ask; all when empty.
        error_consumed_families: Families of the error bank whose columns the build consumes,
            or None to ask only the bank's observations-view propositions. When set, every
            selected question of the error bank (``error_families``) is asked over the
            observations view, the full matrix is saved as ``error_matrix``, and only these
            declared families' columns enter the error tags, the clustered tag space and the
            probe ranking.
        clustering: How the world-learning tag space (every tagged view plus the error tags) is
            clustered, its cluster count chosen by the data; None skips clustering.
        cluster_admission: Which clusters, and how many traces of each, enter world learning.
            By default every eligible member of every cluster is admitted. It narrows both the
            construction and retrieval corpora unless an explicit rule overrides that corpus.
        cluster_naming: How an injected naming model names the clusters, or None to leave them
            unnamed; naming needs a namer passed to the run.
        environment_families: How an injected reviewer turns the clusters into environment
            families, each given its own scoped index, world model and rule-memory inputs; None
            keeps one simulator for the whole build. Needs clustering and a reviewer.
        probe_limit: Most mining-probe tasks selected for error mining, most error-laden first;
            None selects every probe task, still ranked.
        construction_rule: Optional explicit tag rule for the construction corpus; it overrides
            cluster admission for that corpus.
        retrieval_rule: Optional explicit tag rule for the retrieval corpus; it overrides
            cluster admission for that corpus.
        construction: Reconstruct the world model over the construction scope. Off by default:
            reconstruction makes paid calls and needs an injected constructor.
        threshold: Probability strictly above which a proposition fires, for the index, the
            query tags and the policy alike.
        tag_weight: Weight of the task-tag overlap in the treatment policy.
        tag_mode: ``boost`` re-ranks; ``require_shared_family`` also filters by family.
        tag_families: Families of the filter, required by ``require_shared_family``.
        concurrency: Decision calls in flight at once. It changes no result, so it is left out
            of every identity.
        tagging_deadline_hours: Wall-clock allowance of the admitted decision transport; left
            out of every identity.
    """

    decision_alias: str = Field(default=DEFAULT_DECISION_ALIAS, min_length=1)
    task_bank: str = DEFAULT_TASK_BANK
    task_families: tuple[str, ...] = ()
    observations_bank: str | None = None
    observations_families: tuple[str, ...] = ()
    seed: int = Field(default=0, ge=0)
    probe_fraction: float = Field(default=0.3, gt=0.0, lt=1.0)
    error_bank: str | None = ERROR_BANK_ID
    error_families: tuple[str, ...] = ()
    error_consumed_families: tuple[str, ...] | None = Field(
        default=None, min_length=1, exclude_if=lambda value: value is None
    )
    clustering: ClusteringSpec | None = Field(default_factory=ClusteringSpec)
    cluster_admission: ClusterAdmissionSpec = Field(default_factory=ClusterAdmissionSpec)
    cluster_naming: ClusterNamingSpec | None = None
    environment_families: EnvironmentFamilySpec | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    probe_limit: int | None = Field(default=None, ge=1)
    construction_rule: ScopeRule | None = None
    retrieval_rule: ScopeRule | None = None
    construction: ConstructionSettings | None = None
    threshold: float = Field(default=DEFAULT_FIRE_THRESHOLD, gt=0.0, lt=1.0)
    tag_weight: float = Field(default=0.25, ge=0.0)
    tag_mode: TagRetrievalMode = "boost"
    tag_families: tuple[str, ...] = ()
    concurrency: int = Field(default=DEFAULT_TAGGING_CONCURRENCY, ge=1, le=64)
    tagging_deadline_hours: float = Field(default=12.0, gt=0.0)

    def identity(self, code_revision: str) -> str:
        """Bind the registration and the producing code.

        Args:
            code_revision: Producing source identity.

        Returns:
            Stable identity of the registration.
        """
        return stable_id(
            "variance-build-spec",
            {
                "spec": sha256_json(
                    self.model_dump(mode="json", exclude={"concurrency", "tagging_deadline_hours"})
                ),
                "code": code_revision,
            },
        )


class RoleCount(ContractModel):
    """How much of the task set one role holds.

    Attributes:
        lineages: Leakage lineages reserved for the role.
        traces: Source traces of those lineages.
        tasks: Representative tasks of those lineages.
    """

    lineages: int = Field(ge=0)
    traces: int = Field(ge=0)
    tasks: int = Field(ge=0)


class ViewPlan(ContractModel):
    """The decisions one view's tagging may buy.

    Attributes:
        view: Rendered view.
        bank_id: Bank asked about it.
        bank_version: Revision of that bank.
        bank_sha256: Digest of the exact selected questions.
        questions: Questions asked per unit.
        roles: Roles whose traces are tagged on this view.
        traces: Traces tagged.
        distinct_units: Distinct rendered texts, each decided once.
        calls_per_unit: Decision calls per unit, ``ceil(questions / 32)``.
        decision_calls: ``distinct_units * calls_per_unit``; saved decisions replay free.
    """

    view: TagView
    bank_id: str
    bank_version: int
    bank_sha256: str
    questions: int = Field(ge=1)
    roles: tuple[Role, ...]
    traces: int = Field(ge=0)
    distinct_units: int = Field(ge=0)
    calls_per_unit: int = Field(ge=1)
    decision_calls: int = Field(ge=0)


class VarianceBuildPlan(ContractModel):
    """The credential-free summary of one variance build, computed before consent.

    Attributes:
        identity: Run identity the report is saved under.
        saved: The report already exists and replays without any call.
        decision_model: Identity of the decision model.
        roles: Exposure counts by role.
        views: Decisions each view may buy.
        decision_calls: Calls across every view.
        decision_attempts: Physical attempts the calls may take under the retry policy.
        embedder: Identity of the build's embedder, reused for the scoped index.
        embedding_inputs: Distinct embedding inputs of every world-learning transition.
        embedding_inputs_saved: Of those, the inputs already in the build's checkpoint; the
            scoped index is a subset, so it needs no embedding call when these are equal.
        construction: Whether paid reconstruction was requested.
        errors: Decisions the error tags may buy, when an error bank is named; included in
            ``decision_calls``.
        naming_calls: Upper bound on cluster-naming calls, at the largest cluster count.
        review_calls: Upper bound on environment-family review calls over world learning.
    """

    identity: str
    saved: bool
    decision_model: ModelSnapshot
    roles: dict[Role, RoleCount]
    views: tuple[ViewPlan, ...]
    decision_calls: int = Field(ge=0)
    decision_attempts: int = Field(ge=0)
    embedder: ModelSnapshot
    embedding_inputs: int = Field(ge=0)
    embedding_inputs_saved: int = Field(ge=0)
    construction: bool
    errors: ViewPlan | None = None
    naming_calls: int = Field(default=0, ge=0)
    review_calls: int = Field(default=0, ge=0)


class TagMatrixRecord(ContractModel):
    """What one saved tag matrix answered, beside the matrix itself.

    Attributes:
        bank_id: Bank asked.
        bank_version: Revision of that bank.
        bank_sha256: Digest of the exact selected questions.
        view: Rendered view the questions were asked about.
        roles: Roles whose traces were tagged.
        decision_model: Identity of the decision model that answered.
        matrix_sha256: Digest of the saved matrix.
        traces: Rows in the matrix.
    """

    bank_id: str
    bank_version: int
    bank_sha256: str
    view: TagView
    roles: tuple[Role, ...]
    decision_model: ModelSnapshot
    matrix_sha256: str
    traces: int = Field(ge=0)


class TagFailure(ContractModel):
    """One trace whose tags could not be bought or read, recorded instead of redrawn.

    Attributes:
        trace_id: The trace.
        role: Its exposure role.
        view: The view that failed.
        error: The tagger's failure message.
    """

    trace_id: str
    role: Role
    view: TagView
    error: str


class ViewTagging(ContractModel):
    """Coverage of one tagged view.

    Attributes:
        plan: What the view could buy.
        matrix: The saved matrix.
        tagged: Traces with a row, by role.
        failures: Traces with no row, in unit order.
    """

    plan: ViewPlan
    matrix: ArtifactInput
    tagged: dict[Role, int]
    failures: tuple[TagFailure, ...]


class ProbeBinding(ContractModel):
    """The query tags one mining-probe task retrieves with.

    Attributes:
        task_id: The probe task.
        lineage_id: Its lineage.
        trace_id: Source trace whose task-view row supplied the tags, or None when no source
            trace was tagged; such a task retrieves by cosine similarity alone.
        fired: Number of fired task-view propositions.
    """

    task_id: str
    lineage_id: str
    trace_id: str | None
    fired: int = Field(ge=0)


class TraceAdmission(ContractModel):
    """Why one world-learning trace entered the world-learning corpora.

    Attributes:
        trace_id: The trace.
        corpora: The corpora it entered.
        cluster_id: Its cluster, when cluster admission chose it.
        reason: One sentence per corpus, joined.
    """

    trace_id: str
    corpora: tuple[Literal["construction", "retrieval"], ...]
    cluster_id: str | None = None
    reason: str


class ProbePlacement(ContractModel):
    """Where one mining-probe task was placed among the environment families.

    Attributes:
        task_id: The probe task.
        trace_id: Its source trace placed under the clustering, when one was tagged.
        cluster_id: The cluster the mixture placed it in.
        posterior: That cluster's posterior probability.
        family_id: The family it joined, or None when it stays unplaced.
        reason: Why it was placed there, or why not.
    """

    task_id: str
    trace_id: str | None = None
    cluster_id: str | None = None
    posterior: float | None = None
    family_id: str | None = None
    reason: str


class FamilyScope(ContractModel):
    """One environment family's own simulator inputs, keyed by its identity.

    Attributes:
        family_id: The family.
        status: ``compatible``, or ``admitted_uncertain`` when explicitly admitted.
        cluster_id: The cluster it came from.
        path: Its position in the review tree.
        members: Member world-learning traces.
        retrieval_scope: The members the family's index holds (tagged, in the build's scope).
        index: The family's scoped, tagged fit-only index.
        world_model: The family's world model over that index.
        tag_retrieval: Treatment binding over the family's probe tasks, when it has any.
        rules_traces: World-learning traces its one rule memory is learned from.
        tools: Tools of its reconciled contracts.
        tasks: World-learning tasks and placed probe tasks, every one sharing this simulator.
    """

    family_id: str
    status: FamilyStatus
    cluster_id: str
    path: str
    members: tuple[str, ...]
    retrieval_scope: CorpusScope
    index: ArtifactInput
    world_model: ArtifactInput
    tag_retrieval: ArtifactInput | None = None
    rules_traces: tuple[str, ...]
    tools: tuple[str, ...] = ()
    tasks: tuple[str, ...] = ()


class FamilyScoping(ContractModel):
    """Per-family scoping of a variance build.

    Attributes:
        families: The saved ``environment-families`` record.
        scopes: One scope per admitted family, in review order.
        uncertain: Uncertain groups reported and left out of training.
        admitted_uncertain: Uncertain groups the spec explicitly admitted.
        left_out: Admitted families without a scope, with why.
        placements: Every mining-probe task's placement.
    """

    families: ArtifactInput
    scopes: tuple[FamilyScope, ...] = ()
    uncertain: tuple[str, ...] = ()
    admitted_uncertain: tuple[str, ...] = ()
    left_out: dict[str, str] = Field(default_factory=dict)
    placements: tuple[ProbePlacement, ...] = ()

    def scope(self, family_id: str) -> FamilyScope:
        """Return one family's scope.

        Raises:
            KeyError: No scope has that family identity.
        """
        for row in self.scopes:
            if row.family_id == family_id:
                return row
        raise KeyError(f"no scope for environment family {family_id}")

    def references(self) -> tuple[ArtifactInput, ...]:
        """Return every artifact the scoping saved or read, for the report's inputs."""
        rows = (
            ref
            for scope in self.scopes
            for ref in (scope.index, scope.world_model, scope.tag_retrieval)
            if ref is not None
        )
        return (self.families, *rows)


class VarianceBuildReport(ContractModel):
    """The complete record of one variance build.

    Attributes:
        spec: The registration.
        identity: Run identity, naming the saved report.
        build: The completed build it was derived from.
        plan: The credential-free summary the run was consented under.
        exposure: Saved exposure plan.
        tagging: Coverage of every tagged view.
        construction_scope: Corpus the world model's knowledge may be constructed from.
        retrieval_scope: Corpus the scoped index holds.
        construction_selection: Rule-selected construction traces flagged as uncertain.
        retrieval_selection: Rule-selected retrieval traces flagged as uncertain.
        index: Scoped, tagged fit-only retrieval index.
        index_transitions: Transitions it holds.
        constructed: World model reconstructed over the construction scope, when requested.
        world_model: Scoped world model over the index; the pipeline's simulator.
        tag_retrieval: Saved treatment binding.
        control_retrieval: Saved weight-zero control binding.
        probes: Query tags of every mining-probe task, in task order.
        rules_stage_traces: World-learning traces for ``run_rules_stage``: the construction
            scope's traces.
        next_steps: How the rules stage and actor-critic runs consume these artifacts.
        errors: Saved ``error-tags`` record over world-learning and mining-probe traces.
        error_failures: Traces whose error tags were refused; recorded, never redrawn.
        error_matrix: Saved full error-bank matrix, every asked column, when the spec names
            ``error_consumed_families``.
        clustering: Saved ``tag-clusters`` record over world learning.
        cluster_names: Saved ``tag-cluster-names`` record, when naming was requested.
        admission: The cluster admission, when clustering ran.
        admissions: Why every world-learning trace in either corpus is there.
        probe_ranking: Every mining-probe task ranked by its source traces' error tags.
        selected_probes: Probe tasks selected for error mining, most error-laden first.
        environment_families: Per-family scopes, when environment families were requested.
    """

    spec: VarianceBuildSpec
    identity: str
    build: ProjectBuildArtifacts
    plan: VarianceBuildPlan
    exposure: ArtifactInput
    tagging: tuple[ViewTagging, ...]
    construction_scope: CorpusScope
    retrieval_scope: CorpusScope
    construction_selection: dict[str, tuple[str, ...]] = Field(default_factory=dict)
    retrieval_selection: dict[str, tuple[str, ...]] = Field(default_factory=dict)
    index: ArtifactInput
    index_transitions: int = Field(ge=1)
    constructed: ArtifactInput | None
    world_model: ArtifactInput
    tag_retrieval: ArtifactInput
    control_retrieval: ArtifactInput
    probes: tuple[ProbeBinding, ...]
    rules_stage_traces: tuple[str, ...]
    next_steps: tuple[str, ...]
    errors: ArtifactInput | None = None
    error_failures: tuple[TagFailure, ...] = ()
    error_matrix: ArtifactInput | None = Field(default=None, exclude_if=lambda value: value is None)
    clustering: ArtifactInput | None = None
    cluster_names: ArtifactInput | None = None
    admission: ClusterAdmission | None = None
    admissions: tuple[TraceAdmission, ...] = ()
    probe_ranking: tuple[TaskErrorRank, ...] = ()
    selected_probes: tuple[str, ...] = ()
    environment_families: FamilyScoping | None = Field(
        default=None, exclude_if=lambda value: value is None
    )


def view_plan(
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


def role_counts(plan: ExposurePlan, tasks: Sequence[TaskCase]) -> dict[Role, RoleCount]:
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


def corpus_scope(
    plan: ExposurePlan,
    corpus: str,
    selection: TagSelection | None,
    admission: ClusterAdmission | None,
) -> CorpusScope:
    """Declare one world-learning corpus from its rule, else cluster admission, else the role.

    Raises:
        ValueError: An admitted trace is outside world learning, or nothing is selected.
    """
    name = f"world-learning-{corpus}"
    if selection is not None or admission is None:
        return world_learning_scope(plan, name=name, selection=selection)
    plan.require(admission.trace_ids(), "world_learning")
    return CorpusScope.declare(
        name=name, kind="trace", ids=admission.trace_ids(), source_plan=sha256_json(plan)
    )


def trace_admissions(
    construction: tuple[CorpusScope, ScopeRule | None],
    retrieval: tuple[CorpusScope, ScopeRule | None],
    admission: ClusterAdmission | None,
) -> tuple[TraceAdmission, ...]:
    """Record why every trace of either corpus was admitted."""
    admitted = {row.trace_id: row for row in (admission.admitted if admission else ())}
    rows: dict[str, list[tuple[Corpus, str]]] = {}
    corpora: tuple[tuple[Corpus, tuple[CorpusScope, ScopeRule | None]], ...] = (
        ("construction", construction),
        ("retrieval", retrieval),
    )
    for label, (scope, rule) in corpora:
        for trace in scope.ids:
            if rule is not None:
                why = f"tag rule {rule.rule.name!r} over the {rule.view} view selected it"
            elif admission is not None:
                why = admitted[trace].reason
            else:
                why = "every world-learning trace (no rule and no clustering)"
            rows.setdefault(trace, []).append((label, f"{label}: {why}"))
    return tuple(
        TraceAdmission(
            trace_id=trace,
            corpora=tuple(label for label, _ in rows[trace]),
            cluster_id=admitted[trace].cluster_id if trace in admitted else None,
            reason="; ".join(why for _, why in rows[trace]),
        )
        for trace in sorted(rows)
    )


def probe_selection(
    plan: ExposurePlan,
    tasks: Sequence[TaskCase],
    errors: ErrorTags | None,
    limit: int | None,
) -> tuple[tuple[TaskErrorRank, ...], tuple[str, ...], tuple[TagFailure, ...]]:
    """Rank mining-probe tasks by their source traces' error tags and select the top ones.

    Args:
        plan: Exposure plan naming the probe lineages.
        tasks: Build tasks.
        errors: Error record, or None when no error bank was asked.
        limit: Most tasks selected, or None for all of them.

    Returns:
        The ranking, the selected task identities and every trace whose error tags were refused.
    """
    if errors is None:
        return (), (), ()
    probes = {
        task.task_id: task.source_trace_ids
        for task in tasks
        if plan.roles.get(task.lineage_group_id) == "mining_probes"
    }
    ranking = rank_error_tasks(errors, probes)
    failures = tuple(
        TagFailure(trace_id=trace, role=plan.role_of(trace), view="observations", error=error)
        for trace, error in errors.failed.items()
    )
    selected = ranking if limit is None else ranking[:limit]
    return ranking, tuple(row.task_id for row in selected), failures


def next_steps(
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


def load_exposure_plan(project: ProjectStore, reference: ArtifactInput) -> ExposurePlan:
    """Read a saved exposure plan, as a variance build report names it.

    Args:
        project: Project holding the plan.
        reference: The ``variance-exposure-plan`` artifact.

    Returns:
        The plan.

    Raises:
        ValueError: The reference is not the saved plan it pins.
    """
    return ExposurePlan.model_validate_json(_payload(project, reference, EXPOSURE_TYPE, "exposure"))


def load_tag_matrix(project: ProjectStore, reference: ArtifactInput) -> TagMatrix:
    """Read a saved tag matrix.

    Args:
        project: Project holding the matrix.
        reference: The ``variance-tag-matrix`` artifact.

    Returns:
        The matrix.

    Raises:
        ValueError: The reference is not the saved matrix it pins.
    """
    return TagMatrix.model_validate_json(_payload(project, reference, MATRIX_TYPE, "matrix"))


def load_tag_retrieval_binding(
    project: ProjectStore, reference: ArtifactInput
) -> TagRetrievalBinding:
    """Read a saved tag retrieval binding, to pass to ``load_grounded_world_model``.

    Args:
        project: Project holding the binding.
        reference: The ``tag-retrieval-binding`` artifact.

    Returns:
        The binding.

    Raises:
        ValueError: The reference is not the saved binding it pins.
    """
    return TagRetrievalBinding.model_validate_json(
        _payload(project, reference, BINDING_TYPE, "binding")
    )


def _payload(project: ProjectStore, reference: ArtifactInput, kind: str, name: str) -> bytes:
    """Return one pinned artifact's ``<name>.json`` after checking its pin and type.

    Args:
        project: Project holding the artifact.
        reference: The pinned artifact.
        kind: Its required artifact type.
        name: Its data file, without the extension.

    Returns:
        The file's bytes.

    Raises:
        ValueError: The artifact changed or has another type.
    """
    stored = project.artifacts.read(reference.artifact_id)
    if artifact_input(stored.manifest) != reference or stored.manifest.artifact_type != kind:
        raise ValueError(f"artifact {reference.artifact_id} is not the pinned {kind}")
    return stored.payloads[f"{name}.json"]


def save_artifact(
    store: ArtifactStore,
    kind: str,
    files: Mapping[str, BaseModel],
    inputs: tuple[ArtifactInput, ...],
    code_revision: str,
) -> ArtifactInput:
    """Save JSON files as one content-addressed artifact, or adopt the identical saved one.

    Args:
        store: Artifact store.
        kind: Artifact type, also the identity prefix.
        files: Models keyed by file name.
        inputs: Artifacts the files were derived from.
        code_revision: Producing source identity.

    Returns:
        The artifact reference.

    Raises:
        ValueError: An artifact with the same identity holds different content.
    """
    artifact_id = stable_id(kind, {name: sha256_json(model) for name, model in files.items()})
    if not store.exists(artifact_id):
        try:
            manifest = store.write_json(
                artifact_id=artifact_id,
                artifact_type=kind,
                envelope=ArtifactEnvelope(
                    schema_version=1,
                    created_at=datetime.now(UTC),
                    code_revision=code_revision,
                    inputs=tuple(sorted(set(inputs), key=lambda ref: ref.artifact_id)),
                ),
                files=dict(files),
            )
        except ArtifactAlreadyExistsError:
            pass
        else:
            return artifact_input(manifest)
    stored = store.read(artifact_id)
    same = stored.manifest.artifact_type == kind and all(
        type(model).model_validate_json(stored.payloads[name]) == model
        for name, model in files.items()
    )
    if not same:
        raise ValueError(f"saved {kind} {artifact_id} differs from the content this run derived")
    return artifact_input(stored.manifest)
