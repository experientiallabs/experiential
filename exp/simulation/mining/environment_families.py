"""Environment families: tag clusters reviewed into groups that share one simulator.

A tag cluster (:mod:`exp.simulation.mining.tag_clusters`) is a grouping hypothesis. Before its
traces share one learned rule memory and one simulator, every cluster is reviewed by an agentic
compatibility reviewer (:mod:`exp.simulation.mining.environment_review`) under several prompt
variants and a reconcile:

- **compatible**: the group becomes an :class:`EnvironmentFamily` with its member traces, the
  tool contracts parsed over every member (with the reviewers' contract notes), the shared state
  domains and world semantics, the tasks and starting states whose source traces lie in it, the
  compatibility evidence and its status;
- **split**: each proposed sub-group is re-reviewed the same way, recursively, up to
  ``max_depth`` split levels; a sub-group past the bound stays uncertain;
- **uncertain**: the group is kept as an :class:`UncertainGroup` with its open questions. It is
  never forced into a family; :func:`admitted_families` admits one only when the caller names it.

Only declared eligible traces (world learning) are reviewed or rendered. A clustering holding a
held-out trace is refused, and a held-out task is never attached. The record is saved as a
content-addressed ``environment-families`` artifact whose identity binds the clustering, the
reviewer model, the registration, the review protocol and the exact traces and tasks, so
:func:`environment_families` replays it with no call.
"""

from __future__ import annotations

from collections.abc import Collection, Mapping, Sequence
from typing import Literal

from pydantic import Field

from exp.common.core.artifacts import (
    ArtifactInput,
    ContractModel,
    JsonObject,
    sha256_json,
    stable_id,
)
from exp.common.project import ArtifactStore, artifact_input
from exp.common.tasks import TaskCase
from exp.common.traces import Trace
from exp.simulation.mining.environment_candidates import (
    candidate_protocol_sha256,
    candidate_root_bound,
    environment_candidates,
)
from exp.simulation.mining.environment_review import (
    ENVIRONMENT_REVIEW_PROTOCOL,
    CompatibilityReviewer,
    CompatibilitySpec,
    ContractNote,
    GroupReview,
    OpenQuestion,
    StateDomain,
    group_contracts,
    reconcile,
    review_protocol_sha256,
    review_variant,
)
from exp.simulation.mining.tag_clusters import TagClustering, read_record, write_record
from exp.simulation.rules.model import ToolContract

FAMILIES_TYPE = "environment-families"
"""Artifact type of a saved :class:`EnvironmentFamilies` record (``families.json``)."""

FamilyStatus = Literal["compatible", "admitted_uncertain"]
"""How a family was formed: reviewed compatible, or an uncertain group the caller admitted."""

TaskRole = Literal["world_learning", "mining_probes"]
"""The data role a family task belongs to."""


class EnvironmentFamilySpec(ContractModel):
    """How a tag clustering becomes environment families.

    Attributes:
        probe_placement: Explicit placement policy; absent retains historical unique
            tool-name placement.
            all_exact_schemas attaches a task to every schema-covered base without start QA.
        review: The compatibility review registration.
        admit_uncertain: Uncertain group identities explicitly admitted as families; every other
            uncertain group is reported and left out of training.
    """

    probe_placement: Literal["unique", "all_exact_schemas"] | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    review: CompatibilitySpec = Field(default_factory=CompatibilitySpec)
    admit_uncertain: tuple[str, ...] = ()

    def calls(self, traces: int, *, roots: int | None = None) -> int:
        """Return an upper bound of the review calls over ``traces`` eligible traces."""
        return self.review.calls(traces, roots=roots)

    def candidate_calls(self, traces: Sequence[Trace], *, clusters: int) -> int:
        """Quote complete evidence using its registered partition and cluster root bound."""
        roots = candidate_root_bound(traces, self.review, clusters=clusters)
        return self.calls(len(traces), roots=roots)


class StartingState(ContractModel):
    """One starting state of a family task.

    Attributes:
        task_id: The task.
        initial_context_sha256: Digest of the task's initial context, its recorded start.
        checkpoint: Saved continuation checkpoint of the start, once one is captured.
    """

    task_id: str
    initial_context_sha256: str
    checkpoint: ArtifactInput | None = None


class FamilyTask(ContractModel):
    """One task inside a family.

    Attributes:
        task_id: The task.
        lineage_id: Its leakage lineage.
        role: Its data role.
        source_trace_ids: Its source traces.
        tools: Tools the task exposes.
        start: Its starting state.
    """

    task_id: str
    lineage_id: str
    role: TaskRole
    source_trace_ids: tuple[str, ...]
    tools: tuple[str, ...] = ()
    start: StartingState


class CompatibilityEvidence(ContractModel):
    """Why a group's verdict was reached.

    Attributes:
        reviews: Every variant review, usable or not.
        reconcile: The reconcile review, when one ran.
        rationale: The verdict's rationale, or why there is none.
        locators: Locators the verdict cites.
    """

    reviews: tuple[GroupReview, ...] = ()
    reconcile: GroupReview | None = None
    rationale: str
    locators: tuple[str, ...] = ()


class EnvironmentFamily(ContractModel):
    """One environment family: traces that share one rule memory and one simulator.

    Attributes:
        family_id: Content identity of the family (clustering and members).
        cluster_id: The cluster it came from.
        path: Position in the review tree, such as ``c00`` or ``c00/1``.
        status: How it was formed.
        members: Member world-learning traces.
        tool_contracts: Contracts parsed over every member, reconciled by tool name.
        contract_notes: The reviewers' reconciliations and conflicts per tool.
        state_domains: Shared state domains.
        world_semantics: What the same actions do in this environment.
        tasks: Tasks inside the family, each with its starting state.
        evidence: Compatibility evidence.
    """

    family_id: str
    cluster_id: str
    path: str
    status: FamilyStatus
    members: tuple[str, ...] = Field(min_length=1)
    tool_contracts: dict[str, ToolContract] = Field(default_factory=dict)
    contract_notes: tuple[ContractNote, ...] = ()
    state_domains: tuple[StateDomain, ...] = ()
    world_semantics: str = ""
    tasks: tuple[FamilyTask, ...] = ()
    evidence: CompatibilityEvidence

    def task_ids(self) -> tuple[str, ...]:
        """Return the family's task identities in order."""
        return tuple(task.task_id for task in self.tasks)


class UncertainGroup(ContractModel):
    """A group whose compatibility the evidence does not settle; never forced into a family.

    Attributes:
        group_id: Content identity of the group.
        cluster_id: The cluster it came from.
        path: Position in the review tree.
        members: Its traces.
        reason: Why it is uncertain.
        open_questions: Questions that would settle it.
        evidence: Compatibility evidence.
        tasks: Tasks placed in it, kept for when it is explicitly admitted.
    """

    group_id: str
    cluster_id: str
    path: str
    members: tuple[str, ...] = Field(min_length=1)
    reason: str
    open_questions: tuple[OpenQuestion, ...] = ()
    evidence: CompatibilityEvidence
    tasks: tuple[FamilyTask, ...] = ()


class EnvironmentFamilies(ContractModel):
    """The ``environment-families`` record of one clustering.

    Attributes:
        protocol: Review protocol.
        clustering: Identity of the reviewed clustering.
        model: Identity of the reviewer model.
        spec: The review registration.
        evidence_sha256: Digest of the traces and tasks reviewed.
        families: Compatible families, in review order.
        uncertain: Uncertain groups, in review order.
        skipped: Clusters with no eligible member, with why.
    """

    protocol: str = ENVIRONMENT_REVIEW_PROTOCOL
    clustering: str
    model: JsonObject
    spec: CompatibilitySpec
    evidence_sha256: str
    families: tuple[EnvironmentFamily, ...] = ()
    uncertain: tuple[UncertainGroup, ...] = ()
    skipped: dict[str, str] = Field(default_factory=dict)

    def identity(self) -> str:
        """Return the content-addressed identity of this record."""
        return stable_id(FAMILIES_TYPE, {"protocol": self.protocol, "record": sha256_json(self)})


def family_identity(clustering: str, members: Collection[str]) -> str:
    """Return the identity of a family or group of one clustering."""
    return stable_id("environment-family", {"clustering": clustering, "members": sorted(members)})


def family_task(task: TaskCase, role: TaskRole) -> FamilyTask:
    """Describe one task and its recorded starting state for a family.

    Raises:
        ValueError: The task is held out.
    """
    if task.partition == "held_out":
        raise ValueError(f"task {task.task_id} is held out and never enters a family")
    return FamilyTask(
        task_id=task.task_id,
        lineage_id=task.lineage_group_id,
        role=role,
        source_trace_ids=task.source_trace_ids,
        tools=tuple(sorted(tool.name for tool in task.tools)),
        start=StartingState(
            task_id=task.task_id, initial_context_sha256=sha256_json(task.initial_context)
        ),
    )


def families_identity(
    clustering: TagClustering,
    reviewer: CompatibilityReviewer,
    spec: CompatibilitySpec,
    traces: Mapping[str, Trace],
    tasks: Sequence[TaskCase],
) -> str:
    """Return the identity families are saved under, computed before any call."""
    return family_review_identity(
        clustering=clustering.identity(),
        model=reviewer.model.model_dump(mode="json"),
        configuration_sha256=reviewer.configuration_sha256,
        spec=spec,
        evidence_sha256=_evidence_digest(traces, tasks),
        protocol_sha256=review_protocol_sha256(),
    )


def family_review_identity(
    *,
    clustering: str,
    model: JsonObject,
    configuration_sha256: str | None,
    spec: CompatibilitySpec,
    evidence_sha256: str,
    protocol_sha256: str,
) -> str:
    """Bind a review's immutable input identity without loading or reviewing its evidence."""
    return stable_id(
        FAMILIES_TYPE,
        {
            "protocol": protocol_sha256,
            **(
                {}
                if spec.maximum_candidate_members is None
                else {"candidate_protocol": candidate_protocol_sha256()}
            ),
            "clustering": clustering,
            "model": model,
            "configuration": configuration_sha256,
            "spec": sha256_json(spec),
            "evidence": evidence_sha256,
        },
    )


def review_environment_families(
    clustering: TagClustering,
    reviewer: CompatibilityReviewer,
    *,
    traces: Mapping[str, Trace],
    eligible: Collection[str],
    held_out: Collection[str],
    tasks: Sequence[TaskCase] = (),
    spec: CompatibilitySpec | None = None,
) -> EnvironmentFamilies:
    """Review every cluster into compatible families and uncertain groups.

    Args:
        clustering: The clustering whose clusters are reviewed.
        reviewer: Injected reviewer.
        traces: Trace content by identity; only eligible members are read.
        eligible: Traces that may be reviewed (world learning).
        held_out: Final-evaluation traces; a clustering holding one is refused.
        tasks: World-learning tasks to attach; a task joins the family holding every one of its
            eligible source traces.
        spec: Review registration; defaults apply when omitted.

    Raises:
        ValueError: The clustering holds a held-out trace, a task is held out, or an eligible
            member's trace content is missing.
    """
    spec = spec or CompatibilitySpec()
    leaked = sorted(set(clustering.trace_ids) & set(held_out))
    if leaked:
        raise ValueError(f"clustering holds held-out traces {leaked[:5]}; refit without them")
    allowed = set(eligible) - set(held_out)
    families: list[EnvironmentFamily] = []
    uncertain: list[UncertainGroup] = []
    skipped: dict[str, str] = {}
    ident = clustering.identity()
    shown = sorted(allowed & set(clustering.trace_ids))
    missing = [trace for trace in shown if trace not in traces]
    if missing:
        raise ValueError(f"trace content missing for eligible members {missing[:5]}")
    candidates = environment_candidates([traces[trace] for trace in shown], spec)
    for cluster in clustering.clusters:
        members = tuple(sorted(t for t in cluster.members if t in allowed))
        if not members:
            skipped[cluster.cluster_id] = "no eligible member"
            continue
        missing = [t for t in members if t not in traces]
        if missing:
            raise ValueError(f"trace content missing for eligible members {missing[:5]}")
        for candidate in candidates:
            partition = tuple(sorted(set(members) & set(candidate.members)))
            if not partition:
                continue
            path = cluster.cluster_id
            scope = ident
            if spec.maximum_candidate_members is not None:
                path += "/" + candidate.candidate_id
                scope = stable_id(
                    "environment-candidate-review",
                    {"clustering": ident, "candidate": candidate.model_dump(mode="json")},
                )
            _review_group(
                reviewer,
                spec,
                traces,
                scope,
                cluster.cluster_id,
                path,
                partition,
                0,
                families,
                uncertain,
            )
    attached = _attach(families, tasks, allowed)
    return EnvironmentFamilies(
        clustering=ident,
        model=reviewer.model.model_dump(mode="json"),
        spec=spec,
        evidence_sha256=_evidence_digest(traces, tasks, allowed),
        families=attached,
        uncertain=tuple(uncertain),
        skipped=skipped,
    )


def environment_families(
    store: ArtifactStore,
    clustering: TagClustering,
    reviewer: CompatibilityReviewer,
    *,
    traces: Mapping[str, Trace],
    eligible: Collection[str],
    held_out: Collection[str],
    tasks: Sequence[TaskCase] = (),
    spec: CompatibilitySpec | None = None,
    code_revision: str,
    inputs: Sequence[ArtifactInput] = (),
) -> tuple[EnvironmentFamilies, ArtifactInput]:
    """Replay the saved families of exactly this clustering, reviewer and evidence, or review.

    Returns:
        The record and its saved artifact.
    """
    spec = spec or CompatibilitySpec()
    allowed = set(eligible) - set(held_out)
    shown = {t: traces[t] for t in sorted(allowed & set(clustering.trace_ids)) if t in traces}
    identity = families_identity(clustering, reviewer, spec, shown, tasks)
    if store.exists(identity):
        reference = artifact_input(store.read(identity).manifest)
        return load_environment_families(store, reference), reference
    record = review_environment_families(
        clustering,
        reviewer,
        traces=shown,
        eligible=allowed,
        held_out=held_out,
        tasks=tasks,
        spec=spec,
    )
    return record, write_record(
        store, identity, FAMILIES_TYPE, "families.json", record, code_revision, inputs
    )


def persist_environment_families(
    store: ArtifactStore,
    record: EnvironmentFamilies,
    *,
    code_revision: str,
    inputs: Sequence[ArtifactInput] = (),
) -> ArtifactInput:
    """Save a record under its content identity (for example with placed tasks), or adopt it.

    Raises:
        ValueError: A saved artifact under that identity holds a different record.
    """
    identity = record.identity()
    if store.exists(identity):
        reference = artifact_input(store.read(identity).manifest)
        if load_environment_families(store, reference) != record:
            raise ValueError(f"saved environment families {identity} hold a different record")
        return reference
    return write_record(
        store, identity, FAMILIES_TYPE, "families.json", record, code_revision, inputs
    )


def with_placed(
    record: EnvironmentFamilies, placed: Mapping[str, Sequence[FamilyTask]]
) -> EnvironmentFamilies:
    """Return the record with tasks placed in its families and uncertain groups, by identity.

    Raises:
        ValueError: A placement names no family or group, or repeats a task.
    """
    known = {f.family_id for f in record.families} | {g.group_id for g in record.uncertain}
    unknown = sorted(set(placed) - known)
    if unknown:
        raise ValueError(f"tasks placed in unknown families or groups {unknown}")
    families = tuple(with_tasks(f, placed.get(f.family_id, ())) for f in record.families)
    groups = tuple(
        g.model_copy(
            update={
                "tasks": tuple(
                    sorted((*g.tasks, *placed.get(g.group_id, ())), key=lambda t: t.task_id)
                )
            }
        )
        for g in record.uncertain
    )
    return record.model_copy(update={"families": families, "uncertain": groups})


def load_environment_families(
    store: ArtifactStore, reference: ArtifactInput
) -> EnvironmentFamilies:
    """Read saved families, refusing a changed or mistyped artifact."""
    payload = read_record(store, reference, FAMILIES_TYPE, "families.json")
    return EnvironmentFamilies.model_validate_json(payload)


def admitted_families(
    record: EnvironmentFamilies,
    admit_uncertain: Collection[str] = (),
    *,
    traces: Mapping[str, Trace] | None = None,
) -> tuple[EnvironmentFamily, ...]:
    """Return the compatible families plus the uncertain groups explicitly admitted.

    An admitted uncertain group becomes a family with status ``admitted_uncertain``, keeping its
    open questions as its rationale; its contracts are parsed when trace content is given.

    Raises:
        ValueError: An admitted identity names no uncertain group.
    """
    groups = {group.group_id: group for group in record.uncertain}
    unknown = sorted(set(admit_uncertain) - set(groups))
    if unknown:
        raise ValueError(f"admit_uncertain names no uncertain group: {unknown}")
    admitted = [
        EnvironmentFamily(
            family_id=group.group_id,
            cluster_id=group.cluster_id,
            path=group.path,
            status="admitted_uncertain",
            members=group.members,
            tool_contracts={}
            if traces is None
            else group_contracts([traces[t] for t in group.members if t in traces]),
            tasks=group.tasks,
            evidence=group.evidence,
        )
        for group_id, group in groups.items()
        if group_id in set(admit_uncertain)
    ]
    return (*record.families, *admitted)


def with_tasks(family: EnvironmentFamily, tasks: Sequence[FamilyTask]) -> EnvironmentFamily:
    """Return the family with more tasks, keeping task identities distinct and ordered.

    Raises:
        ValueError: A task is already in the family.
    """
    known = set(family.task_ids())
    repeated = sorted(task.task_id for task in tasks if task.task_id in known)
    if repeated:
        raise ValueError(f"tasks {repeated} are already in family {family.family_id}")
    ordered = sorted((*family.tasks, *tasks), key=lambda task: (task.role, task.task_id))
    return family.model_copy(update={"tasks": tuple(ordered)})


def _review_group(
    reviewer: CompatibilityReviewer,
    spec: CompatibilitySpec,
    traces: Mapping[str, Trace],
    clustering: str,
    cluster_id: str,
    path: str,
    members: tuple[str, ...],
    depth: int,
    families: list[EnvironmentFamily],
    uncertain: list[UncertainGroup],
) -> None:
    """Review one group and record it as a family, an uncertain group, or reviewed sub-groups."""
    group_traces = [traces[t] for t in members]
    scope = family_identity(clustering, members)
    reviews = tuple(
        review_variant(reviewer, group_traces, variant=variant, spec=spec, scope=scope)
        for variant in spec.variants
    )
    usable = [row for row in reviews if row.answer is not None]

    def keep_uncertain(
        reason: str, evidence: CompatibilityEvidence, questions: Sequence[OpenQuestion] = ()
    ) -> None:
        """Record the group as uncertain."""
        uncertain.append(
            UncertainGroup(
                group_id=scope,
                cluster_id=cluster_id,
                path=path,
                members=members,
                reason=reason,
                open_questions=tuple(questions),
                evidence=evidence,
            )
        )

    if len(usable) < spec.min_reviews:
        reason = f"{len(usable)} usable reviews, fewer than the {spec.min_reviews} required"
        keep_uncertain(reason, CompatibilityEvidence(reviews=reviews, rationale=reason))
        return
    final = reconcile(reviewer, group_traces, reviews, spec=spec, scope=scope)
    answer = final.answer
    if answer is None:
        reason = f"the reconcile failed: {final.failure}"
        evidence = CompatibilityEvidence(reviews=reviews, reconcile=final, rationale=reason)
        keep_uncertain(reason, evidence)
        return
    evidence = CompatibilityEvidence(
        reviews=reviews,
        reconcile=final,
        rationale=answer.rationale,
        locators=tuple(sorted(answer.locators())),
    )
    if answer.verdict == "uncertain":
        keep_uncertain(answer.rationale, evidence, answer.open_questions)
        return
    if answer.verdict == "compatible":
        families.append(
            EnvironmentFamily(
                family_id=scope,
                cluster_id=cluster_id,
                path=path,
                status="compatible",
                members=members,
                tool_contracts=group_contracts(group_traces),
                contract_notes=answer.contract_notes,
                state_domains=answer.state_domains,
                world_semantics=answer.world_semantics,
                evidence=evidence,
            )
        )
        return
    for position, subgroup in enumerate(answer.subgroups):
        child = tuple(sorted(subgroup.members))
        child_path = f"{path}/{position}"
        if depth + 1 > spec.max_depth:
            question = OpenQuestion(
                question="Is this proposed sub-group one environment? It was not re-reviewed.",
                evidence=subgroup.evidence,
            )
            uncertain.append(
                UncertainGroup(
                    group_id=family_identity(clustering, child),
                    cluster_id=cluster_id,
                    path=child_path,
                    members=child,
                    reason=f"split depth bound {spec.max_depth} reached: {subgroup.rationale}",
                    open_questions=(question,),
                    evidence=evidence,
                )
            )
            continue
        _review_group(
            reviewer,
            spec,
            traces,
            clustering,
            cluster_id,
            child_path,
            child,
            depth + 1,
            families,
            uncertain,
        )


def _attach(
    families: Sequence[EnvironmentFamily], tasks: Sequence[TaskCase], allowed: Collection[str]
) -> tuple[EnvironmentFamily, ...]:
    """Attach each world-learning task to the family holding all its eligible source traces."""
    rows: dict[str, list[FamilyTask]] = {family.family_id: [] for family in families}
    home = {trace: family.family_id for family in families for trace in family.members}
    for task in tasks:
        sources = [t for t in task.source_trace_ids if t in allowed]
        owners = {home.get(t) for t in sources}
        if task.partition == "held_out":
            raise ValueError(f"task {task.task_id} is held out and never enters a family")
        if sources and len(owners) == 1 and None not in owners:
            rows[str(next(iter(owners)))].append(family_task(task, "world_learning"))
    return tuple(with_tasks(family, rows[family.family_id]) for family in families)


def _evidence_digest(
    traces: Mapping[str, Trace], tasks: Sequence[TaskCase], allowed: Collection[str] | None = None
) -> str:
    """Digest exactly the trace content and tasks the review may read."""
    shown = sorted(t for t in traces if allowed is None or t in allowed)
    return sha256_json(
        {
            "traces": {t: sha256_json(traces[t]) for t in shown},
            "tasks": sorted(sha256_json(task) for task in tasks),
        }
    )
