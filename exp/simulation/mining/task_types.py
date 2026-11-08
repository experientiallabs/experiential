"""Task types: discrete sub-clusters of work inside each environment family.

An environment family (:mod:`~exp.simulation.mining.environment_families`) groups traces that
share one simulator; it says nothing about what kinds of task those traces ask for. A family of
spreadsheet episodes may hold reconciliations, chart builds and formula repairs, and each wants
its own task set and, often, its own environment instance. This module finds those task types.

Inside each family the member traces are clustered again on their task-view tags plus any
authored tags (:mod:`~exp.simulation.mining.tag_bank_selection`) and outcome features the caller
joins, with the same Bernoulli-mixture fit and data-chosen count as the first level
(:func:`~exp.simulation.mining.tag_clusters.cluster_tags`): a family whose tags show no structure
stays one task type. A family smaller than ``min_members_to_split`` is never split. Each sub-cluster
is named by a language model in the same agentic draft-and-reconcile loop as the first level
(:func:`~exp.simulation.mining.tag_cluster_names.name_clusters`), under task-type stances
(:data:`DEFAULT_TASK_TYPE_PROMPTS`).

A family task joins the task type holding most of its source traces; a source trace outside the
fit is placed under the fitted mixture without refitting
(:func:`~exp.simulation.mining.tag_clusters.assign_clusters`), and a task none of whose source
traces can be placed is listed as unplaced. :func:`task_type_family` turns one task type into an
:class:`~exp.simulation.mining.environment_families.EnvironmentFamily` narrowed to its members and
tasks, so it can feed its own environment instance and task set through the unchanged family
stages.

Difficulty is a dimension left to fill: :class:`TaskTypeDifficulty` is empty until judged pass
rates per actor tier exist, and :func:`with_difficulty` fills it from per-task
:class:`TaskDifficulty` values, which the curriculum owner computes.

The record is saved as a content-addressed ``task-types`` artifact; :func:`task_types` replays it
with no call, and each family's clustering and names are saved as their own artifacts.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Collection, Mapping, Sequence
from typing import Literal

from pydantic import Field

from exp.common.core.artifacts import (
    ArtifactInput,
    ContractModel,
    JsonObject,
    Sha256,
    sha256_json,
    stable_id,
)
from exp.common.project import ArtifactStore, artifact_input
from exp.simulation.mining.environment_families import (
    CompatibilityEvidence,
    EnvironmentFamily,
    FamilyTask,
    family_identity,
)
from exp.simulation.mining.tag_cluster_names import (
    NAMES_TYPE,
    ClusterNamer,
    ClusterNames,
    ClusterNamingSpec,
    NamingPrompt,
    name_clusters,
    names_identity,
)
from exp.simulation.mining.tag_clusters import (
    ClusteringSpec,
    TagClustering,
    assign_clusters,
    cluster_tags,
    persist_tag_clustering,
    read_record,
    write_record,
)
from exp.simulation.mining.tagging import TagMatrix

TASK_TYPES_PROTOCOL = "task-types-v1"
"""Version of the second-level fit, task placement and naming stances; bump when any changes."""

TASK_TYPES_TYPE = "task-types"
"""Artifact type of a saved :class:`TaskTypes` record (``task_types.json``)."""

DEFAULT_TASK_TYPE_PROMPTS: tuple[NamingPrompt, ...] = (
    NamingPrompt(
        variant_id="task_type",
        stance=(
            "Name the discrete kind of task these episodes ask for: the request, the inputs it "
            "provides and the deliverable it expects. Say nothing about how well it went."
        ),
    ),
    NamingPrompt(
        variant_id="environment_use",
        stance=(
            "Name what these tasks require of the shared environment: which state they read "
            "and which they change."
        ),
    ),
    NamingPrompt(
        variant_id="contrast",
        stance=(
            "Name what separates this task type from the nearest other task types inside the "
            "same environment."
        ),
    ),
)
"""Task-kind, environment-use and contrast stances drafted for every task type."""

DifficultyBasis = Literal["judged_pass_rates"]
"""Where a difficulty value comes from."""


class TaskTypeSpec(ContractModel):
    """How families are split into task types.

    Attributes:
        clustering: The second-level fit and count selection.
        naming: The naming registration, with task-type stances by default.
        min_members_to_split: Fewest members a family needs before it is split at all.
    """

    clustering: ClusteringSpec = Field(
        default_factory=lambda: ClusteringSpec(max_clusters=8, exemplars=3)
    )
    naming: ClusterNamingSpec = Field(
        default_factory=lambda: ClusterNamingSpec(prompts=DEFAULT_TASK_TYPE_PROMPTS)
    )
    min_members_to_split: int = Field(default=6, ge=2, le=10_000)


class TaskDifficulty(ContractModel):
    """One task's difficulty tier from judged attempts of an ordered ladder of actors.

    Levels run from 0 (the weakest actor passes) to ``actors`` (no actor passes).

    Attributes:
        level: Point estimate of the tier.
        label: Readable tier, such as ``weak models pass`` or ``no model passes``.
        low: Easiest tier the attempts plausibly support.
        high: Hardest tier the attempts plausibly support.
        actors: Actors on the ladder.
        score: Mean judged failure share over the rated actors, in [0, 1], for ordering inside
            a tier.
        confident: Whether ``low`` equals ``high``.
        basis: Where the value comes from.
    """

    level: int = Field(ge=0)
    label: str = Field(min_length=1, max_length=200)
    low: int = Field(ge=0)
    high: int = Field(ge=0)
    actors: int = Field(ge=1)
    score: float = Field(ge=0.0, le=1.0)
    confident: bool
    basis: DifficultyBasis = "judged_pass_rates"


class TaskTypeDifficulty(ContractModel):
    """A task type's difficulty, summarized from its tasks.

    Attributes:
        tasks: Each rated task's difficulty.
        level: Lower median of the rated tasks' levels, None while no task is rated.
        label: The label of a task at that level.
        lowest: Easiest rated level.
        highest: Hardest rated level.
        unrated: Tasks with no difficulty yet.
    """

    tasks: dict[str, TaskDifficulty] = Field(default_factory=dict)
    level: int | None = None
    label: str | None = None
    lowest: int | None = None
    highest: int | None = None
    unrated: tuple[str, ...] = ()


class TaskType(ContractModel):
    """One discrete task type inside one environment family.

    Attributes:
        task_type_id: Content identity (family and members).
        family_id: The family it splits.
        cluster_id: Its cluster in the family's second-level fit.
        name: Its language-model name, when naming succeeded.
        description: Its description, when naming succeeded.
        members: Member traces, most typical first.
        exemplars: The most typical members.
        uncertain_members: Members whose assignment probability is low.
        fired: Columns that hold in it.
        distinctive: Columns most over-represented in it relative to its family.
        tasks: Family tasks placed in it.
        difficulty: Its difficulty, filled once judged pass rates exist.
    """

    task_type_id: str
    family_id: str
    cluster_id: str
    name: str | None = None
    description: str | None = None
    members: tuple[str, ...] = Field(min_length=1)
    exemplars: tuple[str, ...] = ()
    uncertain_members: tuple[str, ...] = ()
    fired: tuple[str, ...] = ()
    distinctive: tuple[str, ...] = ()
    tasks: tuple[FamilyTask, ...] = ()
    difficulty: TaskTypeDifficulty = Field(default_factory=TaskTypeDifficulty)

    def task_ids(self) -> tuple[str, ...]:
        """Return the placed task identities in order."""
        return tuple(task.task_id for task in self.tasks)


class FamilyTaskTypes(ContractModel):
    """One family's task types and how they were found.

    Attributes:
        family_id: The family.
        clustering: Identity of its second-level clustering.
        names: Identity of the saved names given to it.
        reason: Why this count of task types was chosen.
        task_types: Its task types, largest first.
        unplaced: Family tasks none of whose source traces could be placed.
    """

    family_id: str
    clustering: str
    names: str
    reason: str
    task_types: tuple[TaskType, ...]
    unplaced: tuple[str, ...] = ()


class TaskTypes(ContractModel):
    """The ``task-types`` record over a set of environment families.

    Attributes:
        protocol: Task-types protocol.
        families: Identity of the environment-families record split.
        model: Identity of the naming model.
        spec: The registration.
        evidence_sha256: Digest of the families, matrices, question and exemplar texts used.
        by_family: Each family's task types, in family order.
    """

    protocol: str = TASK_TYPES_PROTOCOL
    families: str
    model: JsonObject
    spec: TaskTypeSpec
    evidence_sha256: Sha256
    by_family: tuple[FamilyTaskTypes, ...]

    def identity(self) -> str:
        """Return the content-addressed identity of this record."""
        return stable_id(TASK_TYPES_TYPE, {"protocol": self.protocol, "record": sha256_json(self)})

    def all(self) -> tuple[TaskType, ...]:
        """Return every task type, family by family."""
        return tuple(task_type for row in self.by_family for task_type in row.task_types)

    def task_type_of(self, task_id: str) -> TaskType | None:
        """Return the task type a task was placed in, or None when it was not placed."""
        return next((t for t in self.all() if task_id in t.task_ids()), None)


def family_clustering_spec(family: EnvironmentFamily, spec: TaskTypeSpec) -> ClusteringSpec:
    """Return the fit a family gets: one type below the split floor, else the registration."""
    if len(family.members) < spec.min_members_to_split:
        return spec.clustering.model_copy(update={"min_clusters": 1, "max_clusters": 1})
    return spec.clustering


def family_matrices(members: Collection[str], matrices: Sequence[TagMatrix]) -> list[TagMatrix]:
    """Return the matrices that tag some member: the shared banks and the family's own banks."""
    wanted = set(members)
    return [m for m in matrices if any(row.trace_id in wanted for row in m.rows)]


def split_family(
    family: EnvironmentFamily,
    matrices: Sequence[TagMatrix],
    namer: ClusterNamer,
    *,
    questions: Mapping[str, str],
    exemplar_texts: Mapping[str, str],
    spec: TaskTypeSpec | None = None,
) -> tuple[FamilyTaskTypes, TagClustering, ClusterNames]:
    """Cluster one family's members into task types, name them and place the family's tasks.

    Args:
        family: The family to split.
        matrices: Task-view, authored and outcome matrices with distinct columns; only the
            family's members are clustered.
        namer: Injected naming model.
        questions: Column or question identity mapped to its proposition text.
        exemplar_texts: Rendered text of member traces, by trace, shown to the namer.
        spec: Registration; defaults apply when omitted.

    Matrices that tag none of the family's members (another family's authored bank) are left
    out of its space, so each family is split over the shared banks and its own authored bank.

    Raises:
        ValueError: No member is tagged in every matrix the family is tagged in.
    """
    spec = spec or TaskTypeSpec()
    own = family_matrices(family.members, matrices)
    clustering = cluster_tags(own, family_clustering_spec(family, spec), trace_ids=family.members)
    names = name_clusters(
        clustering, namer, questions=questions, exemplar_texts=exemplar_texts, spec=spec.naming
    )
    saved = names_identity(clustering, namer, spec.naming, questions, exemplar_texts)
    return _family_types(family, clustering, names, saved, matrices), clustering, names


def discover_task_types(
    families: Sequence[EnvironmentFamily],
    matrices: Sequence[TagMatrix],
    namer: ClusterNamer,
    *,
    families_record: str,
    questions: Mapping[str, str],
    exemplar_texts: Mapping[str, str],
    spec: TaskTypeSpec | None = None,
) -> TaskTypes:
    """Split every family into task types.

    Args:
        families: Admitted families, in order.
        matrices: Matrices joined into each family's tag space.
        namer: Injected naming model.
        families_record: Identity of the environment-families record they came from.
        questions: Column or question identity mapped to its proposition text.
        exemplar_texts: Rendered member text by trace.
        spec: Registration; defaults apply when omitted.
    """
    spec = spec or TaskTypeSpec()
    rows = tuple(
        split_family(
            family,
            matrices,
            namer,
            questions=questions,
            exemplar_texts=exemplar_texts,
            spec=spec,
        )[0]
        for family in families
    )
    return TaskTypes(
        families=families_record,
        model=namer.model.model_dump(mode="json"),
        spec=spec,
        evidence_sha256=_evidence_digest(families, matrices, questions, exemplar_texts),
        by_family=rows,
    )


def task_types_identity(
    families: Sequence[EnvironmentFamily],
    matrices: Sequence[TagMatrix],
    namer: ClusterNamer,
    *,
    families_record: str,
    questions: Mapping[str, str],
    exemplar_texts: Mapping[str, str],
    spec: TaskTypeSpec,
) -> str:
    """Return the identity task types are saved under, computed before any call."""
    return stable_id(
        TASK_TYPES_TYPE,
        {
            "protocol": TASK_TYPES_PROTOCOL,
            "families": families_record,
            "model": namer.model.model_dump(mode="json"),
            "spec": sha256_json(spec),
            "evidence": _evidence_digest(families, matrices, questions, exemplar_texts),
        },
    )


def task_types(
    store: ArtifactStore,
    families: Sequence[EnvironmentFamily],
    matrices: Sequence[TagMatrix],
    namer: ClusterNamer,
    *,
    families_record: str,
    questions: Mapping[str, str],
    exemplar_texts: Mapping[str, str],
    spec: TaskTypeSpec | None = None,
    code_revision: str,
    inputs: Sequence[ArtifactInput] = (),
) -> tuple[TaskTypes, ArtifactInput]:
    """Replay saved task types for exactly these families, matrices and namer, or find them.

    Each family's clustering is saved as a ``tag-clusters`` artifact and its names as a
    ``tag-cluster-names`` artifact, both cited by the record.

    Returns:
        The record and its saved artifact.
    """
    spec = spec or TaskTypeSpec()
    identity = task_types_identity(
        families,
        matrices,
        namer,
        families_record=families_record,
        questions=questions,
        exemplar_texts=exemplar_texts,
        spec=spec,
    )
    if store.exists(identity):
        reference = artifact_input(store.read(identity).manifest)
        return load_task_types(store, reference), reference
    rows: list[FamilyTaskTypes] = []
    cited: list[ArtifactInput] = list(inputs)
    for family in families:
        row, clustering, names = split_family(
            family,
            matrices,
            namer,
            questions=questions,
            exemplar_texts=exemplar_texts,
            spec=spec,
        )
        rows.append(row)
        cited.append(persist_tag_clustering(store, clustering, code_revision=code_revision))
        if not store.exists(row.names):
            write_record(store, row.names, NAMES_TYPE, "names.json", names, code_revision, ())
        cited.append(artifact_input(store.read(row.names).manifest))
    record = TaskTypes(
        families=families_record,
        model=namer.model.model_dump(mode="json"),
        spec=spec,
        evidence_sha256=_evidence_digest(families, matrices, questions, exemplar_texts),
        by_family=tuple(rows),
    )
    return record, write_record(
        store, identity, TASK_TYPES_TYPE, "task_types.json", record, code_revision, cited
    )


def load_task_types(store: ArtifactStore, reference: ArtifactInput) -> TaskTypes:
    """Read saved task types, refusing a changed or mistyped artifact."""
    payload = read_record(store, reference, TASK_TYPES_TYPE, "task_types.json")
    return TaskTypes.model_validate_json(payload)


def with_difficulty(record: TaskTypes, difficulty: Mapping[str, TaskDifficulty]) -> TaskTypes:
    """Return the record with each task type's difficulty filled from per-task difficulty.

    Args:
        record: Task types.
        difficulty: Difficulty by task identity; tasks it omits stay unrated.
    """
    rows = []
    for row in record.by_family:
        filled = tuple(
            t.model_copy(update={"difficulty": summarize_difficulty(t.task_ids(), difficulty)})
            for t in row.task_types
        )
        rows.append(row.model_copy(update={"task_types": filled}))
    return record.model_copy(update={"by_family": tuple(rows)})


def summarize_difficulty(
    task_ids: Sequence[str], difficulty: Mapping[str, TaskDifficulty]
) -> TaskTypeDifficulty:
    """Summarize the difficulty of a task type's tasks by the lower median of their levels."""
    rated = {task: difficulty[task] for task in task_ids if task in difficulty}
    unrated = tuple(task for task in task_ids if task not in difficulty)
    if not rated:
        return TaskTypeDifficulty(unrated=unrated)
    ordered = sorted(rated.values(), key=lambda value: (value.level, value.score))
    middle = ordered[(len(ordered) - 1) // 2]
    return TaskTypeDifficulty(
        tasks=rated,
        level=middle.level,
        label=middle.label,
        lowest=ordered[0].level,
        highest=ordered[-1].level,
        unrated=unrated,
    )


def task_type_family(family: EnvironmentFamily, task_type: TaskType) -> EnvironmentFamily:
    """Return a family narrowed to one task type, to feed its own environment and task set.

    The narrowed family keeps the parent's tool contracts, state domains and world semantics
    (the environment is shared), and carries only the task type's members and tasks.

    Raises:
        ValueError: The task type belongs to another family.
    """
    if task_type.family_id != family.family_id:
        raise ValueError(
            f"task type {task_type.task_type_id} splits family {task_type.family_id}, "
            f"not {family.family_id}"
        )
    rationale = f"task type {task_type.name or task_type.cluster_id} of {family.family_id}"
    return family.model_copy(
        update={
            "family_id": task_type.task_type_id,
            "path": f"{family.path}/{task_type.cluster_id}",
            "members": task_type.members,
            "tasks": task_type.tasks,
            "evidence": CompatibilityEvidence(
                reviews=family.evidence.reviews,
                reconcile=family.evidence.reconcile,
                rationale=f"{rationale}; {family.evidence.rationale}"[:4_000],
                locators=family.evidence.locators,
            ),
        }
    )


def _family_types(
    family: EnvironmentFamily,
    clustering: TagClustering,
    names: ClusterNames,
    names_ref: str,
    matrices: Sequence[TagMatrix],
) -> FamilyTaskTypes:
    """Build one family's task types from its fit and names, placing its tasks."""
    by_cluster = {row.cluster_id: row for row in names.names}
    outside = {t for task in family.tasks for t in task.source_trace_ids} - set(
        clustering.assignments
    )
    placed = assign_clusters(clustering, matrices, outside) if outside else {}
    tasks: dict[str, list[FamilyTask]] = {}
    unplaced: list[str] = []
    for task in family.tasks:
        votes = Counter(
            clustering.assignments.get(t) or placed.get(t, (None, 0.0))[0]
            for t in task.source_trace_ids
        )
        votes.pop(None, None)
        if not votes:
            unplaced.append(task.task_id)
            continue
        top = max(votes.values())
        chosen = min(str(cluster) for cluster, count in votes.items() if count == top)
        tasks.setdefault(chosen, []).append(task)
    types = []
    for cluster in clustering.clusters:
        named = by_cluster.get(cluster.cluster_id)
        types.append(
            TaskType(
                task_type_id=family_identity(
                    f"{family.family_id}/{cluster.cluster_id}", cluster.members
                ),
                family_id=family.family_id,
                cluster_id=cluster.cluster_id,
                name=None if named is None else named.name,
                description=None if named is None else named.description,
                members=cluster.members,
                exemplars=cluster.exemplars,
                uncertain_members=cluster.uncertain_members,
                fired=cluster.fired,
                distinctive=cluster.distinctive,
                tasks=tuple(tasks.get(cluster.cluster_id, ())),
            )
        )
    return FamilyTaskTypes(
        family_id=family.family_id,
        clustering=clustering.identity(),
        names=names_ref,
        reason=clustering.reason,
        task_types=tuple(types),
        unplaced=tuple(unplaced),
    )


def _evidence_digest(
    families: Sequence[EnvironmentFamily],
    matrices: Sequence[TagMatrix],
    questions: Mapping[str, str],
    exemplar_texts: Mapping[str, str],
) -> str:
    """Digest the families, matrices, question texts and member texts a split may read."""
    members = {t for family in families for t in family.members}
    return sha256_json(
        {
            "families": [sha256_json(family) for family in families],
            "matrices": [sha256_json(matrix) for matrix in matrices],
            "questions": dict(sorted(questions.items())),
            "exemplars": {t: exemplar_texts[t] for t in sorted(members) if t in exemplar_texts},
        }
    )
