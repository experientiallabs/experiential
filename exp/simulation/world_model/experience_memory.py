"""Versioned cross-trajectory memory over the append-only experience library.

A lineage is a numbered sequence of immutable versions. Each learning version consumes one
distinct completed trajectory: its disclosed native replies become a source observation and
its session investigations become unresolved research. Neither becomes a verified rule.
Version identities are derived from the lineage and number, so a concurrent or changed
publication of the same number fails instead of forking the lineage.
"""

import json
from collections.abc import Sequence
from datetime import datetime
from typing import Literal

from filelock import FileLock
from pydantic import Field, ValidationError

from exp.common.core.artifacts import (
    ArtifactEnvelope,
    ArtifactInput,
    ContractModel,
    JsonObject,
    JsonValue,
    Sha256,
    canonical_json_bytes,
    sha256_bytes,
    sha256_json,
    stable_id,
)
from exp.common.project import ArtifactStore, artifact_input
from exp.simulation.reconstruction.library import (
    LibraryEntry,
    LibrarySnapshot,
    extend_library,
    resolve_library_entry,
)
from exp.simulation.world_model.artifact import GroundedWorldModelArtifact
from exp.simulation.world_model.memory_tree import (
    Block,
    LineDraft,
    LineKind,
    append_lines,
    containing_blocks,
    line_id,
    read_line,
)
from exp.simulation.world_model.runtime import load_grounded_world_model_artifact

_TYPE = "experience-memory-version"


class MemoryVersion(ArtifactEnvelope):
    """One frozen checkpoint of accumulated experience for a single simulator.

    Attributes:
        lineage: Caller-named memory sequence within one project.
        version: Number of consumed distinct trajectories; 0 is never persisted.
        world_model: Serving simulator whose predictions produced this version's research.
        knowledge_identity: What the simulator knows (knowledge and serving corpus). This is
            the only compatibility gate: any model, top-k, reranker, prompt, prediction
            guidance or memory selector may read or extend the lineage, so serving
            treatments compare on one bank.
        knowledge_identity_scheme: Current scheme 2 treats prediction guidance as serving policy.
        serving_policy: How this version's trajectory was served; provenance, never a gate.
        log_length: Memory log lines covered by this version.
        appended: The log lines this version adds, ending at ``log_length``. The version is
            the commit point: positional line artifacts are materialized from it afterwards,
            so an interrupted publication leaves no line behind.
        selection_policy_sha256: Memory selector used to deliver these records.
        library: Experience library snapshot holding every admitted entry.
        consumed: Ordered digests of consumed trajectories, one per version.
        publisher: Run that consumed this version's trajectory, for exactly-once recovery.
        summary_generations: Current generation of every block forgotten at or before this
            version, keyed ``"lo-hi"``; absent blocks are generation 0. Readers pinned to this
            version read exactly these generations.
        forgotten: Blocks this version forgot (``"lo-hi"``): summaries over a line retired by
            this trajectory's investigations, or named by an operator forget recorded
            against the parent version, each with every ancestor.
    """

    lineage: str = Field(pattern=r"^[a-z][a-z0-9-]{0,63}$")
    version: int = Field(ge=1, strict=True)
    world_model: ArtifactInput
    selection_policy_sha256: Sha256
    library: ArtifactInput
    consumed: tuple[Sha256, ...]
    publisher: str
    knowledge_identity: Sha256
    knowledge_identity_scheme: Literal[2]
    serving_policy: JsonObject
    log_length: int = Field(ge=0)
    appended: tuple[LineDraft, ...]
    summary_generations: dict[str, int] | None = Field(
        default=None, exclude_if=lambda value: not value
    )
    forgotten: tuple[str, ...] | None = Field(default=None, exclude_if=lambda value: not value)


def summary_generations(version: MemoryVersion | None) -> dict[Block, int]:
    """Block generations a reader pinned to ``version`` uses; absent blocks are 0."""
    if version is None or not version.summary_generations:
        return {}
    found: dict[Block, int] = {}
    for key, generation in version.summary_generations.items():
        lo, _, hi = key.partition("-")
        found[(int(lo), int(hi))] = generation
    return found


def retired_lines(store: ArtifactStore, version: MemoryVersion | None) -> frozenset[int]:
    """Log positions whose hypothesis was retired by an investigation in this version."""
    if version is None or not version.log_length:
        return frozenset()
    retired = set(memory_view(store, version, Eligibility()).retired)
    if not retired:
        return frozenset()
    return frozenset(
        index
        for index in range(version.log_length)
        if read_line(store, version.lineage, index).origin in retired
    )


def forget_summary(
    store: ArtifactStore,
    lineage: str,
    block: Block,
    *,
    reason: str,
    created_at: datetime,
    code_revision: str,
) -> str:
    """Record an operator decision to forget one block summary and rebuild it.

    The record is pinned to the current head. A published checkpoint never changes, so the
    forget takes effect in the next published version, which raises the generation of the
    block and of every ancestor; the next maintenance rebuilds them. Repeating the same
    forget against the same head is a no-op.

    Args:
        store: Project artifact owner.
        lineage: Memory lineage.
        block: Aligned ``(lo, hi)`` block of at least two lines within the head's log.
        reason: Operator explanation, kept with the record.
        created_at: Aware record time.
        code_revision: Recording implementation identity.

    Returns:
        The forget record's artifact identity.

    Raises:
        ValueError: The block is not an aligned power-of-two block of the head's log.
    """
    lo, hi = block
    size = hi - lo
    with memory_lock(store, lineage):
        head = memory_head(store, lineage)
        total = 0 if head is None else head.log_length or 0
        if size < 2 or size & (size - 1) or lo % size or hi > total:
            raise ValueError(
                f"block #{lo}-{hi - 1} is not a complete aligned block of memory {lineage!r}"
            )
        base = 0 if head is None else head.version
        identity = stable_id(
            "memory-forget", {"lineage": lineage, "base_version": base, "lo": lo, "hi": hi}
        )
        if not store.exists(identity):
            store.write_json(
                artifact_id=identity,
                artifact_type="memory-forget",
                envelope=ArtifactEnvelope(
                    schema_version=1, created_at=created_at, code_revision=code_revision
                ),
                files={
                    "forget.json": {
                        "lineage": lineage,
                        "base_version": base,
                        "lo": lo,
                        "hi": hi,
                        "reason": reason,
                    }
                },
            )
        return identity


def _forgotten_blocks(
    store: ArtifactStore, lineage: str, parent: MemoryVersion | None, research: Sequence[str]
) -> list[Block]:
    """Blocks the next version forgets: over retired lines, and operator forgets of the parent.

    Only complete blocks of the parent's log can have been read, so only they need a new
    generation; blocks that complete later are built from current lines anyway.
    """
    total = 0 if parent is None else parent.log_length or 0
    if total < 2 or parent is None:
        return []
    superseded: set[str] = set()
    for identity in research:
        if not store.exists(identity):
            continue
        record = json.loads(store.read_bytes(identity, "record.json"))
        investigation = record.get("investigation") if isinstance(record, dict) else None
        if isinstance(investigation, dict):
            superseded.update(str(key) for key in investigation.get("supersedes") or ())
    positions = [i for i in range(total) if read_line(store, lineage, i).origin in superseded]
    blocks = set(containing_blocks(total, positions))
    for identity in store.list_ids(artifact_type="memory-forget"):
        row = json.loads(store.read_bytes(identity, "forget.json"))
        if row.get("lineage") != lineage or row.get("base_version") != parent.version:
            continue
        lo, hi = int(row["lo"]), int(row["hi"])
        blocks.update(
            block for block in containing_blocks(total, [lo]) if block[0] <= lo and hi <= block[1]
        )
    return sorted(blocks, key=lambda block: (block[1] - block[0], block[0]))


def memory_version_id(lineage: str, version: int) -> str:
    """Derive the only artifact identity a lineage version may occupy."""
    return stable_id("experience-memory", {"lineage": lineage, "version": version})


def read_memory_version(store: ArtifactStore, lineage: str, version: int) -> MemoryVersion:
    """Load one verified checkpoint, rejecting a different lineage or number."""
    identity = memory_version_id(lineage, version)
    if not store.exists(identity):
        raise ValueError(f"memory lineage {lineage!r} has no version {version}")
    try:
        saved = MemoryVersion.model_validate_json(store.read_bytes(identity, "memory.json"))
    except ValidationError as exc:
        raise ValueError(
            "memory checkpoint does not match the current format; build into a fresh "
            "project ID to learn new memory while retaining existing artifacts"
        ) from exc
    if saved.lineage != lineage or saved.version != version:
        raise ValueError("memory version artifact differs from its lineage position")
    return saved


def memory_head(store: ArtifactStore, lineage: str) -> MemoryVersion | None:
    """Return the latest contiguous version, or None before any trajectory is consumed."""
    head = None
    while store.exists(memory_version_id(lineage, (head.version if head else 0) + 1)):
        head = read_memory_version(store, lineage, (head.version if head else 0) + 1)
    return head


KNOWLEDGE_IDENTITY_SCHEME: Literal[2] = 2


def knowledge_identity(artifact: GroundedWorldModelArtifact) -> Sha256:
    """Identify what a simulator knows: its behavioral knowledge and serving evidence corpus.

    A memory-first simulator (``original_evidence == "none"``) is identified apart from a
    retrieving one over the same corpus.

    Prediction guidance is tunable serving text, so the identity leaves it out.
    """
    knowledge = artifact.knowledge
    # A memory-first simulator never reads the corpus, so it knows something different; the
    # key is absent for retrieving simulators, keeping their recorded identities unchanged.
    profile: JsonObject = (
        {} if artifact.original_evidence == "retrieved" else {"original_evidence": "none"}
    )
    return sha256_json(
        {
            **profile,
            "knowledge": None
            if knowledge is None
            else knowledge.model_dump(mode="json", exclude={"prediction_guidance"}),
            "serving_rag": artifact.serving_rag.model_dump(mode="json"),
        }
    )


def serving_policy(
    artifact: GroundedWorldModelArtifact, world_model: ArtifactInput, memory_selector: str
) -> JsonObject:
    """How a run served that knowledge; recorded with each version, never compared."""
    return {
        "world_model": world_model.model_dump(mode="json"),
        "model_id": artifact.model.model_id,
        "top_k": artifact.top_k,
        "selection_policy_sha256": artifact.selection_policy_sha256,
        "prompt_version": artifact.prompt_version,
        "prompt_sha256": artifact.prompt_sha256,
        "prediction_guidance_sha256": sha256_json(
            "" if artifact.knowledge is None else artifact.knowledge.prediction_guidance
        ),
        "memory_selector_sha256": memory_selector,
    }


def require_memory_scope(
    version: MemoryVersion | None,
    *,
    world_model: ArtifactInput,
    artifact: GroundedWorldModelArtifact,
    selection_policy_sha256: str,
) -> None:
    """Refuse experience built on different knowledge under the current identity scheme."""
    if version is not None and version.knowledge_identity != knowledge_identity(artifact):
        raise ValueError("memory lineage belongs to different knowledge")


def memory_lock(store: ArtifactStore, lineage: str) -> FileLock:
    """Serialize publication and reservation decisions for one lineage."""
    return FileLock(store.project_directory / f"memory-{lineage}.lock", timeout=60)


def materialize_log(store: ArtifactStore, version: MemoryVersion | None) -> None:
    """Write any committed log lines still missing, so readers see the whole committed log."""
    if version is None:
        return
    with memory_lock(store, version.lineage):
        _materialize(store, version)


def _materialize(store: ArtifactStore, version: MemoryVersion) -> None:
    """Materialize committed lines newest first; the caller holds the lineage lock.

    Lines come only from committed versions, with the version's own time and revision, so
    repeating this is an exact replay. Ancestors are materialized before a child commits,
    so the walk stops at the first version whose lines all exist.
    """
    current: MemoryVersion | None = version
    while current is not None and current.log_length:
        start = current.log_length - len(current.appended)
        if all(
            store.exists(line_id(current.lineage, start + i)) for i in range(len(current.appended))
        ):
            return
        append_lines(
            store,
            current.lineage,
            start,
            current.appended,
            created_at=current.created_at,
            code_revision=current.code_revision,
        )
        current = (
            read_memory_version(store, current.lineage, current.version - 1)
            if current.version > 1
            else None
        )


def reservation_id(lineage: str, task_lineage: str) -> str:
    """Identity of one declared reserved task lineage within a memory lineage."""
    return stable_id("memory-reservation", {"lineage": lineage, "task_lineage": task_lineage})


def is_reserved(store: ArtifactStore, lineage: str, task_lineage: str) -> bool:
    """Report whether evaluation declared this task lineage as never learnable."""
    return store.exists(reservation_id(lineage, task_lineage))


def consumed_task_lineages(store: ArtifactStore, version: MemoryVersion | None) -> set[str]:
    """Task lineages of every trajectory a version has consumed."""
    if version is None:
        return set()
    found: set[str] = set()
    for digest in version.consumed:
        source = _consumed(store, digest)
        if source is not None and isinstance(source.get("task_lineage"), str):
            found.add(str(source["task_lineage"]))
    return found


def reserve_task_lineage(
    store: ArtifactStore,
    lineage: str,
    task_lineage: str,
    *,
    created_at: datetime,
    code_revision: str,
) -> None:
    """Declare a task lineage reserved for evaluation; refuse if it was already learned.

    The decision runs under the lineage lock, so a reservation and a consumption of the
    same task lineage cannot both succeed. Re-declaring an existing reservation is a no-op.
    """
    with memory_lock(store, lineage):
        if is_reserved(store, lineage, task_lineage):
            return
        if task_lineage in consumed_task_lineages(store, memory_head(store, lineage)):
            raise ValueError(
                f"task lineage {task_lineage!r} was already consumed by memory {lineage!r}; "
                "it cannot serve as reserved evaluation"
            )
        store.write_json(
            artifact_id=reservation_id(lineage, task_lineage),
            artifact_type="memory-reservation",
            envelope=ArtifactEnvelope(
                schema_version=1, created_at=created_at, code_revision=code_revision
            ),
            files={"reservation.json": {"lineage": lineage, "task_lineage": task_lineage}},
        )


def memory_records(store: ArtifactStore, version: MemoryVersion | None) -> dict[str, JsonObject]:
    """Resolve active research, retaining superseded records only in the archive.

    Returns:
        Unresolved research records keyed by their original identity.
    """
    records, _ = memory_catalog(store, version)
    return {k: v for k, v in records.items() if v.get("kind") != "original-observation"}


class Eligibility(ContractModel):
    """Which consumed sources a run may read; applied to records, lines and summaries alike.

    Attributes:
        exclude_public: Public identity of the trajectory about to be predicted.
        reserved_task_lineages: Task lineages declared reserved for evaluation.
        hidden_task_lineage: Task lineage being predicted, hidden from this read so earlier
            trajectories of the same task cannot answer for it (test-then-train learning).
        strict: Reserved evaluation; content without resolvable provenance is ineligible.
    """

    exclude_public: Sha256 | None = None
    reserved_task_lineages: tuple[str, ...] = ()
    hidden_task_lineage: str | None = None
    strict: bool = Field(default=False, strict=True)

    def admits(self, source: JsonObject | None) -> bool:
        """Decide one source trajectory; None means its provenance is unknown."""
        if source is None:
            return not self.strict
        if public_trajectory_identity(source) == self.exclude_public:
            return False
        lineage = source.get("task_lineage")
        if not isinstance(lineage, str) or not lineage.strip():
            return not self.strict
        return lineage not in self.reserved_task_lineages and lineage != self.hidden_task_lineage


RETRIEVED_SOURCE = "retrieved-source"


def retrieved_source(
    store: ArtifactStore,
    *,
    trace_id: str,
    task_lineage: str,
    created_at: datetime,
    code_revision: str,
) -> ArtifactInput:
    """Reference one recorded trace that retrieval delivered to a session, as a typed source.

    Memory written by a session that saw retrieved evidence depends on this source, so the
    retrieved trace's task lineage is withheld wherever it is hidden or reserved, exactly as a
    consumed trajectory's would be: a retrieved fact cannot reach its own task through memory.
    """
    identity = stable_id(RETRIEVED_SOURCE, {"trace_id": trace_id, "task_lineage": task_lineage})
    if not store.exists(identity):
        store.write_json(
            artifact_id=identity,
            artifact_type=RETRIEVED_SOURCE,
            envelope=ArtifactEnvelope(
                schema_version=1, created_at=created_at, code_revision=code_revision
            ),
            files={"source.json": {"trace_id": trace_id, "task_lineage": task_lineage}},
        )
    return artifact_input(store.read(identity).manifest)


def load_source(store: ArtifactStore, reference: ArtifactInput | None) -> JsonObject | None:
    """Resolve a typed source reference to its consumed trajectory, verifying the manifest.

    A retrieved-source reference resolves to its recorded trace and task lineage.
    """
    if reference is None or not store.exists(reference.artifact_id):
        return None
    stored = store.read(reference.artifact_id)
    if artifact_input(stored.manifest) != reference:
        raise ValueError("source trajectory artifact differs from its recorded reference")
    if stored.manifest.artifact_type == RETRIEVED_SOURCE:
        source = json.loads(store.read_bytes(reference.artifact_id, "source.json"))
        return source if isinstance(source, dict) else None
    if stored.manifest.artifact_type != "consumed-trajectory":
        return None
    payload = json.loads(store.read_bytes(reference.artifact_id, "trajectory.json"))
    trajectory = payload.get("trajectory") if isinstance(payload, dict) else None
    return trajectory if isinstance(trajectory, dict) else None


def reserved_task_lineages(store: ArtifactStore, lineage: str) -> tuple[str, ...]:
    """Every task lineage declared reserved for this memory lineage."""
    found = []
    for identity in store.list_ids(artifact_type="memory-reservation"):
        row = json.loads(store.read_bytes(identity, "reservation.json"))
        if row.get("lineage") == lineage:
            found.append(str(row["task_lineage"]))
    return tuple(sorted(found))


def memory_catalog(
    store: ArtifactStore,
    version: MemoryVersion | None,
    *,
    exclude_public: str | None = None,
    eligibility: Eligibility | None = None,
) -> tuple[dict[str, JsonObject], list[str]]:
    """Return everything a later trajectory may retrieve, and what was withheld.

    Three ordered stages keep ineligible evidence inert:

    1. Eligibility: each entry's typed ``derived_from`` reference is resolved and judged by
       ``eligibility``. Retrieval cues never decide eligibility. Unknown provenance is
       withheld under strict (reserved) evaluation and admitted otherwise.
    2. Supersession: only eligible investigations may retire earlier research. A withheld
       investigation therefore cannot change the run's context indirectly.
    3. Projection: eligible, unretired research is returned unchanged, and each consumed
       step becomes one labelled ``original-observation`` of a different completed instance.

    Retired research remains in the immutable library archive.

    Args:
        store: Project artifact owner.
        version: Pinned memory version, or None before any consumption.
        exclude_public: Shorthand for an eligibility excluding one trajectory.
        eligibility: Full eligibility decision; overrides ``exclude_public``.

    Returns:
        Eligible records keyed by identity, and the sorted withheld identities.
    """
    view = memory_view(store, version, eligibility or Eligibility(exclude_public=exclude_public))
    return view.records, view.withheld


class MemoryView(ContractModel):
    """One run's eligible view of a pinned memory version.

    Attributes:
        records: Eligible, unretired research and per-step original observations.
        withheld: Identities excluded by eligibility (own trajectory, reserved, dependencies,
            unknown provenance under strict evaluation).
        retired: Research retired by an eligible investigation; archived, never served.
        sources: For each served record, every source trajectory it depends on.
    """

    records: dict[str, JsonObject]
    withheld: list[str]
    retired: list[str]
    sources: dict[str, tuple[ArtifactInput, ...]]


def memory_view(
    store: ArtifactStore, version: MemoryVersion | None, eligibility: Eligibility
) -> MemoryView:
    """Eligibility, then supersession from eligible investigations, then projection.

    An entry is eligible only if its own source and every ``depends_on`` source are
    admitted, so research written after seeing withheld memory is withheld with it.
    """
    if version is None:
        return MemoryView(records={}, withheld=[], retired=[], sources={})
    eligible: list[tuple[LibraryEntry, JsonObject, JsonObject | None]] = []
    withheld: list[str] = []
    for entry in _library(store, version).entries:
        payload = resolve_library_entry(store, entry)
        if not isinstance(payload, dict):
            continue
        source = load_source(store, entry.derived_from)
        admitted = eligibility.admits(source) and all(
            eligibility.admits(load_source(store, dep)) for dep in entry.depends_on
        )
        if admitted:
            eligible.append((entry, payload, source))
        elif entry.kind == "trajectory" and source is not None:
            withheld.extend(_observation_ids(entry, source))
        else:
            withheld.append(entry.entry_id)
    retired: set[str] = set()
    for entry, payload, _ in eligible:
        investigation = payload.get("investigation")
        if entry.kind == "research" and isinstance(investigation, dict):
            retired.update(str(key) for key in investigation.get("supersedes", ()))
    records: dict[str, JsonObject] = {}
    sources: dict[str, tuple[ArtifactInput, ...]] = {}
    for entry, payload, source in eligible:
        closure = tuple(ref for ref in (entry.derived_from, *entry.depends_on) if ref is not None)
        if entry.kind == "research" and entry.entry_id not in retired:
            records[entry.entry_id] = payload
            sources[entry.entry_id] = closure
        elif entry.kind == "trajectory" and source is not None:
            for identity, row in _observations(entry, source).items():
                records[identity] = row
                sources[identity] = closure
    return MemoryView(
        records=records, withheld=sorted(withheld), retired=sorted(retired), sources=sources
    )


def provenance_gaps(store: ArtifactStore, version: MemoryVersion | None) -> list[str]:
    """Entries whose source cannot be resolved; visibly ineligible for reserved evaluation."""
    if version is None:
        return []
    return sorted(
        entry.entry_id
        for entry in _library(store, version).entries
        if load_source(store, entry.derived_from) is None
    )


def _observation_ids(entry: LibraryEntry, source: JsonObject) -> list[str]:
    """Identities of one consumed trajectory's per-step observations."""
    return list(_observations(entry, source))


def _observations(entry: LibraryEntry, source: JsonObject) -> dict[str, JsonObject]:
    """Project a consumed trajectory into labelled per-step native observations."""
    actions, replies = source.get("actions"), source.get("observed_replies")
    if not isinstance(actions, list) or not isinstance(replies, list):
        raise ValueError("consumed trajectory lacks its action and reply inventory")
    rows: dict[str, JsonObject] = {}
    for step, (action, reply) in enumerate(zip(actions, replies, strict=True)):
        identity = f"{entry.entry_id}-step-{step}"
        rows[identity] = {
            "id": identity,
            "kind": "original-observation",
            "trajectory": entry.entry_id,
            "task_lineage": source.get("task_lineage"),
            "step": step,
            "action": action,
            "observed_reply": reply,
            "authority": "Original native observation from a different completed "
            "instance; its values describe that instance, not the current one",
        }
    return rows


def public_trajectory_identity(trajectory: JsonObject) -> Sha256:
    """Identify a trajectory by what is known before prediction: task, context and actions."""
    return sha256_json(
        {
            "task": trajectory.get("task"),
            "initial_context": trajectory.get("initial_context"),
            "actions": trajectory.get("actions"),
        }
    )


def _consumed(store: ArtifactStore, digest: str | None) -> JsonObject | None:
    """Load the consumed trajectory a library entry was derived from, if recorded."""
    if digest is None:
        return None
    identity = stable_id("consumed-trajectory", {"digest": digest})
    if not store.exists(identity):
        return None
    payload = json.loads(store.read_bytes(identity, "trajectory.json"))
    trajectory = payload.get("trajectory") if isinstance(payload, dict) else None
    return trajectory if isinstance(trajectory, dict) else None


def consumed_trajectory_digest(trajectory: JsonObject) -> Sha256:
    """Identify a completed trajectory by its public actions and disclosed outcomes only."""
    return sha256_json(trajectory)


def publish_trajectory(
    store: ArtifactStore,
    *,
    lineage: str,
    parent: MemoryVersion | None,
    world_model: ArtifactInput,
    selection_policy_sha256: Sha256,
    trajectory: JsonObject,
    research: Sequence[str],
    lines: Sequence[tuple[LineKind, str, str]] = (),
    created_at: datetime,
    code_revision: str,
    publisher: str,
) -> tuple[MemoryVersion, bool]:
    """Consume one completed trajectory into the next lineage version.

    Args:
        store: Project artifact owner holding the session research records.
        lineage: Memory sequence to extend.
        parent: Pinned version the trajectory was predicted against, or None for the first.
        world_model: Serving simulator that produced the predictions; it is loaded here, so
            the recorded knowledge identity always belongs to this reference.
        selection_policy_sha256: Selector that delivered parent memory.
        trajectory: Task lineage, public actions (``{name, arguments}``) and one disclosed
            native reply per action, without run identity.
        research: Session research record artifact IDs created from this trajectory.
        lines: Log lines (kind, one-line text, full source identity) appended in order.
        created_at: Aware publication time.
        code_revision: Publishing implementation identity.
        publisher: Stable run identity; a restart of the same run recovers its own version.

    Returns:
        The new (or recovered own) version and True, or the existing head and False when
        another run already consumed this exact trajectory. Repeated draws never count twice.

    Raises:
        ReservedTaskError: The trajectory's task lineage is reserved for evaluation.
        ValueError: The trajectory lacks a task lineage, or its actions and replies are
            malformed or unequal in number.
    """
    _require_admissible(trajectory)
    artifact = load_grounded_world_model_artifact(store, world_model)
    with memory_lock(store, lineage):
        return _publish(
            store,
            lineage=lineage,
            parent=parent,
            world_model=world_model,
            artifact=artifact,
            selection_policy_sha256=selection_policy_sha256,
            trajectory=trajectory,
            research=research,
            lines=lines,
            created_at=created_at,
            code_revision=code_revision,
            publisher=publisher,
        )


def _require_admissible(trajectory: JsonObject) -> None:
    """Refuse a trajectory whose ownership or inventory a later reader could not trust."""
    lineage = trajectory.get("task_lineage")
    if not isinstance(lineage, str) or not lineage.strip():
        raise ValueError("a consumed trajectory requires its task lineage")
    actions, replies = trajectory.get("actions"), trajectory.get("observed_replies")
    if not isinstance(actions, list) or not isinstance(replies, list):
        raise ValueError("a consumed trajectory requires action and reply lists")
    if len(actions) != len(replies):
        raise ValueError("a consumed trajectory requires one disclosed reply per action")
    for action in actions:
        if (
            not isinstance(action, dict)
            or set(action) != {"name", "arguments"}
            or not isinstance(action["name"], str)
            or not action["name"]
            or not isinstance(action["arguments"], dict)
        ):
            raise ValueError("each consumed action is exactly {name, arguments}")


class ReservedTaskError(ValueError):
    """A reserved task lineage can be evaluated but never consumed into memory."""


def _publish(
    store: ArtifactStore,
    *,
    lineage: str,
    parent: MemoryVersion | None,
    world_model: ArtifactInput,
    artifact: GroundedWorldModelArtifact,
    selection_policy_sha256: Sha256,
    trajectory: JsonObject,
    research: Sequence[str],
    lines: Sequence[tuple[LineKind, str, str]],
    created_at: datetime,
    code_revision: str,
    publisher: str,
) -> tuple[MemoryVersion, bool]:
    """Publication body; the caller holds the lineage lock."""
    if parent is not None and (
        parent.lineage != lineage
        or not store.exists(memory_version_id(lineage, parent.version))
        or read_memory_version(store, lineage, parent.version) != parent
    ):
        raise ValueError("parent is not the persisted version of this lineage")
    require_memory_scope(
        parent,
        world_model=world_model,
        artifact=artifact,
        selection_policy_sha256=selection_policy_sha256,
    )
    task_lineage = trajectory.get("task_lineage")
    if isinstance(task_lineage, str) and is_reserved(store, lineage, task_lineage):
        raise ReservedTaskError(f"task lineage {task_lineage!r} is reserved for evaluation")
    digest = consumed_trajectory_digest(trajectory)
    head = memory_head(store, lineage)
    if head is not None and digest in head.consumed:
        index = head.consumed.index(digest)
        own = read_memory_version(store, lineage, index + 1)
        if own.publisher == publisher and index == (parent.version if parent else 0):
            _materialize(store, own)
            return own, True
        return head, False
    if (head.version if head else 0) != (parent.version if parent else 0):
        raise ValueError("memory advanced after this trajectory was pinned")
    number = 1 if parent is None else parent.version + 1
    source_id = stable_id("consumed-trajectory", {"digest": digest})
    payload: JsonObject = {
        "trajectory": trajectory,
        "authority": "Disclosed native replies after prediction; not a behavioral rule",
    }
    if not store.exists(source_id):
        store.write_json(
            artifact_id=source_id,
            artifact_type="consumed-trajectory",
            envelope=ArtifactEnvelope(
                schema_version=1, created_at=created_at, code_revision=code_revision
            ),
            files={"trajectory.json": payload},
        )
    source_ref = artifact_input(store.read(source_id).manifest)
    entries: list[LibraryEntry] = [
        _entry(
            store, source_id, "trajectory.json", kind="trajectory", cue=digest, source=source_ref
        )
    ]
    prior = set() if parent is None else {e.entry_id for e in _library(store, parent).entries}
    entries.extend(
        _entry(
            store,
            identity,
            "record.json",
            kind="research",
            cue=digest,
            source=source_ref,
            depends=_record_dependencies(store, identity),
        )
        for identity in research
        if identity not in prior
    )
    if parent is not None:
        _materialize(store, parent)
    forgotten = _forgotten_blocks(
        store, lineage, parent, [identity for identity in research if identity not in prior]
    )
    generations = summary_generations(parent)
    for block in forgotten:
        generations[block] = generations.get(block, 0) + 1
    drafts = tuple(
        LineDraft(
            kind=kind,
            text=text,
            origin=origin,
            derived_from=source_ref,
            depends_on=_record_dependencies(store, origin) if kind == "hypothesis" else (),
        )
        for kind, text, origin in lines
    )
    snapshot = extend_library(
        store,
        entries,
        parent=None if parent is None else _library(store, parent),
        created_at=created_at,
        code_revision=code_revision,
    )
    version = MemoryVersion(
        schema_version=1,
        created_at=created_at,
        inputs=tuple(
            sorted(
                [artifact_input(store.read(snapshot.library_id).manifest)]
                + ([] if parent is None else [_input(store, lineage, parent.version)]),
                key=lambda row: row.artifact_id,
            )
        ),
        code_revision=code_revision,
        lineage=lineage,
        version=number,
        world_model=world_model,
        selection_policy_sha256=selection_policy_sha256,
        library=artifact_input(store.read(snapshot.library_id).manifest),
        consumed=(*(() if parent is None else parent.consumed), digest),
        publisher=publisher,
        knowledge_identity=knowledge_identity(artifact),
        knowledge_identity_scheme=KNOWLEDGE_IDENTITY_SCHEME,
        serving_policy=serving_policy(artifact, world_model, selection_policy_sha256),
        log_length=((parent.log_length or 0) if parent else 0) + len(drafts),
        appended=drafts,
        summary_generations={f"{lo}-{hi}": n for (lo, hi), n in sorted(generations.items())}
        or None,
        forgotten=tuple(f"{lo}-{hi}" for lo, hi in forgotten) or None,
    )
    persisted, _ = store.write_or_replay(
        artifact_id=memory_version_id(lineage, number),
        artifact_type=_TYPE,
        envelope=version,
        envelope_path="memory.json",
        envelope_type=MemoryVersion,
        files={"memory.json": canonical_json_bytes(version)},
    )
    _materialize(store, persisted)
    return persisted, True


def _entry(
    store: ArtifactStore,
    identity: str,
    path: str,
    *,
    kind: str,
    cue: str,
    source: ArtifactInput,
    depends: tuple[ArtifactInput, ...] = (),
) -> LibraryEntry:
    """Reference one persisted record with its disposition, source and dependencies."""
    return LibraryEntry(
        entry_id=identity,
        kind="trajectory" if kind == "trajectory" else "research",
        artifact=artifact_input(store.read(identity).manifest),
        path=path,
        payload_sha256=sha256_bytes(store.read_bytes(identity, path)),
        disposition="source" if kind == "trajectory" else "unresolved",
        cues=("trajectory:" + cue,),
        derived_from=source,
        depends_on=depends,
    )


def _record_dependencies(store: ArtifactStore, identity: str) -> tuple[ArtifactInput, ...]:
    """Sources a session research record declared as supplied to its producing work."""
    if not store.exists(identity):
        return ()
    payload = json.loads(store.read_bytes(identity, "record.json"))
    rows = payload.get("depends_on") if isinstance(payload, dict) else None
    refs = {ArtifactInput.model_validate(row) for row in rows or ()}
    return tuple(sorted(refs, key=lambda ref: ref.artifact_id))


def _input(store: ArtifactStore, lineage: str, version: int) -> ArtifactInput:
    """Reference a persisted version as provenance for its child."""
    return artifact_input(store.read(memory_version_id(lineage, version)).manifest)


def _library(store: ArtifactStore, version: MemoryVersion) -> LibrarySnapshot:
    """Load the exact library snapshot pinned by a version."""
    if artifact_input(store.read(version.library.artifact_id).manifest) != version.library:
        raise ValueError("memory library manifest changed")
    raw: JsonValue = json.loads(store.read_bytes(version.library.artifact_id, "library.json"))
    return LibrarySnapshot.model_validate(raw)
