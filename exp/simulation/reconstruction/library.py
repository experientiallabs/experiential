"""Append-only, source-bound shared experience across reconstructed environments."""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from datetime import datetime
from typing import Literal

from pydantic import Field, model_validator

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
from exp.common.traces import Trace
from exp.simulation.reconstruction.models import ReconstructionBundle
from exp.simulation.reconstruction.sources import from_canonical_traces
from exp.simulation.retrieval.contracts import RAGLineageBinding
from exp.simulation.retrieval.transitions import extract_fit_transitions


class LibraryEntry(ContractModel):
    """One immutable source, abstraction or investigation, retaining its authority.

    Attributes:
        entry_id: Stable original identity, never overwritten by later revisions.
        kind: Original trajectory, generated rule/skill or investigation record.
        artifact: Verified manifest containing the referenced payload.
        path: Manifest-bound artifact file, never an arbitrary filesystem path.
        payload_sha256: Exact content, independent of its descriptive cues.
        disposition: Source observation, development-accepted revision or unaccepted attempt.
        cues: Indexable descriptions; matching them does not establish transfer correctness.
        supersedes: Earlier entry replaced by an accepted child; old evidence remains retained.
        derived_from: Source trajectory artifact this entry was produced from. Eligibility
            decisions use this typed reference, never ``cues``; absent means unknown provenance.
        depends_on: Other source trajectories whose memory was supplied to the work that
            produced this entry. Eligibility requires every one of them.
    """

    entry_id: str = Field(min_length=1)
    kind: Literal["trajectory", "rule", "skill", "research"]
    artifact: ArtifactInput
    path: str = Field(min_length=1)
    payload_sha256: Sha256
    disposition: Literal["source", "accepted", "rejected", "unresolved"]
    cues: tuple[str, ...] = ()
    supersedes: str | None = None
    derived_from: ArtifactInput | None = Field(default=None, exclude_if=lambda value: value is None)
    depends_on: tuple[ArtifactInput, ...] = Field(default=(), exclude_if=lambda value: not value)

    @model_validator(mode="after")
    def authority(self) -> LibraryEntry:
        """Source observations and accepted hypotheses remain distinct evidence kinds."""
        if (self.kind == "trajectory") != (self.disposition == "source"):
            raise ValueError("only original trajectories have source disposition")
        if self.supersedes is not None and self.disposition != "accepted":
            raise ValueError("an unaccepted revision cannot supersede active knowledge")
        return self


class LibrarySnapshot(ArtifactEnvelope):
    """Immutable growing library; scored sessions pin one snapshot.

    Attributes:
        library_id: Identity of entries, parent snapshot and publishing implementation.
        parent: Prior snapshot, absent only for the first generation.
        entries: All admitted entries, including rejected research and superseded versions.
    """

    library_id: str
    parent: ArtifactInput | None
    entries: tuple[LibraryEntry, ...]


def extend_library(
    store: ArtifactStore,
    additions: Sequence[LibraryEntry],
    *,
    parent: LibrarySnapshot | None = None,
    created_at: datetime,
    code_revision: str,
) -> LibrarySnapshot:
    """Retain new experience without mutating observations or silently activating failures.

    Args:
        store: Existing shared immutable ArtifactStore, not another database.
        additions: References to already persisted, qualified-scope artifacts.
        parent: Verified prior library snapshot, fixed throughout current scored sessions.
        created_at: Aware publication timestamp.
        code_revision: Exact publishing implementation identity.

    Returns:
        Persisted append-only child. Development acceptance does not authorize deployment.
    """
    parent_ref = None
    previous: tuple[LibraryEntry, ...] = ()
    if parent is not None:
        parent_ref = artifact_input(store.read(parent.library_id).manifest)
        if json.loads(store.read_bytes(parent.library_id, "library.json")) != parent.model_dump(
            mode="json"
        ):
            raise ValueError("parent library differs from its persisted snapshot")
        previous = parent.entries
    entries = (*previous, *additions)
    by_id = {entry.entry_id: entry for entry in previous}
    for entry in additions:
        if entry.entry_id in by_id:
            raise ValueError("library entries are immutable; use a new child identity")
        if entry.supersedes is not None:
            old = by_id.get(entry.supersedes)
            if old is None or old.kind != entry.kind:
                raise ValueError("supersession must name an earlier entry of the same kind")
        resolve_library_entry(store, entry)
        by_id[entry.entry_id] = entry
    if not additions:
        raise ValueError("a new library version requires newly admitted experience")
    refs = {entry.artifact.artifact_id: entry.artifact for entry in entries}
    if parent_ref is not None:
        refs[parent_ref.artifact_id] = parent_ref
    identity = stable_id(
        "experience-library",
        {
            "parent": parent_ref.model_dump(mode="json") if parent_ref else None,
            "code_revision": code_revision,
            "entries": [entry.model_dump(mode="json") for entry in entries],
        },
    )
    snapshot = LibrarySnapshot(
        schema_version=1,
        created_at=created_at,
        inputs=tuple(refs[key] for key in sorted(refs)),
        code_revision=code_revision,
        library_id=identity,
        parent=parent_ref,
        entries=entries,
    )
    persisted, _ = store.write_or_replay(
        artifact_id=identity,
        artifact_type="experience-library",
        envelope=snapshot,
        envelope_path="library.json",
        envelope_type=LibrarySnapshot,
        files={"library.json": canonical_json_bytes(snapshot)},
    )
    return persisted


def retrieve_library(
    store: ArtifactStore,
    snapshot: LibrarySnapshot,
    target_traces: Sequence[Trace],
    *,
    select: Callable[[JsonObject, tuple[LibraryEntry, ...]], tuple[str, ...]],
    top_k: int,
) -> JsonObject:
    """Select prior experience using only new-environment trajectories.

    Args:
        store: Shared artifact owner, resolving exact original payload bytes.
        snapshot: Frozen library version from earlier completed investigations.
        target_traces: Raw new-task observations; no hand-coded family label or future truth.
        select: Explicit ranking policy, with caller-owned JEV/provider accounting if used.
        top_k: Registered bound on delivered entries.

    Returns:
        Typed, source-linked transfer context. No selected value initializes current state.
    """
    if not target_traces or top_k < 1:
        raise ValueError("library selection requires observed target traces and a positive bound")
    stored = json.loads(store.read_bytes(snapshot.library_id, "library.json"))
    if stored != snapshot.model_dump(mode="json"):
        raise ValueError("selected library snapshot differs from persisted contents")
    superseded = {entry.supersedes for entry in snapshot.entries if entry.supersedes is not None}
    target_payloads = {
        sha256_bytes(canonical_json_bytes(_source_payload(trace))) for trace in target_traces
    }
    repeated = {
        entry.entry_id
        for entry in snapshot.entries
        if entry.kind == "trajectory" and entry.payload_sha256 in target_payloads
    }
    eligible = tuple(
        entry
        for entry in snapshot.entries
        if entry.entry_id not in superseded
        and entry.entry_id not in repeated
        and (entry.kind == "research" or entry.disposition in ("source", "accepted"))
    )
    query: JsonObject = {
        "target_traces": [t.model_dump(mode="json") for t in target_traces],
        "authority": "Infer applicability from trajectories; prior values are not current facts.",
    }
    selected = select(query, eligible)
    allowed = {entry.entry_id: entry for entry in eligible}
    if (
        len(selected) > top_k
        or len(set(selected)) != len(selected)
        or not set(selected) <= set(allowed)
    ):
        raise ValueError("selector returned duplicate, unavailable or excessive library entries")
    return {
        "library_id": snapshot.library_id,
        "excluded_exact_target_originals": sorted(repeated),
        "query_sha256": sha256_json(query),
        "selected": [
            {
                "entry": allowed[key].model_dump(mode="json"),
                "payload": resolve_library_entry(store, allowed[key]),
            }
            for key in selected
        ],
        "authority": "Historical observations and hypotheses, not current-instance facts",
    }


def resolve_library_entry(store: ArtifactStore, entry: LibraryEntry) -> JsonValue:
    """Resolve an admitted library record, verifying manifest identity and payload bytes.

    Args:
        store: Existing immutable artifact owner.
        entry: Reference already admitted for the caller's evidence scope.

    Returns:
        Exact JSON payload. Resolution alone does not authorize its use or prove its claims.
    """
    if artifact_input(store.read(entry.artifact.artifact_id).manifest) != entry.artifact:
        raise ValueError("library source manifest changed")
    payload = store.read_bytes(entry.artifact.artifact_id, entry.path)
    if sha256_bytes(payload) != entry.payload_sha256:
        raise ValueError("library payload differs from its registered bytes")
    return json.loads(payload)


def remember_reconstruction(
    store: ArtifactStore,
    bundle: ReconstructionBundle,
    traces: Sequence[Trace],
    *,
    parent: LibrarySnapshot | None,
    created_at: datetime,
    code_revision: str,
) -> LibrarySnapshot:
    """Add actual source episodes and an unqualified construction to the shared base.

    Args:
        store: Shared artifact owner used for construction and the library.
        bundle: Completed source-linked reconstruction, not a behavioral acceptance claim.
        traces: Exact target episodes consumed by that construction.
        parent: Previously pinned shared base, absent for the first environment.
        created_at: Aware publication timestamp.
        code_revision: Implementation identity for this knowledge-clock event.

    Returns:
        New library snapshot with one selectable entry per original episode; generated rules
        remain inactive until development selection. Duplicate episodes are not republished.
    """
    raw = from_canonical_traces(
        traces,
        description=bundle.sources.description,
        published_contracts=bundle.sources.published_contracts,
        prior_context=bundle.sources.prior_context,
    )
    if raw != bundle.sources:
        raise ValueError("remembered traces differ from the reconstruction's actual source input")
    bundle_ref = artifact_input(store.read(bundle.artifact_id).manifest)
    sources = remember_trajectories(
        store, traces, parent=parent, created_at=created_at, code_revision=code_revision
    )
    if any(entry.entry_id == bundle.artifact_id for entry in sources.entries):
        return sources
    return extend_library(
        store,
        (
            LibraryEntry(
                entry_id=bundle.artifact_id,
                kind="rule",
                artifact=bundle_ref,
                path="artifact.json",
                payload_sha256=sha256_bytes(store.read_bytes(bundle.artifact_id, "artifact.json")),
                disposition="unresolved",
                cues=tuple(tool.name for tool in bundle.interface.tools),
            ),
        ),
        parent=sources,
        created_at=created_at,
        code_revision=code_revision,
    )


def remember_trajectories(
    store: ArtifactStore,
    traces: Sequence[Trace],
    *,
    parent: LibrarySnapshot | None,
    created_at: datetime,
    code_revision: str,
) -> LibrarySnapshot:
    """Admit exact whole originals independently of whether model construction succeeds.

    Args:
        store: Shared immutable artifact owner.
        traces: Caller-admitted episodes; native qualification remains separate.
        parent: Earlier frozen snapshot, absent when opening a new source index.
        created_at: Aware admission time.
        code_revision: Source-processing implementation identity.

    Returns:
        Append-only snapshot, without duplicate episodes or accepted generated behavior.
    """
    if not traces or len({trace.trace_id for trace in traces}) != len(traces):
        raise ValueError("source admission requires distinct nonempty whole episodes")
    additions: list[LibraryEntry] = []
    for trace in traces:
        payload = _source_payload(trace)
        source_id = stable_id("library-originals", payload)
        if not store.exists(source_id):
            store.write_json(
                artifact_id=source_id,
                artifact_type="library-original-trajectories",
                envelope=ArtifactEnvelope(
                    schema_version=1, created_at=created_at, code_revision=code_revision
                ),
                files={"sources.json": payload},
            )
        elif json.loads(store.read_bytes(source_id, "sources.json")) != payload:
            raise ValueError("remembered original trajectory bytes changed")
        transitions = extract_fit_transitions(
            (trace,),
            (
                RAGLineageBinding(
                    trace_id=trace.trace_id,
                    lineage_id=stable_id(
                        "library-lineage", {"id": trace.conversation_id or trace.trace_id}
                    ),
                    partition="fit",
                ),
            ),
        )
        source_ref = artifact_input(store.read(source_id).manifest)
        additions.append(
            LibraryEntry(
                entry_id=source_id,
                kind="trajectory",
                artifact=source_ref,
                path="sources.json",
                payload_sha256=sha256_bytes(store.read_bytes(source_id, "sources.json")),
                disposition="source",
                cues=(
                    trace.task,
                    *sorted(
                        {t.action.tool_name for t in transitions if t.action.tool_name is not None}
                    ),
                ),
            )
        )
    existing = set() if parent is None else {entry.entry_id for entry in parent.entries}
    fresh = tuple(entry for entry in additions if entry.entry_id not in existing)
    if not fresh and parent is not None:
        return parent
    return extend_library(
        store, fresh, parent=parent, created_at=created_at, code_revision=code_revision
    )


def _source_payload(trace: Trace) -> JsonObject:
    """Identify one exact original episode, independent of construction order or batch size."""
    return {
        "traces": [trace.model_dump(mode="json")],
        "authority": "Original caller-admitted trajectories; native qualification is separate",
    }
