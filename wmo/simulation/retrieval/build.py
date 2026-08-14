"""Build digest-verified immutable RAG artifacts from real traces only.

The artifact has no mutation surface, so generated world-model predictions cannot become
retrieval demonstrations.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Literal

from pydantic import BaseModel

from wmo.common.core.artifacts import ArtifactInput, canonical_json_bytes, stable_id
from wmo.common.project import (
    ArtifactAlreadyExistsError,
    ArtifactManifest,
    ArtifactStore,
    artifact_input,
)
from wmo.common.traces import Trace, load_trace_dataset
from wmo.simulation.retrieval.contracts import (
    RAG_ARTIFACT_TYPE,
    RAG_INDEX_PATH,
    RAG_KEY_SCHEMA_VERSION,
    RAG_TRANSITIONS_PATH,
    RAG_VECTORS_PATH,
    RAGIndex,
    RAGLineageBinding,
    RAGSourceRef,
    RAGTransition,
    RAGVector,
    RealTraceSourceIdentity,
)
from wmo.simulation.retrieval.embedding import (
    RAGEmbedderBinding,
    default_rag_embedder,
    embed_rag_texts,
)
from wmo.simulation.retrieval.store import load_rag_index
from wmo.simulation.retrieval.transitions import extract_real_transitions


@dataclass(frozen=True)
class PersistedRAGIndex:
    """Completed RAG index with the exact records and manifest persisted for it."""

    index: RAGIndex
    manifest: ArtifactManifest
    transitions: tuple[RAGTransition, ...]
    vectors: tuple[RAGVector, ...]


def persist_trace_rag(
    store: ArtifactStore,
    source_inputs: Sequence[ArtifactInput],
    lineage_bindings: Sequence[RAGLineageBinding],
    *,
    created_at: datetime,
    code_revision: str,
    embedder: RAGEmbedderBinding | None = None,
    rag_id: str | None = None,
    default_top_k: int = 5,
    included_partitions: frozenset[Literal["fit", "held_out"]] = frozenset({"fit"}),
) -> PersistedRAGIndex:
    """Build a read-only index from selected partitions of real imported traces.

    Trace count is not a product restriction. Roughly 100 to 1,000 traces is a useful common
    starting range, but one trace and corpora larger than 1,000 follow the same contract.

    Args:
        store: Project-local immutable artifact store.
        source_inputs: Exact manifest references for verified trace-dataset artifacts.
        lineage_bindings: Frozen fit or held-out lineage assignment for every source trace.
        created_at: Time this completed artifact is materialized.
        code_revision: Exact WMO revision producing the artifact.
        embedder: Explicit semantic embedding client and snapshot. A deterministic local hashing
            embedder is used when omitted.
        rag_id: Optional explicit artifact ID. Content addressing is used when omitted.
        default_top_k: Default number of matches returned by the loaded retriever.
        included_partitions: Frozen lineage partitions eligible for this index. Fit-only indexes
            use ``{"fit"}``; serving indexes use both real-evidence partitions.

    Returns:
        The immutable envelope, manifest, observed transitions, and persisted vectors.

    Raises:
        ValueError: A source is not verified real trace evidence, lineage is incomplete, no real
            included transition exists, or the embedder violates its contract.
    """
    if not source_inputs:
        raise ValueError("RAG construction needs at least one verified real trace source")
    if default_top_k <= 0:
        raise ValueError("RAG default top_k must be positive")
    sources, traces = _load_real_sources(store, source_inputs)
    invalid_partitions = included_partitions.difference({"fit", "held_out"})
    if "fit" not in included_partitions or invalid_partitions:
        raise ValueError("RAG included_partitions must contain fit and may also contain held_out")
    transitions = extract_real_transitions(
        traces,
        lineage_bindings,
        included_partitions=included_partitions,
    )
    if not transitions:
        raise ValueError(
            "verified included traces contain no real action-to-subsequent-observation transitions"
        )
    binding = embedder or default_rag_embedder()
    embedded = embed_rag_texts(binding, tuple(item.key_text for item in transitions))
    vectors = tuple(
        RAGVector(transition_id=transition.transition_id, values=values)
        for transition, values in zip(transitions, embedded, strict=True)
    )
    dimensions = len(vectors[0].values)
    transitions_payload = _jsonl_bytes(transitions)
    vectors_payload = _jsonl_bytes(vectors)
    fit_lineages = tuple(
        sorted({binding.lineage_id for binding in lineage_bindings if binding.partition == "fit"})
    )
    included_lineages = tuple(
        sorted(
            {
                binding.lineage_id
                for binding in lineage_bindings
                if binding.partition in included_partitions
            }
        )
    )
    if not fit_lineages or not included_lineages:
        raise ValueError("RAG construction needs at least one included lineage")
    content = {
        "code_revision": code_revision,
        "default_top_k": default_top_k,
        "embedder": binding.snapshot.model_dump(mode="json"),
        "embedding_dimension": dimensions,
        "fit_lineage_ids": list(fit_lineages),
        "included_lineage_ids": list(included_lineages),
        "included_partitions": sorted(included_partitions),
        "key_schema_version": RAG_KEY_SCHEMA_VERSION,
        "schema_version": 1,
        "sources": [source.model_dump(mode="json") for source in sources],
        "transition_count": len(transitions),
        "transition_ids": [item.transition_id for item in transitions],
        "transitions_path": RAG_TRANSITIONS_PATH,
        "transitions_sha256": hashlib.sha256(transitions_payload).hexdigest(),
        "vectors_path": RAG_VECTORS_PATH,
        "vectors_sha256": hashlib.sha256(vectors_payload).hexdigest(),
    }
    resolved_id = rag_id or stable_id("trace-rag", content)
    index = RAGIndex(
        schema_version=1,
        created_at=created_at,
        inputs=tuple(source.artifact_input for source in sources),
        code_revision=code_revision,
        source=None,
        rag_id=resolved_id,
        sources=sources,
        embedder=binding.snapshot,
        transitions_sha256=hashlib.sha256(transitions_payload).hexdigest(),
        vectors_sha256=hashlib.sha256(vectors_payload).hexdigest(),
        transition_ids=tuple(item.transition_id for item in transitions),
        fit_lineage_ids=fit_lineages,
        included_lineage_ids=included_lineages,
        included_partitions=tuple(sorted(included_partitions)),
        embedding_dimension=dimensions,
        transition_count=len(transitions),
        default_top_k=default_top_k,
    )
    files = {
        RAG_INDEX_PATH: canonical_json_bytes(index),
        RAG_TRANSITIONS_PATH: transitions_payload,
        RAG_VECTORS_PATH: vectors_payload,
    }
    destination = store.project_directory / "artifacts" / index.rag_id
    if destination.exists():
        return _load_exact_replay(store, index, transitions, vectors, files)
    try:
        manifest = store.write(
            artifact_id=index.rag_id,
            artifact_type=RAG_ARTIFACT_TYPE,
            envelope=index,
            files=files,
        )
    except ArtifactAlreadyExistsError:
        return _load_exact_replay(store, index, transitions, vectors, files)
    return PersistedRAGIndex(index, manifest, transitions, vectors)


def _load_real_sources(
    store: ArtifactStore,
    supplied_inputs: Sequence[ArtifactInput],
) -> tuple[tuple[RAGSourceRef, ...], tuple[Trace, ...]]:
    """Verify exact source manifests and reject generated or evaluation evidence.

    Args:
        store: Project artifact store containing every supplied dataset.
        supplied_inputs: Exact immutable trace-dataset manifest pointers.

    Returns:
        Sorted typed real-source references and all verified canonical trace records.

    Raises:
        ValueError: Inputs repeat, drift, use another artifact type, or include any dataset or
            individual trace without typed real-observation provenance.
    """
    by_id = {item.artifact_id: item for item in supplied_inputs}
    if len(by_id) != len(supplied_inputs):
        raise ValueError("RAG source inputs must not repeat an artifact")
    sources: list[RAGSourceRef] = []
    traces: list[Trace] = []
    for source_input in sorted(supplied_inputs, key=lambda item: item.artifact_id):
        stored = store.read(source_input.artifact_id)
        if artifact_input(stored.manifest) != source_input:
            raise ValueError(
                f"RAG source {source_input.artifact_id} does not match its supplied manifest digest"
            )
        if stored.manifest.artifact_type != "trace-dataset":
            raise ValueError(
                f"RAG source {source_input.artifact_id} has forbidden artifact type "
                f"{stored.manifest.artifact_type!r}; only verified real trace datasets are "
                "supported"
            )
        loaded = load_trace_dataset(store, source_input.artifact_id)
        source = loaded.dataset.source
        if source is None:
            raise ValueError(
                f"RAG source {source_input.artifact_id} has missing provenance; "
                "generated, simulation, teacher, judgment, evaluation, and manual evidence "
                "cannot ground retrieval"
            )
        try:
            real_source = RealTraceSourceIdentity.model_validate(source.model_dump(mode="json"))
        except ValueError as exc:
            raise ValueError(
                f"RAG source {source_input.artifact_id} has forbidden provenance "
                f"{source.kind!r}; generated, simulation, teacher, judgment, evaluation, and "
                "manual evidence cannot ground retrieval"
            ) from exc
        for trace in loaded.traces:
            try:
                RealTraceSourceIdentity.model_validate(
                    trace.source.identity.model_dump(mode="json")
                )
            except ValueError as exc:
                raise ValueError(
                    f"RAG source {source_input.artifact_id} trace {trace.trace_id!r} has "
                    f"forbidden provenance {trace.source.identity.kind!r}"
                ) from exc
        sources.append(
            RAGSourceRef(
                kind="trace_dataset",
                artifact_input=source_input,
                source=real_source,
                records_sha256=loaded.dataset.traces_sha256,
                trace_ids=tuple(sorted(loaded.dataset.trace_ids)),
            )
        )
        traces.extend(loaded.traces)
    return tuple(sources), tuple(traces)


def _jsonl_bytes(records: Sequence[BaseModel]) -> bytes:
    """Serialize Pydantic records as deterministic newline-terminated JSONL."""
    payloads = [canonical_json_bytes(record) for record in records]
    return b"\n".join(payloads) + b"\n"


def _load_exact_replay(
    store: ArtifactStore,
    expected: RAGIndex,
    transitions: tuple[RAGTransition, ...],
    vectors: tuple[RAGVector, ...],
    files: dict[str, bytes],
) -> PersistedRAGIndex:
    """Reuse a content-identical immutable index after complete verification."""
    loaded = load_rag_index(store, expected.rag_id)
    replay = expected.model_copy(update={"created_at": loaded.index.created_at})
    if loaded.index != replay:
        raise ValueError("existing RAG index differs from replayed real-trace evidence")
    if loaded.transitions != transitions or loaded.vectors != vectors:
        raise ValueError("existing RAG records differ from replayed real-trace evidence")
    for path, payload in files.items():
        if path == RAG_INDEX_PATH:
            continue
        if store.read_bytes(expected.rag_id, path) != payload:
            raise ValueError(f"existing RAG payload {path} differs from replayed evidence")
    return PersistedRAGIndex(loaded.index, loaded.manifest, transitions, vectors)
