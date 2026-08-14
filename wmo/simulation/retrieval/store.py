"""Digest-verified loading of immutable real-trace RAG indexes."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

from pydantic import ValidationError

from wmo.common.core.artifacts import JsonValue, stable_id
from wmo.common.project import (
    ArtifactCorruptionError,
    ArtifactManifest,
    ArtifactStore,
    artifact_input,
)
from wmo.common.traces import Trace, load_trace_dataset
from wmo.simulation.retrieval.contracts import (
    RAG_ARTIFACT_TYPE,
    RAG_INDEX_PATH,
    RAGIndex,
    RAGTransition,
    RAGVector,
)
from wmo.simulation.retrieval.transitions import render_rag_key

_REAL_SOURCE_KINDS = frozenset({"file", "otlp", "production"})


@dataclass(frozen=True)
class LoadedRAGIndex:
    """Verified immutable RAG envelope, manifest, transitions, and vectors."""

    index: RAGIndex
    manifest: ArtifactManifest
    transitions: tuple[RAGTransition, ...]
    vectors: tuple[RAGVector, ...]


def load_rag_index(store: ArtifactStore, rag_id: str) -> LoadedRAGIndex:
    """Load and fully verify one completed immutable RAG artifact.

    Args:
        store: Project-local artifact store that owns the index and its exact inputs.
        rag_id: Immutable RAG artifact identity.

    Returns:
        Verified envelope and exact ordered observed-transition records.

    Raises:
        ArtifactCorruptionError: Any type, hash, dimension, ID, lineage, source, or numeric
            invariant fails closed.
    """
    stored = store.read(rag_id)
    if stored.manifest.artifact_type != RAG_ARTIFACT_TYPE:
        raise ArtifactCorruptionError(f"artifact {rag_id} is not a trace RAG index")
    try:
        index = RAGIndex.model_validate_json(store.read_bytes(rag_id, RAG_INDEX_PATH))
    except (ValidationError, ValueError) as exc:
        raise ArtifactCorruptionError(f"RAG index {rag_id} has an invalid envelope") from exc
    if index.rag_id != rag_id:
        raise ArtifactCorruptionError(
            f"RAG envelope ID {index.rag_id} does not match artifact {rag_id}"
        )
    manifest_identity = (
        stored.manifest.schema_version,
        stored.manifest.created_at,
        stored.manifest.inputs,
        stored.manifest.code_revision,
        stored.manifest.source,
    )
    envelope_identity = (
        index.schema_version,
        index.created_at,
        index.inputs,
        index.code_revision,
        index.source,
    )
    if manifest_identity != envelope_identity:
        raise ArtifactCorruptionError(f"RAG index {rag_id} differs from its artifact manifest")
    transitions_payload = store.read_bytes(rag_id, index.transitions_path)
    vectors_payload = store.read_bytes(rag_id, index.vectors_path)
    if hashlib.sha256(transitions_payload).hexdigest() != index.transitions_sha256:
        raise ArtifactCorruptionError(f"RAG index {rag_id} transition digest does not match")
    if hashlib.sha256(vectors_payload).hexdigest() != index.vectors_sha256:
        raise ArtifactCorruptionError(f"RAG index {rag_id} vector digest does not match")
    transitions = _parse_transitions(rag_id, transitions_payload)
    vectors = _parse_vectors(rag_id, vectors_payload)
    _verify_records(index, transitions, vectors)
    _verify_sources(store, index, transitions)
    return LoadedRAGIndex(index, stored.manifest, transitions, vectors)


def _parse_transitions(rag_id: str, payload: bytes) -> tuple[RAGTransition, ...]:
    """Parse all transition JSONL records with strict Pydantic contracts."""
    try:
        return tuple(
            RAGTransition.model_validate_json(line)
            for line in payload.decode("utf-8").splitlines()
            if line
        )
    except (UnicodeDecodeError, ValidationError, ValueError) as exc:
        raise ArtifactCorruptionError(f"RAG index {rag_id} has invalid transitions") from exc


def _parse_vectors(rag_id: str, payload: bytes) -> tuple[RAGVector, ...]:
    """Parse all vector JSONL records and reject NaN, infinity, and non-unit values."""
    try:
        return tuple(
            RAGVector.model_validate_json(line)
            for line in payload.decode("utf-8").splitlines()
            if line
        )
    except (UnicodeDecodeError, ValidationError, ValueError) as exc:
        raise ArtifactCorruptionError(f"RAG index {rag_id} has invalid vectors") from exc


def _verify_records(
    index: RAGIndex,
    transitions: tuple[RAGTransition, ...],
    vectors: tuple[RAGVector, ...],
) -> None:
    """Verify ordering, IDs, key hashes, lineage membership, and vector dimensions."""
    transition_ids = tuple(item.transition_id for item in transitions)
    vector_ids = tuple(item.transition_id for item in vectors)
    if transition_ids != index.transition_ids or vector_ids != index.transition_ids:
        raise ArtifactCorruptionError("RAG records do not match ordered transition IDs")
    if len(transitions) != index.transition_count or len(vectors) != index.transition_count:
        raise ArtifactCorruptionError("RAG record counts do not match the envelope")
    included_lineages = set(index.included_lineage_ids or index.fit_lineage_ids)
    for transition, vector in zip(transitions, vectors, strict=True):
        if transition.lineage_id not in included_lineages:
            raise ArtifactCorruptionError("RAG transition is outside the frozen included lineages")
        key_text = render_rag_key(
            task=transition.task,
            initial_context=transition.initial_context,
            action=transition.action,
        )
        if key_text != transition.key_text:
            raise ArtifactCorruptionError("RAG transition key differs from its versioned fields")
        if hashlib.sha256(key_text.encode("utf-8")).hexdigest() != transition.key_sha256:
            raise ArtifactCorruptionError("RAG transition key hash does not match")
        material: JsonValue = {
            "action": transition.action.model_dump(mode="json", exclude_none=False),
            "action_span_id": transition.action_span_id,
            "key_schema_version": index.key_schema_version,
            "lineage_id": transition.lineage_id,
            "observation": transition.observation.model_dump(mode="json"),
            "observation_span_id": transition.observation_span_id,
            "trace_id": transition.trace_id,
        }
        if stable_id("rag-transition", material) != transition.transition_id:
            raise ArtifactCorruptionError("RAG transition ID does not match its observed evidence")
        if len(vector.values) != index.embedding_dimension:
            raise ArtifactCorruptionError("RAG vector dimension does not match the envelope")


def _verify_sources(
    store: ArtifactStore,
    index: RAGIndex,
    transitions: tuple[RAGTransition, ...],
) -> None:
    """Reopen every exact real source and bind transitions to its trace and span identities."""
    traces: dict[str, Trace] = {}
    for source in index.sources:
        stored = store.read(source.artifact_input.artifact_id)
        if artifact_input(stored.manifest) != source.artifact_input:
            raise ArtifactCorruptionError("RAG source manifest digest no longer matches")
        if stored.manifest.artifact_type != "trace-dataset" or source.kind != "trace_dataset":
            raise ArtifactCorruptionError("RAG source is not a supported real trace dataset")
        loaded = load_trace_dataset(store, source.artifact_input.artifact_id)
        if (
            loaded.dataset.source != source.source
            or loaded.dataset.traces_sha256 != source.records_sha256
            or tuple(sorted(loaded.dataset.trace_ids)) != source.trace_ids
        ):
            raise ArtifactCorruptionError("RAG source reference differs from its trace dataset")
        if source.source.kind not in _REAL_SOURCE_KINDS:
            raise ArtifactCorruptionError("RAG source provenance is not verified real evidence")
        for trace in loaded.traces:
            if trace.trace_id in traces:
                raise ArtifactCorruptionError("RAG sources repeat a trace ID")
            traces[trace.trace_id] = trace
    for transition in transitions:
        trace = traces.get(transition.trace_id)
        if trace is None:
            raise ArtifactCorruptionError("RAG transition names an unknown source trace")
        if (
            transition.task != trace.task
            or transition.initial_context != trace.initial_context
            or transition.conversation_id != trace.conversation_id
        ):
            raise ArtifactCorruptionError("RAG transition differs from its source trace fields")
        spans = {span.span_id: span for span in trace.spans}
        action_span = spans.get(transition.action_span_id)
        observation_span = spans.get(transition.observation_span_id)
        if action_span is None or observation_span is None:
            raise ArtifactCorruptionError("RAG transition names an unknown source span")
        if observation_span.started_at < action_span.ended_at:
            raise ArtifactCorruptionError("RAG observation precedes its source action")
