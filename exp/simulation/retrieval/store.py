"""Digest-verified loading of immutable real-trace RAG indexes."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from pydantic import ValidationError

from exp.common.core.artifacts import JsonValue, envelope_matches_manifest, stable_id
from exp.common.project import (
    ArtifactCorruptionError,
    ArtifactManifest,
    ArtifactStore,
    artifact_input,
)
from exp.common.traces import Trace, load_trace_dataset
from exp.simulation.retrieval.contracts import (
    RAG_ARTIFACT_TYPE,
    RAG_INDEX_PATH,
    RAGIndex,
    RAGTransition,
    RAGVector,
    RealTraceSourceIdentity,
)
from exp.simulation.retrieval.tags import verify_index_tags
from exp.simulation.retrieval.transitions import render_rag_key


@dataclass(frozen=True, eq=False)
class LoadedRAGIndex:
    """Verified immutable RAG envelope, manifest, transitions, and vector matrix.

    Vectors are held as one read-only float64 matrix with a row per transition, in transition
    order. Float64 keeps every persisted JSON value bit-exact, so retrieval can reproduce the
    exact persisted cosine scores and :attr:`vectors` can rebuild the persisted records. Equality
    is identity: two loads are compared through their envelopes, not their matrices.

    Attributes:
        index: Verified immutable envelope.
        manifest: Manifest the envelope was verified against.
        transitions: Ordered observed-transition records.
        matrix: Read-only ``(transition_count, embedding_dimension)`` float64 unit vectors.
    """

    index: RAGIndex
    manifest: ArtifactManifest
    transitions: tuple[RAGTransition, ...]
    matrix: npt.NDArray[np.float64]

    @property
    def vectors(self) -> tuple[RAGVector, ...]:
        """Rebuild the exact persisted vector records from the matrix.

        This materializes every vector as Python floats. Retrieval never calls it; it serves
        verification paths that compare complete records.

        Returns:
            One record per transition, equal to the persisted JSONL records.
        """
        return tuple(
            RAGVector(transition_id=transition.transition_id, values=tuple(row.tolist()))
            for transition, row in zip(self.transitions, self.matrix, strict=True)
        )


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
        raise ArtifactCorruptionError(
            f"RAG index {rag_id} has an invalid or unsupported envelope; "
            "rebuild the project's indexes"
        ) from exc
    if index.rag_id != rag_id:
        raise ArtifactCorruptionError(
            f"RAG envelope ID {index.rag_id} does not match artifact {rag_id}"
        )
    if not envelope_matches_manifest(index, stored.manifest):
        raise ArtifactCorruptionError(f"RAG index {rag_id} differs from its artifact manifest")
    transitions_payload = store.read_bytes(rag_id, index.transitions_path)
    vectors_payload = store.read_bytes(rag_id, index.vectors_path)
    if hashlib.sha256(transitions_payload).hexdigest() != index.transitions_sha256:
        raise ArtifactCorruptionError(f"RAG index {rag_id} transition digest does not match")
    if hashlib.sha256(vectors_payload).hexdigest() != index.vectors_sha256:
        raise ArtifactCorruptionError(f"RAG index {rag_id} vector digest does not match")
    transitions = _parse_transitions(rag_id, transitions_payload)
    vector_ids, matrix = _parse_vectors(rag_id, vectors_payload, index.embedding_dimension)
    _verify_records(index, transitions, vector_ids)
    _verify_sources(store, index, transitions)
    return LoadedRAGIndex(index, stored.manifest, transitions, matrix)


def _parse_transitions(rag_id: str, payload: bytes) -> tuple[RAGTransition, ...]:
    """Parse all transition JSONL records with strict Pydantic contracts."""
    try:
        return tuple(
            RAGTransition.model_validate_json(line)
            for line in payload.decode("utf-8").split("\n")
            if line
        )
    except (UnicodeDecodeError, ValidationError, ValueError) as exc:
        raise ArtifactCorruptionError(f"RAG index {rag_id} has invalid transitions") from exc


def _parse_vectors(
    rag_id: str, payload: bytes, dimension: int
) -> tuple[tuple[str, ...], npt.NDArray[np.float64]]:
    """Parse vector JSONL into ordered IDs and one read-only float64 matrix.

    Each record is validated by the strict :class:`RAGVector` contract (finite unit values),
    copied into its matrix row, and discarded, so peak memory is the matrix plus one record.

    Args:
        rag_id: Index identity used in error messages.
        payload: Digest-verified vector JSONL bytes.
        dimension: Envelope embedding dimension every record must have.

    Returns:
        Transition IDs in record order and the matching ``(records, dimension)`` matrix.

    Raises:
        ArtifactCorruptionError: A record is invalid or its dimension differs from the envelope.
    """
    try:
        lines = [line for line in payload.decode("utf-8").split("\n") if line]
    except UnicodeDecodeError as exc:
        raise ArtifactCorruptionError(f"RAG index {rag_id} has invalid vectors") from exc
    matrix = np.empty((len(lines), dimension), dtype=np.float64)
    ids: list[str] = []
    for row, line in enumerate(lines):
        try:
            record = RAGVector.model_validate_json(line)
        except (ValidationError, ValueError) as exc:
            raise ArtifactCorruptionError(f"RAG index {rag_id} has invalid vectors") from exc
        if len(record.values) != dimension:
            raise ArtifactCorruptionError("RAG vector dimension does not match the envelope")
        matrix[row] = record.values
        ids.append(record.transition_id)
    matrix.flags.writeable = False
    return tuple(ids), matrix


def _verify_records(
    index: RAGIndex,
    transitions: tuple[RAGTransition, ...],
    vector_ids: tuple[str, ...],
) -> None:
    """Verify ordering, IDs, key hashes, lineage, scope and tag coverage.

    Vector dimensions are verified while the vectors are parsed.

    Args:
        index: Persisted immutable RAG envelope.
        transitions: Parsed ordered transition records.
        vector_ids: Transition IDs of the parsed vectors, in record order.

    Raises:
        ArtifactCorruptionError: Any record contradicts the frozen index identity.
    """
    transition_ids = tuple(item.transition_id for item in transitions)
    if transition_ids != index.transition_ids or vector_ids != index.transition_ids:
        raise ArtifactCorruptionError("RAG records do not match ordered transition IDs")
    if len(transitions) != index.transition_count or len(vector_ids) != index.transition_count:
        raise ArtifactCorruptionError("RAG record counts do not match the envelope")
    included_lineages = set(index.included_lineage_ids or index.fit_lineage_ids)
    for transition in transitions:
        if transition.lineage_id not in included_lineages:
            raise ArtifactCorruptionError("RAG transition is outside the frozen included lineages")
        if index.corpus_scope is not None and not index.corpus_scope.admits(
            trace_id=transition.trace_id, lineage_id=transition.lineage_id
        ):
            raise ArtifactCorruptionError("RAG transition is outside the index's corpus scope")
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
    if index.trace_tags is not None:
        try:
            verify_index_tags(index.trace_tags, (item.trace_id for item in transitions))
        except ValueError as exc:
            raise ArtifactCorruptionError(f"RAG index tag record is inconsistent: {exc}") from exc


def _verify_sources(
    store: ArtifactStore,
    index: RAGIndex,
    transitions: tuple[RAGTransition, ...],
) -> None:
    """Reopen exact real sources and bind transitions to trace and span identities.

    Args:
        store: Project artifact store containing the index sources.
        index: Loaded retrieval envelope with exact source pointers.
        transitions: Parsed transition records that must trace back to source spans.

    Raises:
        ArtifactCorruptionError: A source, trace, transition field, span, order, or typed
            provenance differs from the persisted retrieval contract.
    """
    traces: dict[str, Trace] = {}
    for source in index.sources:
        stored = store.read(source.artifact_input.artifact_id)
        if artifact_input(stored.manifest) != source.artifact_input:
            raise ArtifactCorruptionError("RAG source manifest digest no longer matches")
        if stored.manifest.artifact_type != "trace-dataset" or source.kind != "trace_dataset":
            raise ArtifactCorruptionError("RAG source is not a supported real trace dataset")
        loaded = load_trace_dataset(store, source.artifact_input.artifact_id)
        if (
            loaded.dataset.source is None
            or loaded.dataset.source.model_dump(mode="json")
            != source.source.model_dump(mode="json")
            or loaded.dataset.traces_sha256 != source.records_sha256
            or tuple(sorted(loaded.dataset.trace_ids)) != source.trace_ids
        ):
            raise ArtifactCorruptionError("RAG source reference differs from its trace dataset")
        for trace in loaded.traces:
            try:
                RealTraceSourceIdentity.model_validate(
                    trace.source.identity.model_dump(mode="json")
                )
            except ValueError as exc:
                raise ArtifactCorruptionError(
                    "RAG source trace provenance is not verified real evidence"
                ) from exc
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
