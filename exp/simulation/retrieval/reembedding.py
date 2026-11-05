"""Provider-free inventories for re-embedding a verified immutable retrieval graph."""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

from pydantic import Field

from exp.common.core.artifacts import (
    ArtifactInput,
    ContractModel,
    envelope_matches_manifest,
    sha256_json,
)
from exp.common.models import ModelSnapshot
from exp.common.project import (
    ArtifactCorruptionError,
    ArtifactManifest,
    ArtifactStore,
    artifact_input,
)
from exp.simulation.retrieval.contracts import (
    RAG_ARTIFACT_TYPE,
    RAG_INDEX_PATH,
    RAGIndex,
    RAGTransition,
)
from exp.simulation.retrieval.embedding_inputs import (
    MAXIMUM_BATCH_BYTES,
    MAXIMUM_BATCH_INPUTS,
    embedding_batches,
    plan_rag_embedding_inputs,
    validate_batch_bounds,
)
from exp.simulation.retrieval.store import LoadedRAGIndex, _parse_transitions
from exp.simulation.retrieval.transitions import render_rag_key


class ReembeddingNode(ContractModel):
    """One unchanged evidence membership requiring a new vector artifact.

    Attributes:
        source: Exact original index manifest, never a proposed replacement.
        transitions: Number of observed transitions retained by this node.
        included_partitions: Original partition membership.
        corpus_scope_sha256: Original explicit scope digest, or None when unscoped.
    """

    source: ArtifactInput
    transitions: int = Field(ge=1)
    included_partitions: tuple[str, ...]
    corpus_scope_sha256: str | None


class RAGReembeddingPlan(ContractModel):
    """Cold-cache physical input inventory, with no execution or spend authority.

    Attributes:
        protocol: Version of this inventory contract.
        original_embedder: Old representation; none of its vectors count as target cache hits.
        indexes: Exact original memberships to rebuild.
        maximum_chunk_bytes: Unchanged build/query chunk semantics.
        unique_inputs: Distinct inputs across all memberships.
        conservative_input_tokens: Native UTF-8 upper bound across those inputs.
        input_digest: Canonical ordered input digest, without exposing source text.
        maximum_batch_inputs: Requested provider input-count bound.
        maximum_batch_bytes: Requested provider UTF-8 batch bound.
        batches: Native request-batch count before transport attempts.
        attempts_per_batch: Explicit proposed maximum physical transport attempts.
        physical_attempt_bound: Batches multiplied by attempts, not observed spend.
        input_verification: Distinguishes decoded old indexes from transition-only inventories.
        admission: This inventory never admits a migration or establishes readiness.
    """

    protocol: Literal["rag-reembedding-plan-v2"] = "rag-reembedding-plan-v2"
    original_embedder: ModelSnapshot
    indexes: tuple[ReembeddingNode, ...]
    maximum_chunk_bytes: int = Field(ge=4, le=2048)
    unique_inputs: int = Field(ge=1)
    conservative_input_tokens: int = Field(ge=1)
    input_digest: str
    maximum_batch_inputs: int = Field(ge=1)
    maximum_batch_bytes: int = Field(ge=2048)
    batches: int = Field(ge=1)
    attempts_per_batch: int = Field(ge=1)
    physical_attempt_bound: int = Field(ge=1)
    input_verification: Literal["fully_loaded_indexes", "manifest_bound_transition_inputs"]
    admission: Literal["not_admitted"] = "not_admitted"


@dataclass(frozen=True)
class RAGReembeddingInput:
    """Manifest-bound original inputs for costing, without qualifying old vector values.

    Attributes:
        index: Original immutable index envelope.
        manifest: Exact original artifact manifest.
        transitions: Digest-bound action/result records; no new source evidence is authored.
    """

    index: RAGIndex
    manifest: ArtifactManifest
    transitions: tuple[RAGTransition, ...]


def load_reembedding_input(
    store: ArtifactStore,
    reference: ArtifactInput,
) -> RAGReembeddingInput:
    """Read a saved transition inventory while leaving full vector/source qualification separate.

    Artifact payload hashes, envelope, transition membership and canonical query keys are checked.
    Old vector bytes are hash-checked but not numerically decoded, and source trace payloads are
    not recursively requalified. This is only a physical-input inventory, never execution consent.

    Args:
        store: Immutable owner of the original index.
        reference: Exact reviewed original manifest pointer.

    Returns:
        Original records for provider-free cold-input planning.

    Raises:
        ArtifactCorruptionError: Reference, payload, envelope, membership or query-key checks fail.
    """
    stored = store.read(reference.artifact_id)
    if (
        artifact_input(stored.manifest) != reference
        or stored.manifest.artifact_type != RAG_ARTIFACT_TYPE
    ):
        raise ArtifactCorruptionError("re-embedding input differs from its index reference")
    index = RAGIndex.model_validate_json(stored.payloads[RAG_INDEX_PATH])
    if index.rag_id != reference.artifact_id or not envelope_matches_manifest(
        index, stored.manifest
    ):
        raise ArtifactCorruptionError("re-embedding input envelope differs from its manifest")
    transitions_payload = stored.payloads[index.transitions_path]
    if (
        hashlib.sha256(transitions_payload).hexdigest() != index.transitions_sha256
        or hashlib.sha256(stored.payloads[index.vectors_path]).hexdigest() != index.vectors_sha256
    ):
        raise ArtifactCorruptionError("re-embedding input payload differs from its envelope")
    transitions = _parse_transitions(index.rag_id, transitions_payload)
    if (
        tuple(row.transition_id for row in transitions) != index.transition_ids
        or len(transitions) != index.transition_count
    ):
        raise ArtifactCorruptionError("re-embedding transition membership differs from envelope")
    traces = {trace for source in index.sources for trace in source.trace_ids}
    lineages = set(index.included_lineage_ids or index.fit_lineage_ids)
    for row in transitions:
        if row.trace_id not in traces or row.lineage_id not in lineages:
            raise ArtifactCorruptionError("re-embedding input names undeclared source evidence")
        if index.corpus_scope is not None and not index.corpus_scope.admits(
            trace_id=row.trace_id,
            lineage_id=row.lineage_id,
        ):
            raise ArtifactCorruptionError("re-embedding input lies outside its original scope")
        key = render_rag_key(task=row.task, initial_context=row.initial_context, action=row.action)
        if row.key_text != key or row.key_sha256 != hashlib.sha256(key.encode()).hexdigest():
            raise ArtifactCorruptionError("re-embedding input has a noncanonical query key")
    return RAGReembeddingInput(index=index, manifest=stored.manifest, transitions=transitions)


def plan_rag_reembedding(
    indexes: Sequence[LoadedRAGIndex],
    *,
    maximum_batch_inputs: int = MAXIMUM_BATCH_INPUTS,
    maximum_batch_bytes: int = MAXIMUM_BATCH_BYTES,
    attempts_per_batch: int = 1,
) -> RAGReembeddingPlan:
    """Count new-representation inputs for one verified serving/fit/scoped graph.

    Uses native key decomposition and lossless batching. Supplied indexes must already have
    passed load_rag_index. One full serving root and only its exact fit-only descendants are
    accepted; this function neither creates replacement indexes nor binds a target model.

    Args:
        indexes: Fully verified original serving root and any required fit/scoped descendants.
        maximum_batch_inputs: Proposed input-count bound for each native embedding request.
        maximum_batch_bytes: Proposed UTF-8 input-byte bound per request.
        attempts_per_batch: Explicit finite transport attempt allowance for each request.

    Returns:
        A cold target-cache inventory. Prices, target readiness, and original-operation
        migration admission require separate native contracts.

    Raises:
        ValueError: The graph mixes roots, representations, evidence, scopes or invalid bounds.
    """
    return _plan(
        tuple(RAGReembeddingInput(row.index, row.manifest, row.transitions) for row in indexes),
        maximum_batch_inputs=maximum_batch_inputs,
        maximum_batch_bytes=maximum_batch_bytes,
        attempts_per_batch=attempts_per_batch,
        input_verification="fully_loaded_indexes",
    )


def plan_rag_reembedding_inputs(
    inputs: Sequence[RAGReembeddingInput],
    *,
    maximum_batch_inputs: int = MAXIMUM_BATCH_INPUTS,
    maximum_batch_bytes: int = MAXIMUM_BATCH_BYTES,
    attempts_per_batch: int = 1,
) -> RAGReembeddingPlan:
    """Inventory manifest-bound transitions without claiming full source/vector qualification.

    Args:
        inputs: Original records from load_reembedding_input, not newly generated evidence.
        maximum_batch_inputs: Proposed native input-count bound per request.
        maximum_batch_bytes: Proposed native UTF-8 byte bound per request.
        attempts_per_batch: Explicit finite physical-attempt bound.

    Returns:
        An unadmitted cost inventory requiring separate full migration/readiness validation.
    """
    return _plan(
        inputs,
        maximum_batch_inputs=maximum_batch_inputs,
        maximum_batch_bytes=maximum_batch_bytes,
        attempts_per_batch=attempts_per_batch,
        input_verification="manifest_bound_transition_inputs",
    )


def _plan(
    indexes: Sequence[RAGReembeddingInput],
    *,
    maximum_batch_inputs: int,
    maximum_batch_bytes: int,
    attempts_per_batch: int,
    input_verification: Literal["fully_loaded_indexes", "manifest_bound_transition_inputs"],
) -> RAGReembeddingPlan:
    """Apply the shared original-graph checks and native cold input/batch planner."""
    validate_batch_bounds(maximum_inputs=maximum_batch_inputs, maximum_bytes=maximum_batch_bytes)
    if isinstance(attempts_per_batch, bool) or attempts_per_batch < 1:
        raise ValueError("re-embedding requires a positive physical attempt allowance")
    ordered = tuple(sorted(indexes, key=lambda row: row.index.rag_id))
    if not ordered or len({row.index.rag_id for row in ordered}) != len(ordered):
        raise ValueError("re-embedding requires distinct verified indexes")
    roots = tuple(
        row
        for row in ordered
        if row.index.included_partitions == ("fit", "held_out") and row.index.corpus_scope is None
    )
    if len(roots) != 1:
        raise ValueError("re-embedding requires exactly one full original serving root")
    root = roots[0]
    originals = {row.transition_id: row for row in root.transitions}
    for loaded in ordered:
        index = loaded.index
        if (
            index.sources != root.index.sources
            or index.embedder != root.index.embedder
            or index.embedding_dimension != root.index.embedding_dimension
            or index.default_top_k != root.index.default_top_k
            or index.embedding_chunk_bytes != root.index.embedding_chunk_bytes
            or index.key_schema_version != root.index.key_schema_version
        ):
            raise ValueError("re-embedding indexes must share the original serving representation")
        if loaded is not root and (
            index.included_partitions != ("fit",)
            or not set(index.fit_lineage_ids) <= set(root.index.fit_lineage_ids)
        ):
            raise ValueError("re-embedding descendants must remain within original fit evidence")
        for transition in loaded.transitions:
            if originals.get(transition.transition_id) != transition:
                raise ValueError("re-embedding cannot change or add original transition evidence")
    inputs = plan_rag_embedding_inputs(
        tuple(row.key_text for row in root.transitions),
        maximum_chunk_bytes=root.index.embedding_chunk_bytes,
    )
    batches = sum(
        1
        for _ in embedding_batches(
            inputs.texts,
            maximum_inputs=maximum_batch_inputs,
            maximum_bytes=maximum_batch_bytes,
        )
    )
    return RAGReembeddingPlan(
        input_verification=input_verification,
        original_embedder=root.index.embedder,
        indexes=tuple(
            ReembeddingNode(
                source=artifact_input(loaded.manifest),
                transitions=loaded.index.transition_count,
                included_partitions=loaded.index.included_partitions,
                corpus_scope_sha256=None
                if loaded.index.corpus_scope is None
                else sha256_json(loaded.index.corpus_scope),
            )
            for loaded in ordered
        ),
        maximum_chunk_bytes=root.index.embedding_chunk_bytes,
        unique_inputs=len(inputs.texts),
        conservative_input_tokens=inputs.maximum_input_tokens,
        input_digest=sha256_json(list(inputs.texts)),
        maximum_batch_inputs=maximum_batch_inputs,
        maximum_batch_bytes=maximum_batch_bytes,
        batches=batches,
        attempts_per_batch=attempts_per_batch,
        physical_attempt_bound=batches * attempts_per_batch,
    )
