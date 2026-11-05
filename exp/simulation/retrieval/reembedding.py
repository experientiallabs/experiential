"""Provider-free inventories for re-embedding a verified immutable retrieval graph."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

from pydantic import Field

from exp.common.core.artifacts import ArtifactInput, ContractModel, sha256_json
from exp.common.models import ModelSnapshot
from exp.common.project import artifact_input
from exp.simulation.retrieval.embedding_inputs import (
    MAXIMUM_BATCH_BYTES,
    MAXIMUM_BATCH_INPUTS,
    embedding_batches,
    plan_rag_embedding_inputs,
    validate_batch_bounds,
)
from exp.simulation.retrieval.store import LoadedRAGIndex


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
        admission: This inventory never admits a migration or establishes readiness.
    """

    protocol: Literal["rag-reembedding-plan-v1"] = "rag-reembedding-plan-v1"
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
    admission: Literal["not_admitted"] = "not_admitted"


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
