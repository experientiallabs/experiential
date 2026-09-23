"""Lossless, deduplicated text planning for component-pooled retrieval embeddings.

Task, starting context, and action are embedded independently, so repeated instructions do not
need another provider call for every transition. The versioned key still retains every byte;
chunking changes the embedding representation, never the stored observed evidence.
"""

from __future__ import annotations

import json
from collections.abc import Iterator, Sequence
from dataclasses import dataclass

from exp.common.core.artifacts import canonical_json_bytes
from exp.simulation.retrieval.contracts import RAG_KEY_SCHEMA_VERSION

MAXIMUM_CHUNK_BYTES = 2_048
# UTF-8 bytes conservatively bound text tokens. These batches remain below OpenAI's 300,000
# total-token and 2,048-input limits and Gemini's 100-input batch bound, without guessing
# a tokenizer for other providers.
MAXIMUM_BATCH_BYTES = 262_144
MAXIMUM_BATCH_INPUTS = 100


@dataclass(frozen=True)
class RAGEmbeddingInputPlan:
    """Exact unique provider inputs and their per-key, per-component chunk assignments."""

    texts: tuple[str, ...]
    keys: tuple[tuple[tuple[str, ...], ...], ...]

    @property
    def maximum_input_tokens(self) -> int:
        """Return the conservative token bound for one pass over all unique chunks."""
        return sum(len(text.encode("utf-8")) for text in self.texts)


def embedding_chunk_bytes(maximum_input_tokens: int | None) -> int:
    """Choose a lossless chunk size within the configured embedder's context window."""
    limit = min(
        MAXIMUM_CHUNK_BYTES,
        MAXIMUM_CHUNK_BYTES if maximum_input_tokens is None else maximum_input_tokens,
    )
    if limit < 4:
        raise ValueError("RAG embeddings require a context window of at least four tokens")
    return limit


def plan_rag_embedding_inputs(
    keys: Sequence[str], *, maximum_chunk_bytes: int = MAXIMUM_CHUNK_BYTES
) -> RAGEmbeddingInputPlan:
    """Plan bounded UTF-8 chunks once per exact text, preserving all key components.

    Args:
        keys: Complete canonical keys rendered by the current retrieval schema.
        maximum_chunk_bytes: Frozen input chunk bound used for both index and query vectors.

    Returns:
        Unique chunks in first-occurrence order and assignments for reconstructing every key.

    Raises:
        ValueError: Keys do not match the supported schema or the chunk bound is invalid.
    """
    if not 4 <= maximum_chunk_bytes <= MAXIMUM_CHUNK_BYTES:
        raise ValueError("RAG embedding chunk size must be between 4 and 2048 UTF-8 bytes")
    unique: dict[str, None] = {}
    assignments: list[tuple[tuple[str, ...], ...]] = []
    for key in keys:
        components = _key_components(key)
        chunks = tuple(_split_utf8(component, maximum_chunk_bytes) for component in components)
        assignments.append(chunks)
        for component in chunks:
            for text in component:
                unique.setdefault(text, None)
    return RAGEmbeddingInputPlan(texts=tuple(unique), keys=tuple(assignments))


def embedding_batches(texts: Sequence[str]) -> Iterator[tuple[str, ...]]:
    """Batch bounded chunks by both total UTF-8 bytes and number of inputs."""
    batch: list[str] = []
    size = 0
    for text in texts:
        length = len(text.encode("utf-8"))
        if not text or length > MAXIMUM_CHUNK_BYTES:
            raise ValueError("RAG embedding batches require nonempty bounded text chunks")
        if batch and (len(batch) == MAXIMUM_BATCH_INPUTS or size + length > MAXIMUM_BATCH_BYTES):
            yield tuple(batch)
            batch = []
            size = 0
        batch.append(text)
        size += length
    if batch:
        yield tuple(batch)


def _key_components(key: str) -> tuple[str, ...]:
    """Separate complete task, starting context, and action without using the target result."""
    value = json.loads(key)
    if not isinstance(value, dict) or value.get("key_schema_version") != RAG_KEY_SCHEMA_VERSION:
        raise ValueError("unsupported RAG embedding key schema; rebuild the project's indexes")
    if set(value) != {"task", "initial_context", "action", "key_schema_version"}:
        raise ValueError("RAG embedding key must contain exactly task, initial context, and action")
    # Empty context has no semantic content. Complete nonempty fields receive one pooled vote,
    # so a long repeated system prompt cannot outweigh the action merely because it is longer.
    return tuple(
        canonical_json_bytes({field: value[field]}).decode("utf-8")
        for field in ("task", "initial_context", "action")
        if value[field] not in (None, {}, "")
    )


def _split_utf8(text: str, maximum_bytes: int) -> tuple[str, ...]:
    """Split on Unicode character boundaries, retaining every character exactly once."""
    chunks: list[str] = []
    start = 0
    size = 0
    for index, character in enumerate(text):
        length = len(character.encode("utf-8"))
        if size + length > maximum_bytes:
            chunks.append(text[start:index])
            start = index
            size = 0
        size += length
    if start < len(text):
        chunks.append(text[start:])
    return tuple(chunks)
