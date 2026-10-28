"""Tests for request descriptors and the similarity input that mining embeds."""

from __future__ import annotations

from collections.abc import Sequence

from exp.common.tasks import ToolSchema
from exp.common.traces.ingest.chat_json_test import _inline_history, _two_request_trace
from exp.simulation.mining.descriptors import (
    RoutingDescriptor,
    coverage_descriptor,
    normalized_vectors,
    routing_descriptor,
)


def _descriptor(*tools: str) -> RoutingDescriptor:
    """One request declaring the named tools."""
    return RoutingDescriptor(
        trace_id="trace-1",
        intent="Cancel order #W1001",
        initial_context={"task_lineage": "task-1"},
        tools=tuple(
            ToolSchema(name=name, description=f"Run {name}.", input_schema={"type": "object"})
            for name in tools
        ),
        tags=("domain:retail",),
    )


class RecordingEmbedder:
    """Records the texts it embeds and returns one fixed vector per text.

    Attributes:
        texts: Every text embedded so far, in order.
    """

    def __init__(self) -> None:
        """Start with no recorded text."""
        self.texts: list[str] = []

    def embed(self, texts: Sequence[str]) -> tuple[tuple[float, ...], ...]:
        """Record the texts and return a unit vector for each."""
        self.texts.extend(texts)
        return tuple((1.0, 0.0) for _ in texts)


def test_similarity_text_compares_task_content_without_the_tool_catalogue() -> None:
    shared, other = _descriptor("get_order", "cancel_order"), _descriptor("escalate")

    assert shared.similarity_text() == other.similarity_text()
    assert "get_order" not in shared.similarity_text()
    assert "Cancel order #W1001" in shared.similarity_text()
    assert shared.similarity_text() == "Cancel order #W1001"
    assert "task-1" not in shared.similarity_text()
    assert "domain:retail" not in shared.similarity_text()


def test_exact_duplicates_still_require_the_same_declared_tools() -> None:
    assert _descriptor("get_order").fingerprint() != _descriptor("escalate").fingerprint()
    assert "get_order" in _descriptor("get_order").embedding_text()


def test_mining_vectors_embed_the_similarity_text() -> None:
    embedder = RecordingEmbedder()
    descriptor = _descriptor("get_order")

    normalized_vectors(embedder, (descriptor,))

    assert embedder.texts == [descriptor.similarity_text()]


def test_descriptors_match_for_shared_and_inline_history() -> None:
    """Storing chat history once leaves routing and coverage descriptors unchanged."""
    trace = _two_request_trace()
    inline = _inline_history(trace)

    assert routing_descriptor(trace) == routing_descriptor(inline)
    assert coverage_descriptor(trace) == coverage_descriptor(inline)
