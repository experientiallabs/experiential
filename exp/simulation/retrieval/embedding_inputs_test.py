"""Regression coverage for bounded, lossless, shared RAG embedding inputs."""

import json

import pytest

from exp.common.core.artifacts import canonical_json_bytes
from exp.simulation.retrieval.contracts import RAGAction
from exp.simulation.retrieval.embedding_inputs import (
    MAXIMUM_BATCH_BYTES,
    MAXIMUM_BATCH_INPUTS,
    embedding_batches,
    embedding_chunk_bytes,
    plan_rag_embedding_inputs,
)
from exp.simulation.retrieval.transitions import render_rag_key


def test_shared_task_and_context_are_embedded_once_without_losing_long_unicode() -> None:
    """Repeated instructions reuse chunks, while oversized fields keep their entire contents."""
    task = "Research these organizations. " + "会社😀\u2028" * 8_000
    context = {
        "instruction_messages": [{"role": "developer", "content": "Carefully verify. " * 900}]
    }
    actions = tuple(
        RAGAction(kind="tool_call", tool_name="search", tool_arguments={"query": f"company {i}"})
        for i in range(10)
    )
    keys = tuple(
        render_rag_key(task=task, initial_context=context, action=action) for action in actions
    )
    plan = plan_rag_embedding_inputs(keys)

    assert all(0 < len(text.encode("utf-8")) <= 2_048 for text in plan.texts)
    assert len(plan.texts) == len(set(plan.texts))
    assert plan.maximum_input_tokens < sum(len(key.encode("utf-8")) for key in keys) / 5
    for action, components in zip(actions, plan.keys, strict=True):
        recovered = {}
        for chunks in components:
            recovered.update(json.loads("".join(chunks)))
        assert recovered == {
            "task": task,
            "initial_context": context,
            "action": action.model_dump(mode="json", exclude_none=False),
        }


def test_batches_bound_count_and_total_bytes_without_omitting_inputs() -> None:
    """A corpus larger than one provider request is sent in complete ordered batches."""
    texts = tuple(f"{index:04d}" + "x" * 2_044 for index in range(2_100))
    batches = tuple(embedding_batches(texts))

    assert len(batches) > 1
    assert tuple(text for batch in batches for text in batch) == texts
    assert all(len(batch) <= MAXIMUM_BATCH_INPUTS for batch in batches)
    assert all(
        sum(len(text.encode("utf-8")) for text in batch) <= MAXIMUM_BATCH_BYTES for batch in batches
    )


def test_old_or_response_bearing_keys_cannot_enter_the_embedding_plan() -> None:
    """The retrieval lookup excludes observations and refuses stale representation semantics."""
    key = json.loads(
        render_rag_key(
            task="task", initial_context={}, action=RAGAction(kind="message", content="act")
        )
    )
    with pytest.raises(ValueError, match="exactly task"):
        plan_rag_embedding_inputs(
            (canonical_json_bytes({**key, "observation": "future"}).decode(),)
        )
    key["key_schema_version"] = "observed-transition-key-v1"
    with pytest.raises(ValueError, match="rebuild"):
        plan_rag_embedding_inputs((canonical_json_bytes(key).decode(),))


@pytest.mark.parametrize("limit", [None, 8_192, 2_048, 64])
def test_chunk_bound_respects_the_configured_context(limit: int | None) -> None:
    """Smaller configured embedding windows reduce chunks without reducing document length."""
    assert embedding_chunk_bytes(limit) == min(limit or 2_048, 2_048)


@pytest.mark.parametrize("limit", [-1, 0, 1, 3])
def test_invalid_context_is_rejected_before_embedding(limit: int) -> None:
    """An invalid limit cannot silently become an unlimited provider input."""
    with pytest.raises(ValueError, match="at least four"):
        embedding_chunk_bytes(limit)
