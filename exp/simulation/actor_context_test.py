"""Context provenance describes actually attempted actor requests, including conflicts."""

from pathlib import Path

import pytest

from exp.common.models import ModelMessage, ModelRequest, ModelResponse
from exp.common.tasks import TaskCase, ToolSchema
from exp.simulation.actor_context import (
    ActorContextFragment,
    ContextRecordingClient,
    with_actor_context,
)


class UnavailableClient:
    """Fail after the request reaches the recording boundary."""

    def complete(self, request: ModelRequest) -> ModelResponse:
        """Preserve the failure rather than manufacturing an actor response."""
        raise RuntimeError("fixture unavailable")


def test_manifest_captures_delivered_conflict_and_marks_absent_sources(tmp_path: Path) -> None:
    """Conflicting hooks and rules are located by source without claiming simulator memory."""
    task = TaskCase(
        task_id="t",
        lineage_group_id="owner",
        partition="held_out",
        workload_weight=1,
        source_trace_ids=("fixture",),
        instruction="Read order.",
        tools=(
            ToolSchema(name="read", description="Read an order", input_schema={"type": "object"}),
        ),
    )
    fragments = (
        ActorContextFragment(
            fragment_id="rule",
            source="system_prompt",
            role="system",
            content="Read before answering.",
        ),
        ActorContextFragment(
            fragment_id="hook", source="hook", role="developer", content="Never call read."
        ),
    )
    public = with_actor_context(task, fragments)
    client = ContextRecordingClient(UnavailableClient(), public)
    request = ModelRequest(
        messages=(
            ModelMessage(role="system", content="Read before answering."),
            ModelMessage(role="developer", content="Never call read."),
            ModelMessage(role="user", content="Read order."),
        ),
        tools=task.tools,
    )
    with pytest.raises(RuntimeError, match="fixture unavailable"):
        client.complete(request)
    manifest = client.manifest()
    assert manifest["absent_sources"] == [
        "skill_description",
        "reminder",
        "memory",
        "appended_prompt",
    ]
    rows = manifest["fragments"]
    assert isinstance(rows, list)
    assert {(r["source"], r["content"]) for r in rows if isinstance(r, dict)} >= {
        ("hook", "Never call read."),
        ("system_prompt", "Read before answering."),
    }
    assert task.initial_context == {}
