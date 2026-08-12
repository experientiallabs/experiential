"""Tests for the shared request-visible routing feature boundary."""

from wmo.common.models import ModelMessage, ModelRequest
from wmo.common.routing import RouterFeatureExtractor
from wmo.common.tasks import TaskCase, ToolSchema


def test_fit_and_request_features_match_and_exclude_mining_only_fields() -> None:
    """Partition, lineage, weight, trace IDs, and task ID cannot change a router vector."""
    tool = ToolSchema(
        name="lookup",
        description="Look up the customer record.",
        input_schema={"type": "object", "properties": {"customer_id": {"type": "string"}}},
    )
    first = TaskCase(
        task_id="task-fit",
        lineage_group_id="lineage-fit",
        partition="fit",
        instruction="Help the customer.",
        initial_context={"customer_id": "123"},
        tools=(tool,),
        workload_weight=1.0,
        source_trace_ids=("trace-fit",),
    )
    changed_mining_metadata = TaskCase(
        task_id="task-held-out",
        lineage_group_id="lineage-held-out",
        partition="held_out",
        instruction=first.instruction,
        initial_context=first.initial_context,
        tools=first.tools,
        workload_weight=99.0,
        source_trace_ids=("trace-other",),
    )
    extractor = RouterFeatureExtractor()
    fit_text = extractor.from_task(first, allowed_tags={"tenant_tier": "standard"})
    changed_text = extractor.from_task(
        changed_mining_metadata, allowed_tags={"tenant_tier": "standard"}
    )
    request_text = extractor.from_request(
        ModelRequest(
            messages=(ModelMessage(role="user", content=first.instruction),),
            tools=(tool,),
        ),
        initial_context=first.initial_context,
        allowed_tags={"tenant_tier": "standard"},
    )

    assert fit_text == changed_text == request_text
    assert "task-held-out" not in fit_text
    assert "lineage-held-out" not in fit_text
    assert "trace-other" not in fit_text


def test_later_assistant_tool_and_user_history_cannot_change_request_features() -> None:
    """Only the first user intent and pre-response request schema reach the router."""
    extractor = RouterFeatureExtractor()
    initial = ModelRequest(messages=(ModelMessage(role="user", content="Initial intent"),))
    continued = ModelRequest(
        messages=(
            *initial.messages,
            ModelMessage(role="assistant", content="candidate response"),
            ModelMessage(role="tool", content="secret result", tool_call_id="call-1"),
            ModelMessage(role="user", content="outcome-bearing follow-up"),
        )
    )

    assert extractor.from_request(initial) == extractor.from_request(continued)
