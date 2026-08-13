"""Request-time capability eligibility for the frozen router runtime."""

import pytest

from wmo.common.models import AssistantAction, ModelMessage, ModelRequest, ToolCall
from wmo.runtime.router import RouterModelCapabilityError
from wmo.runtime.router.runtime_test import _request, _runtime


def test_selected_model_must_prove_tool_capability_before_dispatch() -> None:
    """A tool-bearing request never reaches an incapable selected model."""
    runtime, client = _runtime(candidate_tools=False)

    with pytest.raises(RouterModelCapabilityError, match="does not support tool calls"):
        runtime.complete(_request(tool_name="read"), episode_id="tool-episode")

    assert client.complete_calls == 0


def test_selected_model_must_prove_output_capacity_before_dispatch() -> None:
    """An explicit output limit never reaches a model with unknown capacity."""
    runtime, client = _runtime()
    request = ModelRequest(
        messages=(ModelMessage(role="user", content="route me"),),
        maximum_output_tokens=100,
    )

    with pytest.raises(RouterModelCapabilityError, match="output-token capacity"):
        runtime.complete(request, episode_id="capacity-episode")

    assert client.complete_calls == 0


def test_replayed_tool_history_requires_tool_capability() -> None:
    """A tool result cannot bypass eligibility by omitting current tool definitions."""
    runtime, client = _runtime(candidate_tools=False)
    request = ModelRequest(
        messages=(
            ModelMessage(role="user", content="read it"),
            ModelMessage(
                role="assistant",
                assistant_action=AssistantAction(
                    tool_calls=(ToolCall(call_id="call-a", name="read", arguments={"path": "a"}),)
                ),
            ),
            ModelMessage(role="tool", content="done", tool_call_id="call-a"),
        )
    )

    with pytest.raises(RouterModelCapabilityError, match="does not support tool calls"):
        runtime.complete(request, episode_id="history-episode")

    assert client.complete_calls == 0
