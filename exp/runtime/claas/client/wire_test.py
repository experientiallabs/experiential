"""Caller-visible message and sampling fidelity for the learning client."""

import pytest

from exp.common.models import AssistantAction, ModelMessage, ModelRequest, ToolCall, ToolChoice
from exp.runtime.claas.client.wire import chat_payload


def test_chat_history_preserves_tool_call_and_observation_identity() -> None:
    """The bridge forwards the scaffold's original complete tool conversation."""
    call = ToolCall(call_id="call-1", name="lookup", arguments={"key": "x"})
    request = ModelRequest(
        messages=(
            ModelMessage(role="user", content="look up x"),
            ModelMessage(role="assistant", assistant_action=AssistantAction(tool_calls=(call,))),
            ModelMessage(role="tool", tool_call_id="call-1", content="result"),
        ),
        maximum_output_tokens=17,
    )
    payload = chat_payload("student", request, maximum_output_tokens=512)
    assert payload["max_tokens"] == 17
    assert payload["messages"] == [
        {"role": "user", "content": "look up x"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "call-1",
                    "type": "function",
                    "function": {"name": "lookup", "arguments": call.arguments_json()},
                }
            ],
        },
        {"role": "tool", "content": "result", "tool_call_id": "call-1"},
    ]


@pytest.mark.parametrize(
    "changes",
    [
        {"temperature": 0.7},
        {"top_p": 0.9},
        {"top_k": 5},
        {"logprobs": True},
        {"reasoning_effort": "low"},
        {"tool_choice": ToolChoice(name="lookup")},
    ],
)
def test_unsupported_controls_fail_before_dispatch(changes: dict[str, object]) -> None:
    """A client never silently strips requested policy controls before training capture."""
    request = ModelRequest(messages=(ModelMessage(role="user", content="task"),)).model_copy(
        update=changes
    )
    with pytest.raises(ValueError, match="learner"):
        chat_payload("student", request, maximum_output_tokens=512)
