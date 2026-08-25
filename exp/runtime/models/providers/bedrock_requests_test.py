"""Tests for the shared native Bedrock Converse payload builders."""

from exp.common.models import AssistantAction, ModelMessage, ModelRequest, ToolCall, ToolChoice
from exp.common.tasks import ToolSchema
from exp.runtime.models.providers.bedrock_requests import converse_request


def _tool_transcript_request() -> ModelRequest:
    """Build a visible transcript containing an earlier tool call and result."""
    return ModelRequest(
        messages=(
            ModelMessage(role="system", content="You are precise."),
            ModelMessage(role="user", content="Create a ticket."),
            ModelMessage(
                role="assistant",
                assistant_action=AssistantAction(
                    tool_calls=(
                        ToolCall(
                            call_id="call-old",
                            name="create_ticket",
                            arguments={"priority": "normal"},
                        ),
                    )
                ),
            ),
            ModelMessage(role="tool", content="created", tool_call_id="call-old"),
        ),
        tools=(
            ToolSchema(
                name="create_ticket",
                description="Create one support ticket.",
                input_schema={"type": "object"},
            ),
        ),
        tool_choice=ToolChoice(name="create_ticket"),
        temperature=0.1,
        maximum_output_tokens=256,
    )


def test_converse_request_preserves_tool_ids_and_named_choice() -> None:
    """Converse keeps exact tool-use IDs and forwards named tool choice."""
    payload = converse_request("us.anthropic.claude-sonnet-4-5", _tool_transcript_request())

    assert payload["modelId"] == "us.anthropic.claude-sonnet-4-5"
    assert payload["system"] == [{"text": "You are precise."}]
    assert payload["inferenceConfig"] == {"maxTokens": 256, "temperature": 0.1}
    tool_config = payload["toolConfig"]
    assert isinstance(tool_config, dict)
    assert tool_config["toolChoice"] == {"tool": {"name": "create_ticket"}}
    messages = payload["messages"]
    assert isinstance(messages, list)
    assistant = messages[1]
    tool_result = messages[2]
    assert isinstance(assistant, dict)
    assert isinstance(tool_result, dict)
    assistant_content = assistant["content"]
    result_content = tool_result["content"]
    assert isinstance(assistant_content, list)
    assert isinstance(result_content, list)
    tool_use = assistant_content[0]
    result_block = result_content[0]
    assert isinstance(tool_use, dict)
    assert isinstance(result_block, dict)
    tool_use_block = tool_use["toolUse"]
    result_payload = result_block["toolResult"]
    assert isinstance(tool_use_block, dict)
    assert isinstance(result_payload, dict)
    assert tool_use_block["toolUseId"] == "call-old"
    assert tool_result["role"] == "user"
    assert result_payload["toolUseId"] == "call-old"


def test_converse_request_gates_top_p_and_model_specific_top_k() -> None:
    """Bedrock keeps top-p standard and puts certified top-k in the model extension map."""
    request = _tool_transcript_request().model_copy(update={"top_p": 0.8, "top_k": 20})
    payload = converse_request(
        "us.anthropic.claude-sonnet-4-5",
        request,
        supports_top_k=True,
    )
    assert payload["inferenceConfig"]["topP"] == 0.8
    assert payload["additionalModelRequestFields"] == {"top_k": 20}


def test_converse_request_omits_tools_when_choice_is_none() -> None:
    """A ``none`` tool choice drops the tool configuration entirely."""
    request = ModelRequest(
        messages=(ModelMessage(role="user", content="hi"),),
        tools=(
            ToolSchema(
                name="lookup",
                description="find",
                input_schema={"type": "object"},
            ),
        ),
        tool_choice="none",
    )
    payload = converse_request("model", request)
    assert "toolConfig" not in payload
