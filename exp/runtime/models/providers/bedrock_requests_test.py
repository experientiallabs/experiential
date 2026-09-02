"""Tests for the shared native Bedrock Converse payload builders."""

from typing import cast

from exp.common.core.artifacts import JsonObject
from exp.common.models import AssistantAction, ModelMessage, ModelRequest, ToolCall, ToolChoice
from exp.common.models.content import DocumentContentPart, TextContentPart
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
    inference_config = cast("dict[str, object]", payload["inferenceConfig"])
    assert inference_config["topP"] == 0.8
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


def test_converse_request_adds_stop_schema_and_strict_tool_fields() -> None:
    """Shared Converse builders use AWS's exact structured generation fields."""
    schema: JsonObject = {
        "type": "object",
        "properties": {"answer": {"type": "string"}},
    }

    payload = converse_request(
        "model",
        _tool_transcript_request(),
        stop_sequences=("DONE",),
        structured_output_name="answer",
        structured_output_description="Return one answer.",
        structured_output_schema=schema,
        strict_tool_names=("create_ticket",),
    )

    inference_config = cast("dict[str, object]", payload["inferenceConfig"])
    assert inference_config["stopSequences"] == ["DONE"]
    tool_config = cast("dict[str, object]", payload["toolConfig"])
    tools = cast("list[dict[str, object]]", tool_config["tools"])
    tool_spec = cast("dict[str, object]", tools[0]["toolSpec"])
    assert tool_spec["strict"] is True
    assert payload["outputConfig"] == {
        "textFormat": {
            "type": "json_schema",
            "structure": {
                "jsonSchema": {
                    "schema": '{"properties":{"answer":{"type":"string"}},"type":"object"}',
                    "name": "answer",
                    "description": "Return one answer.",
                }
            },
        }
    }


def test_converse_request_emits_named_document_blocks_in_caller_order() -> None:
    """PDF parts become ``document`` blocks with per-turn ordinal names when unnamed."""
    pdf = "JVBERi0xLjQKJSBtaW5pbWFsIHBkZgo="
    request = ModelRequest(
        messages=(
            ModelMessage(
                role="user",
                content="compare these",
                content_parts=(
                    DocumentContentPart(data=pdf, name="Report (Q3).pdf"),
                    TextContentPart(text="compare these"),
                    DocumentContentPart(data="JVBERi0xLjcK"),
                ),
            ),
        ),
    )
    payload = converse_request("anthropic.claude-fixture", request)
    messages = cast(list[JsonObject], payload["messages"])
    assert messages[0]["content"] == [
        {
            "document": {
                "name": "Report (Q3)-pdf",
                "format": "pdf",
                "source": {"bytes": pdf},
            }
        },
        {"text": "compare these"},
        {
            "document": {
                "name": "document-2",
                "format": "pdf",
                "source": {"bytes": "JVBERi0xLjcK"},
            }
        },
    ]
