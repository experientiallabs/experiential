"""Instruction-role emission per OpenAI-family wire; the rest of the builders
are exercised in streaming_requests_test.py."""

from __future__ import annotations

from typing import cast

from exp.common.core.artifacts import JsonObject
from exp.common.models.model import ToolCall
from exp.runtime.gateway.contracts import (
    ExposedReasoningContentBlock,
    GatewayApiSurface,
    GatewayMessage,
    GatewayRequest,
    GatewayToolDefinition,
)
from exp.runtime.models.providers.openai_payloads import (
    openai_compatible_stream_payload,
    openai_responses_stream_payload,
)


def _developer_conversation() -> GatewayRequest:
    """Build one request with a leading and a mid-conversation developer turn."""
    return GatewayRequest(
        surface=GatewayApiSurface.RESPONSES,
        messages=(
            GatewayMessage(role="developer", content="Follow policy."),
            GatewayMessage(role="user", content="hi"),
            GatewayMessage(role="assistant", content="hello"),
            GatewayMessage(role="developer", content="Now be terse."),
            GatewayMessage(role="user", content="go"),
        ),
        stream=True,
        include_usage=True,
    )


def test_chat_wire_emits_developer_instructions_as_system() -> None:
    """The Chat Completions wire folds ``developer`` into ``system`` losslessly.

    OpenAI defines both roles identically (developer-provided instructions the
    model follows regardless of user messages), and the third-party
    OpenAI-compatible servers behind this dialect enumerate only the classic
    roles (an Azure AI Foundry DeepSeek rung answered ``developer is not one of
    ['system', 'assistant', 'user', 'tool', 'function']`` in production), so
    the fold needs no disclosure and applies to every provider on the wire.
    """
    payload = openai_compatible_stream_payload("deepseek-v4-flash", _developer_conversation())
    messages = payload["messages"]
    assert isinstance(messages, list)
    assert [message["role"] for message in messages] == [
        "system",
        "user",
        "assistant",
        "system",
        "user",
    ]
    assert messages[0]["content"] == "Follow policy."
    assert messages[3]["content"] == "Now be terse."
    assert "developer" not in str(payload)


def test_responses_wire_keeps_the_developer_role_it_defines() -> None:
    """The native Responses wire owns the ``developer`` role: leading instructions
    ride ``instructions`` and a later developer turn keeps its role verbatim."""
    payload = openai_responses_stream_payload(
        "gpt-5.4", _developer_conversation(), supports_temperature=True
    )
    assert payload["instructions"] == "Follow policy."
    items = payload["input"]
    assert isinstance(items, list)
    assert {"role": "developer", "content": "Now be terse."} in items


def test_replayed_responses_items_drop_the_output_only_status_field() -> None:
    """A replayed input MESSAGE loses the output-only ``status`` a client copied
    from a prior response (OpenAI: "Unknown parameter: 'input[N].status'");
    every other item, hosted tool echoes included, re-emits verbatim."""
    replayed: JsonObject = {
        "type": "message",
        "role": "user",
        "content": [{"type": "input_text", "text": "run it again"}],
        "status": "completed",
    }
    hosted: JsonObject = {"type": "web_search_call", "id": "ws_1", "status": "completed"}
    request = GatewayRequest(
        surface=GatewayApiSurface.RESPONSES,
        messages=(
            GatewayMessage(role="user", content="run it"),
            GatewayMessage(role="user", provider_native_item=replayed),
            GatewayMessage(role="assistant", provider_native_item=hosted),
        ),
        stream=True,
        include_usage=True,
    )
    payload = openai_responses_stream_payload("gpt-6-astra", request, supports_temperature=False)
    items = payload["input"]
    assert isinstance(items, list)
    assert {
        "type": "message",
        "role": "user",
        "content": [{"type": "input_text", "text": "run it again"}],
    } in items
    # A hosted tool echo keeps its status: that schema defines the field.
    assert hosted in items
    # The caller's own item object is left untouched.
    assert replayed["status"] == "completed"


def test_replayed_items_with_foreign_ids_are_repaired_for_the_openai_wire() -> None:
    """An ``item_``-prefixed id (another gateway's emulation) is dropped from a
    replayed message and function call, a reasoning item carrying one is dropped
    whole, and OpenAI's own ids pass untouched."""
    foreign_message: JsonObject = {
        "type": "message",
        "role": "assistant",
        "id": "item_4762e0563d44cba8b5696951",
        "status": "completed",
        "content": [{"type": "output_text", "text": "hello", "annotations": []}],
    }
    foreign_call: JsonObject = {
        "type": "function_call",
        "id": "item_9f1c",
        "call_id": "call_1",
        "name": "read",
        "arguments": "{}",
    }
    foreign_reasoning: JsonObject = {"type": "reasoning", "id": "item_ab12", "summary": []}
    own_reasoning: JsonObject = {"type": "reasoning", "id": "rs_1", "summary": []}
    request = GatewayRequest(
        surface=GatewayApiSurface.RESPONSES,
        messages=(
            GatewayMessage(role="user", content="go"),
            GatewayMessage(role="assistant", provider_native_item=foreign_reasoning),
            GatewayMessage(role="assistant", provider_native_item=foreign_message),
            GatewayMessage(role="assistant", provider_native_item=foreign_call),
            GatewayMessage(role="assistant", provider_native_item=own_reasoning),
        ),
        stream=True,
        include_usage=True,
    )
    payload = openai_responses_stream_payload("gpt-5.6-sol", request, supports_temperature=False)
    items = payload["input"]
    assert isinstance(items, list)
    assert foreign_reasoning not in items
    assert {key: value for key, value in foreign_message.items() if key != "id"} in items
    assert {key: value for key, value in foreign_call.items() if key != "id"} in items
    assert own_reasoning in items
    assert all("item_" not in str(item.get("id", "")) for item in items if isinstance(item, dict))


def _agent_loop(*, reasoning: tuple[str | None, ...]) -> GatewayRequest:
    """One tools request whose history holds two assistant tool-call turns and a final text turn.

    ``reasoning`` gives the ``reasoning_content`` each assistant turn replays
    (``None`` = the field was absent, the shape an OpenAI-compatible SDK or a
    history started on another provider produces).
    """
    turns: tuple[tuple[str | None, tuple[str, JsonObject] | None], ...] = (
        ("", ("call-1", {"path": "a.txt"})),
        (None, ("call-2", {"path": "b.txt"})),
        ("Done reading.", None),
    )
    messages: list[GatewayMessage] = [GatewayMessage(role="user", content="read a and b")]
    for (content, call), replayed in zip(turns, reasoning, strict=True):
        blocks = (ExposedReasoningContentBlock(content=replayed),) if replayed is not None else ()
        if call is None:
            messages.append(
                GatewayMessage(role="assistant", content=content, provider_reasoning=blocks)
            )
            continue
        call_id, arguments = call
        messages.append(
            GatewayMessage(
                role="assistant",
                content=content or None,
                tool_calls=(ToolCall(call_id=call_id, name="read_file", arguments=arguments),),
                provider_reasoning=blocks,
            )
        )
        messages.append(GatewayMessage(role="tool", content="ok", tool_call_id=call_id))
    messages.append(GatewayMessage(role="user", content="now summarize"))
    return GatewayRequest(
        surface=GatewayApiSurface.CHAT_COMPLETIONS,
        messages=tuple(messages),
        tools=(
            GatewayToolDefinition(
                name="read_file",
                description="Read one file.",
                parameters={"type": "object", "properties": {"path": {"type": "string"}}},
            ),
        ),
        stream=True,
        include_usage=True,
    )


def _assistant_turns(payload: JsonObject) -> list[JsonObject]:
    messages = cast("list[JsonObject]", payload["messages"])
    return [message for message in messages if message["role"] == "assistant"]


def test_deepseek_origin_backfills_empty_reasoning_on_tool_call_turns_only() -> None:
    """A tool-call turn with no reasoning gets ``reasoning_content: ""``; text turns do not.

    DeepSeek's thinking mode 400s a tools request unless every assistant
    tool-call turn carries the field (``The `reasoning_content` in the thinking
    mode must be passed back to the API.``) and accepts an empty string exactly
    like real reasoning (verified live 2026-09-10). The final text turn is left
    alone: the provider does not require the field there and the wire stays
    minimal.
    """
    payload = openai_compatible_stream_payload(
        "deepseek-flash", _agent_loop(reasoning=(None, None, None)), deepseek_reasoning_history=True
    )
    first_call, second_call, final_text = _assistant_turns(payload)
    assert first_call["tool_calls"] and first_call["reasoning_content"] == ""
    assert second_call["tool_calls"] and second_call["reasoning_content"] == ""
    assert "tool_calls" not in final_text
    assert "reasoning_content" not in final_text


def test_deepseek_origin_forwards_caller_reasoning_verbatim_without_the_exposure_stamp() -> None:
    """Caller plaintext replays byte-for-byte on plain AND tool-call turns, empty included.

    No ``reasoning_output_exposed`` stamp is involved: the platform's DeepSeek
    lane is not stamped, and replay is what the provider REQUIRES, not what the
    catalog chose to expose.
    """
    payload = openai_compatible_stream_payload(
        "deepseek-flash",
        _agent_loop(reasoning=("", "I should read b next.", "Both files are read.")),
        deepseek_reasoning_history=True,
    )
    first_call, second_call, final_text = _assistant_turns(payload)
    assert first_call["reasoning_content"] == ""
    assert second_call["reasoning_content"] == "I should read b next."
    assert final_text["reasoning_content"] == "Both files are read."


def test_deepseek_origin_leaves_a_history_with_no_tool_calls_untouched() -> None:
    """A plain conversation on the DeepSeek origin gets no injected field at all."""
    request = GatewayRequest(
        surface=GatewayApiSurface.CHAT_COMPLETIONS,
        messages=(
            GatewayMessage(role="user", content="hi"),
            GatewayMessage(role="assistant", content="hello"),
            GatewayMessage(role="user", content="again"),
        ),
        stream=True,
        include_usage=True,
    )
    payload = openai_compatible_stream_payload(
        "deepseek-flash", request, deepseek_reasoning_history=True
    )
    (assistant,) = _assistant_turns(payload)
    assert assistant == {"role": "assistant", "content": "hello"}


def test_other_compatible_origins_keep_the_exposure_gated_behaviour() -> None:
    """Off the DeepSeek origin nothing changes: no backfill, plaintext still exposure-gated."""
    request = _agent_loop(reasoning=("", "I should read b next.", None))
    stripped = openai_compatible_stream_payload("deepseek-v4-flash", request)
    assert all("reasoning_content" not in turn for turn in _assistant_turns(stripped))
    exposed = openai_compatible_stream_payload(
        "hy4-preview", request, reasoning_output_exposed=True
    )
    first_call, second_call, final_text = _assistant_turns(exposed)
    assert first_call["reasoning_content"] == ""
    assert second_call["reasoning_content"] == "I should read b next."
    assert "reasoning_content" not in final_text
