"""Tests for the wmh rendering seam over tinker-cookbook renderers.

Conversion-only logic is tested without the cookbook; everything that touches
a real renderer is gated on the cookbook being importable (it is part of the
distill extra), so the suite still passes in an environment without it.
"""

from __future__ import annotations

import json
import re
import sys

import pytest
from llm_waterfall.types import (
    ChatFunctionCall,
    ChatFunctionDefinition,
    ChatMessage,
    ChatTool,
    ChatToolCall,
)

from wmh.distill.rendering import (
    build_renderer,
    renderer_messages_from_chat,
    tool_specs_from_chat,
)


class _CharTokenizer:
    """Char-level tokenizer whose listed special strings map to single ids."""

    _SPECIALS = {"<|im_start|>": 300000, "<|im_end|>": 300001}

    bos_token: str | None = None
    eos_token_id: int | None = None
    name_or_path = "fake/char"

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        """Encode text; each special string becomes one id, other chars one id each."""
        del add_special_tokens
        pattern = "(" + "|".join(re.escape(special) for special in self._SPECIALS) + ")"
        ids: list[int] = []
        for piece in re.split(pattern, text):
            special = self._SPECIALS.get(piece)
            if special is not None:
                ids.append(special)
            else:
                ids.extend(ord(ch) for ch in piece)
        return ids

    def decode(self, token_ids: list[int]) -> str:
        """Decode ids back to text, restoring special strings."""
        reverse = {v: k for k, v in self._SPECIALS.items()}
        return "".join(reverse.get(t) or chr(t) for t in token_ids)


def _tool(name: str) -> ChatTool:
    return ChatTool(
        function=ChatFunctionDefinition(
            name=name,
            description=f"run {name}",
            parameters={"type": "object", "properties": {"path": {"type": "string"}}},
        )
    )


def test_tool_specs_from_chat_converts_openai_tools() -> None:
    specs = tool_specs_from_chat([_tool("ls")])
    assert specs == [
        {
            "name": "ls",
            "description": "run ls",
            "parameters": {"type": "object", "properties": {"path": {"type": "string"}}},
        }
    ]


def test_renderer_messages_from_chat_converts_roles_content_and_tools() -> None:
    pytest.importorskip("tinker_cookbook")
    from tinker_cookbook.renderers.base import ToolCall

    messages = [
        ChatMessage(role="system", content="be terse"),
        ChatMessage(
            role="user",
            content=[{"type": "text", "text": "hello "}, {"type": "text", "text": "world"}],
        ),
        ChatMessage(
            role="assistant",
            content=None,
            tool_calls=[
                ChatToolCall(
                    id="call_7",
                    function=ChatFunctionCall(name="ls", arguments='{"path": "/tmp"}'),
                )
            ],
        ),
        ChatMessage.model_validate(
            {"role": "tool", "content": "ok", "tool_call_id": "call_7", "name": "ls"}
        ),
    ]
    converted = renderer_messages_from_chat(messages)
    assert [msg["role"] for msg in converted] == ["system", "user", "assistant", "tool"]
    assert converted[1]["content"] == "hello world"
    assert converted[2]["content"] == ""
    tool_calls = converted[2]["tool_calls"]
    assert isinstance(tool_calls[0], ToolCall)
    assert tool_calls[0].id == "call_7"
    assert tool_calls[0].function.name == "ls"
    assert tool_calls[0].function.arguments == '{"path": "/tmp"}'
    assert converted[3]["tool_call_id"] == "call_7"
    assert converted[3]["name"] == "ls"


def test_renderer_messages_from_chat_rejects_non_text_parts() -> None:
    pytest.importorskip("tinker_cookbook")
    messages = [
        ChatMessage(
            role="user",
            content=[{"type": "image_url", "image_url": {"url": "http://x/y.png"}}],
        )
    ]
    with pytest.raises(ValueError, match="text-only"):
        renderer_messages_from_chat(messages)


def test_build_renderer_role_colon_round_trip() -> None:
    # meta-llama/Llama-3.2-1B is a base model, so the cookbook maps it to the
    # role_colon renderer; the generation prompt must carry the user content.
    pytest.importorskip("tinker_cookbook")
    tokenizer = _CharTokenizer()
    rendering = build_renderer("meta-llama/Llama-3.2-1B", tokenizer)

    prompt_ids = rendering.build_generation_prompt(
        [ChatMessage(role="user", content="hello world")]
    )
    prompt_text = rendering.decode(prompt_ids)
    assert "User: hello world" in prompt_text
    assert prompt_text.endswith("Assistant:")
    assert rendering.stop_sequences == ["\n\nUser:"]

    stopped = rendering.parse_response(tokenizer.encode("hi there\n\nUser:"))
    assert stopped.text == "hi there"
    assert stopped.tool_calls == []
    assert stopped.stopped is True

    truncated = rendering.parse_response(tokenizer.encode("cut off mid"))
    assert truncated.stopped is False


def test_role_colon_eos_termination_is_a_clean_stop() -> None:
    # role_colon reports a bare EOS token as a clean end of turn (ParseTermination.EOS);
    # it must parse as stopped, not truncation, or a finished answer would reach the
    # agent as finish_reason "length".
    pytest.importorskip("tinker_cookbook")

    class _EosCharTokenizer(_CharTokenizer):
        eos_token_id = 300002

    tokenizer = _EosCharTokenizer()
    rendering = build_renderer("meta-llama/Llama-3.2-1B", tokenizer)
    ended = rendering.parse_response([*tokenizer.encode("all done"), 300002])
    assert ended.text == "all done"
    assert ended.stopped is True


def test_build_renderer_qwen3_tools_prompt_and_tool_call_parse() -> None:
    pytest.importorskip("tinker_cookbook")
    tokenizer = _CharTokenizer()
    rendering = build_renderer("Qwen/Qwen3-8B", tokenizer)

    prompt_ids = rendering.build_generation_prompt(
        [
            ChatMessage(role="system", content="be brief"),
            ChatMessage(role="user", content="list files"),
        ],
        tools=[_tool("ls")],
    )
    prompt_text = rendering.decode(prompt_ids)
    assert "<tools>" in prompt_text
    assert '"ls"' in prompt_text
    assert "be brief" in prompt_text
    assert "list files" in prompt_text
    assert prompt_text.endswith("<|im_start|>assistant\n")
    assert rendering.stop_sequences == [_CharTokenizer._SPECIALS["<|im_end|>"]]

    sampled = tokenizer.encode(
        "<think>plan</think>listing now\n"
        '<tool_call>\n{"name": "ls", "arguments": {"path": "/tmp"}}\n</tool_call>'
        "<|im_end|>"
    )
    parsed = rendering.parse_response(sampled)
    assert parsed.stopped is True
    assert "<think>plan</think>" in parsed.text
    assert "listing now" in parsed.text
    assert len(parsed.tool_calls) == 1
    call = parsed.tool_calls[0]
    assert call.id == "call_0"
    assert call.function.name == "ls"
    assert json.loads(call.function.arguments) == {"path": "/tmp"}

    truncated = rendering.parse_response(tokenizer.encode("ran out of bud"))
    assert truncated.stopped is False


def test_build_renderer_unknown_model_is_actionable() -> None:
    pytest.importorskip("tinker_cookbook")
    with pytest.raises(ValueError, match="unknown-org/mystery"):
        build_renderer("unknown-org/mystery", _CharTokenizer())


def test_build_renderer_missing_extra_names_the_extra(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Simulate the distill extra being absent: None entries make every import
    # of the package (and its already-imported submodules) raise ImportError.
    for name in list(sys.modules):
        if name == "tinker_cookbook" or name.startswith("tinker_cookbook."):
            monkeypatch.setitem(sys.modules, name, None)
    monkeypatch.setitem(sys.modules, "tinker_cookbook", None)
    with pytest.raises(ImportError, match="uv sync --extra distill"):
        build_renderer("Qwen/Qwen3-8B", _CharTokenizer())
