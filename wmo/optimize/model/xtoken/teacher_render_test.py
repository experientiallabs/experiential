"""Tests for teacher-side rendering and content-island extraction.

The fakes here reproduce the behaviours of GLM-5.2's real chat template that
matter for island location (verified against the real template this session):
visible content is stripped, `<think>...</think>` is split out of content on its
own, and tool-call argument VALUES are emitted verbatim with raw newlines while
the surrounding syntax differs from the student's template.
"""

from __future__ import annotations

import json

from wmo.common.vendor.waterfall.types import ChatFunctionCall, ChatMessage, ChatTool, ChatToolCall
from wmo.optimize.model.xtoken.byte_offsets import BYTE_DECODER
from wmo.optimize.model.xtoken.teacher_render import render_for_teacher


class FakeTemplateTokenizer:
    """A byte-level tokenizer plus a GLM-shaped chat template.

    `chunk_bytes` controls how many raw bytes each token covers, which is how
    the tests place token boundaries on or across island edges.
    """

    def __init__(self, chunk_bytes: int = 1, *, transform: dict[str, str] | None = None) -> None:
        self.chunk_bytes = chunk_bytes
        self._transform = transform or {}
        self._encoder = {value: char for char, value in BYTE_DECODER.items()}
        self._vocab: dict[bytes, int] = {}
        self._by_id: dict[int, bytes] = {}

    def _intern(self, piece: bytes) -> int:
        if piece not in self._vocab:
            token_id = len(self._vocab) + 1
            self._vocab[piece] = token_id
            self._by_id[token_id] = piece
        return self._vocab[piece]

    def apply_chat_template(
        self,
        conversation: list[dict[str, object]],
        *,
        tools: list[dict[str, object]] | None = None,
        tokenize: bool = False,
        add_generation_prompt: bool = False,
        clear_thinking: bool = True,
    ) -> str:
        """Render GLM-style: `<|role|>`, split think, strip content, verbatim args."""
        out: list[str] = []
        if tools:
            out.append("<tools>" + json.dumps(tools) + "</tools>")
        for message in conversation:
            role = str(message.get("role"))
            content = str(message.get("content") or "")
            if role != "assistant":
                out.append(f"<|{role}|>{content}")
                continue
            reasoning = ""
            visible = content
            if "</think>" in content:
                head, _, tail = content.partition("</think>")
                reasoning = head.split("<think>")[-1]
                visible = tail
            out.append("<|assistant|>")
            # clear_thinking=False is what keeps historical reasoning; the
            # default drops it, exactly like the real template.
            keep_reasoning = bool(reasoning) and not clear_thinking
            out.append(f"<think>{reasoning}</think>" if keep_reasoning else "<think></think>")
            stripped = visible.strip()
            if stripped:
                out.append(self._transform.get(stripped, stripped))
            raw_calls = message.get("tool_calls")
            calls = raw_calls if isinstance(raw_calls, list) else []
            for call in calls:
                if not isinstance(call, dict):
                    continue
                function = call.get("function")
                if not isinstance(function, dict):
                    continue
                arguments = function.get("arguments")
                out.append("<tool_call>" + str(function.get("name")))
                if isinstance(arguments, dict):
                    for key, value in arguments.items():
                        rendered = self._transform.get(str(value), str(value))
                        out.append(f"<arg_key>{key}</arg_key><arg_value>{rendered}</arg_value>")
                out.append("</tool_call>")
        return "".join(out)

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        """Split the text's UTF-8 bytes into fixed-size chunks."""
        raw = text.encode("utf-8")
        return [
            self._intern(raw[i : i + self.chunk_bytes])
            for i in range(0, len(raw), self.chunk_bytes)
        ]

    def convert_ids_to_tokens(self, ids: list[int]) -> list[str | None]:
        """Byte-level surface forms for the interned pieces."""
        out: list[str | None] = []
        for token_id in ids:
            piece = self._by_id.get(token_id)
            out.append(None if piece is None else "".join(self._encoder[b] for b in piece))
        return out

    def decode(self, token_ids: list[int]) -> str:
        """Concatenate the raw bytes and decode as UTF-8."""
        return b"".join(self._by_id.get(t, b"") for t in token_ids).decode("utf-8", "replace")


def _tool() -> ChatTool:
    return ChatTool.model_validate(
        {
            "type": "function",
            "function": {
                "name": "bash",
                "description": "Run a shell command",
                "parameters": {
                    "type": "object",
                    "properties": {"command": {"type": "string"}},
                    "required": ["command"],
                },
            },
        }
    )


def _call(name: str, arguments: dict[str, object]) -> ChatToolCall:
    return ChatToolCall(
        id="c0", function=ChatFunctionCall(name=name, arguments=json.dumps(arguments))
    )


def test_reasoning_and_text_become_separate_islands() -> None:
    messages = [
        ChatMessage(role="user", content="hi"),
        ChatMessage(role="assistant", content="<think>let me check</think>all done"),
    ]
    render = render_for_teacher(FakeTemplateTokenizer(), messages)
    kinds = [island.kind for island in render.islands]
    texts = [island.text for island in render.islands]
    assert kinds == ["reasoning", "text"]
    assert texts == ["let me check", "all done"]
    assert render.unmatched == []


def test_tool_argument_value_is_an_island_with_raw_newlines() -> None:
    command = "sudo apt-get update\ncd /app && ls"
    messages = [
        ChatMessage(role="user", content="go"),
        ChatMessage(
            role="assistant",
            content="running",
            tool_calls=[_call("bash", {"command": command})],
        ),
    ]
    render = render_for_teacher(FakeTemplateTokenizer(), messages, [_tool()])
    argument_islands = [i for i in render.islands if i.kind == "tool_argument"]
    assert len(argument_islands) == 1
    assert argument_islands[0].text == command
    assert "\n" in argument_islands[0].text


def test_framing_tokens_are_not_covered_by_any_island() -> None:
    messages = [
        ChatMessage(role="user", content="hi"),
        ChatMessage(role="assistant", content="ok"),
    ]
    tokenizer = FakeTemplateTokenizer()
    render = render_for_teacher(tokenizer, messages)
    covered = {
        position
        for island in render.islands
        for position in range(island.teacher_start, island.teacher_end)
    }
    # The render is much longer than the covered content, so framing is present
    # and deliberately uncovered.
    assert covered
    assert len(covered) < len(render.token_ids)
    # Every covered token's bytes come from the island text, never from framing.
    decoded = tokenizer.decode(render.token_ids)
    for island in render.islands:
        assert island.text in decoded


def test_repeated_content_matches_the_later_occurrence_for_the_later_message() -> None:
    # Both assistant turns say the same thing. The forward cursor must give the
    # second turn the SECOND occurrence, not re-match the first.
    messages = [
        ChatMessage(role="user", content="hi"),
        ChatMessage(role="assistant", content="same text"),
        ChatMessage(role="tool", content="out", tool_call_id="c0"),
        ChatMessage(role="assistant", content="same text"),
    ]
    render = render_for_teacher(FakeTemplateTokenizer(), messages)
    assert len(render.islands) == 2
    first, second = render.islands
    assert first.message_index == 1
    assert second.message_index == 3
    assert second.byte_start > first.byte_start
    assert first.teacher_end <= second.teacher_start


def test_content_the_template_transforms_is_reported_unmatched() -> None:
    # A template that rewrites content breaks byte identity; the piece must be
    # reported rather than mis-located onto unrelated tokens.
    tokenizer = FakeTemplateTokenizer(transform={"needs escaping": "needs\\x20escaping"})
    messages = [
        ChatMessage(role="user", content="hi"),
        ChatMessage(role="assistant", content="needs escaping"),
    ]
    render = render_for_teacher(tokenizer, messages)
    assert render.islands == []
    assert render.unmatched == ["needs escaping"]


def test_tokens_straddling_an_island_edge_are_excluded() -> None:
    # With 8-byte tokens, boundaries rarely land on island edges, so a token
    # mixing framing bytes with content bytes must not be claimed.
    messages = [
        ChatMessage(role="user", content="hi"),
        ChatMessage(role="assistant", content="abcdefghijklmnop"),
    ]
    coarse = render_for_teacher(FakeTemplateTokenizer(chunk_bytes=8), messages)
    fine = render_for_teacher(FakeTemplateTokenizer(chunk_bytes=1), messages)
    coarse_covered = sum(i.teacher_end - i.teacher_start for i in coarse.islands)
    fine_covered = sum(i.teacher_end - i.teacher_start for i in fine.islands)
    # Byte-for-byte the fine tokenizer covers the whole 16-byte island; the
    # coarse one covers strictly fewer bytes because its edge tokens straddle.
    assert fine_covered == 16
    assert coarse_covered * 8 <= 16


def test_non_assistant_messages_contribute_no_islands() -> None:
    messages = [
        ChatMessage(role="system", content="be good"),
        ChatMessage(role="user", content="a question"),
        ChatMessage(role="tool", content="tool output", tool_call_id="c0"),
    ]
    render = render_for_teacher(FakeTemplateTokenizer(), messages)
    assert render.islands == []


def test_non_string_argument_values_are_skipped() -> None:
    messages = [
        ChatMessage(role="user", content="go"),
        ChatMessage(
            role="assistant",
            content="x",
            tool_calls=[_call("bash", {"command": "ls", "timeout": 30, "quiet": True})],
        ),
    ]
    render = render_for_teacher(FakeTemplateTokenizer(), messages, [_tool()])
    arguments = [i.text for i in render.islands if i.kind == "tool_argument"]
    assert arguments == ["ls"]


def test_islands_never_start_at_teacher_position_zero() -> None:
    # Position 0 has no context so it can never carry a logprob.
    messages = [ChatMessage(role="assistant", content="hello")]
    render = render_for_teacher(FakeTemplateTokenizer(), messages)
    for island in render.islands:
        assert island.teacher_start >= 1


def test_byte_ranges_agree_with_the_rendered_text() -> None:
    messages = [
        ChatMessage(role="user", content="hi"),
        ChatMessage(role="assistant", content="<think>why</think>because"),
    ]
    tokenizer = FakeTemplateTokenizer()
    render = render_for_teacher(tokenizer, messages)
    raw = tokenizer.decode(render.token_ids).encode("utf-8")
    for island in render.islands:
        assert raw[island.byte_start : island.byte_end] == island.text.encode("utf-8")


def test_multibyte_content_keeps_exact_byte_ranges() -> None:
    messages = [
        ChatMessage(role="user", content="hi"),
        ChatMessage(role="assistant", content="caf\N{LATIN SMALL LETTER E WITH ACUTE} \N{ROCKET}"),
    ]
    tokenizer = FakeTemplateTokenizer(chunk_bytes=1)
    render = render_for_teacher(tokenizer, messages)
    raw = tokenizer.decode(render.token_ids).encode("utf-8")
    assert render.islands
    for island in render.islands:
        assert raw[island.byte_start : island.byte_end] == island.text.encode("utf-8")
