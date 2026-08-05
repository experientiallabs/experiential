"""Teacher-side rendering: the same conversation under the teacher's chat template.

The teacher cannot score the student's token ids (different vocabulary), so it
scores its own tokenization of the same conversation. This module renders the
canonical message list with the TEACHER's chat template and reports which
teacher token ranges cover byte-identical message content, which is what the
chunk aligner pairs against the student's sampled tokens.

Only message CONTENT can be compared. The two templates frame turns
differently (Qwen writes `<|im_start|>assistant`, GLM writes `<|assistant|>`;
Qwen writes tool calls as `<function=bash><parameter=command>`, GLM as
`<tool_call>bash<arg_key>command</arg_key><arg_value>`), so framing tokens have
no counterpart on the other side and are deliberately left uncovered. Measured
on the headline run's real spans, framing is 4.5% of sampled tokens, so ~95.5%
remains scoreable.

Two properties of GLM-5.2's template drive the render options:

- `clear_thinking=False` keeps `reasoning_content` on EVERY assistant turn.
  The default drops it from historical turns (`loop.index0 > ns.last_user_index`),
  which would make 73% of the run's think tokens unscoreable and, worse, would
  have the teacher condition on a history the student never saw. Passing
  `clear_thinking=False` also short-circuits that `last_user_index` test, which
  makes the render prefix-stable as a side benefit.
- The template STRIPS visible content (`content.strip()`) and splits
  `<think>...</think>` out of content on its own. Islands are therefore located
  by searching for the stripped form; leading and trailing whitespace the
  student sampled has no counterpart and stays uncovered.
"""

from __future__ import annotations

import json
import logging
from typing import Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field

from wmo.distill.xtoken.byte_offsets import span_byte_ends
from wmo.utils.waterfall.types import ChatMessage, ChatTool

logger = logging.getLogger(__name__)

IslandKind = Literal["reasoning", "text", "tool_argument"]
"""Which part of an assistant message an island covers."""


class TemplateTokenizer(Protocol):
    """The tokenizer slice teacher rendering needs.

    HuggingFace fast tokenizers satisfy this structurally; the tests use small
    deterministic fakes.
    """

    def apply_chat_template(
        self,
        conversation: list[dict[str, object]],
        *,
        tools: list[dict[str, object]] | None = ...,
        tokenize: bool = ...,
        add_generation_prompt: bool = ...,
        clear_thinking: bool = ...,
    ) -> str:
        """Render a conversation to a string with the model's chat template."""
        ...

    def encode(self, text: str, add_special_tokens: bool = ...) -> list[int]:
        """Encode text to token ids."""
        ...

    def convert_ids_to_tokens(self, ids: list[int]) -> list[str | None]:
        """The surface form of each token id; needed for exact byte offsets."""
        ...

    def decode(self, token_ids: list[int]) -> str:
        """Decode token ids back to text; used to verify the byte reconstruction."""
        ...


class ContentIsland(BaseModel):
    """One byte-identical content region, located in the teacher's token stream."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    kind: IslandKind
    message_index: int = Field(ge=0)
    """Index into the rendered conversation's message list."""

    text: str = Field(min_length=1)
    """The content bytes this island covers, identical on both sides."""

    teacher_start: int = Field(ge=1)
    """First teacher token fully inside the island. Position 0 can never be
    scored (no context), so an island never starts there."""

    teacher_end: int = Field(gt=1)
    """One past the last teacher token fully inside the island."""

    byte_start: int = Field(ge=0)
    """Island start as a byte offset into the rendered text."""

    byte_end: int = Field(gt=0)
    """Island end as a byte offset into the rendered text."""


class TeacherRender(BaseModel):
    """A conversation rendered and tokenized under the teacher's template."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    token_ids: list[int]
    islands: list[ContentIsland] = Field(default_factory=list)
    unmatched: list[str] = Field(default_factory=list)
    """Content pieces the render did not contain verbatim (the template
    transformed them), reported so the caller can meter the fallback rate
    instead of silently losing signal."""


def _byte_prefixes(text: str) -> list[int]:
    """Byte offset of each character index, plus a final total.

    Entry i is the number of UTF-8 bytes in `text[:i]`, so a character range
    `[a, b)` maps to the byte range `[out[a], out[b])`. Built in one pass
    because doing `len(text[:i].encode())` per token is quadratic.
    """
    out = [0] * (len(text) + 1)
    total = 0
    for index, char in enumerate(text):
        total += len(char.encode("utf-8"))
        out[index + 1] = total
    return out


def _content_pieces(message: ChatMessage) -> list[tuple[IslandKind, str]]:
    """The comparable content of one assistant message, in render order.

    Reasoning comes first (the template emits `<think>` before visible text),
    then visible text, then each tool call's argument VALUES. Argument values
    are compared, not the JSON around them: both templates emit the values
    with raw newlines and no JSON escaping (verified for Qwen3.6 and GLM-5.2),
    but the surrounding syntax differs entirely.
    """
    pieces: list[tuple[IslandKind, str]] = []
    raw = message.content if isinstance(message.content, str) else ""
    if "</think>" in raw:
        head, _, tail = raw.partition("</think>")
        reasoning = head.split("<think>")[-1]
        if reasoning.strip():
            pieces.append(("reasoning", reasoning.strip()))
        visible = tail
    else:
        visible = raw
    if visible.strip():
        pieces.append(("text", visible.strip()))
    for call in message.tool_calls or []:
        arguments = call.function.arguments
        if not isinstance(arguments, str):
            continue
        # The template renders each argument VALUE; the JSON envelope differs
        # per template so it is never an island.
        for value in _argument_values(arguments):
            if value.strip():
                pieces.append(("tool_argument", value))
    return pieces


def _argument_values(arguments: str) -> list[str]:
    """The string values of a tool call's JSON arguments object, in order.

    Non-string values are skipped: templates render them through their own
    formatting (`true` vs `True`, number spacing), so byte identity is not
    guaranteed and a mismatched island is worse than an absent one.
    """
    try:
        parsed = json.loads(arguments)
    except (TypeError, ValueError):
        logger.debug("tool call arguments are not valid JSON; no islands from them")
        return []
    if not isinstance(parsed, dict):
        return []
    return [value for value in parsed.values() if isinstance(value, str)]


def _renderer_messages(messages: list[ChatMessage]) -> list[dict[str, object]]:
    """Convert canonical chat messages into the dict shape chat templates expect.

    Tool call arguments are passed as a parsed dict: both Qwen3.6's and
    GLM-5.2's templates iterate `arguments.items()` and raise on a string.
    """
    out: list[dict[str, object]] = []
    for message in messages:
        entry: dict[str, object] = {
            "role": message.role,
            "content": message.content if isinstance(message.content, str) else "",
        }
        if message.tool_calls:
            calls: list[dict[str, object]] = []
            for call in message.tool_calls:
                try:
                    arguments = json.loads(call.function.arguments)
                except (TypeError, ValueError):
                    arguments = {}
                calls.append(
                    {
                        "type": "function",
                        "function": {"name": call.function.name, "arguments": arguments},
                    }
                )
            entry["tool_calls"] = calls
        out.append(entry)
    return out


def _tool_specs(tools: list[ChatTool]) -> list[dict[str, object]]:
    """Convert canonical tool definitions into chat-template tool dicts."""
    return [
        {
            "type": "function",
            "function": {
                "name": tool.function.name,
                "description": tool.function.description,
                "parameters": dict(tool.function.parameters),
            },
        }
        for tool in tools
    ]


def render_for_teacher(
    tokenizer: TemplateTokenizer,
    messages: list[ChatMessage],
    tools: list[ChatTool] | None = None,
) -> TeacherRender:
    """Render a conversation with the teacher's template and locate content islands.

    Islands are found by scanning the rendered text forward with a monotonic
    cursor, so a content string that also appears earlier can never be matched
    to the wrong occurrence. Only teacher tokens FULLY inside an island are
    reported: a token straddling an island edge mixes content bytes with
    framing bytes, so its logprob is not comparable.

    Args:
        tokenizer: The teacher's tokenizer, with its chat template.
        messages: The canonical conversation, in order.
        tools: Tool schemas the conversation was generated with.

    Returns:
        The teacher token ids plus the located islands, and the content pieces
        the render did not contain verbatim.
    """
    rendered = tokenizer.apply_chat_template(
        _renderer_messages(messages),
        tools=_tool_specs(tools) if tools else None,
        tokenize=False,
        add_generation_prompt=False,
        clear_thinking=False,
    )
    token_ids = tokenizer.encode(rendered, add_special_tokens=False)
    token_byte_ends = _token_byte_ends(tokenizer, token_ids, rendered)
    if token_byte_ends is None:
        return TeacherRender(token_ids=token_ids, islands=[], unmatched=[])

    prefixes = _byte_prefixes(rendered)
    islands: list[ContentIsland] = []
    unmatched: list[str] = []
    cursor = 0
    for message_index, message in enumerate(messages):
        if message.role != "assistant":
            continue
        for kind, piece in _content_pieces(message):
            found = rendered.find(piece, cursor)
            if found < 0:
                unmatched.append(piece)
                logger.debug(
                    "content piece of kind %s in message %d is not present verbatim in "
                    "the teacher render; it will not be scored",
                    kind,
                    message_index,
                )
                continue
            cursor = found + len(piece)
            byte_start = prefixes[found]
            byte_end = prefixes[found + len(piece)]
            span = _tokens_inside(token_byte_ends, byte_start, byte_end)
            if span is None:
                unmatched.append(piece)
                continue
            start, end = span
            islands.append(
                ContentIsland(
                    kind=kind,
                    message_index=message_index,
                    text=piece,
                    teacher_start=start,
                    teacher_end=end,
                    byte_start=byte_start,
                    byte_end=byte_end,
                )
            )
    return TeacherRender(token_ids=token_ids, islands=islands, unmatched=unmatched)


def _token_byte_ends(
    tokenizer: TemplateTokenizer, token_ids: list[int], rendered: str
) -> list[int] | None:
    """Cumulative byte-end offset per teacher token over the rendered text.

    Uses the same exact reconstruction as the student side so both sides'
    offsets live in one byte space.
    """
    result = span_byte_ends(tokenizer, token_ids)
    if result is None:
        return None
    ends, span = result
    if span != rendered.encode("utf-8"):
        logger.warning(
            "re-encoding the teacher render does not reproduce it byte for byte "
            "(%d vs %d bytes), so island offsets would be wrong; nothing is scored",
            len(span),
            len(rendered.encode("utf-8")),
        )
        return None
    return ends


def _tokens_inside(
    token_byte_ends: list[int], byte_start: int, byte_end: int
) -> tuple[int, int] | None:
    """The half-open token range fully contained in a byte range.

    A token spans `[ends[i - 1], ends[i])`. Only tokens whose whole span lies
    inside `[byte_start, byte_end)` qualify, and position 0 is excluded because
    it has no context and can never carry a logprob.
    """
    start: int | None = None
    end: int | None = None
    previous = 0
    for index, boundary in enumerate(token_byte_ends):
        if index >= 1 and previous >= byte_start and boundary <= byte_end:
            if start is None:
                start = index
            end = index + 1
        previous = boundary
    if start is None or end is None:
        return None
    return start, end
