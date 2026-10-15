"""Render an exchange into bounded text windows that a decision model can read.

Messages render as compact role-tagged lines::

    [task] Fix the failing date parser test
    [user] It still fails on leap years
    [assistant] Running the tests again.
    [tool_call bash] {"command":"pytest -q"}
    [tool_result bash] 1 failed, 41 passed

The conversation scope gets one opening window: leading instructions, the tool list and the
first user message, shortened to fit. The turn scope covers the new slice of a request, the
messages after its last assistant message plus the response, split at message boundaries into
overlapping windows that each start with the ``[task]`` header. A single message too large for a
window keeps its head and tail around an explicit omission marker.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence

from pydantic import Field, model_validator

from exp.common.core.artifacts import ContractModel
from exp.common.tags.bank import TagScope
from exp.common.tags.exchange import (
    TagExchange,
    TagMessage,
    TagTool,
    first_user_index,
    leading_instructions,
    opening_end,
)

# Conservative characters per token: English prose averages about four, code and JSON about
# three. Mostly non-Latin text runs lower and needs a smaller budget than for_input_tokens gives.
CONSERVATIVE_CHARS_PER_TOKEN = 3
# Largest window. Rendering replaces the control characters JSON would escape as six bytes,
# so a window costs at most 4 bytes of JSON text per character (128,000 bytes here), and even
# a full bank of 32 questions with 300-character, 4-byte texts (about 116,000 bytes) still
# fits a 262,144-byte decision request.
MAX_WINDOW_CHARS = 32_000
TOOL_DESCRIPTION_CHARS = 160
# Fewest characters each opening block keeps before middle blocks are collapsed.
MIN_BLOCK_CHARS = 80

_MARKER = "\n[... {omitted} characters omitted ...]\n"
_BLANK_RUNS = re.compile(r"\n{3,}")
_SURROGATE = re.compile("[\ud800-\udfff]")
_CONTROL = re.compile("[\x00-\x08\x0b-\x1f]")


class TagWindowBudget(ContractModel):
    """Character limits for rendered windows.

    Attributes:
        window_chars: Most characters in one window, header included; 2,000 through 32,000,
            default 24,000.
        overlap_chars: Most characters of trailing messages repeated at the start of the next
            turn window, default 2,000.
        header_chars: Most characters of the ``[task]`` header, default 1,000.
    """

    window_chars: int = Field(default=24_000, ge=2_000, le=MAX_WINDOW_CHARS)
    overlap_chars: int = Field(default=2_000, ge=0)
    header_chars: int = Field(default=1_000, ge=200)

    @model_validator(mode="after")
    def _room_for_new_content(self) -> TagWindowBudget:
        """Keep at least half of every window for messages it has not shown yet.

        Returns:
            The validated budget.

        Raises:
            ValueError: Header plus overlap exceed half the window.
        """
        if self.header_chars + self.overlap_chars > self.window_chars // 2:
            raise ValueError("header_chars plus overlap_chars may use at most half of a window")
        return self

    @classmethod
    def for_input_tokens(
        cls, input_tokens: int, *, reserved_tokens: int = 2_048
    ) -> TagWindowBudget:
        """Size windows for a decision server that accepts ``input_tokens`` per request.

        Args:
            input_tokens: The server's input token limit.
            reserved_tokens: Tokens kept for the questions and protocol framing.

        Returns:
            A budget at ``CONSERVATIVE_CHARS_PER_TOKEN`` characters per token, capped at
            ``MAX_WINDOW_CHARS``, with a twelfth of the window as overlap.

        Raises:
            ValueError: The limit leaves fewer than 2,000 characters for a window.
        """
        window = min(
            (input_tokens - reserved_tokens) * CONSERVATIVE_CHARS_PER_TOKEN, MAX_WINDOW_CHARS
        )
        if window < 2_000:
            raise ValueError(
                f"{input_tokens} input tokens leave no room for a tag window after reserving "
                f"{reserved_tokens}; serve the decision model with a larger context"
            )
        return cls(
            window_chars=window, overlap_chars=window // 12, header_chars=min(1_000, window // 8)
        )


class TagWindow(ContractModel):
    """One rendered window and the messages it covers.

    Attributes:
        scope: Conversation for the opening window, turn for turn windows.
        index: Zero-based position within the scope.
        start_message: First covered message position.
        end_message: One past the last covered message position.
        text: Rendered text, at most the budget's ``window_chars``.
    """

    scope: TagScope
    index: int = Field(ge=0)
    start_message: int = Field(ge=0)
    end_message: int = Field(ge=1)
    text: str = Field(min_length=1)


def clip_text(text: str, limit: int) -> str:
    """Shorten text to ``limit`` characters, keeping its head and tail.

    Args:
        text: Text to shorten.
        limit: Most characters to return.

    Returns:
        The text unchanged when it fits, otherwise its head, an omission marker naming how many
        characters were dropped, and its tail.
    """
    if len(text) <= limit:
        return text
    marker_bound = len(_MARKER.format(omitted=len(text)))
    keep = limit - marker_bound
    if keep <= 0:
        return text[: max(limit, 0)]
    tail = keep // 2
    head = keep - tail
    omitted = len(text) - head - tail
    return text[:head] + _MARKER.format(omitted=omitted) + (text[-tail:] if tail else "")


def _visible(text: str) -> str:
    """Replace control characters other than newline and tab with spaces.

    Args:
        text: Rendered window text.

    Returns:
        Text of the same length whose JSON encoding needs no six-byte escapes.
    """
    return _CONTROL.sub(" ", text)


def _compact(text: str) -> str:
    """Trim text and collapse runs of blank lines.

    Args:
        text: Raw message text.

    Returns:
        The compacted text.
    """
    return _BLANK_RUNS.sub("\n\n", text.replace("\r\n", "\n").strip())


def _arguments(text: str) -> str:
    """Render tool call arguments as compact JSON when they parse.

    Args:
        text: Arguments as emitted.

    Returns:
        Compact JSON, or the compacted original text when it is not JSON, nests too deeply to
        parse, or escapes a lone surrogate that cannot appear in rendered text.
    """
    try:
        compact = json.dumps(json.loads(text), ensure_ascii=False, separators=(",", ":"))
    except (ValueError, RecursionError):
        return _compact(text)
    return _compact(text) if _SURROGATE.search(compact) else compact


def render_message(message: TagMessage, names: Mapping[str, str]) -> str:
    """Render one message as role-tagged lines.

    Args:
        message: The message.
        names: Tool names by call id, for labeling tool results.

    Returns:
        ``[role] text`` for ordinary messages, ``[tool_result name] text`` for tool results and
        one ``[tool_call name] {args}`` line per assistant tool call.
    """
    text = _compact(message.text())
    lines: list[str] = []
    if message.role == "tool":
        name = names.get(message.tool_call_id or "")
        lines.append(f"{f'[tool_result {name}]' if name else '[tool_result]'} {text or '(empty)'}")
    elif text or not message.tool_calls:
        lines.append(f"[{message.role}] {text or '(empty)'}")
    lines.extend(
        f"[tool_call {call.function.name}] {_arguments(call.function.arguments)}"
        for call in message.tool_calls
    )
    return "\n".join(lines)


def _render_all(messages: Sequence[TagMessage]) -> list[str]:
    """Render messages in order, naming each tool result after the latest earlier call.

    Resolving names in message order keeps a later call that reuses a call id from relabeling
    an earlier result.

    Args:
        messages: Messages in position order, starting at position 0.

    Returns:
        One rendered block per message.
    """
    names: dict[str, str] = {}
    blocks: list[str] = []
    for message in messages:
        blocks.append(render_message(message, names))
        names.update((call.id, call.function.name) for call in message.tool_calls)
    return blocks


def _tool_lines(tools: Sequence[TagTool]) -> str:
    """Render the offered tools, one line each with a short description.

    Args:
        tools: Tools offered in the request.

    Returns:
        ``[tool name] description`` lines.
    """
    lines: list[str] = []
    for tool in tools:
        summary = next((line for line in _compact(tool.description or "").splitlines()), "")
        if len(summary) > TOOL_DESCRIPTION_CHARS:
            summary = summary[: TOOL_DESCRIPTION_CHARS - 3] + "..."
        lines.append(f"[tool {tool.name}] {summary}".rstrip())
    return "\n".join(lines)


def _fit(blocks: list[str], budget: int) -> list[str]:
    """Shorten the largest blocks equally until the joined blocks fit the budget.

    Blocks smaller than their fair share keep their full text; the rest are clipped to one
    shared cap. When the blocks do not fit and there are too many for each to keep
    ``MIN_BLOCK_CHARS``, the middle ones collapse into one omission line, keeping the first and
    last blocks.

    Args:
        blocks: Rendered blocks, joined with newlines.
        budget: Most characters of the joined result.

    Returns:
        Blocks in their original order whose newline-joined length fits the budget.
    """
    if sum(len(block) for block in blocks) + len(blocks) - 1 <= budget:
        return blocks
    most = max(budget // MIN_BLOCK_CHARS, 3)
    if len(blocks) > most:
        head = (most - 1) // 2
        tail = most - 1 - head
        omitted = len(blocks) - head - tail
        blocks = [*blocks[:head], f"[... {omitted} messages omitted ...]", *blocks[-tail:]]
    available = max(budget - (len(blocks) - 1), 0)
    if sum(len(block) for block in blocks) <= available:
        return blocks
    remaining = available
    lengths = sorted(len(block) for block in blocks)
    cap = 0
    for position, length in enumerate(lengths):
        share = remaining // (len(lengths) - position)
        if length > share:
            cap = share
            break
        remaining -= length
    return [clip_text(block, cap) if len(block) > cap else block for block in blocks]


def opening_window(exchange: TagExchange, budget: TagWindowBudget | None = None) -> TagWindow:
    """Render the conversation opening as one window.

    The window covers every message up to and including the first user message (or the
    leading instructions when there is no user message) and lists the offered tools before
    that user message. Oversized parts share the budget, each keeping its head and tail.

    Args:
        exchange: Any exchange of the conversation.
        budget: Character limits; the default budget when None.

    Returns:
        The conversation-scope window with index 0.
    """
    budget = budget or TagWindowBudget()
    first = first_user_index(exchange.messages)
    end = opening_end(exchange.messages)
    blocks = _render_all(exchange.messages[:end])
    if exchange.tools:
        blocks.insert(
            len(blocks) - 1 if first is not None else len(blocks), _tool_lines(exchange.tools)
        )
    return TagWindow(
        scope=TagScope.CONVERSATION,
        index=0,
        start_message=0,
        end_message=end,
        text=_visible(clip_text("\n".join(_fit(blocks, budget.window_chars)), budget.window_chars)),
    )


def turn_start(exchange: TagExchange) -> int:
    """Return the position where the exchange's new slice begins.

    The slice starts after the last assistant message of the request. The request's leading
    system and developer messages are never part of it: the opening window covers them.

    Args:
        exchange: The exchange.

    Returns:
        The first message position of the turn slice.
    """
    last_assistant = max(
        (i for i, message in enumerate(exchange.messages) if message.role == "assistant"),
        default=-1,
    )
    return max(last_assistant + 1, leading_instructions(exchange.messages))


def _task_header(exchange: TagExchange, limit: int) -> str:
    """Render the ``[task]`` header from the first user message.

    Args:
        exchange: The exchange.
        limit: Most characters of the header.

    Returns:
        The header line, shortened to ``limit``.
    """
    first = first_user_index(exchange.messages)
    task = "(no user message)" if first is None else _compact(exchange.messages[first].text())
    prefix = "[task] "
    return prefix + clip_text(task or "(empty)", limit - len(prefix))


def turn_windows(
    exchange: TagExchange, budget: TagWindowBudget | None = None
) -> tuple[TagWindow, ...]:
    """Split the exchange's new slice into overlapping windows.

    Windows break only between messages. Each starts with the ``[task]`` header and repeats up
    to ``overlap_chars`` of the previous window's trailing messages, while always adding at
    least one message the previous window did not show.

    Args:
        exchange: The exchange.
        budget: Character limits; the default budget when None.

    Returns:
        One or more turn-scope windows in order.
    """
    budget = budget or TagWindowBudget()
    start = turn_start(exchange)
    header = _task_header(exchange, budget.header_chars)
    body = budget.window_chars - len(header) - 1
    blocks = [clip_text(block, body) for block in _render_all(exchange.all_messages())[start:]]
    windows: list[TagWindow] = []
    first = 0
    while True:
        last = first + 1
        size = len(blocks[first])
        while last < len(blocks) and size + 1 + len(blocks[last]) <= body:
            size += 1 + len(blocks[last])
            last += 1
        windows.append(
            TagWindow(
                scope=TagScope.TURN,
                index=len(windows),
                start_message=start + first,
                end_message=start + last,
                text=_visible("\n".join((header, *blocks[first:last]))),
            )
        )
        if last == len(blocks):
            return tuple(windows)
        following = len(blocks[last])
        carried = 0
        resume = last
        while resume - 1 > first:
            step = len(blocks[resume - 1]) + 1
            if carried + step > budget.overlap_chars or carried + step + following > body:
                break
            carried += step
            resume -= 1
        first = resume


__all__ = [
    "CONSERVATIVE_CHARS_PER_TOKEN",
    "MAX_WINDOW_CHARS",
    "TagWindow",
    "TagWindowBudget",
    "clip_text",
    "opening_window",
    "render_message",
    "turn_start",
    "turn_windows",
]
