"""Tests for window rendering, the opening window and turn window splitting."""

from __future__ import annotations

import json
from collections.abc import Mapping

import pytest
from pydantic import ValidationError

from exp.common.tags import windows as windows_module
from exp.common.tags.bank import TagScope
from exp.common.tags.exchange import (
    TagExchange,
    TagFunction,
    TagImagePart,
    TagMessage,
    TagTextPart,
    TagTool,
    TagToolCall,
    opening_key,
)
from exp.common.tags.windows import (
    CONSERVATIVE_CHARS_PER_TOKEN,
    MAX_WINDOW_CHARS,
    TagWindowBudget,
    clip_text,
    opening_window,
    render_message,
    turn_start,
    turn_windows,
)

SMALL = TagWindowBudget(window_chars=2_000, overlap_chars=300, header_chars=200)


def _call(call_id: str, name: str, arguments: str) -> TagToolCall:
    """Build one tool call.

    Args:
        call_id: Call id.
        name: Tool name.
        arguments: Argument text.

    Returns:
        The call.
    """
    return TagToolCall(id=call_id, function=TagFunction(name=name, arguments=arguments))


def _agent_exchange(results: int = 1, result_chars: int = 40) -> TagExchange:
    """Build a coding-agent exchange whose new slice holds tool results and a reply.

    Args:
        results: Number of tool calls and results in the new slice.
        result_chars: Characters of each tool result.

    Returns:
        The exchange.
    """
    calls = tuple(_call(f"c{i}", "bash", f'{{"command": "step {i}"}}') for i in range(results))
    return TagExchange(
        exchange_id="req",
        tools=(TagTool(name="bash", description="Run a shell command.\nMore detail."),),
        messages=(
            TagMessage(role="system", content="You are a coding agent."),
            TagMessage(role="user", content="Fix the failing date parser test"),
            TagMessage(role="assistant", content="Looking.", tool_calls=calls),
            *(
                TagMessage(role="tool", tool_call_id=f"c{i}", content=str(i % 10) * result_chars)
                for i in range(results)
            ),
        ),
        response=(TagMessage(role="assistant", content="All tests pass now."),),
    )


def test_render_message_uses_compact_role_tags() -> None:
    """Each role renders as a tagged line and tool calls render compact arguments."""
    names = {"c1": "bash"}
    assert render_message(TagMessage(role="user", content="  hi\r\n\n\n\nthere  "), names) == (
        "[user] hi\n\nthere"
    )
    assistant = TagMessage(
        role="assistant",
        content="Running it.",
        tool_calls=(
            _call("c1", "bash", '{ "command" : "pytest -q" }'),
            _call("c2", "edit", "not json"),
        ),
    )
    assert render_message(assistant, names) == (
        '[assistant] Running it.\n[tool_call bash] {"command":"pytest -q"}\n'
        "[tool_call edit] not json"
    )
    calls_only = TagMessage(role="assistant", tool_calls=(_call("c1", "bash", "{}"),))
    assert render_message(calls_only, names) == "[tool_call bash] {}"
    result = TagMessage(role="tool", tool_call_id="c1", content="exit 1")
    assert render_message(result, names) == "[tool_result bash] exit 1"
    orphan = TagMessage(role="tool", tool_call_id="missing", content="")
    assert render_message(orphan, names) == "[tool_result] (empty)"
    picture = TagMessage(
        role="user", content=(TagTextPart(text="see "), TagImagePart(media_type="image/png"))
    )
    assert render_message(picture, names) == "[user] see [image]"
    assert render_message(TagMessage(role="assistant"), names) == "[assistant] (empty)"


def test_clip_text_keeps_head_and_tail_with_an_exact_marker() -> None:
    """Clipped text fits the limit and says how much was dropped."""
    text = "H" * 500 + "M" * 1_000 + "T" * 500
    clipped = clip_text(text, 300)
    assert len(clipped) <= 300
    assert clipped.startswith("H" * 100) and clipped.endswith("T" * 100)
    head, marker, tail = clipped.partition("\n[... ")
    omitted = int(tail.split(" characters omitted")[0])
    assert omitted == len(text) - len(head) - len(clipped.rsplit("...]\n", 1)[1])
    assert marker
    assert clip_text("short", 300) == "short"


def test_opening_window_covers_instructions_tools_and_first_request() -> None:
    """The opening lists tools before the first user message and stops there."""
    window = opening_window(_agent_exchange())
    assert (window.scope, window.index) == (TagScope.CONVERSATION, 0)
    assert (window.start_message, window.end_message) == (0, 2)
    assert window.text.splitlines() == [
        "[system] You are a coding agent.",
        "[tool bash] Run a shell command.",
        "[user] Fix the failing date parser test",
    ]


def test_opening_window_shares_the_budget_and_keeps_short_parts_whole() -> None:
    """A huge system prompt is clipped while the short first request stays whole."""
    exchange = TagExchange(
        exchange_id="req",
        messages=(
            TagMessage(role="system", content="S" * 10_000),
            TagMessage(role="user", content="Summarize the attached report."),
        ),
        response=(TagMessage(role="assistant", content="Summary."),),
    )
    window = opening_window(exchange, SMALL)
    assert len(window.text) <= SMALL.window_chars
    assert window.text.endswith("[user] Summarize the attached report.")
    assert "characters omitted" in window.text


def test_opening_window_without_a_user_message_covers_instructions() -> None:
    """An instruction-only request opens with its leading instructions."""
    exchange = TagExchange(
        exchange_id="req",
        messages=(TagMessage(role="developer", content="Classify the input."),),
        response=(TagMessage(role="assistant", content="ok"),),
    )
    window = opening_window(exchange)
    assert (window.start_message, window.end_message) == (0, 1)
    assert window.text == "[developer] Classify the input."


def test_turn_slice_starts_after_the_last_assistant_message() -> None:
    """Only new messages and the response are in the turn window, under the task header."""
    exchange = _agent_exchange()
    assert turn_start(exchange) == 3
    (window,) = turn_windows(exchange)
    assert (window.scope, window.index, window.start_message, window.end_message) == (
        TagScope.TURN,
        0,
        3,
        5,
    )
    assert window.text.splitlines() == [
        "[task] Fix the failing date parser test",
        "[tool_result bash] " + "0" * 40,
        "[assistant] All tests pass now.",
    ]


def test_first_turn_skips_the_leading_instructions() -> None:
    """On the first request the opening's system prompt is not repeated in the turn."""
    exchange = TagExchange(
        exchange_id="req",
        messages=(
            TagMessage(role="system", content="Be terse."),
            TagMessage(role="user", content="What is 2 + 2?"),
        ),
        response=(TagMessage(role="assistant", content="4"),),
    )
    (window,) = turn_windows(exchange)
    assert (window.start_message, window.end_message) == (1, 3)
    assert window.text == "[task] What is 2 + 2?\n[user] What is 2 + 2?\n[assistant] 4"


def test_long_turns_split_at_message_boundaries_with_overlap() -> None:
    """Windows fit the budget, start with the header, overlap, and cover every message."""
    exchange = _agent_exchange(results=24, result_chars=200)
    windows = turn_windows(exchange, SMALL)
    assert len(windows) > 2
    assert [window.index for window in windows] == list(range(len(windows)))
    assert windows[0].start_message == turn_start(exchange)
    assert windows[-1].end_message == len(exchange.all_messages())
    for previous, current in zip(windows, windows[1:], strict=False):
        assert current.start_message < previous.end_message < current.end_message
    for window in windows:
        assert len(window.text) <= SMALL.window_chars
        assert window.text.startswith("[task] Fix the failing date parser test\n")
        assert "characters omitted" not in window.text


def test_zero_overlap_windows_tile_the_slice() -> None:
    """Without overlap each message appears in exactly one window."""
    budget = TagWindowBudget(window_chars=2_000, overlap_chars=0, header_chars=200)
    windows = turn_windows(_agent_exchange(results=12, result_chars=400), budget)
    for previous, current in zip(windows, windows[1:], strict=False):
        assert current.start_message == previous.end_message


def test_one_oversized_message_keeps_head_and_tail() -> None:
    """A tool result larger than a window is clipped inside its own window."""
    exchange = _agent_exchange(results=1, result_chars=0).model_copy(
        update={
            "messages": (
                *_agent_exchange().messages[:3],
                TagMessage(role="tool", tool_call_id="c0", content="BEGIN" + "x" * 50_000 + "END"),
            )
        }
    )
    windows = turn_windows(exchange, SMALL)
    assert len(windows) == 2
    oversized = windows[0]
    assert (oversized.start_message, oversized.end_message) == (3, 4)
    assert len(oversized.text) <= SMALL.window_chars
    assert "[tool_result bash] BEGIN" in oversized.text
    assert oversized.text.endswith("END")
    assert "characters omitted" in oversized.text
    assert windows[1].text.endswith("[assistant] All tests pass now.")


def test_task_header_is_clipped_to_its_budget() -> None:
    """A long first request is shortened in every turn header."""
    exchange = TagExchange(
        exchange_id="req",
        messages=(
            TagMessage(role="user", content="R" * 5_000),
            TagMessage(role="assistant", content="ok"),
            TagMessage(role="user", content="next"),
        ),
        response=(TagMessage(role="assistant", content="done"),),
    )
    (window,) = turn_windows(exchange, SMALL)
    header = window.text.split("\n[user] next")[0]
    assert len(header) <= SMALL.header_chars
    assert header.startswith("[task] RRR")


def test_budget_validation_and_token_sizing() -> None:
    """Budgets keep room for new content and windows never outgrow the token or byte limit."""
    with pytest.raises(ValidationError, match="at most half"):
        TagWindowBudget(window_chars=2_000, overlap_chars=900, header_chars=200)
    budget = TagWindowBudget.for_input_tokens(8_192)
    assert budget.window_chars * CONSERVATIVE_CHARS_PER_TOKEN <= 8_192 - 2_048
    assert TagWindowBudget.for_input_tokens(200_000).window_chars <= MAX_WINDOW_CHARS
    with pytest.raises(ValueError, match="larger context"):
        TagWindowBudget.for_input_tokens(2_500)


def test_reused_call_ids_keep_the_name_of_the_earlier_call() -> None:
    """A later call reusing an id does not relabel an earlier tool result."""
    exchange = TagExchange(
        exchange_id="req",
        messages=(
            TagMessage(role="user", content="Copy the file"),
            TagMessage(role="assistant", tool_calls=(_call("call_1", "read_file", "{}"),)),
            TagMessage(role="tool", tool_call_id="call_1", content="contents"),
        ),
        response=(TagMessage(role="assistant", tool_calls=(_call("call_1", "write_file", "{}"),)),),
    )
    (window,) = turn_windows(exchange)
    assert "[tool_result read_file] contents" in window.text
    assert window.text.endswith("[tool_call write_file] {}")


def test_windows_hold_only_text_a_request_can_carry() -> None:
    """Control characters and decoded lone surrogates render as spaces.

    JSON then costs at most four bytes per character and the text is valid UTF-8.
    """
    budget = TagWindowBudget(window_chars=MAX_WINDOW_CHARS)
    exchange = TagExchange(
        exchange_id="req",
        messages=(TagMessage(role="system", content="\x00\x1b" * 40_000),),
        response=(
            TagMessage(
                role="assistant",
                content="a\tb\x07c",
                tool_calls=(_call("c", "write", '{"text":"cut \\ud83d"}'),),
            ),
        ),
    )
    windows = (opening_window(exchange, budget), *turn_windows(exchange, budget))
    for window in windows:
        encoded = json.dumps(window.text, ensure_ascii=False).encode("utf-8")
        assert len(encoded) <= 4 * len(window.text) + 2
        assert "\x00" not in window.text and "\x1b" not in window.text
    assert windows[-1].text.endswith('[assistant] a\tb c\n[tool_call write] {"text":"cut  "}')


def test_tool_results_fall_back_to_their_own_name() -> None:
    """A chat tool result naming its tool is labeled even when its call is not in view."""
    result = TagMessage.model_validate(
        {"role": "tool", "tool_call_id": "call_9", "name": "kb_search", "content": "3 articles"}
    )
    assert result.name == "kb_search"
    assert render_message(result, {}) == "[tool_result kb_search] 3 articles"
    assert render_message(result, {"call_9": "search_v2"}) == "[tool_result search_v2] 3 articles"
    named_user = TagMessage.model_validate({"role": "user", "name": "alice", "content": "hi"})
    assert render_message(named_user, {}) == "[user] hi"
    assert TagMessage.model_validate_json(result.model_dump_json()) == result


def test_opening_lists_tools_in_canonical_order() -> None:
    """Caller tool order changes neither the rendered opening nor its key."""
    tools = tuple(TagTool(name=f"tool_{i:02d}", description="d" * 150) for i in range(25))
    forward = TagExchange(
        exchange_id="req",
        tools=tools,
        messages=(TagMessage(role="user", content="Go"),),
        response=(TagMessage(role="assistant", content="ok"),),
    )
    backward = forward.model_copy(update={"tools": tools[::-1]})
    assert opening_window(forward, SMALL).text == opening_window(backward, SMALL).text
    assert opening_key(forward) == opening_key(backward)


def test_turn_windows_never_render_earlier_history(monkeypatch: pytest.MonkeyPatch) -> None:
    """Only the new slice is rendered; earlier messages only name their calls.

    Args:
        monkeypatch: Replaces the renderer to record which messages it renders.
    """
    rendered: list[TagMessage] = []
    original = windows_module.render_message

    def recording(message: TagMessage, names: Mapping[str, str]) -> str:
        """Record and render one message.

        Args:
            message: Message to render.
            names: Tool names by call id.

        Returns:
            The rendered message.
        """
        rendered.append(message)
        return original(message, names)

    monkeypatch.setattr(windows_module, "render_message", recording)
    exchange = _agent_exchange(results=2)
    turn_windows(exchange)
    assert rendered == list(exchange.all_messages()[turn_start(exchange) :])
