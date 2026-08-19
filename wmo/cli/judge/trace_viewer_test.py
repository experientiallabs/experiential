"""Tests for the full-screen scrollable trace proposal viewer."""

from __future__ import annotations

import io

import pytest
from rich.console import Console

from wmo.cli.judge import trace_viewer
from wmo.cli.judge.review_test import _proposal, _trace


def _plain(renderable: object) -> str:
    """Render one Rich renderable to plain text.

    Args:
        renderable: Any Rich renderable.

    Returns:
        Style-free rendered text.
    """
    console = Console(width=100, color_system=None, file=io.StringIO(), highlight=False)
    with console.capture() as capture:
        console.print(renderable)
    return capture.get()


def test_blocks_cover_task_conversation_outcome_and_judge_proposals() -> None:
    """Every conversation role, the outcome, and each axis proposal become styled blocks."""
    proposal = _proposal(
        _trace("trace-a", completion="Looked up the account.", failed=True),
        multi_axis=True,
    )

    blocks = trace_viewer.proposal_blocks(proposal)

    rendered = "".join(_plain(block.renderable) for block in blocks)
    assert "Original user request" in rendered
    assert "Resolve the customer's support request." in rendered
    assert "Please resolve customer issue trace-a." in rendered
    assert "Looked up the account." in rendered
    assert "Tool call · search" in rendered
    assert "Output · search" in rendered
    assert "Found the relevant account record." in rendered
    assert "Final response" in rendered
    assert "Final outcome" in rendered
    assert "Customer request failed" in rendered
    assert "Axis 1 of 2: Task success" in rendered
    assert "Axis 2 of 2: Policy compliance" in rendered
    assert "The response completed the requested account lookup." in rendered
    anchors = tuple(block.anchor for block in blocks if block.anchor is not None)
    assert anchors[0] == "Task"
    assert "Step 1" in anchors
    assert anchors[-2:] == ("Axis 1", "Axis 2")


def test_pairwise_blocks_separate_both_candidates() -> None:
    """Pairwise proposals label candidate A and candidate B sections and anchors."""
    proposal = _proposal(
        _trace("trace-a", completion="Candidate A finished.", failed=False),
        reference=_trace("trace-b", completion="Candidate B finished.", failed=False),
    )

    blocks = trace_viewer.proposal_blocks(proposal)

    anchors = tuple(block.anchor for block in blocks if block.anchor is not None)
    assert "Candidate A" in anchors
    assert "Candidate B" in anchors
    assert "A:Task" in anchors
    assert "B:Task" in anchors
    rendered = "".join(_plain(block.renderable) for block in blocks)
    assert "Candidate A finished." in rendered
    assert "Candidate B finished." in rendered


def test_single_command_tool_calls_render_as_shell_lines() -> None:
    """A bash-style single-command argument object is shown as one shell line."""
    text = trace_viewer._tool_call_text("bash", '{"command": "ls -la /tmp"}')

    assert text.plain == "$ ls -la /tmp"


def test_other_tool_arguments_render_as_indented_json() -> None:
    """Structured tool arguments keep their exact indented JSON form."""
    text = trace_viewer._tool_call_text("search", '{"query":"customer issue"}')

    assert '"query": "customer issue"' in text.plain


def test_render_block_lines_tracks_anchor_offsets_in_order() -> None:
    """Rendered anchors point at strictly increasing line offsets."""
    proposal = _proposal(_trace("trace-a", completion="Done.", failed=False))

    lines, anchors = trace_viewer.render_block_lines(
        trace_viewer.proposal_blocks(proposal), width=80
    )

    assert lines
    offsets = tuple(offset for offset, _ in anchors)
    assert offsets[0] == 0
    assert offsets == tuple(sorted(offsets))
    assert offsets[-1] < len(lines)


@pytest.mark.parametrize(
    ("key", "start", "expected"),
    [
        ("j", 0, 1),
        ("down", 3, 4),
        ("k", 3, 2),
        ("up", 0, 0),
        (" ", 0, 10),
        ("pgdn", 0, 10),
        ("b", 15, 5),
        ("d", 0, 5),
        ("u", 8, 3),
        ("g", 9, 0),
        ("home", 9, 0),
        ("G", 0, 20),
        ("end", 0, 20),
        ("x", 4, 4),
    ],
)
def test_apply_key_moves_and_clamps_the_scroll_offset(key: str, start: int, expected: int) -> None:
    """Line, page, half-page, and jump keys land on the clamped expected offset.

    Args:
        key: Normalized navigation key.
        start: Scroll offset before the key.
        expected: Clamped scroll offset after the key.
    """
    result = trace_viewer._apply_key(key, top=start, view_height=10, maximum_top=20, anchors=())

    assert result == expected


def test_apply_key_steps_between_anchor_offsets() -> None:
    """The step keys jump to the nearest anchor after or before the viewport."""
    anchors = ((0, "Task"), (7, "Step 1"), (14, "Outcome"))

    forward = trace_viewer._apply_key("n", top=0, view_height=5, maximum_top=30, anchors=anchors)
    backward = trace_viewer._apply_key(
        "p", top=forward, view_height=5, maximum_top=30, anchors=anchors
    )
    past_end = trace_viewer._apply_key("n", top=14, view_height=5, maximum_top=30, anchors=anchors)

    assert forward == 7
    assert backward == 0
    assert past_end == 30


def test_view_trace_proposal_draws_and_exits_on_q() -> None:
    """One scripted session paints the header, content, and footer then returns."""
    buffer = io.StringIO()
    console = Console(
        file=buffer,
        width=100,
        height=30,
        force_terminal=True,
        color_system="truecolor",
    )
    proposal = _proposal(_trace("trace-a", completion="Done.", failed=False), total=3)
    keys = iter(("j", " ", "n", "G", "q"))

    trace_viewer.view_trace_proposal(proposal, console=console, key_reader=lambda: next(keys))

    output = buffer.getvalue()
    assert "\x1b[?1049h" in output
    assert "\x1b[?1049l" in output
    assert "Trace 1 of 3" in output
    assert "q continue" in output


def test_view_trace_proposal_restores_the_screen_on_interrupt() -> None:
    """Ctrl-C leaves the alternate screen before the interrupt propagates."""
    buffer = io.StringIO()
    console = Console(
        file=buffer,
        width=100,
        height=30,
        force_terminal=True,
        color_system="truecolor",
    )
    proposal = _proposal(_trace("trace-a", completion="Done.", failed=False))

    with pytest.raises(KeyboardInterrupt):
        trace_viewer.view_trace_proposal(proposal, console=console, key_reader=lambda: "\x03")

    assert "\x1b[?1049l" in buffer.getvalue()
