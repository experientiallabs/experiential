"""Tests for the shared CLI progress renderer."""

from __future__ import annotations

import io

import pytest
from rich.console import Console

from exp.cli.shared.progress import progress_display, qualified
from exp.common.progress import ProgressEvent


def _plain_console(*, interactive: bool) -> tuple[Console, io.StringIO]:
    """Return a capture console with an explicitly forced interactivity mode."""
    buffer = io.StringIO()
    console = Console(
        file=buffer,
        force_terminal=interactive,
        force_interactive=interactive,
        width=100,
        color_system=None,
        legacy_windows=False,
    )
    return console, buffer


def test_noninteractive_output_is_stable_lines_without_cursor_control() -> None:
    """A piped stream receives one plain appended line per update."""
    console, buffer = _plain_console(interactive=False)
    with progress_display(console) as observe:
        observe(ProgressEvent(stage="normalization"))
        observe(ProgressEvent(stage="embeddings", completed=2, total=5, detail="serving index"))
    output = buffer.getvalue()
    assert output == ("  . normalization\n  . embeddings (serving index) 2/5\n")
    assert "\x1b[" not in output


def test_interactive_display_prints_each_finished_stage_once() -> None:
    """The in-place line finalizes a stage row when the next stage begins."""
    console, buffer = _plain_console(interactive=True)
    with progress_display(console) as observe:
        observe(ProgressEvent(stage="embeddings", completed=0, total=4))
        observe(ProgressEvent(stage="embeddings", completed=4, total=4))
        observe(ProgressEvent(stage="RAG"))
    output = buffer.getvalue()
    assert output.count("[x] embeddings 4/4") == 1
    assert output.count("[x] RAG") == 1
    assert "[x] embeddings 0/4" not in output


def test_interactive_repeated_stage_updates_stay_on_one_row() -> None:
    """Same-stage count updates replace the live line instead of stacking rows."""
    console, buffer = _plain_console(interactive=True)
    with progress_display(console) as observe:
        observe(ProgressEvent(stage="judgments", completed=1, total=3, detail="fit"))
        observe(ProgressEvent(stage="judgments", completed=2, total=3, detail="fit"))
        observe(ProgressEvent(stage="judgments", completed=3, total=3, detail="fit"))
    output = buffer.getvalue()
    assert output.count("[x] judgments (fit)") == 1


def test_interactive_failure_never_marks_the_active_stage_finished() -> None:
    """A raising command leaves its interrupted stage marked unfinished, not completed."""
    console, buffer = _plain_console(interactive=True)
    with pytest.raises(RuntimeError, match="boom"):
        with progress_display(console) as observe:
            observe(ProgressEvent(stage="embeddings", completed=4, total=4))
            observe(ProgressEvent(stage="grounded model"))
            raise RuntimeError("boom")
    output = buffer.getvalue()
    assert "[x] embeddings 4/4" in output
    assert "[ ] grounded model" in output
    assert "[x] grounded model" not in output


def test_noninteractive_failure_adds_no_extra_lines() -> None:
    """A raising command leaves the appended stream exactly as already written."""
    console, buffer = _plain_console(interactive=False)
    with pytest.raises(RuntimeError, match="boom"):
        with progress_display(console) as observe:
            observe(ProgressEvent(stage="grounded model"))
            raise RuntimeError("boom")
    assert buffer.getvalue() == "  . grounded model\n"


def test_interactive_countable_stage_renders_a_bar_and_rate_based_eta(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A countable stage shows a progress bar and an ETA once a rate is observable."""
    clock = iter([0.0, 10.0, 10.0])
    monkeypatch.setattr("exp.cli.shared.progress.monotonic", lambda: next(clock))
    console, buffer = _plain_console(interactive=True)
    with progress_display(console) as observe:
        observe(ProgressEvent(stage="judgments", completed=0, total=4))
        observe(ProgressEvent(stage="judgments", completed=2, total=4))
    output = buffer.getvalue()
    assert "\u2501" in output
    assert "2/4" in output
    assert "eta 10s" in output


def test_interactive_first_countable_update_has_no_eta() -> None:
    """No ETA is invented before any progress rate is observable."""
    console, buffer = _plain_console(interactive=True)
    with progress_display(console) as observe:
        observe(ProgressEvent(stage="judgments", completed=0, total=4))
    assert "eta" not in buffer.getvalue()


def test_single_line_display_prints_no_permanent_finished_rows() -> None:
    """A single-line section keeps one in-place line across every stage change."""
    console, buffer = _plain_console(interactive=True)
    with progress_display(console, single_line=True) as observe:
        observe(ProgressEvent(stage="preflight"))
        observe(ProgressEvent(stage="judgments", completed=1, total=2))
        observe(ProgressEvent(stage="artifact publication"))
    assert "[x]" not in buffer.getvalue()


def test_single_line_failure_still_marks_the_active_stage_unfinished() -> None:
    """A raising single-line section leaves its interrupted stage visible."""
    console, buffer = _plain_console(interactive=True)
    with pytest.raises(RuntimeError, match="boom"):
        with progress_display(console, single_line=True) as observe:
            observe(ProgressEvent(stage="fitting"))
            raise RuntimeError("boom")
    assert "[ ] fitting" in buffer.getvalue()


def test_qualified_attaches_a_detail_and_preserves_counts() -> None:
    """The wrapper forwards events with the owning command's qualifier attached."""
    seen: list[ProgressEvent] = []
    hook = qualified(seen.append, "serving index")
    assert hook is not None
    hook(ProgressEvent(stage="embeddings", completed=1, total=2))
    hook(ProgressEvent(stage="embeddings", completed=2, total=2, detail="existing"))
    assert seen == [
        ProgressEvent(stage="embeddings", completed=1, total=2, detail="serving index"),
        ProgressEvent(stage="embeddings", completed=2, total=2, detail="existing"),
    ]


def test_qualified_without_observer_is_absent() -> None:
    """No wrapper is constructed when nobody is watching."""
    assert qualified(None, "serving index") is None
