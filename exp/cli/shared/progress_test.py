"""Tests for the shared CLI progress renderer."""

from __future__ import annotations

import io

import pytest
from rich.console import Console
from rich.live import Live
from rich.text import Text

from exp.cli.shared.progress import (
    Phase,
    ProgressDisplay,
    progress_display,
    progress_phases,
    qualified,
)
from exp.common.progress import ProgressEvent
from exp.simulation.retrieval.contracts import RAGAction
from exp.simulation.retrieval.embedding import (
    RAGEmbeddingCache,
    default_rag_embedder,
    embed_rag_texts,
)
from exp.simulation.retrieval.transitions import render_rag_key


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
    """A log gets each stage once when it starts and once when its counts complete."""
    console, buffer = _plain_console(interactive=False)
    with progress_display(console) as observe:
        observe(ProgressEvent(stage="normalization"))
        observe(ProgressEvent(stage="embeddings", completed=2, total=5, detail="serving index"))
        observe(ProgressEvent(stage="embeddings", completed=3, total=5, detail="serving index"))
        observe(ProgressEvent(stage="embeddings", completed=5, total=5, detail="serving index"))
    output = buffer.getvalue()
    assert output == (
        "  . normalization\n"
        "  . embeddings (serving index) 2/5\n"
        "  . embeddings (serving index) 5/5\n"
    )
    assert "\x1b[" not in output


def test_embedding_batches_retain_the_serving_and_fit_index_labels() -> None:
    """Chunk progress must preserve the index qualifier, including a fully reused fit index."""
    console, buffer = _plain_console(interactive=False)
    binding = default_rag_embedder()
    cache = RAGEmbeddingCache(binding, maximum_chunk_bytes=2_048)
    keys = (
        render_rag_key(
            task="task", initial_context={}, action=RAGAction(kind="message", content="action")
        ),
    )
    with progress_display(console) as observe:
        embed_rag_texts(binding, keys, cache=cache, progress=qualified(observe, "serving index"))
        embed_rag_texts(binding, keys, cache=cache, progress=qualified(observe, "fit-only index"))
    output = buffer.getvalue()
    assert "embeddings (serving index) 2/2" in output
    assert "embeddings (fit-only index) 2/2" in output


def test_interactive_display_keeps_one_line_and_leaves_no_stage_rows() -> None:
    """Stage changes replace the single line; nothing piles up above it."""
    console, buffer = _plain_console(interactive=True)
    with progress_display(console) as observe:
        observe(ProgressEvent(stage="embeddings", completed=0, total=4))
        observe(ProgressEvent(stage="embeddings", completed=4, total=4))
        observe(ProgressEvent(stage="RAG"))
    output = Text.from_ansi(buffer.getvalue()).plain
    assert "[x]" not in output
    assert "RAG" in output


def test_interactive_failure_marks_the_interrupted_status_unfinished() -> None:
    """A raising command leaves its interrupted status visible as unfinished."""
    console, buffer = _plain_console(interactive=True)
    with pytest.raises(RuntimeError, match="boom"):
        with progress_display(console) as observe:
            observe(ProgressEvent(stage="embeddings", completed=4, total=4))
            observe(ProgressEvent(stage="grounded model"))
            raise RuntimeError("boom")
    output = Text.from_ansi(buffer.getvalue()).plain
    assert "[ ] grounded model" in output
    assert "[x]" not in output


def test_noninteractive_failure_adds_no_extra_lines() -> None:
    """A raising command leaves the appended stream exactly as already written."""
    console, buffer = _plain_console(interactive=False)
    with pytest.raises(RuntimeError, match="boom"):
        with progress_display(console) as observe:
            observe(ProgressEvent(stage="grounded model"))
            raise RuntimeError("boom")
    assert buffer.getvalue() == "  . grounded model\n"


def _manual(console: Console, phases: tuple[Phase, ...]) -> tuple[ProgressDisplay, Live]:
    """A started display, and its line, which refreshes only when the test asks."""
    display = ProgressDisplay(console, phases)
    live = display._live
    assert live is not None
    live.auto_refresh = False
    display.start()
    return display, live


def test_countable_stage_renders_a_bar_counts_and_rate_based_eta(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The bar follows the running stage's counts, with an estimate once a rate exists."""
    monkeypatch.setenv("TERM", "xterm-256color")
    now = [0.0]
    monkeypatch.setattr("exp.cli.shared.progress.monotonic", lambda: now[0])
    console, buffer = _plain_console(interactive=True)
    display, live = _manual(console, (Phase("only", "", 1.0),))
    observe = display.phase("only")
    try:
        observe(ProgressEvent(stage="judgments", completed=0, total=4))
        live.refresh()
        assert "eta" not in Text.from_ansi(buffer.getvalue()).plain, "no rate observed yet"
        now[0] = 10.0
        observe(ProgressEvent(stage="judgments", completed=2, total=4))
        live.refresh()
        line = Text.from_ansi(buffer.getvalue()).plain
        assert "\u2501" in line and "50%" in line and "judgments 2/4" in line
        assert "eta 10s" in line and "10s elapsed" in line
    finally:
        display.stop()


def test_elapsed_time_stays_live_between_completed_work(monkeypatch: pytest.MonkeyPatch) -> None:
    """A long provider request keeps the line moving without inventing completed work."""
    monkeypatch.setenv("TERM", "xterm-256color")
    now = [100.0]
    monkeypatch.setattr("exp.cli.shared.progress.monotonic", lambda: now[0])
    console, buffer = _plain_console(interactive=True)
    display, live = _manual(console, (Phase("only", "", 1.0),))
    try:
        display.phase("only")(ProgressEvent(stage="Rollouts", completed=0, total=100))
        live.refresh()
        previous = len(buffer.getvalue())
        now[0] = 165.0
        live.refresh()
        current = Text.from_ansi(buffer.getvalue()[previous:]).plain
        assert "Rollouts 0/100" in current and "1m05s elapsed" in current
        assert "eta" not in current
    finally:
        display.stop()
    assert "\x1b[?1049h" not in buffer.getvalue()
    assert "\x1b[?25h" in buffer.getvalue()


_PHASES = (
    Phase("index", "Building indexes", 0.1),
    Phase("learning", "Learning", 0.7, driver="Learning from traces", unit="traces"),
    Phase("retest", "Retesting misses", 0.2, driver="Replaying traces", unit="traces"),
)


def test_phases_fill_one_bar_by_weight_and_show_counts_and_the_inner_stage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Driver counts move the phase; other stages only name what is running inside it."""
    monkeypatch.setenv("TERM", "xterm-256color")
    console, buffer = _plain_console(interactive=True)
    display, live = _manual(console, _PHASES)
    try:
        display.phase("index")(ProgressEvent(stage="serving index"))
        display.complete("index")
        learn = display.phase("learning")
        learn(ProgressEvent(stage="Learning from traces", completed=4, total=12))
        learn(ProgressEvent(stage="Simulating turns", completed=3, total=8, detail="trace 5/12"))
        live.refresh()
        line = Text.from_ansi(buffer.getvalue()).plain
        assert "33%" in line, "index 0.1 + learning 0.7 x 4/12"
        assert "Learning 4/12 traces · Simulating turns (trace 5/12) 3/8" in line
        with display.paused():
            console.print("a question")
        display.phase("retest")(ProgressEvent(stage="Replaying traces", completed=2, total=9))
        live.refresh()
        assert "84%" in Text.from_ansi(buffer.getvalue()).plain
    finally:
        display.stop()


def test_phases_in_a_log_print_each_phase_and_its_completion_only() -> None:
    """A log names each phase; a counted phase logs its completion, not its inner stages."""
    console, buffer = _plain_console(interactive=False)
    with progress_phases(console, _PHASES) as display:
        display.phase("index")(ProgressEvent(stage="serving index"))
        learn = display.phase("learning")
        learn(ProgressEvent(stage="Learning from traces", completed=0, total=2))
        learn(ProgressEvent(stage="Simulating turns", completed=1, total=8, detail="trace 1/2"))
        learn(ProgressEvent(stage="Learning from traces", completed=2, total=2))
    assert buffer.getvalue() == (
        "  Building indexes...\n  . serving index\n  Learning...\n  . Learning from traces 2/2\n"
    )


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
