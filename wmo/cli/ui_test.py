"""Tests for shared terminal UX."""

from __future__ import annotations

import importlib

import pytest
import typer
from rich.console import Console

from wmo.cli.ui import _decode_key, _step_selection, models_table, select_model
from wmo.common.config import ModelInfo

ui_module = importlib.import_module("wmo.cli.ui")


def _scripted_reader(answers: list[str]):  # noqa: ANN202 - returns a PromptReader
    """Return successive `answers`, ignoring the rendered prompt text."""
    it = iter(answers)
    return lambda _prompt: next(it)


def test_models_table_renders_names() -> None:
    console = Console(force_terminal=False, no_color=True, width=120)
    table = models_table([ModelInfo(name="airline", serve_provider="bedrock", serve_model="opus")])
    with console.capture() as cap:
        console.print(table)
    assert "airline" in cap.get()


def test_decode_key_maps_arrows_and_passes_plain_chars() -> None:
    assert _decode_key("\x1b[A") == "up"
    assert _decode_key("\x1bOA") == "up"  # application cursor mode
    assert _decode_key("\x1b[B") == "down"
    assert _decode_key("\x1b[1;5A") == "esc"  # modified arrows are inert, not a stray '5'
    assert _decode_key("\x1b[5~") == "esc"  # PgUp is inert
    assert _decode_key("\x1b") == "esc"
    assert _decode_key("j") == "j"
    assert _decode_key("\r") == "\r"


def test_arrow_select_moves_pointer_and_accepts(monkeypatch) -> None:  # noqa: ANN001
    keys = iter(["\x1b[B", "\x1b[1;5A", "\r"])  # down, inert modified arrow, Enter
    monkeypatch.setattr(ui_module.click, "getchar", lambda: next(keys))
    console = Console(force_terminal=False, no_color=True, width=100, record=True)
    assert ui_module._arrow_select(console, ["a", "b", "c"], 0) == 1
    assert "❯ b" in console.export_text()  # pointer painted on the accepted row


def test_split_keys_separates_batched_sequences() -> None:
    assert ui_module._split_keys("\x1b[B\x1b[B") == ["\x1b[B", "\x1b[B"]
    assert ui_module._split_keys("\x1b[1;5A") == ["\x1b[1;5A"]
    assert ui_module._split_keys("jk\r") == ["j", "k", "\r"]
    assert ui_module._split_keys("\x1bOA5") == ["\x1bOA", "5"]
    assert ui_module._split_keys("\x1b") == ["\x1b"]


def test_arrow_select_reveals_hidden_rows_on_navigation(monkeypatch) -> None:  # noqa: ANN001
    """Reveal collapsed picker rows when the selection reaches the affordance."""
    keys = iter(["\x1b[B", "\x1b[B", "\x1b[B", "\r"])
    monkeypatch.setattr(ui_module.click, "getchar", lambda: next(keys))
    console = Console(force_terminal=False, no_color=True, width=100, record=True)
    chosen = ui_module._arrow_select(console, ["a", "b"], 0, ["c", "d"])
    assert chosen == 3
    out = console.export_text()
    assert "… 2 more" in out
    assert "❯ d" in out


def test_select_collapsed_keeps_numbered_fallback_complete() -> None:
    """Keep all options available to deterministic non-TTY selection."""
    console = Console(force_terminal=False, no_color=True, width=100, record=True)
    chosen = ui_module._select(
        console,
        _scripted_reader(["4"]),
        "Pick",
        ["a", "b", "c", "d"],
        "a",
        interactive=False,
        collapsed=2,
    )
    assert chosen == "d"
    assert "4." in console.export_text()


def test_arrow_select_aborts_on_eof(monkeypatch) -> None:  # noqa: ANN001
    monkeypatch.setattr(ui_module.click, "getchar", lambda: "")
    console = Console(force_terminal=False, no_color=True, width=100)
    with pytest.raises(typer.Abort):
        ui_module._arrow_select(console, ["a", "b"], 0)


def test_step_selection_navigates_wraps_and_accepts() -> None:
    assert _step_selection("down", 0, 3) == (1, False)
    assert _step_selection("j", 1, 3) == (2, False)
    assert _step_selection("down", 2, 3) == (0, False)
    assert _step_selection("up", 0, 3) == (2, False)
    assert _step_selection("k", 2, 3) == (1, False)
    assert _step_selection("\r", 1, 3) == (1, True)
    assert _step_selection("2", 0, 3) == (1, True)
    assert _step_selection("9", 0, 3) == (0, False)
    assert _step_selection("x", 1, 3) == (1, False)


def test_select_model_single_returns_without_prompting() -> None:
    console = Console(force_terminal=False, no_color=True, width=100)
    info = ModelInfo(name="only", serve_provider="bedrock", serve_model="opus")
    assert select_model(console, [info]) == "only"


def test_select_model_picks_by_number() -> None:
    console = Console(force_terminal=False, no_color=True, width=100)
    infos = [
        ModelInfo(name="airline", serve_provider="bedrock", serve_model="opus"),
        ModelInfo(name="retail", serve_provider="bedrock", serve_model="opus"),
    ]
    assert select_model(console, infos, reader=_scripted_reader(["2"])) == "retail"


def test_select_model_reprompts_then_accepts_name() -> None:
    console = Console(force_terminal=False, no_color=True, width=100)
    infos = [
        ModelInfo(name="airline", serve_provider="bedrock", serve_model="opus"),
        ModelInfo(name="retail", serve_provider="bedrock", serve_model="opus"),
    ]
    chosen = select_model(console, infos, reader=_scripted_reader(["9", "airline"]))
    assert chosen == "airline"


def test_select_model_survives_unicode_digit_input() -> None:
    console = Console(force_terminal=False, no_color=True, width=100)
    infos = [
        ModelInfo(name="airline", serve_provider="bedrock", serve_model="opus"),
        ModelInfo(name="retail", serve_provider="bedrock", serve_model="opus"),
    ]
    chosen = select_model(console, infos, reader=_scripted_reader(["²", "1"]))
    assert chosen == "airline"
