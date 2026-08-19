"""Selection prompt tests for the keyboard screens on a real terminal and the typed fallback."""

from __future__ import annotations

import io
from collections.abc import Callable

import pytest
from rich.console import Console

from wmo.cli.shared.picker import (
    PickerAction,
    PickerEvent,
    PickerKey,
    PickerOption,
    interpret_key_bytes,
    select_many,
    select_many_list,
    select_one,
    select_one_list,
)
from wmo.conftest import TerminalRun

_DOWN = "\x1b[B"
_UP = "\x1b[A"
_ENTER = "\r"
_SPACE = " "


class ScriptedConsole(Console):
    """One console that answers every prompt from a script and records its output."""

    def __init__(self, answers: str, *, width: int = 100) -> None:
        """Build a console reading scripted lines into an in-memory transcript.

        Args:
            answers: Newline-separated answers the screens read in order.
            width: Reported terminal width used for wrapping and narrow-list layout.
        """
        self._transcript = io.StringIO()
        super().__init__(
            file=self._transcript,
            force_terminal=False,
            width=width,
            no_color=True,
            highlight=False,
        )
        self._answers = iter(answers.splitlines())

    def input(self, prompt: object = "", **_: object) -> str:
        """Answer one prompt from the script instead of reading a terminal.

        Args:
            prompt: Prompt text or rich renderable the screen printed.
            _: Prompt options the scripted console ignores, such as ``password``.

        Returns:
            The next scripted answer.
        """
        self._transcript.write(str(prompt))
        return next(self._answers)

    @property
    def output(self) -> str:
        """Return everything the screens printed so far."""
        return self._transcript.getvalue()


def _console(answers: str) -> ScriptedConsole:
    """Build one scripted console for a selection screen.

    Args:
        answers: Newline-separated answers the screen reads in order.

    Returns:
        The scripted console, whose transcript records every printed row.
    """
    return ScriptedConsole(answers)


def _options(count: int) -> tuple[PickerOption, ...]:
    """Build a numbered option list for list rendering and filtering tests.

    Args:
        count: Number of rows to build.

    Returns:
        Rows named ``model-1`` through ``model-<count>``.
    """
    return tuple(
        PickerOption(value=f"model-{index}", label=f"model-{index}", detail=f"detail {index}")
        for index in range(1, count + 1)
    )


def test_numbers_and_ranges_toggle_rows_before_an_empty_line_accepts() -> None:
    """A user selects by number and range, deselects by repeating, then accepts."""
    console = _console("1,3\n2-3\n\n")

    result = select_many(console, title="Models", options=_options(4))

    assert result.action is None
    assert result.values == ("model-1", "model-2")


def test_all_and_none_words_select_and_clear_the_visible_rows() -> None:
    """The list words select everything visible and clear the whole selection."""
    console = _console("all\nnone\nall\n\n")

    result = select_many(console, title="Models", options=_options(3))

    assert result.values == ("model-1", "model-2", "model-3")


def test_typed_text_filters_a_long_list_and_all_selects_only_matches() -> None:
    """Typed text narrows a long list, so selection stays inside the filtered rows."""
    console = _console("model-12\nall\n\n")

    result = select_many(console, title="Models", options=_options(20))

    assert result.values == ("model-12",)
    assert "8 more (type 'more' to show all" in console.output
    assert "filter: model-12" in console.output


def test_preselected_values_survive_reentering_a_screen() -> None:
    """Answers already given stay selected when a screen is shown again."""
    console = _console("\n")

    result = select_many(
        console,
        title="Models",
        options=_options(3),
        preselected=("model-2", "absent"),
    )

    assert result.values == ("model-2",)
    assert "[x] 2. model-2" in console.output


def test_multi_select_rejects_an_empty_selection_below_the_minimum() -> None:
    """A required screen explains the minimum instead of accepting nothing."""
    console = _console("\n2\n\n")

    result = select_many(console, title="Models", options=_options(3))

    assert result.values == ("model-2",)
    assert "Select at least 1." in console.output


def test_multi_select_can_accept_nothing_when_the_role_is_optional() -> None:
    """An optional screen accepts an empty line as an empty selection."""
    console = _console("\n")

    result = select_many(console, title="Candidates", options=_options(3), minimum=0)

    assert result.values == ()


@pytest.mark.parametrize("answer", ["b", "back", "B"])
def test_back_words_return_back_navigation(answer: str) -> None:
    """Any back word navigates back rather than selecting a row.

    Args:
        answer: Back word the user typed.
    """
    console = _console(f"{answer}\n")

    assert select_many(console, title="Models", options=_options(2)).action is PickerAction.BACK


@pytest.mark.parametrize("answer", ["q", "quit", "cancel"])
def test_cancel_words_return_cancel_navigation(answer: str) -> None:
    """Any cancel word cancels the screen.

    Args:
        answer: Cancel word the user typed.
    """
    console = _console(f"{answer}\n")

    result = select_one(console, title="World model", options=_options(2))

    assert result.action is PickerAction.CANCEL


def test_single_select_keeps_the_default_on_an_empty_line() -> None:
    """A prior answer is accepted with an empty line and shown in the prompt."""
    console = _console("\n")

    result = select_one(console, title="Judge", options=_options(3), default="model-3")

    assert result.values == ("model-3",)
    assert "empty line keeps model-3" in console.output


def test_single_select_rejects_more_than_one_number() -> None:
    """One role takes one model, so a multiple answer is refused and retried."""
    console = _console("1,2\n2\n")

    result = select_one(console, title="Judge", options=_options(3))

    assert result.values == ("model-2",)
    assert "Enter exactly one number." in console.output


def test_unmatched_filter_text_is_reported_and_the_list_stays_visible() -> None:
    """A filter matching nothing is explained instead of emptying the screen."""
    console = _console("nothing\n1\n")

    result = select_one(console, title="Judge", options=_options(3))

    assert result.values == ("model-1",)
    assert "No row matches 'nothing'." in console.output


def test_more_expands_a_collapsed_list() -> None:
    """The collapsed long list expands on request without losing the selection."""
    console = _console("more\n20\n\n")

    result = select_many(console, title="Models", options=_options(20))

    assert result.values == ("model-20",)
    assert "20. model-20" in console.output


def test_a_screen_without_rows_is_a_programming_error() -> None:
    """Callers filter rows before showing a screen, so an empty screen fails loudly."""
    console = _console("\n")

    with pytest.raises(ValueError, match="has no available choices"):
        select_many(console, title="Models", options=())


def _keys(*actions: PickerKey | PickerEvent) -> Callable[[], PickerKey | PickerEvent]:
    """Return a key source that yields the scripted keyboard events in order.

    Args:
        actions: Decoded keys or full events the list should consume.

    Returns:
        Zero-argument reader used by ``select_many_list``.
    """
    iterator = iter(actions)
    return lambda: next(iterator)


def _typed(text: str) -> tuple[PickerEvent, ...]:
    """Return one text event per character, as a terminal emits them while typing.

    Args:
        text: Characters typed in order.

    Returns:
        The decoded events carrying each character.
    """
    return tuple(PickerEvent(PickerKey.TEXT, char) for char in text)


def _picker_script(body: str) -> str:
    """Wrap one picker call in a child program that reports what it chose.

    Args:
        body: Statements that assign ``result`` from a picker screen.

    Returns:
        Source for a child interpreter attached to a pseudo-terminal.
    """
    return (
        "from rich.console import Console\n"
        "from wmo.cli.shared.picker import PickerOption, choose_many, choose_one\n"
        "console = Console()\n"
        f"{body}\n"
        'print("RESULT:" + ",".join(result.values) + "|" + str(result.action), flush=True)\n'
    )


def _model_options(count: int) -> str:
    """Return source building model rows that keep provider, role, and pricing metadata.

    Args:
        count: Number of model rows the child screen offers.

    Returns:
        A Python expression evaluating to the option tuple.
    """
    return (
        "tuple(\n"
        "    PickerOption(\n"
        '        value=f"model-{index}",\n'
        '        label=f"model-{index} (openai/gpt-{index})",\n'
        '        detail=f"roles: judge, world_model; pricing: api",\n'
        "    )\n"
        f"    for index in range(1, {count + 1})\n"
        ")"
    )


def _assert_single_region(run: TerminalRun, *, title: str, rows: tuple[str, ...]) -> None:
    """Assert the visible screen kept one copy of the heading and of every named row.

    Args:
        run: Completed pseudo-terminal session.
        title: Screen heading that must appear exactly once.
        rows: Row labels that must each appear exactly once.
    """
    text = run.screen_text()
    assert text.count(title) == 1, text
    for row in rows:
        assert text.count(row) == 1, text
    assert "^[[" not in run.transcript, run.transcript


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        (b"\x1b[A", PickerEvent(PickerKey.UP)),
        (b"\x1bOA", PickerEvent(PickerKey.UP)),
        (b"\x1b[B", PickerEvent(PickerKey.DOWN)),
        (b"\x1bOB", PickerEvent(PickerKey.DOWN)),
        (b"\r", PickerEvent(PickerKey.ENTER)),
        (b"\n", PickerEvent(PickerKey.ENTER)),
        (b" ", PickerEvent(PickerKey.SPACE, " ")),
        (b"\x7f", PickerEvent(PickerKey.BACKSPACE)),
        (b"\x08", PickerEvent(PickerKey.BACKSPACE)),
        (b"\x03", PickerEvent(PickerKey.CANCEL)),
        (b"\x1b", PickerEvent(PickerKey.ESCAPE)),
        (b"b", PickerEvent(PickerKey.TEXT, "b")),
        (b"q", PickerEvent(PickerKey.TEXT, "q")),
        (b"/", PickerEvent(PickerKey.TEXT, "/")),
        (b"x", PickerEvent(PickerKey.TEXT, "x")),
        ("\u00e9".encode(), PickerEvent(PickerKey.TEXT, "\u00e9")),
        ("\u4e16".encode(), PickerEvent(PickerKey.TEXT, "\u4e16")),
        (b"\x00", PickerEvent(PickerKey.IGNORE)),
        (b"\xff", PickerEvent(PickerKey.IGNORE)),
    ],
)
def test_interpret_key_bytes_maps_terminal_events(raw: bytes, expected: PickerEvent) -> None:
    """Each raw key sequence used by the list becomes one stable event.

    Args:
        raw: Bytes a terminal emits for one key press.
        expected: Decoded picker event, carrying the typed character when there is one.
    """
    assert interpret_key_bytes(raw) == expected


@pytest.mark.parametrize("ready_marker", ["Providers", "q cancels."])
def test_provider_multi_select_redraws_one_region_for_repeated_and_batched_keys(
    ready_marker: str,
    python_terminal_child: Callable[..., TerminalRun],
) -> None:
    """Repeated and batched arrow keys move focus inside a single redrawn region.

    Args:
        ready_marker: Early or fully rendered output that triggers the first key batch.
        python_terminal_child: Runner for one inline script under a pseudo-terminal.
    """
    script = _picker_script(
        "result = choose_many(\n"
        "    console,\n"
        '    title="Providers",\n'
        f"    options={_model_options(6)},\n"
        ")"
    )

    run = python_terminal_child(
        script,
        steps=[
            (ready_marker, _DOWN + _DOWN + _SPACE),
            (None, _DOWN + _DOWN + _UP + _SPACE),
            (None, _DOWN + _DOWN + _DOWN + _ENTER),
        ],
    )

    assert "RESULT:model-3,model-4|None" in run.transcript
    _assert_single_region(
        run,
        title="Providers",
        rows=("model-1 (openai/gpt-1)", "model-6 (openai/gpt-6)"),
    )
    assert "[x] model-3 (openai/gpt-3)" in run.screen_text()
    assert [line for line in run.screen if line.strip() == "> Complete"]


def test_model_single_select_confirms_the_focused_row_with_enter(
    python_terminal_child: Callable[..., TerminalRun],
) -> None:
    """Enter immediately confirms the focused model and keeps its metadata visible.

    Args:
        python_terminal_child: Runner for one inline script under a pseudo-terminal.
    """
    script = _picker_script(
        "result = choose_one(\n"
        "    console,\n"
        '    title="World model for the build",\n'
        f"    options={_model_options(4)},\n"
        ')\nassert result.values, "single select returned nothing"'
    )

    run = python_terminal_child(
        script,
        steps=[("World model for the build", _DOWN), (None, _DOWN + _ENTER)],
    )

    assert "RESULT:model-3|None" in run.transcript
    _assert_single_region(
        run,
        title="World model for the build",
        rows=("model-1 (openai/gpt-1)", "model-4 (openai/gpt-4)"),
    )
    assert "roles: judge, world_model; pricing: api" in run.screen_text()
    assert "[ ]" not in run.screen_text()


def test_candidate_multi_select_accepts_an_empty_optional_selection(
    python_terminal_child: Callable[..., TerminalRun],
) -> None:
    """An optional candidate screen submits nothing from Complete without typed commands.

    Args:
        python_terminal_child: Runner for one inline script under a pseudo-terminal.
    """
    script = _picker_script(
        "result = choose_many(\n"
        "    console,\n"
        '    title="Router candidates",\n'
        f"    options={_model_options(3)},\n"
        "    minimum=0,\n"
        ")"
    )

    run = python_terminal_child(
        script,
        steps=[
            ("Router candidates", _SPACE),
            (None, _SPACE),
            (None, _DOWN + _DOWN + _DOWN + _ENTER),
        ],
    )

    assert "RESULT:|None" in run.transcript
    _assert_single_region(run, title="Router candidates", rows=("model-2 (openai/gpt-2)",))
    assert "[x]" not in run.screen_text()


def test_a_long_model_list_scrolls_inside_the_terminal(
    python_terminal_child: Callable[..., TerminalRun],
) -> None:
    """A list taller than the terminal scrolls instead of growing the region.

    Args:
        python_terminal_child: Runner for one inline script under a pseudo-terminal.
    """
    script = _picker_script(
        "result = choose_one(\n"
        "    console,\n"
        '    title="Judge model",\n'
        f"    options={_model_options(40)},\n"
        ")"
    )

    run = python_terminal_child(
        script,
        steps=[("Judge model", _DOWN * 12), (None, _ENTER)],
        size=(16, 100),
    )

    assert "RESULT:model-13|None" in run.transcript
    text = run.screen_text()
    assert text.count("Judge model") == 1
    assert "more above" in text
    assert "more below" in text
    assert "model-40" not in text
    assert len(run.screen) <= 16, run.screen


def test_a_narrow_terminal_keeps_the_hint_and_metadata_readable(
    python_terminal_child: Callable[..., TerminalRun],
) -> None:
    """A narrow terminal splits the hint and still redraws one region.

    Args:
        python_terminal_child: Runner for one inline script under a pseudo-terminal.
    """
    script = _picker_script(
        "result = choose_many(\n"
        "    console,\n"
        '    title="Providers",\n'
        f"    options={_model_options(3)},\n"
        ")"
    )

    run = python_terminal_child(
        script,
        steps=[("Providers", _SPACE), (None, _DOWN + _DOWN + _DOWN + _ENTER)],
        size=(24, 44),
    )

    assert "RESULT:model-1|None" in run.transcript
    text = run.screen_text()
    assert text.count("Providers") == 1
    assert "Activate Complete to submit." in text
    assert "roles: judge, world_model; pricing:" in text


def test_a_single_select_search_narrows_a_huge_catalog_on_a_terminal(
    python_terminal_child: Callable[..., TerminalRun],
) -> None:
    """Typing a search on a real terminal narrows a large catalog and Enter confirms the match.

    Args:
        python_terminal_child: Runner for one inline script under a pseudo-terminal.
    """
    script = _picker_script(
        "result = choose_one(\n"
        "    console,\n"
        '    title="Judge model",\n'
        f"    options={_model_options(40)},\n"
        ")"
    )

    run = python_terminal_child(
        script,
        steps=[("Judge model", "/gpt-39"), (None, _ENTER)],
        size=(16, 100),
    )

    assert "RESULT:model-39|None" in run.transcript
    assert "Search: gpt-39_" in run.transcript
    assert "Enter confirms the focused match" in run.transcript


def test_a_search_accepts_multibyte_characters_on_a_terminal(
    python_terminal_child: Callable[..., TerminalRun],
) -> None:
    """A typed non-ASCII character reaches the query and narrows the rows.

    Args:
        python_terminal_child: Runner for one inline script under a pseudo-terminal.
    """
    script = _picker_script(
        "result = choose_one(\n"
        "    console,\n"
        '    title="Judge model",\n'
        "    options=(\n"
        '        PickerOption(value="modele", label="mod\\u00e8le (openrouter/mod\\u00e8le)"),\n'
        '        PickerOption(value="plain", label="plain (openai/gpt-4)"),\n'
        "    ),\n"
        ")"
    )

    run = python_terminal_child(
        script,
        steps=[("Judge model", "/mod\u00e8le"), (None, _ENTER)],
    )

    assert "RESULT:modele|None" in run.transcript
    assert "Search: mod\u00e8le_" in run.transcript


def test_a_multi_select_search_selects_a_match_and_submits_on_a_terminal(
    python_terminal_child: Callable[..., TerminalRun],
) -> None:
    """A candidate screen narrows by search, keeps the match, and submits from Complete.

    Args:
        python_terminal_child: Runner for one inline script under a pseudo-terminal.
    """
    script = _picker_script(
        "result = choose_many(\n"
        "    console,\n"
        '    title="Router candidates",\n'
        f"    options={_model_options(30)},\n"
        ")"
    )

    run = python_terminal_child(
        script,
        steps=[
            ("Router candidates", "/gpt-27"),
            (None, _ENTER),
            (None, _SPACE),
            (None, _DOWN + _ENTER),
        ],
        size=(16, 100),
    )

    assert "RESULT:model-27|None" in run.transcript
    assert "Search: gpt-27_" in run.transcript


def test_cancelling_a_screen_restores_the_terminal(
    python_terminal_child: Callable[..., TerminalRun],
) -> None:
    """Cancelling leaves canonical input, a visible cursor, and one region behind.

    Args:
        python_terminal_child: Runner for one inline script under a pseudo-terminal.
    """
    script = _picker_script(
        "result = choose_many(\n"
        "    console,\n"
        '    title="Providers",\n'
        f"    options={_model_options(3)},\n"
        ")\n"
        "import sys, termios\n"
        'print("CANON:" + str(bool(termios.tcgetattr(sys.stdin.fileno())[3] & termios.ICANON)))'
    )

    run = python_terminal_child(script, steps=[("Providers", _DOWN + "q")])

    assert "RESULT:|cancel" in run.transcript
    assert "CANON:True" in run.transcript
    assert "\x1b[?25h" in run.transcript
    assert run.transcript.count("\x1b[?25l") == 1


def test_an_exception_during_a_screen_restores_the_terminal(
    python_terminal_child: Callable[..., TerminalRun],
) -> None:
    """A failure raised while a screen waits for keys still releases the region.

    Args:
        python_terminal_child: Runner for one inline script under a pseudo-terminal.
    """
    script = (
        "import signal, sys, termios\n"
        "from rich.console import Console\n"
        "from wmo.cli.shared.picker import PickerOption, choose_many\n"
        "def fail(signum, frame):\n"
        '    """Interrupt the blocked key read with a failure.\n\n'
        "    Args:\n"
        "        signum: Signal number delivered by the timer.\n"
        "        frame: Interrupted stack frame.\n\n"
        "    Raises:\n"
        "        RuntimeError: Always, standing in for a failing screen.\n"
        '    """\n'
        '    raise RuntimeError("screen failed")\n'
        "signal.signal(signal.SIGALRM, fail)\n"
        "signal.setitimer(signal.ITIMER_REAL, 0.75)\n"
        "try:\n"
        "    choose_many(\n"
        "        Console(),\n"
        '        title="Providers",\n'
        '        options=(PickerOption("a", "provider-a"), PickerOption("b", "provider-b")),\n'
        "    )\n"
        "except RuntimeError as error:\n"
        '    print("FAILED:" + str(error))\n'
        'print("CANON:" + str(bool(termios.tcgetattr(sys.stdin.fileno())[3] & termios.ICANON)))\n'
    )

    run = python_terminal_child(script, steps=[])

    assert "FAILED:screen failed" in run.transcript
    assert "CANON:True" in run.transcript
    assert "\x1b[?25h" in run.transcript
    assert run.screen_text().count("Providers") == 1


def test_keyboard_single_select_starts_on_a_prior_answer() -> None:
    """A prior answer is focused first so one Enter keeps it."""
    console = _console("")

    result = select_one_list(
        console,
        title="Judge",
        options=_options(3),
        default="model-3",
        read_key=_keys(PickerKey.ENTER),
    )

    assert result.values == ("model-3",)
    assert "> model-3" in console.output


def test_keyboard_single_select_navigates_and_can_leave_the_screen() -> None:
    """Up and Down wrap around the rows, and b or q leave without a value."""
    console = _console("")

    result = select_one_list(
        console,
        title="Judge",
        options=_options(3),
        read_key=_keys(PickerKey.UP, PickerKey.DOWN, PickerKey.DOWN, PickerKey.ENTER),
    )
    assert result.values == ("model-2",)

    assert (
        select_one_list(
            console, title="Judge", options=_options(2), read_key=_keys(PickerKey.BACK)
        ).action
        is PickerAction.BACK
    )
    assert (
        select_one_list(
            console, title="Judge", options=_options(2), read_key=_keys(PickerKey.CANCEL)
        ).action
        is PickerAction.CANCEL
    )


def test_keyboard_list_moves_focus_toggles_selection_and_submits_from_complete() -> None:
    """Enter toggles the focused row; only Complete submits the current selection."""
    console = _console("")

    result = select_many_list(
        console,
        title="Providers",
        options=_options(3),
        read_key=_keys(
            PickerKey.ENTER,
            PickerKey.DOWN,
            PickerKey.DOWN,
            PickerKey.ENTER,
            PickerKey.ENTER,
            PickerKey.DOWN,
            PickerKey.ENTER,
        ),
    )

    assert result.action is None
    assert result.values == ("model-1",)
    assert "> [x] model-1" in console.output
    assert "> [ ] model-3" in console.output
    assert "> Complete" in console.output


def test_keyboard_list_keeps_preselected_rows_and_back_cancel_keys() -> None:
    """Reentering the list keeps current marks, and b or q still navigate away."""
    console = _console("")

    result = select_many_list(
        console,
        title="Providers",
        options=_options(2),
        preselected=("model-2",),
        read_key=_keys(PickerKey.BACK),
    )

    assert result.action is PickerAction.BACK
    assert "[x] model-2" in console.output

    cancelled = select_many_list(
        console,
        title="Providers",
        options=_options(2),
        read_key=_keys(PickerKey.CANCEL),
    )
    assert cancelled.action is PickerAction.CANCEL


def test_keyboard_list_refuses_complete_until_the_minimum_is_met() -> None:
    """Complete does nothing useful until the required number of rows is selected."""
    console = _console("")

    result = select_many_list(
        console,
        title="Providers",
        options=_options(2),
        read_key=_keys(
            PickerKey.DOWN,
            PickerKey.DOWN,
            PickerKey.ENTER,
            PickerKey.UP,
            PickerKey.UP,
            PickerKey.ENTER,
            PickerKey.DOWN,
            PickerKey.DOWN,
            PickerKey.ENTER,
        ),
    )

    assert result.values == ("model-1",)
    assert "Select at least 1." in console.output


def test_keyboard_multi_select_search_narrows_and_keeps_hidden_selections() -> None:
    """Slash search narrows a long list while selections hidden by the query survive."""
    console = _console("")

    result = select_many_list(
        console,
        title="Router candidates",
        options=_options(20),
        preselected=("model-3",),
        read_key=_keys(
            *_typed("/"),
            *_typed("model-12"),
            PickerEvent(PickerKey.ENTER),
            PickerEvent(PickerKey.ENTER),
            PickerEvent(PickerKey.DOWN),
            PickerEvent(PickerKey.ENTER),
        ),
    )

    assert result.values == ("model-3", "model-12")
    assert "Search: model-12_" in console.output
    assert "Filter: model-12" in console.output


def test_keyboard_multi_select_escape_clears_the_search_and_restores_the_list() -> None:
    """A query without matches is explained, and Esc returns to the full list."""
    console = _console("")

    result = select_many_list(
        console,
        title="Router candidates",
        options=_options(5),
        read_key=_keys(
            *_typed("/"),
            *_typed("zz"),
            PickerEvent(PickerKey.ESCAPE),
            PickerEvent(PickerKey.ENTER),
            PickerEvent(PickerKey.UP),
            PickerEvent(PickerKey.ENTER),
        ),
    )

    assert result.values == ("model-1",)
    assert "No row matches 'zz'." in console.output


def test_keyboard_multi_select_backspace_reopens_a_retained_filter() -> None:
    """Backspace on a closed filter reopens the search line with the last character removed."""
    console = _console("")

    result = select_many_list(
        console,
        title="Router candidates",
        options=_options(12),
        read_key=_keys(
            *_typed("/"),
            *_typed("model-12"),
            PickerEvent(PickerKey.ENTER),
            PickerEvent(PickerKey.BACKSPACE),
            PickerEvent(PickerKey.ESCAPE),
            PickerEvent(PickerKey.TEXT, "b"),
        ),
    )

    assert result.action is PickerAction.BACK
    assert "Filter: model-12" in console.output
    assert "Search: model-1_" in console.output


def test_keyboard_single_select_search_confirms_the_focused_match() -> None:
    """Enter while searching confirms the focused match directly."""
    console = _console("")

    result = select_one_list(
        console,
        title="Judge",
        options=_options(40),
        read_key=_keys(
            *_typed("/"),
            *_typed("model-39"),
            PickerEvent(PickerKey.ENTER),
        ),
    )

    assert result.values == ("model-39",)
    assert "Search: model-39_" in console.output


def test_keyboard_single_select_search_backspace_edits_and_arrows_move_the_focus() -> None:
    """Backspace edits the open query, and arrows keep moving focus through the matches."""
    console = _console("")

    result = select_one_list(
        console,
        title="Judge",
        options=_options(15),
        read_key=_keys(
            *_typed("/"),
            *_typed("model-15"),
            PickerEvent(PickerKey.BACKSPACE),
            PickerEvent(PickerKey.DOWN),
            PickerEvent(PickerKey.ENTER),
        ),
    )

    assert result.values == ("model-10",)


def test_keyboard_search_still_cancels_on_ctrl_c() -> None:
    """Ctrl-C cancels the whole screen even while the search line is open."""
    console = _console("")

    result = select_many_list(
        console,
        title="Router candidates",
        options=_options(3),
        read_key=_keys(
            *_typed("/"),
            *_typed("mod"),
            PickerEvent(PickerKey.CANCEL),
        ),
    )

    assert result.action is PickerAction.CANCEL


def test_keyboard_list_keeps_details_readable_on_a_narrow_terminal() -> None:
    """Details sit on their own line so a narrow terminal does not collide with labels."""
    console = ScriptedConsole("", width=40)

    result = select_many_list(
        console,
        title="Providers",
        options=_options(1),
        read_key=_keys(PickerKey.ENTER, PickerKey.DOWN, PickerKey.ENTER),
    )

    assert result.values == ("model-1",)
    assert "      detail 1" in console.output
    assert "Activate Complete to submit." in console.output
