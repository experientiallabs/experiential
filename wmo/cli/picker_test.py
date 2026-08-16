"""Line-based selection prompt tests."""

from __future__ import annotations

import io

import pytest
from rich.console import Console

from wmo.cli.picker import (
    PickerAction,
    PickerOption,
    select_many,
    select_one,
)


class ScriptedConsole(Console):
    """One console that answers every prompt from a script and records its output."""

    def __init__(self, answers: str) -> None:
        """Build a console reading scripted lines into an in-memory transcript.

        Args:
            answers: Newline-separated answers the screens read in order.
        """
        self._transcript = io.StringIO()
        super().__init__(
            file=self._transcript,
            force_terminal=False,
            width=100,
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
