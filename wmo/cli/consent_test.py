"""The shared spend boundary: consent is said, never inferred from the absence of someone to ask."""

from __future__ import annotations

import io
import re
import sys

import pytest
import typer
from rich.console import Console

from wmo.cli import consent as consent_module
from wmo.cli.consent import NO_CONSENT_EXIT_CODE, can_prompt, require_spend_consent


class _TerminalStdin(io.StringIO):
    """A stdin that claims to be a terminal, with `keystrokes` already queued for the prompt."""

    def isatty(self) -> bool:
        return True


class _Answer:
    """A `rich.prompt.Confirm` stand-in recording what it was asked, and with what default."""

    def __init__(self, answer: bool) -> None:
        self._answer = answer
        self.asked: list[str] = []
        self.defaults: list[bool] = []

    def ask(self, prompt: str, *, default: bool = True) -> bool:
        self.asked.append(prompt)
        self.defaults.append(default)
        return self._answer


def _console(*, terminal: bool) -> tuple[Console, io.StringIO]:
    buffer = io.StringIO()
    return Console(file=buffer, force_terminal=terminal, width=200), buffer


def _flat(buffer: io.StringIO) -> str:
    """The buffer as one line of plain text: a forced-terminal console also emits ANSI."""
    return " ".join(_ANSI.sub("", buffer.getvalue()).split())


_ANSI = re.compile(r"\x1b\[[0-9;]*m")


def test_yes_consents_without_asking_even_at_a_terminal(monkeypatch: pytest.MonkeyPatch) -> None:
    answer = _Answer(False)
    monkeypatch.setattr(consent_module, "Confirm", answer)
    console, buffer = _console(terminal=True)

    assert require_spend_consent(console, yes=True, spend="~$12.00", command="wmo optimize model")
    assert answer.asked == []
    assert buffer.getvalue() == ""


def test_a_terminal_is_asked_and_a_no_declines(
    monkeypatch: pytest.MonkeyPatch, interactive_stdin: None
) -> None:
    answer = _Answer(False)
    monkeypatch.setattr(consent_module, "Confirm", answer)
    console, _buffer = _console(terminal=True)

    assert not require_spend_consent(
        console, yes=False, spend="~$12.00", command="wmo optimize model"
    )
    assert len(answer.asked) == 1


def test_a_terminal_is_asked_and_a_yes_consents(
    monkeypatch: pytest.MonkeyPatch, interactive_stdin: None
) -> None:
    answer = _Answer(True)
    monkeypatch.setattr(consent_module, "Confirm", answer)
    console, _buffer = _console(terminal=True)

    assert require_spend_consent(console, yes=False, spend="~$12.00", command="wmo optimize model")
    assert len(answer.asked) == 1


def test_no_terminal_and_no_yes_refuses_naming_the_spend_and_the_flag(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The money bug: a pipe, CI, or cron used to be read as consent. It now exits 2.

    The refusal has to be actionable on its own, so it carries what would have been spent, the
    command that would have spent it, and the flag that authorizes it.
    """
    answer = _Answer(True)
    monkeypatch.setattr(consent_module, "Confirm", answer)
    console, buffer = _console(terminal=False)

    with pytest.raises(typer.Exit) as caught:
        require_spend_consent(
            console,
            yes=False,
            spend="~$12.00 across 240 cell(s)",
            command="wmo optimize model",
        )

    assert caught.value.exit_code == NO_CONSENT_EXIT_CODE
    assert answer.asked == []  # nothing was asked, because there was nobody to ask
    flat = _flat(buffer)
    assert "cannot ask for spend consent" in flat
    assert "wmo optimize model would spend ~$12.00 across 240 cell(s) here" in flat
    assert "Re-run the same command with --yes to consent explicitly." in flat


def test_the_refusal_names_a_second_way_out_when_the_command_has_one() -> None:
    console, buffer = _console(terminal=False)

    with pytest.raises(typer.Exit):
        require_spend_consent(
            console,
            yes=False,
            spend="~$1.65",
            command="wmo optimize model",
            alternative="--dry-run to see the plan without spending",
        )

    assert "or with --dry-run to see the plan without spending" in _flat(buffer)


# -- the input stream is half of "interactive" --------------------------------------------------
# Checking only the console asked the wrong stream: the console reports on stdout while the
# prompt reads stdin, so `wmo optimize model < /dev/null` at a terminal passed the gate
# and then had a redirect answer the money question for the absent human.


def test_a_terminal_stdout_with_redirected_stdin_refuses(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The hole: a TTY stdout is not a human if stdin comes from a file, a pipe, or a heredoc.

    Under pytest `sys.stdin` is already a non-terminal stub, which is exactly the redirected
    case, so only the console is forced here.
    """
    answer = _Answer(True)
    monkeypatch.setattr(consent_module, "Confirm", answer)
    console, buffer = _console(terminal=True)

    with pytest.raises(typer.Exit) as caught:
        require_spend_consent(console, yes=False, spend="~$12.00", command="wmo optimize model")

    assert caught.value.exit_code == NO_CONSENT_EXIT_CODE
    assert answer.asked == []  # never offered, so a redirect could not answer it
    flat = _flat(buffer)
    assert "cannot ask for spend consent" in flat
    # The reader IS at a terminal, so the refusal has to name the stream that is not one.
    assert "needs a terminal on stdin as well as stdout" in flat


def test_can_prompt_needs_both_streams(monkeypatch: pytest.MonkeyPatch) -> None:
    """Either stream alone is not an interactive session."""
    terminal, _ = _console(terminal=True)
    piped, _ = _console(terminal=False)

    monkeypatch.setattr(sys, "stdin", _TerminalStdin())
    assert can_prompt(terminal)
    assert not can_prompt(piped)  # terminal stdin, redirected stdout

    monkeypatch.setattr(sys, "stdin", io.StringIO("y\n"))
    assert not can_prompt(terminal)  # terminal stdout, redirected stdin
    assert not can_prompt(piped)


def test_a_closed_stdin_is_not_an_interactive_session(monkeypatch: pytest.MonkeyPatch) -> None:
    """`isatty()` raises on a closed stream; that is nobody to ask, not a crash."""
    closed = io.StringIO()
    closed.close()  # a later isatty() raises ValueError
    monkeypatch.setattr(sys, "stdin", closed)
    console, _buffer = _console(terminal=True)

    assert not can_prompt(console)


def test_a_blank_answer_refuses_instead_of_authorizing_the_spend(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The real `Confirm`, a real blank line: the cheapest possible input must not buy anything.

    `default=True` made pressing Enter (and every blank line a redirect can supply) an approval.
    The default is the answer given when nothing was said, so it has to be the refusal.
    """
    monkeypatch.setattr(sys, "stdin", _TerminalStdin("\n"))
    console, _buffer = _console(terminal=True)

    assert not require_spend_consent(
        console, yes=False, spend="~$12.00", command="wmo optimize model"
    )


def test_eof_at_the_prompt_is_the_documented_refusal_not_a_traceback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An input that ends without answering exits 2 with the refusal, not an `EOFError`."""
    monkeypatch.setattr(sys, "stdin", _TerminalStdin(""))  # readline() -> "" -> EOFError
    console, buffer = _console(terminal=True)

    with pytest.raises(typer.Exit) as caught:
        require_spend_consent(
            console,
            yes=False,
            spend="~$12.00 across 240 cell(s)",
            command="wmo optimize model",
        )

    assert caught.value.exit_code == NO_CONSENT_EXIT_CODE
    flat = _flat(buffer)
    assert "cannot ask for spend consent" in flat
    assert "wmo optimize model would spend ~$12.00 across 240 cell(s) here" in flat
    assert "Re-run the same command with --yes to consent explicitly." in flat


def test_the_prompt_defaults_to_refusing(
    monkeypatch: pytest.MonkeyPatch, interactive_stdin: None
) -> None:
    """Belt and braces on the default itself, in the units the stand-in tests are written in."""
    answer = _Answer(False)
    monkeypatch.setattr(consent_module, "Confirm", answer)
    console, _buffer = _console(terminal=True)

    require_spend_consent(console, yes=False, spend="~$12.00", command="wmo optimize model")

    assert answer.defaults == [False]
