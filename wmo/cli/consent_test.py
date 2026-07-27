"""The shared spend boundary: consent is said, never inferred from the absence of a terminal."""

from __future__ import annotations

import io

import pytest
import typer
from rich.console import Console

from wmo.cli import consent as consent_module
from wmo.cli.consent import NO_CONSENT_EXIT_CODE, require_spend_consent


class _Answer:
    """A `rich.prompt.Confirm` stand-in recording what it was asked."""

    def __init__(self, answer: bool) -> None:
        self._answer = answer
        self.asked: list[str] = []

    def ask(self, prompt: str, *, default: bool = True) -> bool:
        self.asked.append(prompt)
        return self._answer


def _console(*, terminal: bool) -> tuple[Console, io.StringIO]:
    buffer = io.StringIO()
    return Console(file=buffer, force_terminal=terminal, width=200), buffer


def _flat(buffer: io.StringIO) -> str:
    return " ".join(buffer.getvalue().split())


def test_yes_consents_without_asking_even_at_a_terminal(monkeypatch: pytest.MonkeyPatch) -> None:
    answer = _Answer(False)
    monkeypatch.setattr(consent_module, "Confirm", answer)
    console, buffer = _console(terminal=True)

    assert require_spend_consent(
        console, yes=True, spend="~$12.00", command="wmo optimize route sweep"
    )
    assert answer.asked == []
    assert buffer.getvalue() == ""


def test_a_terminal_is_asked_and_a_no_declines(monkeypatch: pytest.MonkeyPatch) -> None:
    answer = _Answer(False)
    monkeypatch.setattr(consent_module, "Confirm", answer)
    console, _buffer = _console(terminal=True)

    assert not require_spend_consent(
        console, yes=False, spend="~$12.00", command="wmo optimize route sweep"
    )
    assert len(answer.asked) == 1


def test_a_terminal_is_asked_and_a_yes_consents(monkeypatch: pytest.MonkeyPatch) -> None:
    answer = _Answer(True)
    monkeypatch.setattr(consent_module, "Confirm", answer)
    console, _buffer = _console(terminal=True)

    assert require_spend_consent(
        console, yes=False, spend="~$12.00", command="wmo optimize route sweep"
    )
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
            command="wmo optimize route sweep",
        )

    assert caught.value.exit_code == NO_CONSENT_EXIT_CODE
    assert answer.asked == []  # nothing was asked, because there was nobody to ask
    flat = _flat(buffer)
    assert "cannot ask for spend consent" in flat
    assert "wmo optimize route sweep would spend ~$12.00 across 240 cell(s) here" in flat
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
