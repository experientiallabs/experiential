"""Tests for the shared cost-aware paid-command authorization boundary."""

from __future__ import annotations

import io
import re
import sys
from decimal import Decimal
from pathlib import Path

import pytest
import typer
from rich.console import Console

from wmo.cli.shared import consent as consent_module
from wmo.cli.shared.consent import NO_CONSENT_EXIT_CODE, can_prompt, require_spend_consent
from wmo.common.config.settings import set_maximum_command_cost_usd


class _TerminalStdin(io.StringIO):
    """A stdin buffer that reports itself as an interactive terminal."""

    def isatty(self) -> bool:
        """Report an interactive input stream."""
        return True


class _Answer:
    """Record one or more confirmation prompts and return a fixed answer."""

    def __init__(self, answer: bool) -> None:
        """Configure the fixed confirmation answer.

        Args:
            answer: Boolean returned for every prompt.
        """
        self._answer = answer
        self.asked: list[str] = []
        self.defaults: list[bool] = []

    def ask(
        self,
        prompt: str,
        *,
        default: bool = True,
        console: Console | None = None,
    ) -> bool:
        """Record the prompt contract and return the configured answer.

        Args:
            prompt: User-facing confirmation question.
            default: Answer selected by a blank line.
            console: Console passed by the production boundary.

        Returns:
            The configured fixed answer.
        """
        del console
        self.asked.append(prompt)
        self.defaults.append(default)
        return self._answer


def _console(*, terminal: bool) -> tuple[Console, io.StringIO]:
    """Return a test console and its text buffer.

    Args:
        terminal: Whether the output side reports a TTY.

    Returns:
        Console and captured output buffer.
    """
    buffer = io.StringIO()
    return Console(file=buffer, force_terminal=terminal, width=200), buffer


def _flat(buffer: io.StringIO) -> str:
    """Return captured Rich output as one ANSI-free line."""
    return " ".join(_ANSI.sub("", buffer.getvalue()).split())


def _authorize(
    console: Console,
    root: Path,
    *,
    estimate: float | None,
    yes: bool = False,
    non_interactive: bool = False,
    previously_confirmed: bool = False,
) -> bool:
    """Invoke the shared boundary with one stable command fixture.

    Args:
        console: Capturing console.
        root: WMO settings root.
        estimate: Conservative command estimate.
        yes: Explicit invocation confirmation.
        non_interactive: Whether prompting is forbidden.
        previously_confirmed: Whether immutable command state already records confirmation.

    Returns:
        Shared authorization decision.
    """
    return require_spend_consent(
        console,
        root=root,
        yes=yes,
        estimated_cost_usd=estimate,
        command="wmo optimize model support",
        non_interactive=non_interactive,
        previously_confirmed=previously_confirmed,
    )


_ANSI = re.compile(r"\x1b\[[0-9;]*m")


def test_estimate_at_exactly_half_the_budget_runs_automatically(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Fifty percent belongs to the automatic side of the policy."""
    root = tmp_path / ".wmo"
    set_maximum_command_cost_usd(20.0, root)
    answer = _Answer(False)
    monkeypatch.setattr(consent_module, "Confirm", answer)
    console, buffer = _console(terminal=False)

    assert _authorize(console, root, estimate=10.0)
    assert answer.asked == []
    assert _flat(buffer) == ""


def test_estimate_just_above_half_requires_a_clear_confirmation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    interactive_stdin: None,
) -> None:
    """The first value above fifty percent opens the explicit prompt."""
    root = tmp_path / ".wmo"
    set_maximum_command_cost_usd(20.0, root)
    answer = _Answer(True)
    monkeypatch.setattr(consent_module, "Confirm", answer)
    console, _buffer = _console(terminal=True)

    assert _authorize(console, root, estimate=10.000001)
    assert len(answer.asked) == 1
    prompt = answer.asked[0]
    assert "wmo optimize model support" in prompt
    assert "$10.01" in prompt
    assert answer.defaults == [False]


def test_estimate_at_exactly_the_budget_requires_confirmation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    interactive_stdin: None,
) -> None:
    """One hundred percent is allowed only with explicit confirmation."""
    root = tmp_path / ".wmo"
    set_maximum_command_cost_usd(20.0, root)
    answer = _Answer(True)
    monkeypatch.setattr(consent_module, "Confirm", answer)
    console, _buffer = _console(terminal=True)

    assert _authorize(console, root, estimate=20.0)
    assert len(answer.asked) == 1


def test_estimate_above_budget_fails_even_with_yes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The invocation confirmation never overrides the configured ceiling."""
    root = tmp_path / ".wmo root"
    set_maximum_command_cost_usd(20.0, root)
    answer = _Answer(True)
    monkeypatch.setattr(consent_module, "Confirm", answer)
    console, buffer = _console(terminal=True)

    with pytest.raises(typer.BadParameter) as caught:
        _authorize(console, root, estimate=20.000001, yes=True)

    assert answer.asked == []
    message = str(caught.value)
    assert "exceeds the configured per-command budget" in message
    assert "wmo config budget 20.01 --root" in message
    assert "wmo root" in message
    assert "--yes cannot override" in message


def test_over_budget_interactive_override_defaults_to_no(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    interactive_stdin: None,
) -> None:
    """An interactive over-budget estimate warns and declines on a blank answer."""
    root = tmp_path / ".wmo"
    set_maximum_command_cost_usd(20.0, root)
    answer = _Answer(False)
    monkeypatch.setattr(consent_module, "Confirm", answer)
    console, buffer = _console(terminal=True)

    assert not _authorize(console, root, estimate=35.0)

    assert len(answer.asked) == 1
    assert answer.defaults == [False]
    prompt = answer.asked[0]
    assert "Proceed anyway" in prompt
    assert "warning" in prompt
    assert "$35.00" in prompt
    assert "exceeds the $20.00 budget" in prompt
    assert "No spend was authorized." in _flat(buffer)


def test_over_budget_interactive_explicit_yes_authorizes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    interactive_stdin: None,
) -> None:
    """Only an explicit terminal yes overrides the configured ceiling."""
    root = tmp_path / ".wmo"
    set_maximum_command_cost_usd(20.0, root)
    answer = _Answer(True)
    monkeypatch.setattr(consent_module, "Confirm", answer)
    console, buffer = _console(terminal=True)

    assert _authorize(console, root, estimate=35.0)

    assert len(answer.asked) == 1
    assert answer.defaults == [False]
    assert "No spend was authorized." not in _flat(buffer)


def test_over_budget_noninteractive_fails_even_with_yes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    interactive_stdin: None,
) -> None:
    """The explicit noninteractive flag keeps the over-budget rejection fail-closed."""
    root = tmp_path / ".wmo"
    set_maximum_command_cost_usd(20.0, root)
    answer = _Answer(True)
    monkeypatch.setattr(consent_module, "Confirm", answer)
    console, _buffer = _console(terminal=True)

    with pytest.raises(typer.BadParameter) as caught:
        _authorize(console, root, estimate=35.0, yes=True, non_interactive=True)

    assert answer.asked == []
    assert "--yes cannot override" in str(caught.value)


def test_sub_cent_estimates_are_displayed_conservatively(tmp_path: Path) -> None:
    """Visible sub-cent amounts round up to one cent instead of down to zero."""
    assert consent_module._format_usd(Decimal("0.0000001")) == "$0.01"
    assert consent_module._format_usd(Decimal("0")) == "$0.00"
    assert consent_module._format_usd(Decimal("95.321")) == "$95.33"


def test_noninteractive_above_half_requires_yes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Automation gets an exit-2 instruction instead of an implicit answer."""
    root = tmp_path / ".wmo"
    set_maximum_command_cost_usd(20.0, root)
    answer = _Answer(True)
    monkeypatch.setattr(consent_module, "Confirm", answer)
    console, buffer = _console(terminal=True)

    with pytest.raises(typer.Exit) as caught:
        _authorize(console, root, estimate=15.0, non_interactive=True)

    assert caught.value.exit_code == NO_CONSENT_EXIT_CODE
    assert answer.asked == []
    rendered = _flat(buffer)
    assert "requires explicit confirmation" in rendered
    assert "re-run with --yes" in rendered
    assert "$15.00" in rendered
    assert "$20.00" in rendered


def test_noninteractive_yes_confirms_an_in_budget_estimate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A complete agent invocation can authorize an above-half estimate."""
    root = tmp_path / ".wmo"
    set_maximum_command_cost_usd(20.0, root)
    answer = _Answer(False)
    monkeypatch.setattr(consent_module, "Confirm", answer)
    console, buffer = _console(terminal=False)

    assert _authorize(console, root, estimate=15.0, yes=True, non_interactive=True)
    assert answer.asked == []
    assert _flat(buffer) == ""


def test_prior_immutable_confirmation_avoids_a_repeat_prompt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A paid resume reuses its recorded confirmation while respecting the current ceiling."""
    root = tmp_path / ".wmo"
    set_maximum_command_cost_usd(20.0, root)
    answer = _Answer(False)
    monkeypatch.setattr(consent_module, "Confirm", answer)
    console, buffer = _console(terminal=False)

    assert _authorize(console, root, estimate=15.0, previously_confirmed=True)
    assert answer.asked == []
    assert _flat(buffer) == ""


def test_interactive_decline_returns_false_without_authorizing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    interactive_stdin: None,
) -> None:
    """A clearly worded prompt still permits an explicit refusal."""
    root = tmp_path / ".wmo"
    set_maximum_command_cost_usd(20.0, root)
    answer = _Answer(False)
    monkeypatch.setattr(consent_module, "Confirm", answer)
    console, _buffer = _console(terminal=True)

    assert not _authorize(console, root, estimate=15.0)
    assert len(answer.asked) == 1


def test_zero_budget_allows_only_a_zero_cost_replay(tmp_path: Path) -> None:
    """A zero ceiling disables paid work while preserving deterministic replay."""
    root = tmp_path / ".wmo"
    set_maximum_command_cost_usd(0.0, root)
    console, _buffer = _console(terminal=False)

    assert _authorize(console, root, estimate=0.0)
    with pytest.raises(typer.BadParameter):
        _authorize(console, root, estimate=0.000001, yes=True)


def test_can_prompt_requires_both_terminal_streams(monkeypatch: pytest.MonkeyPatch) -> None:
    """Terminal output alone cannot make redirected input interactive."""
    terminal, _ = _console(terminal=True)
    piped, _ = _console(terminal=False)

    monkeypatch.setattr(sys, "stdin", _TerminalStdin())
    assert can_prompt(terminal)
    assert not can_prompt(piped)

    monkeypatch.setattr(sys, "stdin", io.StringIO("y\n"))
    assert not can_prompt(terminal)
    assert not can_prompt(piped)


def test_eof_at_confirmation_is_an_actionable_refusal(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An unanswered interactive prompt exits cleanly without authorizing spend."""
    root = tmp_path / ".wmo"
    set_maximum_command_cost_usd(20.0, root)
    monkeypatch.setattr(sys, "stdin", _TerminalStdin(""))
    console, buffer = _console(terminal=True)

    with pytest.raises(typer.Exit) as caught:
        _authorize(console, root, estimate=15.0)

    assert caught.value.exit_code == NO_CONSENT_EXIT_CODE
    assert "input ended before confirmation" in _flat(buffer)


@pytest.mark.parametrize("estimate", [-0.01, float("inf"), float("nan")])
def test_invalid_estimate_fails_closed(tmp_path: Path, estimate: float) -> None:
    """Unsafe arithmetic cannot reach an automatic or confirmed decision."""
    console, _buffer = _console(terminal=False)

    with pytest.raises(typer.BadParameter, match="finite and nonnegative"):
        _authorize(console, tmp_path / ".wmo", estimate=estimate, yes=True)


def test_undefined_cost_warns_and_requires_an_explicit_interactive_override(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    interactive_stdin: None,
) -> None:
    """An unpriced model never runs automatically and asks for a manual override."""
    root = tmp_path / ".wmo"
    set_maximum_command_cost_usd(20.0, root)
    answer = _Answer(True)
    monkeypatch.setattr(consent_module, "Confirm", answer)
    console, buffer = _console(terminal=True)

    assert _authorize(console, root, estimate=None)
    assert len(answer.asked) == 1
    assert "undefined amount" in answer.asked[0]
    assert answer.defaults == [False]
    assert "cost of this command is undefined" in _flat(buffer)


def test_undefined_cost_interactive_decline_authorizes_nothing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    interactive_stdin: None,
) -> None:
    """Declining the undefined-cost override refuses the spend."""
    root = tmp_path / ".wmo"
    set_maximum_command_cost_usd(20.0, root)
    answer = _Answer(False)
    monkeypatch.setattr(consent_module, "Confirm", answer)
    console, _buffer = _console(terminal=True)

    assert not _authorize(console, root, estimate=None)
    assert len(answer.asked) == 1


def test_undefined_cost_accepts_an_explicit_yes_flag_with_a_warning(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An explicit --yes is the manual override for an undefined estimate."""
    root = tmp_path / ".wmo"
    set_maximum_command_cost_usd(20.0, root)
    answer = _Answer(False)
    monkeypatch.setattr(consent_module, "Confirm", answer)
    console, buffer = _console(terminal=False)

    assert _authorize(console, root, estimate=None, yes=True)
    assert answer.asked == []
    assert "cost of this command is undefined" in _flat(buffer)


def test_undefined_cost_noninteractive_without_yes_exits_without_spend(
    tmp_path: Path,
) -> None:
    """A session that cannot prompt fails closed on an undefined estimate."""
    root = tmp_path / ".wmo"
    set_maximum_command_cost_usd(20.0, root)
    console, buffer = _console(terminal=False)

    with pytest.raises(typer.Exit) as caught:
        _authorize(console, root, estimate=None, non_interactive=True)

    assert caught.value.exit_code == NO_CONSENT_EXIT_CODE
    assert "manual override" in _flat(buffer)
