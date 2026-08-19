"""Shared cost estimate presentation and authorization for every paid WMO command."""

from __future__ import annotations

import math
import shlex
import sys
from decimal import ROUND_CEILING, Decimal, InvalidOperation
from pathlib import Path
from typing import NoReturn

import typer
from rich.console import Console
from rich.markup import escape
from rich.prompt import Confirm

from wmo.common.config import ARTIFACT_DIR, resolve_command_budget_usd

NO_CONSENT_EXIT_CODE = 2


def _stdin_is_terminal() -> bool:
    """Return whether the input stream is an open terminal."""
    stdin = sys.stdin
    try:
        return stdin is not None and stdin.isatty()
    except ValueError:
        return False


def can_prompt(console: Console) -> bool:
    """Return whether both input and output belong to an interactive terminal.

    Args:
        console: Command-owned output console.

    Returns:
        True only when both terminal streams can reach the same operator.
    """
    return console.is_terminal and _stdin_is_terminal()


def require_spend_consent(
    console: Console,
    *,
    root: str | Path = ARTIFACT_DIR,
    yes: bool,
    estimated_cost_usd: float | None,
    command: str,
    non_interactive: bool = False,
    previously_confirmed: bool = False,
) -> bool:
    """Enforce the configured authorization policy for one conservative cost estimate.

    Estimates at or below half of the configured budget run automatically. Higher estimates up
    to the budget require ``--yes``, a prior immutable confirmation, or an explicit terminal
    answer. An estimate above the budget is a warning that only an explicit interactive answer
    can override, defaulting to no; without a terminal the invocation fails before credentials
    or provider clients, and ``--yes`` never overrides the ceiling.

    An undefined estimate (``None``) means the catalog carries no pricing for a selected model,
    so no ceiling comparison is possible. WMO reports the cost as undefined honestly instead of
    fabricating a number: execution is never automatic and always requires a manual override,
    either ``--yes``, a prior immutable confirmation, or an explicit terminal answer after the
    undefined-cost warning.

    Args:
        console: Command-owned output console.
        root: WMO root that owns ``settings.toml``.
        yes: Explicit invocation confirmation for an in-budget estimate.
        estimated_cost_usd: Conservative upper-bound estimate, or ``None`` when the selected
            models carry no catalog pricing and the cost cannot be estimated.
        command: Complete command identity shown to the operator.
        non_interactive: Whether this invocation forbids terminal questions.
        previously_confirmed: Whether immutable command state records an earlier confirmation.

    Returns:
        True when execution is authorized, or False after an interactive decline.

    Raises:
        typer.BadParameter: Settings or cost arithmetic are invalid, or the estimate exceeds the
            configured budget without a terminal able to override it.
        typer.Exit: No explicit confirmation is available in a noninteractive session.
    """
    if estimated_cost_usd is None:
        return _confirm_undefined_cost(
            console,
            command=command,
            yes=yes,
            non_interactive=non_interactive,
            previously_confirmed=previously_confirmed,
        )
    estimate = _cost_decimal(estimated_cost_usd, label="estimated command cost")
    try:
        configured = resolve_command_budget_usd(root, None)
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from None
    budget = _cost_decimal(configured, label="configured command budget")
    if estimate > budget:
        if non_interactive or not can_prompt(console):
            raise typer.BadParameter(_over_budget_message(root, estimate, budget))
        return _confirm_over_budget(console, command=command, estimate=estimate, budget=budget)
    if estimate <= budget / Decimal(2) or yes or previously_confirmed:
        return True
    if non_interactive or not can_prompt(console):
        _refuse_noninteractive(console, command=command, estimate=estimate, budget=budget)
    prompt = f"Authorize {command} to spend up to {_format_usd(estimate)}?"
    try:
        return Confirm.ask(prompt, default=False, console=console)
    except EOFError:
        _refuse_unanswered(console, command=command, estimate=estimate, budget=budget)


def _confirm_undefined_cost(
    console: Console,
    *,
    command: str,
    yes: bool,
    non_interactive: bool,
    previously_confirmed: bool,
) -> bool:
    """Warn that the cost is undefined and require a manual override before any spend.

    Args:
        console: Command-owned output console.
        command: Complete command identity.
        yes: Explicit invocation confirmation flag.
        non_interactive: Whether this invocation forbids terminal questions.
        previously_confirmed: Whether immutable command state records an earlier confirmation.

    Returns:
        True only after an explicit override; a blank answer or a decline authorizes nothing.

    Raises:
        typer.Exit: No manual override is available in a noninteractive session, or terminal
            input ended before an answer.
    """
    console.print(
        "[yellow]warning[/yellow] the cost of this command is undefined: a selected model has "
        "no catalog pricing, so WMO cannot estimate the spend or enforce the configured budget."
    )
    if yes or previously_confirmed:
        return True
    if non_interactive or not can_prompt(console):
        console.print(
            "authorization: an undefined cost always requires a manual override. This session "
            f"cannot prompt; re-run {escape(command)} with --yes to accept the undefined cost, "
            "or record explicit model pricing to restore budget enforcement."
        )
        raise typer.Exit(NO_CONSENT_EXIT_CODE)
    prompt = f"Authorize {command} to spend an undefined amount?"
    try:
        confirmed = Confirm.ask(prompt, default=False, console=console)
    except EOFError:
        console.print(
            "authorization: input ended before confirmation. No spend was authorized. Re-run "
            f"{escape(command)} with --yes to accept the undefined cost."
        )
        raise typer.Exit(NO_CONSENT_EXIT_CODE) from None
    if confirmed:
        return True
    console.print("No spend was authorized.")
    return False


def _confirm_over_budget(
    console: Console,
    *,
    command: str,
    estimate: Decimal,
    budget: Decimal,
) -> bool:
    """Warn about an over-budget estimate and require an explicit terminal override.

    Args:
        console: Command-owned output console.
        command: Complete command identity.
        estimate: Conservative invocation estimate above the configured budget.
        budget: Configured per-command ceiling.

    Returns:
        True only after an explicit yes; a blank answer or a decline authorizes nothing.

    Raises:
        typer.Exit: Terminal input ended before an answer.
    """
    prompt = (
        f"[yellow]warning[/yellow] estimated {_format_usd(estimate)} exceeds the "
        f"{_format_usd(budget)} budget. Proceed anyway?"
    )
    try:
        confirmed = Confirm.ask(prompt, default=False, console=console)
    except EOFError:
        _refuse_unanswered(console, command=command, estimate=estimate, budget=budget)
    if confirmed:
        return True
    console.print("No spend was authorized.")
    return False


def _cost_decimal(value: float, *, label: str) -> Decimal:
    """Convert one finite nonnegative float into stable decimal policy arithmetic.

    Args:
        value: Numeric USD value.
        label: Field name used in failures.

    Returns:
        Exact decimal representation of the supplied float string.

    Raises:
        typer.BadParameter: The value is negative or non-finite.
    """
    if not math.isfinite(value) or value < 0:
        raise typer.BadParameter(f"{label} must be finite and nonnegative")
    try:
        return Decimal(str(value))
    except InvalidOperation as exc:
        raise typer.BadParameter(f"{label} must be finite and nonnegative") from exc


def _format_usd(value: Decimal) -> str:
    """Format USD with exactly two decimal places, rounding up so coverage never shrinks.

    Args:
        value: Nonnegative finite decimal USD value.

    Returns:
        Dollar-prefixed value with exactly two decimal places.
    """
    rounded = value.quantize(Decimal("0.01"), rounding=ROUND_CEILING)
    return f"${rounded:.2f}"


def _over_budget_message(root: str | Path, estimate: Decimal, budget: Decimal) -> str:
    """Return actionable remediation for a noninteractive over-budget rejection.

    Args:
        root: WMO settings root.
        estimate: Rejected conservative estimate.
        budget: Configured ceiling.

    Returns:
        Error text naming every safe way to proceed.
    """
    amount = _format_usd(estimate).removeprefix("$")
    command = f"wmo config budget {amount} --root {shlex.quote(str(root))}"
    return (
        f"conservative estimate {_format_usd(estimate)} exceeds the configured per-command "
        f"budget {_format_usd(budget)}. Re-run in an interactive terminal to review and "
        f"explicitly override, increase the limit with `{command}`, or reduce this command's "
        "cost inputs. --yes cannot override the ceiling"
    )


def _refuse_noninteractive(
    console: Console,
    *,
    command: str,
    estimate: Decimal,
    budget: Decimal,
) -> NoReturn:
    """Exit when an above-half estimate has no deterministic confirmation.

    Args:
        console: Command-owned output console.
        command: Complete command identity.
        estimate: Conservative invocation estimate.
        budget: Configured per-command ceiling.
    """
    console.print(
        "authorization: this estimate requires explicit confirmation because it exceeds 50% "
        "of the configured budget. This session cannot prompt; re-run with --yes after "
        f"reviewing {escape(command)} ({_format_usd(estimate)} of {_format_usd(budget)})."
    )
    raise typer.Exit(NO_CONSENT_EXIT_CODE)


def _refuse_unanswered(
    console: Console,
    *,
    command: str,
    estimate: Decimal,
    budget: Decimal,
) -> NoReturn:
    """Exit when terminal input ends before confirmation.

    Args:
        console: Command-owned output console.
        command: Complete command identity.
        estimate: Conservative invocation estimate.
        budget: Configured per-command ceiling.
    """
    console.print(
        "authorization: input ended before confirmation. No spend was authorized. Re-run "
        f"{escape(command)} with --yes after reviewing {_format_usd(estimate)} of the "
        f"{_format_usd(budget)} budget."
    )
    raise typer.Exit(NO_CONSENT_EXIT_CODE)
