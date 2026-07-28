"""The one spend boundary every `wmo` command that can cost money passes through.

The rule: consent to spend is SAID, never inferred. It is said by `--yes`, or by answering the
prompt at a terminal. The absence of an interactive session (CI, cron, a pipe, `| tee`, a
redirected `< /dev/null` or heredoc stdin) is not consent, it is the absence of anyone to ask,
so a spending run refuses there instead of starting. Nor is a blank line or an EOF an answer:
the safe direction is refusal, so neither can authorize spend.

"Interactive" means BOTH streams. Prompting reads stdin while the console reports on stdout, so
a terminal stdout with a redirected stdin was a real hole: the gate saw a terminal, the prompt
read whatever the redirect supplied, and a blank line was taken as approval.

This lives in one place because the rule kept being re-implemented per command and a site kept
being missed: `wmo optimize model` shipped proceed-and-note and spent a scripted caller's real
money (#305), `route sweep`, `optimize distill run` and the harbor population search were fixed
next (#307), and the world-model mode of `wmo optimize harness` was still falling through with
no prompt and no notice after both. Every gate now calls `require_spend_consent`, so there is
one behaviour to read and one place a future spend surface has to reach for.
"""

from __future__ import annotations

import sys
from typing import NoReturn

import typer
from rich.console import Console
from rich.prompt import Confirm

# Exit code for "a spending run could not ask for consent". Distinct from the decline path
# (0, the user said no and nothing happened) and from a run failure (1), so a CI job can tell
# "you forgot --yes" apart from "the run broke".
NO_CONSENT_EXIT_CODE = 2

# The two ways there turns out to be nobody to ask. Both open the same refusal, so a caller
# reads one shape of message and always learns the flag that authorizes the spend.
_NO_ONE_TO_ASK = (
    "non-interactive session: cannot ask for spend consent, and consent is never inferred "
    "from the absence of someone to ask. A prompt needs a terminal on stdin as well as "
    "stdout, so a redirected or piped input refuses here too."
)
_NO_ANSWER = (
    "input ended before the spend question was answered: cannot ask for spend consent, and "
    "consent is never inferred from an unanswered prompt."
)


def _stdin_is_terminal() -> bool:
    """Whether the INPUT stream is a terminal.

    Its own function so a test can force the input side the way rich's `force_terminal` forces
    the output side: `click.testing.CliRunner` installs its own non-terminal `sys.stdin` for the
    duration of an `invoke`, so a CLI test cannot fake this by replacing `sys.stdin` itself.
    """
    stdin = sys.stdin
    try:
        return stdin is not None and stdin.isatty()
    except ValueError:
        # A closed stdin. Nobody to ask, which is the same answer as a redirected one.
        return False


def can_prompt(console: Console) -> bool:
    """Whether there is an interactive human to ask, on BOTH ends of this session.

    `console.is_terminal` answers only for the OUTPUT stream. A prompt reads the INPUT one, so
    checking the console alone let `wmo ... < /dev/null`, a heredoc, or `printf y | wmo ...`
    through with a terminal stdout and no human behind stdin. A prompt is only honest when both
    streams belong to the same person, so both have to be a TTY.

    Args:
        console: The console the command prints to, owned by the calling CLI module.

    Returns:
        True only when stdout and stdin are both a terminal.
    """
    return console.is_terminal and _stdin_is_terminal()


def require_spend_consent(
    console: Console,
    *,
    yes: bool,
    spend: str,
    command: str,
    alternative: str | None = None,
) -> bool:
    """Get explicit consent before the first paid call, or refuse to start.

    Args:
        console: The console the command prints to. It is passed in rather than imported
            because each CLI module owns its own `Console`, and tests drive this through a
            non-terminal one. Whether there is anyone to ask is `can_prompt`'s decision, which
            reads this console's stdout state AND stdin.
        yes: The command's `--yes` flag. Consent, already said.
        spend: What this run would spend, in whatever unit the command can honestly quote (a
            projected dollar figure, or the rollout/episode counts when the work is unpriced).
            Printed in the refusal, so a scripted caller learns the size of what it just
            declined to authorize instead of only that something was skipped.
        command: The command being run, e.g. `wmo optimize route sweep`, named in the refusal
            so the message says what to re-run.
        alternative: An optional second way out, phrased as a flag plus what it does, e.g.
            "--dry-run to see the plan without spending".

    Returns:
        True when consent was given, False when the user declined at a terminal. Callers exit
        on False rather than treating it as an error: nothing was run and nothing was spent.

    Raises:
        typer.Exit: Code 2 when `--yes` was not passed and either there was no interactive
            session to ask at, or the prompt reached EOF instead of an answer.
    """
    if yes:
        return True
    if not can_prompt(console):
        _refuse(console, _NO_ONE_TO_ASK, spend=spend, command=command, alternative=alternative)
    try:
        # `default=False`: pressing Enter, and anything else that arrives as a blank line, is a
        # refusal. Defaulting to True made the cheapest possible input authorize the spend.
        return Confirm.ask("\nProceed?", default=False)
    except EOFError:
        # Both streams claimed to be a terminal and then the input ended anyway (Ctrl-D, or a
        # pty that closed under us). Nobody answered, so this is the "could not ask" refusal
        # rather than a considered no, and it exits 2 instead of leaking a traceback.
        _refuse(console, _NO_ANSWER, spend=spend, command=command, alternative=alternative)


def _refuse(
    console: Console,
    lead: str,
    *,
    spend: str,
    command: str,
    alternative: str | None,
) -> NoReturn:
    """Print what would have been spent, what would have spent it, and the flag that allows it."""
    tail = f", or with {alternative}" if alternative else ""
    console.print(
        f"\n{lead}\n"
        f"  [bold]{command}[/bold] would spend {spend} here.\n"
        f"Re-run the same command with --yes to consent explicitly{tail}."
    )
    raise typer.Exit(NO_CONSENT_EXIT_CODE)
