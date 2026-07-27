"""The one spend boundary every `wmo` command that can cost money passes through.

The rule: consent to spend is SAID, never inferred. It is said by `--yes`, or by answering the
prompt at a terminal. The absence of a terminal (CI, cron, a pipe, `| tee`) is not consent, it
is the absence of anyone to ask, so a spending run refuses there instead of starting.

This lives in one place because the rule kept being re-implemented per command and a site kept
being missed: `wmo optimize model` shipped proceed-and-note and spent a scripted caller's real
money (#305), `route sweep`, `optimize distill run` and the harbor population search were fixed
next (#307), and the world-model mode of `wmo optimize harness` was still falling through with
no prompt and no notice after both. Every gate now calls `require_spend_consent`, so there is
one behaviour to read and one place a future spend surface has to reach for.
"""

from __future__ import annotations

import typer
from rich.console import Console
from rich.prompt import Confirm

# Exit code for "a spending run could not ask for consent". Distinct from the decline path
# (0, the user said no and nothing happened) and from a run failure (1), so a CI job can tell
# "you forgot --yes" apart from "the run broke".
NO_CONSENT_EXIT_CODE = 2


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
        console: The console the command prints to. Its `is_terminal` decides whether there is
            anyone to ask, which is why it is passed in rather than imported: each CLI module
            owns its own `Console`, and tests drive this through a non-terminal one.
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
        typer.Exit: Code 2 when there is no terminal to ask at and `--yes` was not passed.
    """
    if yes:
        return True
    if not console.is_terminal:
        tail = f", or with {alternative}" if alternative else ""
        console.print(
            f"\nnon-interactive session: cannot ask for spend consent, and consent is never "
            f"inferred from the absence of a terminal.\n"
            f"  [bold]{command}[/bold] would spend {spend} here.\n"
            f"Re-run the same command with --yes to consent explicitly{tail}."
        )
        raise typer.Exit(NO_CONSENT_EXIT_CODE)
    return Confirm.ask("\nProceed?", default=True)
