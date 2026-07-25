"""`wmh e2b reap`: reclaim E2B sandbox slots held by orphaned harbor trial sandboxes.

The operator-facing half of `wmh.harness.e2b_reap`. It renders the reap candidates and their
evidence, and it never kills anything without `--yes`: a dry run is the default because every
kill is destructive and one wrong id is somebody's running trial.
"""

from __future__ import annotations

from datetime import UTC, datetime

import typer
from rich.console import Console
from rich.table import Table

from wmh.harness.e2b_ledger import read_ledger_files
from wmh.harness.e2b_reap import (
    AliveSandbox,
    ReapCandidate,
    execute_reap,
    kill_sandbox,
    list_alive_sandboxes,
    plan_reap,
    sandbox_cap,
)

_console = Console()


def _now() -> datetime:
    """The instant sandbox ages are measured against (a seam tests pin)."""
    return datetime.now(UTC)


e2b_app = typer.Typer(help="Inspect and reclaim E2B sandbox capacity.", no_args_is_help=True)

# Module-level singletons: a typer.Option call cannot be a default inline (ruff B008).
_REAP_YES = typer.Option(
    False, "--yes", help="Actually kill the candidates. Without it this is a dry run."
)
_REAP_DEAD_OWNERS = typer.Option(
    True,
    "--dead-owners/--no-dead-owners",
    help="Include sandboxes recorded by a wmh run on THIS machine whose process is gone "
    "(exact ids from the local ledger; safe and on by default).",
)
_REAP_STALE_MINUTES = typer.Option(
    None,
    "--stale-minutes",
    min=1,
    help="Also include any harbor trial sandbox on the ACCOUNT started more than N minutes "
    "ago. This match is account-wide and can kill another machine's live run, so it is "
    "opt-in; sandboxes owned by a live local process are still excluded.",
)


@e2b_app.command("reap")
def reap(
    yes: bool = _REAP_YES,
    dead_owners: bool = _REAP_DEAD_OWNERS,
    stale_minutes: int | None = _REAP_STALE_MINUTES,
) -> None:
    """Free E2B concurrency slots held by sandboxes no run is using any more.

    A harbor trial sandbox keeps its slot until its own multi-hour timeout, so a run that dies
    without graceful shutdown (crash, SIGKILL, budget abort, machine sleep) leaves orphans that
    starve every later run at the account cap of 100 concurrent sandboxes (override with
    `$WMH_E2B_SANDBOX_CAP`).

    Two evidence classes, most conservative first:

    - `--dead-owners` (default): unreleased entries in this machine's sandbox ledger whose
      owning process is gone. These are provably orphans of local runs and are killed by exact
      recorded id.
    - `--stale-minutes N` (opt-in): every sandbox ON THE ACCOUNT that carries harbor trial
      metadata and started more than N minutes ago. The match is account-wide, so it can kill
      a run on another machine or in another checkout; use it only when you know no such run
      should be alive. Sandboxes whose local owner process is still running are never selected.

    The default is a dry run that prints the candidates and changes nothing. Pass `--yes` to
    kill them.
    """
    try:
        cap = sandbox_cap()
    except ValueError as error:
        raise typer.BadParameter(str(error)) from error
    if not dead_owners and stale_minutes is None:
        raise typer.BadParameter(
            "--no-dead-owners with no --stale-minutes selects nothing; drop --no-dead-owners "
            "or add --stale-minutes N"
        )
    try:
        alive = list_alive_sandboxes()
    except ImportError as error:
        raise typer.BadParameter(str(error)) from error
    except Exception as error:  # noqa: BLE001 - any provider failure is a usage-level message
        raise typer.BadParameter(
            f"could not list E2B sandboxes ({type(error).__name__}: {error}); check "
            "$E2B_API_KEY and your connection"
        ) from error

    plan = plan_reap(
        alive=alive,
        ledger_files=read_ledger_files(),
        now=_now(),
        dead_owners=dead_owners,
        stale_minutes=stale_minutes,
    )
    _print_usage(alive, cap)
    if not plan.candidates:
        _console.print("[green]nothing to reap[/green]: no orphaned sandbox matched")
        if not plan.vanished:
            return
        if not yes:
            _console.print(
                f"{len(plan.vanished)} ledger record(s) name sandboxes E2B no longer has; "
                "--yes clears them"
            )
            return
        outcome = execute_reap(plan, killer=kill_sandbox)
        _console.print(
            f"released {len(plan.vanished)} stale ledger record(s); "
            f"pruned {len(outcome.pruned_ledgers)} ledger file(s)"
        )
        return

    _console.print(_candidates_table(plan.candidates))
    if not yes:
        _console.print(
            f"[yellow]dry run[/yellow]: {len(plan.candidates)} sandbox(es) would be killed, "
            f"freeing up to {len(plan.candidates)} of {cap} slot(s). Re-run with --yes to do it."
        )
        return

    outcome = execute_reap(plan, killer=kill_sandbox)
    remaining = max(len(alive) - outcome.freed, 0)
    _console.print(
        f"[green]reaped[/green] {outcome.freed} sandbox(es); usage now {remaining}/{cap} "
        f"({max(cap - remaining, 0)} free)"
    )
    if outcome.already_gone:
        _console.print(
            f"{len(outcome.already_gone)} candidate(s) were already gone: "
            f"{', '.join(outcome.already_gone)}"
        )
    for sandbox_id, error in outcome.failed:
        _console.print(f"[red]kill failed[/red] {sandbox_id}: {error}")
    if outcome.pruned_ledgers:
        _console.print(f"pruned {len(outcome.pruned_ledgers)} fully released ledger file(s)")


def _print_usage(alive: list[AliveSandbox], cap: int) -> None:
    """Print current account usage against the cap."""
    trials = sum(1 for sandbox in alive if sandbox.is_harbor_trial())
    _console.print(
        f"E2B usage: [bold]{len(alive)}/{cap}[/bold] concurrent sandbox(es) running "
        f"({trials} harbor trial environment(s)), {max(cap - len(alive), 0)} slot(s) free"
    )


def _candidates_table(candidates: tuple[ReapCandidate, ...]) -> Table:
    """Render one row per reap candidate with the evidence behind it."""
    table = Table(title="Reap candidates")
    table.add_column("Sandbox", no_wrap=True)
    table.add_column("Age", justify="right", no_wrap=True)
    table.add_column("Template", no_wrap=True)
    table.add_column("Source", no_wrap=True)
    table.add_column("Owner", justify="right", no_wrap=True)
    table.add_column("Alive", no_wrap=True)
    table.add_column("Trial")
    for candidate in candidates:
        table.add_row(
            candidate.sandbox_id,
            _age(candidate.age_seconds),
            _template(candidate.template_id),
            candidate.source,
            "unknown" if candidate.owner_pid is None else str(candidate.owner_pid),
            _owner_liveness(candidate.owner_alive),
            candidate.trial_name or "",
        )
    return table


# A wmh harbor alias is `wmh-hb-v1-<64 hex>`: printing it whole squeezes every evidence column
# out of the table, and its head plus tail already identify a template uniquely in practice.
_TEMPLATE_DISPLAY_LEN = 22


def _template(template_id: str) -> str:
    """Shorten a long template alias, keeping both ends so two aliases stay distinguishable."""
    if len(template_id) <= _TEMPLATE_DISPLAY_LEN:
        return template_id
    return f"{template_id[: _TEMPLATE_DISPLAY_LEN - 8]}…{template_id[-7:]}"


def _owner_liveness(owner_alive: bool | None) -> str:
    """Render owner liveness, distinguishing "no recorded owner" from "owner is gone"."""
    if owner_alive is None:
        return "unknown"
    return "yes" if owner_alive else "no"


def _age(seconds: float) -> str:
    """Human-readable sandbox age, e.g. `3h12m`."""
    minutes = int(seconds // 60)
    hours, minutes = divmod(minutes, 60)
    return f"{hours}h{minutes:02d}m" if hours else f"{minutes}m"


def register(app: typer.Typer) -> None:
    """Attach the `wmh e2b` command group to the root CLI."""
    app.add_typer(e2b_app, name="e2b")
