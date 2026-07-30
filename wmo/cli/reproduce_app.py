"""`wmo reproduce`: replay a published benchmark result and compare, one command.

    wmo reproduce list
    wmo reproduce run routerbench
    wmo reproduce run tau-bench --yes   # live providers; forecasts, then spends

The manifests are shipped data (`wmo.reproduce.manifests`); see that package's docs for the
two exactness classes. The command's exit code is the verdict: 0 REPRODUCED, 4 DIVERGED,
so CI can hold a published number honest.
"""

from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from wmo.reproduce import load_manifest, manifest_names, run_reproduction

reproduce_app = typer.Typer(no_args_is_help=True, rich_markup_mode="rich")
_console = Console()

EXIT_DIVERGED = 4


@reproduce_app.command("list")
def list_manifests() -> None:
    """Every benchmark this build can reproduce, with its exactness class."""
    table = Table(show_header=True)
    table.add_column("benchmark")
    table.add_column("exactness")
    table.add_column("data")
    table.add_column("cookbook")
    for name in manifest_names():
        manifest = load_manifest(name)
        table.add_row(name, manifest.exactness, manifest.data.hf_repo, manifest.cookbook)
    _console.print(table)


@reproduce_app.command("run")
def run(
    name: str = typer.Argument(help="A benchmark from `wmo reproduce list`."),
    out: str = typer.Option("reproductions", help="Directory for artifacts and verdict.json."),
    data_dir: str | None = typer.Option(
        None, help="Use an existing data snapshot instead of downloading."
    ),
    yes: bool = typer.Option(
        False, "--yes", help="Approve the spend a live (commands) manifest estimates."
    ),
) -> None:
    """Download the pinned data, replay the pinned protocol, compare, and say which."""
    try:
        manifest = load_manifest(name)
    except KeyError as exc:
        raise typer.BadParameter(str(exc)) from exc

    out_dir = Path(out) / name
    try:
        result = run_reproduction(
            manifest,
            out_dir=out_dir,
            data_dir=Path(data_dir) if data_dir else None,
            approve_spend=yes,
        )
    except PermissionError as exc:
        # The spend gate: state the estimate and stop before anything ran.
        _console.print(f"[yellow]{exc}[/yellow]")
        raise typer.Exit(2) from exc

    table = Table(show_header=True, title=f"{manifest.title} ({manifest.exactness})")
    table.add_column("row")
    table.add_column("field")
    table.add_column("published", justify="right")
    table.add_column("measured", justify="right")
    table.add_column("ok")
    for row in result.rows:
        for field, (published, measured, ok) in row.fields.items():
            table.add_row(
                row.label,
                field,
                f"{published:.6g}",
                f"{measured:.6g}",
                "[green]yes[/green]" if ok else "[red]NO[/red]",
            )
    _console.print(table)
    for note in result.notes:
        _console.print(f"  [dim]note: {note}[/dim]")
    if result.reproduced:
        _console.print(f"[green]REPRODUCED[/green] ({manifest.exactness}) -> {out_dir}")
        return
    _console.print(f"[red]DIVERGED[/red] -> {out_dir}/verdict.json")
    raise typer.Exit(EXIT_DIVERGED)
