"""`wmh harness` — named, versioned agent harnesses under `.wmh/harnesses/<name>/`.

A harness is the scaffold an agent runs with: prompt surfaces, a tool policy, loop parameters, and
skills, stored as immutable numbered versions with movable aliases (`champion` is what runs by
default). `init` writes the baseline as v1; `list`/`show` inspect what exists; run one closed-loop
with `wmh eval closed-loop <tasks> --harness <name>[@ref]`.
"""

from __future__ import annotations

import typer
from rich.console import Console
from rich.table import Table

from wmh.config import ARTIFACT_DIR
from wmh.harness.doc import HarnessDoc
from wmh.harness.store import CHAMPION_ALIAS, HarnessStore

harness_app = typer.Typer(
    help="Named, versioned agent harnesses (.wmh/harnesses): init, list, show.",
    no_args_is_help=True,
)
_console = Console()


@harness_app.command("list")
def list_harnesses(root: str = typer.Option(ARTIFACT_DIR, help="Project dir.")) -> None:
    """List every harness with its versions and aliases."""
    store = HarnessStore(root)
    names = store.list_names()
    if not names:
        _console.print(
            "[yellow]no harnesses yet[/yellow]; `wmh harness init <name>` creates the baseline"
        )
        return
    table = Table(title="Harnesses")
    table.add_column("Name", no_wrap=True)
    table.add_column("Versions", justify="right")
    table.add_column("Aliases")
    table.add_column("Doc hash (champion)")
    broken: list[tuple[str, str]] = []
    for name in names:
        try:
            doc = store.load(name)
            aliases = ", ".join(f"{a}=v{v}" for a, v in sorted(store.aliases(name).items()))
            table.add_row(
                name,
                f"{len(store.versions(name))}",
                aliases or "—",
                doc.doc_hash[:12],
            )
        except (ValueError, FileNotFoundError) as exc:  # one broken dir must not hide the rest
            broken.append((name, str(exc)))
    _console.print(table)
    for name, reason in broken:
        _console.print(f"[red]broken[/red] {name}: {reason}")


@harness_app.command("show")
def show_harness(
    name: str = typer.Argument(..., help="Harness name, optionally name@ref (version or alias)."),
    root: str = typer.Option(ARTIFACT_DIR, help="Project dir."),
) -> None:
    """Print one harness version's surfaces."""
    base, _, ref = name.partition("@")
    try:
        doc = HarnessStore(root).load(base, ref or None)
    except (FileNotFoundError, ValueError) as exc:
        raise typer.BadParameter(str(exc)) from exc
    _console.print(f"[bold]{doc.name}[/bold] v{doc.version}  doc_hash={doc.doc_hash[:12]}")
    for surface in doc.surfaces:
        budget = f"  budget={surface.budget}" if surface.budget is not None else ""
        _console.print(
            f"\n[bold]{surface.id}[/bold]  ({surface.kind.value}, "
            f"hash={surface.content_hash[:12]}{budget})"
        )
        _console.print(surface.content)


@harness_app.command("init")
def init_harness(
    name: str = typer.Argument("baseline", help="Name for the new harness."),
    root: str = typer.Option(ARTIFACT_DIR, help="Project dir."),
) -> None:
    """Write the baseline harness as v1 and point `champion` at it."""
    store = HarnessStore(root)
    try:
        if store.exists(name):
            raise typer.BadParameter(
                f"harness {name!r} already exists; new versions are appended by "
                "`wmh harness create`, and aliases move with `set_alias`"
            )
        doc = store.save_version(HarnessDoc.baseline(name), alias=CHAMPION_ALIAS)
    except ValueError as exc:  # invalid name -> usage error, not a traceback
        raise typer.BadParameter(str(exc)) from exc
    _console.print(
        f"[green]wrote[/green] {name} v{doc.version} (champion) -> {store.dir_for(name)}"
    )
    _console.print(f"run it: [bold]wmh eval closed-loop <tasks.jsonl> --harness {name}[/bold]")
