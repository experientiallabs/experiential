"""Knowledge inspection command for built world models."""

from __future__ import annotations

from typing import TYPE_CHECKING

import typer
from rich.console import Console
from rich.markup import escape

from wmo.common.config import (
    ARTIFACT_DIR,
    ArtifactPaths,
    WorldModelStore,
    load_config,
)

if TYPE_CHECKING:
    pass

from wmo.cli.catalog_cmd import _resolve_model_any

_console = Console()


def knowledge_(
    name: str = typer.Option(None, "--name", help="World model (default: the only one)."),
    root: str = typer.Option(
        ARTIFACT_DIR,
        help="Project dir. Shipped example models are found too while this is the default "
        "project dir; point it elsewhere to search that root alone.",
    ),
) -> None:
    """Show a model's knowledge base: the env's canonical facts, a folder of editable markdown.

    The printed directory IS the editing interface, open it in any editor. `rules.md`/
    `entities.md`/`schemas.md` are seeded at build (with knowledge enabled); `learned.md` collects
    the env's own cross-session notes; `grounded.md` caches web-search groundings. Models are
    resolved across the default project and downloaded benchmark artifacts.

    Args:
        name: World model to inspect, or the only resolvable model.
        root: Project artifact directory, with shipped examples available at the default root.
    """
    from wmo.simulation.model.knowledge import KnowledgeBase

    store_root, resolved = _resolve_model_any(name, root)
    model_dir = WorldModelStore(str(store_root)).resolve(resolved)
    kb = KnowledgeBase(ArtifactPaths(model_dir).knowledge)
    _console.print(f"[bold]{escape(str(kb.directory))}[/bold]")
    build_with_knowledge = f"`wmo build --name {resolved} --file <traces> --knowledge`"
    if kb.is_empty:
        state = "is empty" if kb.directory.is_dir() else "does not exist yet"
        _console.print(
            f"(that directory {state}) seed one with {build_with_knowledge}, which extracts "
            "rules.md / entities.md / schemas.md from the train traces"
        )
        return
    if not load_config(model_dir).knowledge:
        # The files are on disk but `WorldModel.load` gates the KB on config.knowledge, so at the
        # default fidelity nothing here reaches the env prompt. Say so instead of printing them
        # back as if they were live.
        _console.print(
            f"[yellow]inert[/yellow]: {resolved!r} was built without a knowledge base, so "
            "`wmo serve` never renders these files into the environment "
            f"prompt. Activate them by rebuilding with {build_with_knowledge}, or by setting "
            f"`knowledge = true` in {escape(str(ArtifactPaths(model_dir).config))}."
        )
    # Knowledge is hand-edited markdown: `[/items]`, `list[str]` and `[text](url)` are ordinary
    # content, so it is escaped rather than parsed as rich markup (which would crash on the
    # first, and silently delete the other two).
    for file_name, content in kb.files().items():
        _console.print(f"\n[bold]## {escape(file_name)}[/bold]")
        _console.print(escape(content.strip()))
