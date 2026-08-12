"""Table renderers for the terminal CLI."""

from __future__ import annotations

from rich.table import Table

from wmo.common.config import ModelInfo


def models_table(infos: list[ModelInfo]) -> Table:
    """Render every built world model for `wmo list`.

    An artifact the store could not read remains visible as `unreadable`, so a broken artifact
    cannot hide healthy models beside it.

    Args:
        infos: The discovered model artifacts.

    Returns:
        A table with one row for each discovered model artifact.
    """
    table = Table(title="world models")
    table.add_column("name", style="bold")
    table.add_column("serve provider")
    table.add_column("held-out", justify="right")
    table.add_column("rollouts", justify="right")
    table.add_column("frontier", justify="right")
    for info in infos:
        table.add_row(
            info.name,
            "[red]unreadable[/red]"
            if info.error is not None
            else f"{info.serve_provider} ({info.serve_model})",
            "-" if info.held_out_accuracy is None else f"{info.held_out_accuracy:.3f}",
            "-" if info.rollouts_used is None else str(info.rollouts_used),
            "-" if info.frontier_size is None else str(info.frontier_size),
        )
    return table
