"""DeepSWE outcome-matrix conversion command."""

from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console

_console = Console()


def convert_deepswe_cmd(
    source: str = typer.Argument(
        ...,
        help="Directory holding the published DeepSWE v1.1 artifacts: trials.json, tasks.json, "
        "leaderboard-live.json, and the extracted deep-swe-main/tasks/<id>/instruction.md texts.",
    ),
    embedding_cache: str = typer.Option(
        ...,
        "--embedding-cache",
        help="JSON of task id -> recorded embedding vector (Qwen3-Embedding-0.6B, the local "
        "embedder's default model); must cover every task.",
    ),
    out: str = typer.Option(
        "deepswe-bundle",
        "--out",
        help="Directory for the bundle: matrix.json + task_embeddings.npy + "
        "scenario_groups.json (heavy build outputs; published to Hugging Face, never to git).",
    ),
) -> None:
    """Convert published DeepSWE v1.1 trials into an OutcomeMatrix bundle.

    Args:
        source: Directory holding the published DeepSWE artifacts.
        embedding_cache: Recorded task embeddings required for the offline bundle.
        out: Destination directory for normalized outcomes and support artifacts.

    Raises:
        typer.BadParameter: The source data cannot be converted into valid outcomes.
    """
    from wmo.optimize.routing.deepswe import convert_deepswe, top_arm
    from wmo.optimize.routing.outcomes import OutcomeMatrix

    try:
        result = convert_deepswe(Path(source), embedding_cache=Path(embedding_cache), out=Path(out))
    except (FileNotFoundError, ValueError, KeyError) as exc:
        raise typer.BadParameter(str(exc)) from exc
    top = top_arm(OutcomeMatrix.load(result.matrix_path))
    _console.print(
        f"[green]✓[/green] converted {result.models} arms x {result.scenarios} tasks "
        f"({result.scored_outcomes} scored trials, {result.unscored_outcomes} unscored) -> {out}\n"
        f"  integrity gate: {result.crosscheck}\n"
        f"  dropped (unpriced vendors): {len(result.dropped_configs)} configs\n"
        f"  strongest arm {top.name}: graded {top.graded:.3f}, pass@1 {top.pass_at_1:.3f}, "
        f"${top.cost_per_task:.2f}/task over {top.tasks} tasks\n"
        f"  next: `wmo optimize route fit {result.matrix_path} --kind knn "
        f"--fallback {top.name} --embedder local`"
    )


def register(app: typer.Typer) -> None:
    """Register the DeepSWE conversion command on its parent Typer app.

    Args:
        app: Parent Typer application that owns the route command group.
    """
    app.command("convert-deepswe", help="Convert published DeepSWE trials into an outcome matrix.")(
        convert_deepswe_cmd
    )
