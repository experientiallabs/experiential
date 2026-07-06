"""`wmh eval --mode closed-loop` and `wmh eval agreement` — the closed-loop halves of eval.

Kept out of `app.py` so the (large) eval command stays readable; `app.py` routes here.
Closed-loop mode runs the fixed agent against a built world model and scores task success;
`agreement` compares two saved closed-loop reports (e.g. one produced against the world model and
one against a real environment) — the outcome-agreement check docs/reference/closed_loop.md names.
"""

from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console

from wmh.config import WorldModelStore
from wmh.engine import load_world_model
from wmh.evals.agreement import compute_agreement
from wmh.evals.closed_loop import ClosedLoopEval, ClosedLoopReport
from wmh.evals.gold import GoldJudge, GoldVerdict
from wmh.evals.tasks import load_tasks
from wmh.harness.runtime import AgentRuntime


def run_closed_loop(
    console: Console,
    *,
    tasks_file: str,
    name: str | None,
    root: str,
    k: int,
    max_turns: int,
    out: str | None,
) -> None:
    """Run the fixed agent on each task against the world model; print and optionally save."""
    try:
        tasks = load_tasks(tasks_file)
    except (OSError, ValueError) as exc:  # missing file, malformed JSONL, empty, duplicate ids
        raise typer.BadParameter(f"cannot load tasks from {tasks_file!r}: {exc}") from exc
    store = WorldModelStore(root)
    try:
        model_dir = store.resolve(name)
    except (FileNotFoundError, ValueError) as exc:
        raise typer.BadParameter(str(exc)) from exc
    world_model, provider = load_world_model(model_dir)

    console.print(
        f"closed-loop: agent vs world model [bold]{model_dir.name}[/bold] "
        f"on {len(tasks)} task(s), k={k}…"
    )

    def _progress(task_id: str, attempt: int, verdict: GoldVerdict) -> None:
        mark = "[green]pass[/green]" if verdict.passed else "[red]fail[/red]"
        console.print(f"  {task_id} #{attempt}: {mark} ({verdict.rationale})")

    evaluation = ClosedLoopEval(
        tasks,
        world_model,
        provider,
        GoldJudge(provider),
        label=model_dir.name,
        k=k,
        runtime=AgentRuntime(provider, max_turns=max_turns),
        on_progress=_progress,
    )
    report = evaluation.run()
    for task_id, outcome in report.per_task.items():
        console.print(
            f"  {task_id:24} success={outcome.success_rate:.2f} "
            f"assertions={outcome.mean_fraction:.2f}"
        )
    console.print(f"[bold]OVERALL[/bold] {report.summary()}")
    if out:
        Path(out).write_text(report.model_dump_json(indent=2), encoding="utf-8")
        console.print(f"wrote closed-loop report -> {out}")


def run_agreement(console: Console, *, report_a: str, report_b: str, threshold: float) -> None:
    """Compare two saved closed-loop reports task-by-task and print the agreement verdict."""
    a = _load_report(report_a)
    b = _load_report(report_b)
    result = compute_agreement(a, b, pass_threshold=threshold)
    c = result.confusion
    la, lb = result.label_a or "A", result.label_b or "B"
    console.print(f"[bold]task verdict confusion[/bold] ({la} vs {lb}):")
    console.print(f"  {la}-pass & {lb}-pass: {c.a_pass_b_pass}")
    console.print(f"  {la}-pass & {lb}-FAIL: {c.a_pass_b_fail}  (A over-credits these)")
    console.print(f"  {la}-FAIL & {lb}-pass: {c.a_fail_b_pass}")
    console.print(f"  {la}-FAIL & {lb}-FAIL: {c.a_fail_b_fail}")
    console.print(f"[bold]VERDICT[/bold] {result.summary()}")


def _load_report(path: str) -> ClosedLoopReport:
    try:
        return ClosedLoopReport.model_validate_json(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise typer.BadParameter(f"cannot read closed-loop report {path!r}: {exc}") from exc
