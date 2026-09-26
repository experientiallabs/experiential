"""Compact evaluation results with explicit access to reports and accounting."""

import typer
from rich.console import Console
from rich.table import Table
from rich.text import Text

from exp.cli.shared.picker import PickerOption, choose_one
from exp.common.project import ProjectStore
from exp.optimize.evaluation.export import export_report, load_report_evidence
from exp.optimize.evaluation.runs import EvaluationRun


def heading(console: Console, project: str, subtitle: str) -> None:
    """Append a compact project heading while preserving terminal history.

    Args:
        console: Output console receiving the heading in normal scrollback.
        project: Project name rendered as literal text.
        subtitle: Current stage label rendered beside the project name.
    """
    title = Text(f"\nexp eval · {project}", style="bold")
    title.append(f" · {subtitle}", style="dim")
    console.print(title)


def render_report(console: Console, project: ProjectStore, run: EvaluationRun) -> None:
    """Show measured quality, assistant cost, latency, and coverage caveats.

    Args:
        console: Output console receiving the summary table.
        project: Project owning the saved report evidence.
        run: Saved evaluation to render without provider calls.

    Raises:
        OSError: Report evidence cannot be read.
        ValueError: The saved evidence is missing or inconsistent.
    """
    evidence = load_report_evidence(project, run)
    report = evidence.report
    heading(console, project.paths.project_id, "Saved results")
    repeats = run.prepared.setup.repeats
    console.print(
        f"{len(evidence.tasks)} scenarios · {len(report.models)} models · "
        f"{repeats} {'run' if repeats == 1 else 'runs'} per scenario\n"
    )
    table = Table(box=None, padding=(0, 2), expand=False)
    table.add_column("Model", overflow="fold")
    table.add_column("Score /100", justify="right", no_wrap=True)
    table.add_column("$/task", justify="right", no_wrap=True)
    table.add_column("Latency", justify="right", no_wrap=True)
    for metric in report.models:
        table.add_row(
            Text(metric.candidate.alias),
            f"{metric.quality * 100:.1f}" if metric.quality is not None else "unavailable",
            _number(metric.operating_cost_usd, "$"),
            _duration(metric.latency_seconds),
        )
    console.print(table)
    console.print("\n[dim]Assistant cost and latency. Shared valid runs only.[/dim]")
    if report.excluded_cells:
        console.print(
            f"[yellow]{report.excluded_cells} excluded pairs · "
            f"{report.compared_cells} compared[/yellow]"
        )
    if run.prepared.setup.judgment_status == "provisional":
        console.print("[dim]Provisional judge[/dim]")


def inspect_report(console: Console, project: ProjectStore, run: EvaluationRun) -> None:
    """Open the offline trace viewer or accounting details only when requested.

    Args:
        console: Terminal used for the results menu and fallback browser path.
        project: Project owning the evidence and local report exports.
        run: Saved evaluation to inspect until Back or cancellation.

    Raises:
        OSError: Local evidence cannot be read or report files cannot be written.
        ValueError: Saved report evidence is missing or inconsistent.
    """
    _, html_path = export_report(project, run)
    render_report(console, project, run)
    while True:
        choice = choose_one(
            console,
            title="Results",
            options=(
                PickerOption("open", "Open report", "plots and traces"),
                PickerOption("details", "Details"),
                PickerOption("back", "Back"),
            ),
        )
        if not choice.values or choice.values[0] == "back":
            return
        if choice.values[0] == "open":
            if typer.launch(html_path.resolve().as_uri()) != 0:
                console.print(Text(f"Open in your browser: {html_path}"))
        else:
            render_details(console, project, run)


def render_details(console: Console, project: ProjectStore, run: EvaluationRun) -> None:
    """Expose full coverage, separate experiment spend, and portable artifact paths.

    Args:
        console: Output console receiving the detailed accounting and export paths.
        project: Project owning the evidence and local report exports.
        run: Saved evaluation supplying coverage and simulation/judge spend.

    Raises:
        OSError: Local evidence cannot be read or report files cannot be written.
        ValueError: Saved report evidence is missing or inconsistent.
    """
    evidence = load_report_evidence(project, run)
    heading(console, project.paths.project_id, "Result details")
    console.print(Text(f"Run: {run.run_id}"))
    console.print(f"{evidence.report.compared_cells} shared valid scenario/run pairs")
    table = Table("Model", "Valid", "Invalid", "Incomplete", "Not run", box=None)
    for metric in evidence.report.models:
        table.add_row(
            Text(metric.candidate.alias),
            str(metric.scored_cells),
            str(metric.failed_cells),
            str(metric.incomplete_cells),
            str(metric.not_run_cells),
        )
    console.print(table)
    console.print(
        f"\nExperiment spend: simulation {_number(run.simulation_cost_usd, '$')} · "
        f"judge {_number(run.judge_cost_usd, '$')}"
    )
    json_path, html_path = export_report(project, run)
    console.print(Text(f"\nHTML: {html_path}\nJSON: {json_path}"))


def _number(value: float | None, prefix: str, suffix: str = "") -> str:
    """Format measurements without turning missing or small positive costs into zero.

    Args:
        value: Measured value, or ``None`` when unavailable.
        prefix: Unit or currency marker placed before the formatted value.
        suffix: Optional unit marker placed after the formatted value.

    Returns:
        A unit-bearing value with sufficient precision, or ``"unavailable"``.
    """
    if value is None:
        return "unavailable"
    precision = 6 if 0 < abs(value) < 0.001 else 4
    rendered = f"{value:.{precision}f}"
    if value != 0 and float(rendered) == 0:
        rendered = f"{value:.3g}"
    return f"{prefix}{rendered}{suffix}"


def _duration(seconds: float | None) -> str:
    """Show latency in readable units, preserving subsecond measurements.

    Args:
        seconds: Measured latency in seconds, or ``None`` when unavailable.

    Returns:
        Milliseconds, seconds, or minutes with seconds, or ``"unavailable"``.
    """
    if seconds is None:
        return "unavailable"
    if seconds == 0:
        return "0s"
    if seconds < 1:
        return f"{seconds * 1000:.2g}ms"
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes, remainder = divmod(round(seconds), 60)
    return f"{minutes}m {remainder:02d}s"
