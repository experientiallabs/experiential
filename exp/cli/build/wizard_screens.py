"""Provider-free interactive screens for the build wizard.

Every screen here runs before or without paid provider work: the upfront workflow step
picker, the mandatory explicit trace-path prompt, and the completed-replay summary. The
wizard state machine in ``build_wizard`` owns ordering, consent, and execution.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from rich.console import Console
from rich.prompt import Prompt

from exp.common.models import ModelCatalog
from exp.common.project import ProjectModelConfiguration
from exp.common.tasks import TaskCase
from exp.simulation.build import ProjectBuild
from exp.simulation.ingest.sources import CANONICAL_TRACE_SOURCES


@dataclass(frozen=True)
class WizardWorkflowSelection:
    """Wizard steps explicitly selected before any provider or model question."""

    providers: bool = True
    build: bool = True
    judge_rubric: bool = False
    judge_calibration: bool = False
    router: bool = True


_WORKFLOW_STEPS: tuple[tuple[str, str, bool], ...] = (
    ("providers", "connect providers and assign model roles", True),
    ("build", "ingest traces and build the world model", True),
    ("judge rubric", "edit the judge rubric; off keeps the task-success default", False),
    ("judge calibration", "review and approve judge examples by hand", False),
    ("router optimization", "simulate, judge, and fit the router", True),
)


@dataclass(frozen=True)
class WizardBuildPlan:
    """Provider-free grounded-build plan shown before the one spend authorization."""

    trace_path: Path | None
    source: str
    catalog: ModelCatalog
    selected: ProjectModelConfiguration
    tasks: tuple[TaskCase, ...]
    completed: ProjectBuild | None
    accepted_traces: int
    invalid_traces: int
    fit_tasks: int
    held_out_tasks: int
    build_estimate_usd: float | None
    build_reused: bool


def select_workflow(*, console: Console) -> WizardWorkflowSelection:
    """Show every wizard step upfront and read one explicit step selection.

    Args:
        console: Interactive terminal.

    Returns:
        The explicit set of wizard steps the operator chose to run.
    """
    console.print("[bold]Workflow[/bold]")
    for index, (name, summary, default) in enumerate(_WORKFLOW_STEPS, start=1):
        marker = "[green]on[/green] " if default else "[dim]off[/dim]"
        console.print(f"  {index}. {name:<19} {marker} [dim]{summary}[/dim]")
    default_answer = ",".join(
        str(index) for index, step in enumerate(_WORKFLOW_STEPS, start=1) if step[2]
    )
    while True:
        answer = Prompt.ask(
            "Steps to run (comma-separated)",
            default=default_answer,
            console=console,
        )
        tokens = tuple(token.strip() for token in (answer or "").split(",") if token.strip())
        if tokens and all(
            token.isdigit() and 1 <= int(token) <= len(_WORKFLOW_STEPS) for token in tokens
        ):
            chosen = {int(token) for token in tokens}
            return WizardWorkflowSelection(
                providers=1 in chosen,
                build=2 in chosen,
                judge_rubric=3 in chosen,
                judge_calibration=4 in chosen,
                router=5 in chosen,
            )
        console.print(f"[red]error[/red] enter step numbers between 1 and {len(_WORKFLOW_STEPS)}")


def select_trace(initial_source: str, *, console: Console) -> tuple[str, Path]:
    """Select one supported trace source and require an explicit local trace path.

    The wizard never infers a trace file from the working directory; when -t/--traces
    was not given, the operator always names the exact export to use.

    Args:
        initial_source: CLI-provided initial source choice.
        console: Interactive terminal.

    Returns:
        Canonical source name and validated local path.
    """
    source = initial_source.strip().casefold()
    if source not in CANONICAL_TRACE_SOURCES:
        source = Prompt.ask(
            "Trace source",
            choices=list(CANONICAL_TRACE_SOURCES),
            default="otlp",
            console=console,
        )
    while True:
        answer = Prompt.ask(f"Trace path ({source} export)", console=console)
        selected = (answer or "").strip()
        if not selected:
            console.print("[red]error[/red] a local trace path is required")
            continue
        path = Path(selected).expanduser()
        if not path.exists():
            console.print(f"[red]error[/red] trace file not found: {path}")
            continue
        if not path.is_file():
            console.print(f"[red]error[/red] the trace path must name a file: {path}")
            continue
        return source, path


def render_completed_replay(*, console: Console) -> None:
    """Render one verified completed chain without opening planning or consent.

    Args:
        console: Terminal receiving the replay summary.
    """
    console.print("[green]\u2713[/green] reused every verified project artifact")
    console.print("[green]Complete[/green]")
