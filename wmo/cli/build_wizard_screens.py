"""Provider-free interactive screens and plan rendering for the build wizard.

Every screen here runs before or without paid provider work: the upfront workflow step
picker, the mandatory explicit trace-path prompt, the credential-free project plan, and
the completed-replay summary. The wizard state machine in ``build_wizard`` owns ordering,
consent, and execution.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

from rich.console import Console
from rich.prompt import Prompt

from wmo.common.models import ModelCatalog
from wmo.common.project import ProjectModelConfiguration
from wmo.common.tasks import TaskCase
from wmo.optimize.router.automatic.replay import AutomaticRouterReplay
from wmo.optimize.router.automatic.reservations import AutomaticRouterCostPlan
from wmo.simulation.build import ProjectBuild
from wmo.simulation.ingest.sources import CANONICAL_TRACE_SOURCES


@dataclass(frozen=True)
class WizardWorkflowSelection:
    """Wizard steps explicitly selected before any provider or model question."""

    providers: bool = True
    build: bool = True
    judge_rubric: bool = False
    judge_calibration: bool = False
    router: bool = True


_WORKFLOW_STEPS: tuple[tuple[str, str, bool], ...] = (
    ("providers", "connections and model roles", True),
    ("build", "tasks, RAG indexes, world model", True),
    ("judge rubric", "custom rubric; off uses the task-success default", False),
    ("judge calibration", "human-labeled audit", False),
    ("router optimization", "simulate, judge, fit, freeze", True),
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
    build_estimate_usd: float
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


def render_completed_replay(
    project: str,
    replay: AutomaticRouterReplay,
    *,
    console: Console,
) -> None:
    """Render one verified completed chain without opening planning or consent.

    Args:
        project: Local project identifier.
        replay: Exact completed router and report identities.
        console: Terminal receiving the replay summary.
    """
    console.print("[green]\u2713[/green] reused every verified project artifact")
    console.print(f"  router  {replay.policy_id}")
    console.print(f"  report  {replay.report_id}")
    console.print(f"  next    wmo run {project}")
    if replay.judgment_status == "provisional":
        console.print(f"  optional wmo config judge calibrate {project}")
        console.print(f"  after approval wmo build {project}")


def render_plan(
    project: str,
    plan: WizardBuildPlan,
    *,
    catalog: ModelCatalog,
    cost_plan: AutomaticRouterCostPlan | None,
    maximum_build_cost_usd: float,
    router_ceiling: float,
    supplied_router_cap: float | None,
    console: Console,
) -> None:
    """Render one concise full wizard plan before the named spend authorization.

    Args:
        project: Local project identifier.
        plan: Provider-free grounded-build plan.
        catalog: Catalog containing selected router defaults.
        cost_plan: Exact automatic-router reservation, or ``None`` when the router
            optimization step is not selected.
        maximum_build_cost_usd: Strict grounded-build ceiling.
        router_ceiling: Effective automatic-router ceiling.
        supplied_router_cap: Optional user cap checked against the exact required ceiling.
        console: Terminal receiving the plan.
    """
    build_ceiling = 0.0 if plan.build_reused else maximum_build_cost_usd
    console.print("[bold]Plan[/bold]")
    console.print(f"  [dim]project[/dim]      {project}")
    console.print(
        f"  [dim]traces[/dim]       {plan.accepted_traces} accepted, {plan.invalid_traces} "
        f"invalid; {plan.fit_tasks} fit, {plan.held_out_tasks} held out"
    )
    console.print(
        f"  [dim]models[/dim]       world {plan.selected.world_model}, "
        f"judge {plan.selected.judge}, embedder {plan.selected.embedder}"
    )
    console.print(
        f"  [dim]build spend[/dim]  estimate ${plan.build_estimate_usd:.6f}; "
        f"ceiling ${build_ceiling:.4f}"
    )
    if cost_plan is not None:
        console.print(
            f"  [dim]router[/dim]       {', '.join(catalog.roles.candidates)} "
            f"[dim](incumbent {catalog.roles.incumbent})[/dim]"
        )
        console.print(
            f"  [dim]router cost[/dim]  embeddings ${cost_plan.router_embedding_cost_usd:.6f}, "
            f"judgments ${cost_plan.judgment_cost_usd:.6f} ({cost_plan.maximum_judgments}), "
            f"simulation ${cost_plan.simulation_cost_usd:.6f} "
            f"({cost_plan.simulated_episode_count} episodes)"
        )
        console.print(
            f"  [dim]router spend[/dim] required ${cost_plan.required_provider_cost_usd:.6f}; "
            f"ceiling ${router_ceiling:.6f}"
        )
    else:
        console.print("  [dim]router[/dim]       not selected")
    if supplied_router_cap is not None:
        console.print(f"  [dim]supplied cap[/dim] ${supplied_router_cap:.6f}")
    console.print(
        f"  [dim]total[/dim]        ceiling ${math.fsum((build_ceiling, router_ceiling)):.4f}"
    )
