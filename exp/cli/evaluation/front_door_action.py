"""The decision-variance front door: point at traces, consent once, build through the pipeline.

The ``decision-variance`` action with a ``front_door`` file reads one
:class:`~exp.optimize.evaluation.variance.frontdoor.FrontDoorSpec`. By default it prints the
plan: every stage with its status, then the completed work, the remaining priced calls and the
unpriced components (including phases that are priced only once an earlier phase exists). With
``execute`` it asks spend consent once and runs or resumes
:func:`~exp.optimize.evaluation.variance.frontdoor.run_front_door`. A pending project build runs
``exp build --non-interactive --yes`` in a subprocess: the consent to the plan digest, which
names the pending build, is the build's confirmation of its own embedding estimate.
"""

from __future__ import annotations

import subprocess
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from rich.text import Text

from exp.cli.evaluation.continuation_context import ContinuationContext
from exp.cli.evaluation.learning_summary import save_details
from exp.cli.evaluation.spend import review_components
from exp.common.core.artifacts import JsonObject
from exp.common.project import ProjectStore
from exp.optimize.evaluation.executable.model_costs import CostComponent
from exp.optimize.evaluation.variance.frontdoor import (
    FrontDoorApproval,
    FrontDoorDependencies,
    FrontDoorPlan,
    FrontDoorReport,
    FrontDoorSpec,
    TraceSource,
    plan_front_door,
    run_front_door,
)
from exp.optimize.evaluation.variance.frontdoor_states import StageState, statecharts


@dataclass(frozen=True)
class ExpBuild:
    """Run ``exp build`` for one project in a subprocess, never asking a question.

    Attributes:
        root: The EXP root.
        project: The project name.
    """

    root: Path
    project: str

    def __call__(self, project: ProjectStore, source: TraceSource) -> None:
        """Build the project from the source.

        Args:
            project: The project store (its identity must be this builder's project).
            source: The trace source.

        Raises:
            ValueError: The store names another project, or the build exited unsuccessfully.
        """
        if project.paths.project_id != self.project:
            raise ValueError("the project builder was bound to another project")
        command = [
            sys.executable,
            "-m",
            "exp",
            "build",
            self.project,
            "--traces",
            str(source.path),
            "--source",
            source.source,
            "--root",
            str(self.root),
            "--non-interactive",
            # The front door runs its builder only after consent to the exact plan digest,
            # which names this build; that consent is exp build's confirmation.
            "--yes",
        ]
        if source.identity is not None:
            command.extend(["--identity", source.identity])
        completed = subprocess.run(command, check=False)  # noqa: S603 - fixed argument vector
        if completed.returncode != 0:
            raise ValueError(
                f"exp build exited with status {completed.returncode}; rerun "
                f"`{' '.join(command[2:])}` interactively to review its estimate, then rerun "
                "the front door"
            )


def handle_front_door(context: ContinuationContext, path: Path, execute: bool) -> None:
    """Print the front-door plan and, with ``execute``, consent once and run or resume it.

    Args:
        context: Validated ``--continuations`` invocation.
        path: The ``FrontDoorSpec`` file.
        execute: Run after consent; otherwise only the plan is printed.
    """
    store, console, revision = context.store, context.console, context.revision
    spec = FrontDoorSpec.model_validate_json(path.read_bytes())
    dependencies = FrontDoorDependencies(
        builder=ExpBuild(root=context.root, project=context.project)
    )
    plan = plan_front_door(store, spec, code_revision=revision, dependencies=dependencies)
    for line in front_door_lines(plan):
        console.print(Text(line))
    components = front_door_components(plan)
    if components:
        review_components(console, components)
    if not execute or context.dry_run:
        console.print(
            "Dry run: front-door plan and upper-bound costs only; no provider call. Set execute "
            "to run it."
        )
        return
    approval = None
    if not plan.complete:
        unpriced = tuple(row for row in components if row.usd is None)
        if not context.consent_ceiling(
            stage=plan.digest, known_usd=plan.known_usd, unpriced=unpriced, remaining=True
        ):
            return
        ceiling = context.spend.ceiling_usd
        approval = FrontDoorApproval(
            plan_sha256=plan.digest,
            ceiling_usd=plan.known_usd if ceiling is None else ceiling,
            unpriced_attempts=context.spend.unpriced_attempts,
        )
    report = run_front_door(
        store, spec, code_revision=revision, approval=approval, dependencies=dependencies
    )
    identity = f"front-door-{plan.spec_sha256[:16]}"
    details = save_details(store, "front-door", identity, report.model_dump_json(indent=2))
    console.print_json(data=front_door_summary(report))
    console.print(Text(f"Report: {details}"))


def _printer(context: ContinuationContext) -> Callable[[tuple[str, ...]], None]:
    """Print each phase's dry-run view on the command console before it is admitted."""

    def show(lines: tuple[str, ...]) -> None:
        """Print one phase's view."""
        for line in lines:
            context.console.print(Text(line))

    return show


def front_door_lines(plan: FrontDoorPlan) -> tuple[str, ...]:
    """Describe every stage, then completed work, remaining priced calls and unpriced parts.

    Args:
        plan: The credential-free plan.

    Returns:
        The printed lines.
    """
    lines = [f"Front-door plan {plan.digest[:12]} ({len(plan.stages)} stages):"]
    for index, row in enumerate(plan.stages, start=1):
        if row.status == StageState.DONE:
            status = "complete, replays with no call"
        elif row.status == StageState.BLOCKED:
            status = "waiting on an earlier stage"
        else:
            unpriced = f", {len(row.unpriced)} unpriced components" if row.unpriced else ""
            status = f"up to ${row.known_usd:.6f}{unpriced}"
        lines.append(f"{index}. {row.stage}: {status}")
        lines.extend(f"   {line}" for line in row.lines)
    done = [row.stage for row in plan.stages if row.status == StageState.DONE]
    pending = [row for row in plan.stages if row.status == StageState.PLANNED]
    attempts = sum(line.attempts for row in pending for line in row.costs)
    lines.append(f"Completed work: {', '.join(done) if done else 'none'}.")
    lines.append(
        f"Remaining calls: up to ${plan.known_usd:.6f} priced across {len(pending)} pending "
        f"stages, at most {attempts} physical requests."
    )
    unpriced = [f"{line.name} ({line.attempts} attempts)" for line in plan.unpriced]
    unpriced.extend(f"{row.stage}: {note}" for row in plan.stages for note in row.notes)
    lines.append("Unpriced components:" + ("" if unpriced else " none."))
    lines.extend(f"   {line}" for line in unpriced)
    lines.append("State machines (Mermaid):")
    lines.extend(statecharts().splitlines())
    return tuple(lines)


def front_door_components(plan: FrontDoorPlan) -> tuple[CostComponent, ...]:
    """Return every pending stage's cost lines as reviewable components, named by stage."""
    return tuple(
        CostComponent(f"{row.stage}: {line.name}", line.attempts, line.usd)
        for row in plan.stages
        if row.status == StageState.PLANNED
        for line in row.costs
    )


def front_door_summary(report: FrontDoorReport) -> JsonObject:
    """Summarize a front-door run for the terminal.

    Args:
        report: The run report.

    Returns:
        Registered starts, each start's pipeline status and why the run stopped.
    """
    return {
        "complete": report.complete,
        "stopped": report.stopped,
        "preparation": None if report.preparation is None else report.preparation.artifact_id,
        "registered": [binding.task_id for binding in report.bindings],
        "failed": list(report.failed_tasks),
        "refused": list(report.refused_tasks),
        "pipelines": [
            {
                "task": row.task_id,
                "run": None if row.run is None else row.run.artifact_id,
                "complete": row.complete,
                "stopped": row.stopped,
            }
            for row in report.pipelines
        ],
    }
