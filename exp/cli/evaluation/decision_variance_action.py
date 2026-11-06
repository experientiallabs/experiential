"""The ``decision-variance`` action: plan, consent to and run the whole decision-variance pipeline.

The action reads one :class:`~exp.optimize.evaluation.variance.pipeline_spec.DecisionVarianceSpec`
file. By default it only prints the stage plan: every stage's identity, whether it is saved and
replays with no call, what it will do and its upper-bound cost inventory. With ``execute`` it
asks spend consent once for the whole plan (the operation's ceiling then holds every stage's
admitted requests), runs or resumes the pipeline through
:func:`~exp.optimize.evaluation.variance.pipeline.run_decision_variance` and prints each stage's
replay status and measured usage. A run whose stages are all saved replays with no consent and no
call. With a ``front_door`` file instead, the action runs the whole sequence from a trace source
(:mod:`~exp.cli.evaluation.front_door_action`).
"""

from __future__ import annotations

from pathlib import Path
from typing import Self

from pydantic import model_validator
from rich.text import Text

from exp.cli.evaluation.continuation_context import ContinuationContext
from exp.cli.evaluation.front_door_action import handle_front_door
from exp.cli.evaluation.learning_summary import save_details
from exp.cli.evaluation.spend import review_components
from exp.common.core.artifacts import ContractModel
from exp.optimize.evaluation.executable.model_costs import CostComponent
from exp.optimize.evaluation.variance.pipeline import (
    plan_decision_variance,
    run_decision_variance,
)
from exp.optimize.evaluation.variance.pipeline_spec import (
    DecisionVarianceSpec,
    PipelineApproval,
    PipelinePlan,
    PipelineReport,
)


class DecisionVarianceAction(ContractModel):
    """One decision-variance run, from a registered spec or from traces through the front door.

    Attributes:
        spec: ``DecisionVarianceSpec`` JSON file of one registered start's pipeline.
        front_door: ``FrontDoorSpec`` JSON file pointed at a trace source: build, tagging,
            environment families, registered starts and every start's pipeline, in one run.
        execute: Run after spend consent; otherwise only the plan and its cost bounds are
            printed.
    """

    spec: Path | None = None
    front_door: Path | None = None
    execute: bool = False

    @model_validator(mode="after")
    def _one_source(self) -> Self:
        """Require exactly one of a pipeline spec and a front-door spec."""
        if (self.spec is None) == (self.front_door is None):
            raise ValueError("give exactly one of spec and front_door")
        return self


def handle_decision_variance(context: ContinuationContext, action: DecisionVarianceAction) -> None:
    """Print the stage plan and, with ``execute``, consent once and run or resume the pipeline.

    Args:
        context: Validated ``--continuations`` invocation.
        action: The spec file and whether to execute.

    Returns:
        None. The plan, any consent question and the result are printed to the console.
    """
    if action.front_door is not None:
        handle_front_door(context, action.front_door, action.execute)
        return
    assert action.spec is not None
    store, console, revision = context.store, context.console, context.revision
    spec = DecisionVarianceSpec.model_validate_json(action.spec.read_bytes())
    plan = plan_decision_variance(store, spec, code_revision=revision)
    for line in plan_lines(plan):
        console.print(Text(line))
    components = plan_components(plan)
    if components:
        review_components(console, components)
    if not action.execute or context.dry_run:
        console.print(
            "Dry run: stage plan and upper-bound costs only; no provider call. Set execute to "
            "run it."
        )
        return
    approval = None
    if not plan.complete:
        unpriced = tuple(row for row in components if row.usd is None)
        if not context.consent_ceiling(
            stage=plan.digest, known_usd=plan.known_usd, unpriced=unpriced, remaining=True
        ):
            return
        approval = PipelineApproval.for_plan(plan, ceiling_usd=context.spend.ceiling_usd)
    report = run_decision_variance(store, spec, code_revision=revision, approval=approval)
    details = save_details(
        store, "decision-variance", report.run.artifact_id, report.model_dump_json(indent=2)
    )
    console.print_json(data=report_summary(report))
    console.print(Text(f"Report: {details}"))


def plan_lines(plan: PipelinePlan) -> tuple[str, ...]:
    """Describe every stage before consent.

    Args:
        plan: The credential-free plan.

    Returns:
        One heading per stage with its status and bound, followed by its plan lines.
    """
    lines = [f"Decision-variance plan {plan.digest[:12]} ({len(plan.stages)} stages):"]
    for index, row in enumerate(plan.stages, start=1):
        if row.saved:
            status = "saved, replays with no call"
        else:
            unpriced = f", {len(row.unpriced)} unpriced components" if row.unpriced else ""
            status = f"up to ${row.known_usd:.6f}{unpriced}"
        lines.append(f"{index}. {row.stage} [{row.identity[:12]}]: {status}")
        lines.extend(f"   {line}" for line in row.lines)
    total = f"Upper bound: ${plan.known_usd:.6f} priced across unsaved stages"
    if plan.unpriced:
        total += f", plus {len(plan.unpriced)} unpriced components"
    lines.append(total + ".")
    return tuple(lines)


def plan_components(plan: PipelinePlan) -> tuple[CostComponent, ...]:
    """Return every unsaved stage's cost lines as reviewable components, named by stage."""
    return tuple(
        CostComponent(f"{row.stage}: {line.name}", line.attempts, line.usd)
        for row in plan.stages
        if not row.saved
        for line in row.costs
    )


def report_summary(report: PipelineReport) -> dict[str, object]:
    """Summarize a run for the terminal.

    Args:
        report: The run report.

    Returns:
        Each stage's replay status, measured usage and summary, and why the run stopped.
    """
    return {
        "run": report.run.artifact_id,
        "complete": report.complete,
        "stopped": report.stopped,
        "stages": [
            {
                "stage": row.stage,
                "replayed": row.replayed,
                "physical_attempts": row.usage.physical_attempts,
                "accounted_usd": row.usage.accounted_usd,
                "summary": row.summary,
            }
            for row in report.stages
        ],
    }
