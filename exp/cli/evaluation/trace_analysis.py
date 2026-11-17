"""Simulator-only actor councils in the maintained evaluation terminal interface."""

from pathlib import Path

from rich.console import Console
from rich.table import Table
from rich.text import Text

from exp.cli.evaluation.view import heading
from exp.cli.shared.consent import require_spend_consent
from exp.cli.shared.options import usage_error
from exp.cli.shared.progress import progress_display
from exp.cli.shared.theme import EXP_THEME
from exp.common.core.artifacts import JsonObject
from exp.common.project import ProjectStore
from exp.common.release_revision import installed_release_revision
from exp.optimize.evaluation.council.runner import (
    TraceCouncilSpec,
    execute_trace_council,
    prepare_trace_council,
    saved_trace_council,
)


def run_trace_analysis(
    project: str, spec_path: Path, *, root: Path, yes: bool, non_interactive: bool, dry_run: bool
) -> None:
    """Prepare, run or replay one bounded council without any actual-environment adapter.

    Args:
        project: Existing reconstructed simulation project.
        spec_path: Frozen actor, simulator, judge and analyst configuration.
        root: Existing artifact and model-catalog root.
        yes: Explicit authorization of the displayed undefined-cost estimate.
        non_interactive: Disable terminal consent questions.
        dry_run: Prepare without credentials or provider calls.
    """
    console = Console(theme=EXP_THEME)
    with usage_error(OSError, ValueError):
        spec = TraceCouncilSpec.model_validate_json(spec_path.read_bytes())
        store = ProjectStore(root, project)
        prepared = prepare_trace_council(store, spec, code_revision=installed_release_revision())
        previous = saved_trace_council(store, prepared)
        heading(console, project, "Simulator trace analysis")
        console.print(
            Text("Simulator only · actual environment unavailable · proposals are not native truth")
        )
        console.print(
            Text(
                f"{len(spec.tasks)} tasks × {len(spec.actors)} actors · "
                f"{len(spec.analysts)} analysts"
            )
        )
        console.print(Text(f"Simulation: {spec.world_model.artifact_id}"))
        if spec.memory is not None:
            console.print(
                Text(
                    f"Memory: {spec.memory.lineage} frozen v{spec.memory.checkpoint}; "
                    "reserved lineages excluded; no learning"
                )
            )
        console.print(
            Text(f"Physical attempts: at most {spec.maximum_physical_attempts}, including retries")
        )
        console.print(Text(f"Deadline: {spec.deadline.isoformat()}"))
        if dry_run:
            console.print(
                "Prepared without provider calls. Run the same command without --dry-run."
            )
            return
        if previous is None:
            if not require_spend_consent(
                console,
                root=root,
                yes=yes,
                estimated_cost_usd=None,
                cost_is_upper_bound=False,
                command=f"exp eval {project} --trace-analysis {spec_path}",
                non_interactive=non_interactive,
            ):
                return
            with progress_display(console) as progress:
                result = execute_trace_council(store, prepared, progress=progress)
        else:
            console.print("Saved council. No provider calls.")
            result = previous
        render_trace_analysis(console, result)
        console.print(Text(f"Report artifact: {prepared.id}-report"))


def render_trace_analysis(console: Console, result: JsonObject) -> None:
    """Display complete cell coverage and source-located proposals, keeping failures visible.

    Args:
        console: Existing terminal console.
        result: Saved council report, including every planned actor cell.
    """
    table = Table("Actor", "Task", "Tool steps", "Stopped", "Judge", "Failure")
    traces = result.get("traces", [])
    if isinstance(traces, list):
        for row in traces:
            if not isinstance(row, dict):
                continue
            workflow = row.get("workflow")
            evidence = workflow.get("evidence", {}) if isinstance(workflow, dict) else {}
            evidence = evidence if isinstance(evidence, dict) else {}
            steps = evidence.get("steps", [])
            evaluation = row.get("evaluation")
            table.add_row(
                str(row.get("actor_alias")),
                str(row.get("task_id")),
                str(len(steps) if isinstance(steps, list) else 0),
                str(evidence.get("stop_reason", "setup failure")),
                str(evaluation.get("judge_status"))
                if isinstance(evaluation, dict)
                else "unavailable",
                str(row.get("error_type") or "—"),
            )
    console.print(table)
    console.print(
        Text(
            f"Physical attempts: {result.get('physical_attempts')}/"
            f"{result.get('maximum_physical_attempts')}"
        )
    )
    categories = result.get("categories", {})
    if isinstance(categories, dict):
        for category, findings in categories.items():
            if not isinstance(findings, list):
                continue
            console.print(Text(category))
            for finding in findings:
                if isinstance(finding, dict):
                    console.print(Text(f"[{finding.get('severity')}] {finding.get('explanation')}"))
                    console.print(
                        Text(
                            f"Attribution: {finding.get('attribution')} | "
                            f"simulator: {finding.get('simulator_assessment')}"
                        )
                    )
                    console.print(Text(str(finding.get("simulator_explanation"))))
                    console.print(
                        Text(
                            f"Counterfactual: {finding.get('counterfactual_support')} - "
                            f"{finding.get('counterfactual_explanation')}"
                        )
                    )
                    console.print(Text(f"Evidence: {finding.get('evidence_refs')}"))
                    console.print(Text(f"Proposed change: {finding.get('proposed_change')}"))
                    console.print(Text(f"Validation probe: {finding.get('validation_probe')}"))
    console.print(Text(f"Analysis failures: {result.get('analysis_failures')}"))
    console.print("No native accuracy claim. No simulator, verifier or deployment changes applied.")
