"""Native task refinement in the maintained evaluation terminal interface."""

import json
from pathlib import Path

from rich.console import Console
from rich.text import Text

from exp.cli.evaluation.trace_analysis import render_trace_analysis
from exp.cli.evaluation.view import heading
from exp.cli.shared.consent import require_spend_consent
from exp.cli.shared.options import usage_error
from exp.cli.shared.progress import progress_display
from exp.cli.shared.theme import EXP_THEME
from exp.common.core.artifacts import JsonObject
from exp.common.project import ProjectStore
from exp.common.release_revision import installed_release_revision


def run_task_refinement(
    project: str, spec_path: Path, *, root: Path, yes: bool, non_interactive: bool, dry_run: bool
) -> None:
    """Prepare, execute or replay one bounded task refinement.

    Args:
        project: Existing reconstructed simulation project.
        spec_path: Frozen task, challenge and role configuration.
        root: Existing artifact and model-catalog root.
        yes: Consent to the displayed bounded work and undefined-cost warning.
        non_interactive: Disable terminal authorization questions.
        dry_run: Prepare without provider calls or credentials.
    """
    from exp.optimize.evaluation.tasks.refinement import (
        TaskRefinementSpec,
        execute_task_refinement,
        prepare_task_refinement,
        saved_task_refinement,
    )

    console = Console(theme=EXP_THEME)
    with usage_error(OSError, ValueError):
        spec = TaskRefinementSpec.model_validate_json(spec_path.read_bytes())
        store = ProjectStore(root, project)
        prepared = prepare_task_refinement(store, spec, code_revision=installed_release_revision())
        heading(console, project, "Simulator task refinement")
        console.print(Text("Simulator only · actual environment unavailable"))
        console.print(Text("Intent and world remain fixed; model edits are proposals only."))
        console.print(Text(f"Plan artifact: {prepared.id}"))
        console.print(Text(f"Deadline: {prepared.spec.deadline.isoformat()}"))
        console.print(
            Text(f"Refinement-role attempts: at most {prepared.spec.maximum_physical_attempts}")
        )
        console.print(
            Text(
                "Child actor-council attempts: at most "
                f"{prepared.spec.child_maximum_physical_attempts}"
            )
        )
        if dry_run:
            console.print(
                "Prepared without provider calls. Run the same command without --dry-run."
            )
            return
        previous = saved_task_refinement(store, prepared)
        if previous is None:
            if not require_spend_consent(
                console,
                root=root,
                yes=yes,
                estimated_cost_usd=None,
                cost_is_upper_bound=False,
                command=f"exp eval {project} --refine-task {spec_path}",
                non_interactive=non_interactive,
            ):
                return
            with progress_display(console) as progress:
                result = execute_task_refinement(store, prepared, progress=progress)
        else:
            console.print("Saved task refinement. No provider calls.")
            result = execute_task_refinement(store, prepared)
        render_task_refinement(console, result)
        report_manifest = store.artifacts.read(prepared.id + "-report").manifest
        for reference in report_manifest.inputs:
            dependency = store.artifacts.read(reference.artifact_id).manifest
            if dependency.artifact_type == "trace-council-report":
                render_trace_analysis(
                    console,
                    json.loads(store.artifacts.read_bytes(reference.artifact_id, "report.json")),
                )
        console.print(Text(f"Report artifact: {prepared.id}-report"))


def render_task_refinement(console: Console, result: JsonObject) -> None:
    """Render qualification separately from simulator truth.

    Args:
        console: Existing terminal console.
        result: Saved task-refinement report.
    """
    console.print(Text(f"Parent artifact: {result.get('parent_artifact_id', 'unavailable')}"))
    console.print(Text(f"Child artifact: {result.get('child_artifact_id', 'not produced')}"))
    console.print(Text(f"Development: {result.get('development_status', 'incomplete')}"))
    console.print(Text(f"Confirmation: {result.get('confirmation_status', 'not run')}"))
    console.print(
        Text(
            f"Protected checks: gained {result.get('gains', [])} · lost {result.get('losses', [])}"
        )
    )
    console.print(Text(f"Physical attempts: {result.get('physical_attempts', 'unavailable')}"))
    console.print(
        Text(
            "Qualification applies to this simulator task and supplied labels; "
            "it does not establish native environment fidelity or RL release approval."
        )
    )
