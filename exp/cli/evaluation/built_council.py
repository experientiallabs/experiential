"""Build-backed councils through the existing fidelity and actor execution owners."""

from datetime import UTC, datetime
from pathlib import Path

from rich.console import Console
from rich.text import Text

from exp.cli.evaluation.trace_analysis import render_trace_analysis
from exp.cli.evaluation.view import heading
from exp.cli.shared.consent import require_spend_consent
from exp.cli.shared.options import usage_error
from exp.cli.shared.progress import progress_display
from exp.cli.shared.theme import EXP_THEME
from exp.common.core.artifacts import stable_id
from exp.common.project import ProjectStore
from exp.common.release_revision import installed_release_revision
from exp.optimize.evaluation.council.built import (
    execute_built_council,
    prepare_built_council,
)


def run_built_council(
    project: str,
    models: tuple[str, ...] | None,
    *,
    root: Path,
    maximum_concurrency: int = 1,
    yes: bool,
    non_interactive: bool,
    dry_run: bool,
) -> None:
    """Evaluate the current build without a hand-authored council specification.

    Args:
        project: Project with a completed build.
        models: Explicit actor aliases, or None for the five-model default.
        root: Project and model-catalog root.
        maximum_concurrency: Maximum actor/task cells running concurrently.
        yes: Consent to the displayed bounded work and undefined-cost warning.
        non_interactive: Disable terminal authorization questions.
        dry_run: Prepare the immutable plan without provider calls.
    """
    console = Console(theme=EXP_THEME)
    with usage_error(OSError, ValueError):
        store = ProjectStore(root, project)
        prepared = prepare_built_council(
            store,
            models,
            code_revision=installed_release_revision(),
            clock=lambda: datetime.now(UTC),
            maximum_concurrency=maximum_concurrency,
        )
        spec = prepared.council
        heading(console, project, "Build-backed simulator council")
        console.print(Text(f"Fit-only simulator: {spec.world_model.artifact_id}"))
        console.print(
            Text("First measure fidelity on a separate held-out lineage; then run actors.")
        )
        console.print(Text("Actor models: " + ", ".join(actor.alias for actor in spec.actors)))
        console.print(Text(f"Parallel actor sessions: at most {spec.maximum_concurrency}"))
        console.print(
            Text(
                f"{len(spec.actors)} actors · {len(spec.tasks)} tasks · "
                f"{len(spec.analysts)} analysts"
            )
        )
        console.print(
            Text(
                f"Per actor: {spec.maximum_actor_turns} model calls · "
                f"{spec.maximum_tool_operations} tool operations (bounded smoke)"
            )
        )
        if spec.retrieval is not None:
            console.print(
                Text(
                    f"Original evidence: fit-only RAG with {spec.retrieval.jev_alias} cue selection"
                )
            )
        console.print(Text("No learned memory checkpoint is attached to this fresh-build profile."))
        console.print(Text("Simulator only · archived reply references · proposals only"))
        console.print(
            Text(
                f"Fidelity attempts: at most {prepared.calibration.spec.maximum_physical_attempts}"
            )
        )
        console.print(Text(f"Council attempts: at most {spec.maximum_physical_attempts}"))
        if dry_run:
            console.print(
                "Prepared without provider calls. Run the same command without --dry-run."
            )
            return
        report_id = stable_id("trace-council", {"run_id": spec.run_id}) + "-report"
        if not store.artifacts.exists(report_id):
            if not require_spend_consent(
                console,
                root=root,
                yes=yes,
                estimated_cost_usd=None,
                cost_is_upper_bound=False,
                command=f"exp eval {project} --trace-analysis auto",
                non_interactive=non_interactive,
            ):
                return
            with progress_display(console) as progress:
                result = execute_built_council(store, prepared, progress=progress)
        else:
            console.print("Saved council. No provider calls.")
            result = execute_built_council(store, prepared)
        render_trace_analysis(console, result)
        console.print(Text(f"Report artifact: {report_id}"))
