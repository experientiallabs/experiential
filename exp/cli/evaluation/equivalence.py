"""Public CLI composition for paired actor equivalence and its read-only trace council."""

from pathlib import Path

from rich.console import Console

from exp.cli.evaluation.spend import review_cost
from exp.cli.shared.consent import require_spend_consent
from exp.common.progress import ProgressEvent
from exp.common.project import ProjectStore
from exp.optimize.evaluation.equivalence import (
    EquivalenceSpec,
    preflight_equivalence,
    run_equivalence,
    saved_equivalence,
)


def run_equivalence_command(
    store: ProjectStore,
    spec: EquivalenceSpec,
    *,
    console: Console,
    revision: str,
    root: Path,
    project: str,
    spec_path: Path,
    yes: bool,
    non_interactive: bool,
    dry_run: bool,
) -> None:
    """Validate the complete schedule, admit spend once and expose every comparison outcome."""
    saved = saved_equivalence(store, spec, code_revision=revision)
    if saved is None and not store.artifacts.exists(spec.identity(revision)):
        preflight_equivalence(store, spec)
    console.print(
        f"Actor equivalence: {len(spec.cases)} actor/task cases; reference, repeated-reference "
        f"control and candidate. Actor HTTP ceiling: {spec.actor_attempt_bound()}; "
        f"shared judge/analyst HTTP ceiling: {spec.maximum_review_attempts}. "
        f"Each planned review may use at most {spec.review_attempt_allowance()} HTTP attempts. "
        "Simulator provider calls retain their separately pinned runtime ceilings."
    )
    if dry_run:
        console.print("Frozen paired contract validated; no calls or publication.")
        return
    if saved is None and not require_spend_consent(
        console,
        root=root,
        yes=yes,
        estimated_cost_usd=review_cost(console, store, spec),
        cost_is_upper_bound=False,
        command=f"exp eval {project} --continuations {spec_path}",
        non_interactive=non_interactive,
    ):
        return

    def progress(event: ProgressEvent) -> None:
        """Show actual stage and cell progress without exposing private model inputs."""
        console.print(f"{event.stage}: {event.detail or ''}")

    result = saved or run_equivalence(store, spec, code_revision=revision, progress=progress)
    console.print_json(
        data={
            "scope": result.scope,
            "metrics": result.metrics,
            "semantic_judgments": [
                {
                    "case": row.case,
                    "repeat": row.repeat,
                    "condition": row.condition,
                    "status": "unassessable" if row.judgment is None else row.judgment.status,
                    "rubric_score_100_not_accuracy": None
                    if row.judgment is None
                    else row.judgment.score_100,
                }
                for row in result.measurements
            ],
            "analyses": {
                alias: review.model_dump(mode="json") for alias, review in result.analyses.items()
            },
            "analysis_failures": result.analysis_failures,
        }
    )
    console.print(
        f"Saved paired evidence: {spec.identity(revision)}-result. No training or promotion."
    )
