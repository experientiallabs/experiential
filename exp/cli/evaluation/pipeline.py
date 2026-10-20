"""Public manifest-driven development factory using the existing paid stage owners."""

from pathlib import Path

from rich.console import Console

from exp.cli.shared.consent import require_spend_consent
from exp.cli.shared.options import usage_error
from exp.cli.shared.progress import progress_display
from exp.cli.shared.theme import EXP_THEME
from exp.common.project import ProjectStore
from exp.common.release_revision import installed_execution_revision
from exp.optimize.evaluation.task_factory import (
    TaskFactorySpec,
    factory_attempt_bound,
    factory_population,
    factory_requires_execution,
    preflight_task_factory,
    run_task_factory,
    saved_task_factory,
)
from exp.simulation.task_distribution import select_distribution


def run_pipeline(
    project: str, path: Path, *, root: Path, yes: bool, non_interactive: bool, dry_run: bool
) -> None:
    """Validate a frozen mixture, qualify its families and verify delivered package bytes.

    Args:
        project: Existing source project containing seed worlds and independent witnesses.
        path: Typed TaskFactorySpec JSON manifest, with no credentials or generated code.
        root: Existing project store root.
        yes: Explicit authorization of the displayed unknown-price finite allowance.
        non_interactive: Fail rather than prompt when paid work lacks authorization.
        dry_run: Check available source contracts without dispatch or publication.
    """
    console = Console(theme=EXP_THEME)
    with usage_error(OSError, ValueError, RuntimeError):
        spec = TaskFactorySpec.model_validate_json(path.read_bytes())
        store = ProjectStore(root, project)
        revision = installed_execution_revision()
        completed = saved_task_factory(store, spec, code_revision=revision)
        if completed is None:
            preflight_task_factory(store, spec, code_revision=revision)
        planned = select_distribution(spec.distribution, factory_population(store, spec))
        console.print(
            f"Task factory: {len(spec.families)} seeds, {len(planned.components)} distinct "
            f"source/workflow components, {len(spec.actors)} actors; at most "
            f"{factory_attempt_bound(store, spec)} author/actor attempts until "
            f"{spec.deadline.isoformat()}. Runtime inference has separate allowances."
        )
        console.print_json(data={"planned_coverage": planned.model_dump(mode="json")})
        if dry_run:
            console.print(
                "Sources and bounds checked. No new calls, runtime execution or packages."
            )
            return
        if factory_requires_execution(store, spec, code_revision=revision):
            if not require_spend_consent(
                console,
                root=root,
                yes=yes,
                estimated_cost_usd=None,
                cost_is_upper_bound=False,
                command=f"exp eval {project} --pipeline {path}",
                non_interactive=non_interactive,
            ):
                return
        with progress_display(console) as progress:
            result = completed or run_task_factory(
                store, spec, code_revision=revision, progress=progress
            )
        console.print_json(data=result.model_dump(mode="json"))
