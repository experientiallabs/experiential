"""Configless ``wmo optimize router PROJECT`` composition."""

from __future__ import annotations

import time
from datetime import UTC, datetime
from pathlib import Path

import typer
from rich.console import Console

from wmo.cli.consent import can_prompt, require_spend_consent
from wmo.cli.options import ROOT_OPTION, usage_error
from wmo.cli.progress import progress_display
from wmo.cli.router_candidate_setup import collect_router_candidate_setup
from wmo.cli.theme import WMO_THEME
from wmo.common.models import ProviderModelSelection, load_model_catalog
from wmo.common.observability.telemetry import capture_completion_once
from wmo.common.project import ProjectStore
from wmo.common.release_revision import installed_release_revision
from wmo.optimize.router.automatic.preflight import (
    AutomaticRouterOptions,
    AutomaticRouterPreflight,
    preflight_automatic_router,
)
from wmo.optimize.router.automatic.replay import find_completed_automatic_router_replay
from wmo.optimize.router.automatic.service import (
    optimize_project_router,
    persist_router_candidate_setup,
)
from wmo.runtime.models import RuntimeModelCatalog

_console = Console(theme=WMO_THEME)
_CANDIDATE_OPTION = typer.Option(
    None,
    "--candidate",
    help="Repeat a configured completion alias. At least two distinct aliases are required.",
)
_CANDIDATE_MODEL_OPTION = typer.Option(
    None,
    "--candidate-model",
    help=(
        "Advanced: repeat a complete ProviderModelSelection JSON object for a new candidate alias."
    ),
)


def router(
    project: str = typer.Argument(..., metavar="PROJECT", help="Configured local project ID."),
    root: Path = ROOT_OPTION,
    candidate: list[str] | None = _CANDIDATE_OPTION,
    candidate_model: list[str] | None = _CANDIDATE_MODEL_OPTION,
    incumbent: str | None = typer.Option(None, "--incumbent", help="Quality baseline alias."),
    maximum_provider_cost_usd: float = typer.Option(
        25.0,
        "--maximum-provider-cost-usd",
        min=0.000001,
        help="One ceiling for embeddings, simulation, and judging.",
    ),
    maximum_judgments: int = typer.Option(100, "--maximum-judgments", min=1),
    maximum_model_calls: int = typer.Option(8, "--maximum-model-calls", min=1),
    maximum_router_feature_tokens: int = typer.Option(
        8_192, "--maximum-router-feature-tokens", min=1
    ),
    maximum_retrieval_query_tokens: int = typer.Option(
        32_768, "--maximum-retrieval-query-tokens", min=1
    ),
    simulation_maximum_output_tokens: int = typer.Option(
        2_000, "--simulation-maximum-output-tokens", min=256
    ),
    maximum_concurrency: int = typer.Option(1, "--maximum-concurrency", min=1),
    yes: bool = typer.Option(
        False,
        "--yes",
        help="Confirm an in-budget estimate when the shared policy requires it.",
    ),
    non_interactive: bool = typer.Option(False, "--non-interactive"),
) -> None:
    """Optimize a router automatically from the project's completed grounded build.

    Args:
        project: Local project ID below ``<root>/projects``.
        root: Local ``.wmo`` root containing the project and shared model catalog.
        candidate: Repeatable explicit completion candidate aliases.
        candidate_model: Advanced repeatable JSON definitions for new candidate aliases.
        incumbent: Explicit quality incumbent among the selected candidates.
        maximum_provider_cost_usd: Shared ceiling for all optimization provider calls.
        maximum_judgments: Maximum rollout judgments admitted by composition.
        maximum_model_calls: Candidate turns admitted per simulation episode.
        maximum_router_feature_tokens: Input ceiling for each router feature embedding.
        maximum_retrieval_query_tokens: Input ceiling for each grounded retrieval query.
        simulation_maximum_output_tokens: Candidate and world-model output ceiling per turn.
        maximum_concurrency: Maximum simulation workers.
        yes: Explicit confirmation for an in-budget estimate above the automatic threshold.
        non_interactive: Refuse prompts and require complete repeatable inputs.

    Raises:
        typer.BadParameter: Candidate, build, judge, budget, authorization, or artifact input is
            invalid.
    """
    started = time.monotonic()
    now = datetime.now(UTC)
    options = AutomaticRouterOptions(
        maximum_provider_cost_usd=maximum_provider_cost_usd,
        maximum_judgments=maximum_judgments,
        maximum_model_calls=maximum_model_calls,
        maximum_router_feature_tokens=maximum_router_feature_tokens,
        maximum_retrieval_query_tokens=maximum_retrieval_query_tokens,
        simulation_maximum_output_tokens=simulation_maximum_output_tokens,
        maximum_concurrency=maximum_concurrency,
    )
    effective_noninteractive = non_interactive or not can_prompt(_console)
    with usage_error(OSError, ValueError):
        producer_revision = installed_release_revision()
        store = ProjectStore(root, project)
        catalog = load_model_catalog(store.model_catalog_path)
        definitions = tuple(
            ProviderModelSelection.model_validate_json(value)
            for value in tuple(candidate_model or ())
        )
        candidate_plan = collect_router_candidate_setup(
            store.model_catalog_path,
            catalog,
            candidates=tuple(candidate or ()),
            candidate_models=definitions,
            incumbent=incumbent,
            non_interactive=effective_noninteractive,
            console=_console,
            interactive_command=f"wmo optimize router {project} --root {root}",
        )
        preflight = preflight_automatic_router(
            store,
            candidate_plan.selection,
            catalog_override=candidate_plan.prospective_catalog,
            options=options,
        )
        replay = find_completed_automatic_router_replay(
            store,
            preflight,
            options=options,
            code_revision=producer_revision,
        )

    if replay is not None:
        require_spend_consent(
            _console,
            root=root,
            yes=yes,
            estimated_cost_usd=0.0,
            command=f"wmo optimize router {project}",
            assumptions=(
                "verified immutable optimization replay",
                "zero new provider calls",
            ),
            non_interactive=effective_noninteractive,
        )
        with usage_error(OSError, ValueError):
            configured_catalog = persist_router_candidate_setup(store, candidate_plan)
            if configured_catalog != candidate_plan.prospective_catalog:
                raise ValueError("persisted router candidate catalog differs from confirmation")
        _console.print("replay: verified completed optimization")
        _console.print(f"policy: {replay.policy_id}")
        _console.print(f"report: {replay.report_id}")
        return
    _render_preflight(preflight, options)
    if not require_spend_consent(
        _console,
        root=root,
        yes=yes,
        estimated_cost_usd=options.maximum_provider_cost_usd,
        command=f"wmo optimize router {project}",
        assumptions=(
            (
                f"${preflight.router_embedding_reservation.estimated_cost_usd:.4f} router "
                "embedding reservation"
            ),
            f"${preflight.judge_reservation_cost_usd:.4f} judge reservation",
            (
                f"${preflight.remaining_simulation_cost_usd:.4f} candidate, retrieval, and "
                "world-model simulation allocation"
            ),
        ),
        non_interactive=effective_noninteractive,
    ):
        _console.print("Router optimization was not started.")
        return
    with usage_error(OSError, ValueError), progress_display(_console) as progress:
        result = optimize_project_router(
            store,
            candidate_plan,
            RuntimeModelCatalog(catalog),
            options=options,
            provider_spend_consented=True,
            created_at=now,
            code_revision=producer_revision,
            progress=progress,
        )
    capture_completion_once(
        "wmo router completed",
        result.composition.optimization.optimization.report.report_id,
        {
            "success": True,
            "candidate_count": len(preflight.candidates),
            "duration_seconds": max(time.monotonic() - started, 0.0),
        },
        root=root,
    )
    _console.print(f"policy: {result.composition.optimization.optimization.policy.policy_id}")
    _console.print(f"report: {result.composition.optimization.optimization.report.report_id}")


def _render_preflight(
    preflight: AutomaticRouterPreflight,
    options: AutomaticRouterOptions,
) -> None:
    """Render the confirmed scope and one shared provider-spend allocation.

    Args:
        preflight: Verified automatic preflight result.
        options: User-selected bounded controls.
    """
    _console.print("[bold]Router optimization plan[/bold]")
    _console.print(f"provider ceiling: ${options.maximum_provider_cost_usd:.4f}")
    _console.print(f"observed fit traces: {len(preflight.observed_traces)}")
