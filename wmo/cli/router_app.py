"""Configless ``wmo optimize router PROJECT`` composition."""

from __future__ import annotations

import time
from datetime import UTC, datetime
from pathlib import Path

import typer
from rich.console import Console
from rich.prompt import Confirm

from wmo.cli.consent import can_prompt, require_spend_consent
from wmo.cli.options import ROOT_OPTION, usage_error
from wmo.cli.provider_failures import (
    exit_provider_failure,
    router_optimization_retry_command,
)
from wmo.cli.router_candidate_setup import collect_router_candidate_setup
from wmo.common.evaluations import EvaluationCellEvidence, EvaluationPlan
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
from wmo.optimize.router.automatic.service import optimize_project_router
from wmo.optimize.router.composition import (
    FidelityApprovalDecision,
    RouterCompositionBudget,
)
from wmo.runtime.models import RuntimeModelCatalog
from wmo.runtime.models.providers.errors import ProviderError

_console = Console()
_CANDIDATE_OPTION = typer.Option(
    None,
    "--candidate",
    help="Repeat a configured completion alias. At least two distinct aliases are required.",
)
_CANDIDATE_MODEL_OPTION = typer.Option(
    None,
    "--candidate-model",
    help="Repeat a complete ProviderModelSelection JSON object for a new candidate alias.",
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
    preferred_fidelity_overlaps: int = typer.Option(
        10,
        "--preferred-fidelity-overlaps",
        min=1,
        help="Use every available real fit overlap up to this preferred target.",
    ),
    maximum_model_calls: int = typer.Option(8, "--maximum-model-calls", min=1),
    maximum_router_feature_tokens: int = typer.Option(
        8_192, "--maximum-router-feature-tokens", min=1
    ),
    maximum_retrieval_query_tokens: int = typer.Option(
        32_768, "--maximum-retrieval-query-tokens", min=1
    ),
    simulation_maximum_output_tokens: int = typer.Option(
        16_000, "--simulation-maximum-output-tokens", min=8_000
    ),
    maximum_concurrency: int = typer.Option(1, "--maximum-concurrency", min=1),
    yes: bool = typer.Option(
        False,
        "--yes",
        help="Confirm an in-budget estimate when the shared policy requires it.",
    ),
    approve_fidelity: bool = typer.Option(
        False,
        "--approve-fidelity",
        help="Approve passing measured fidelity; required for noninteractive execution.",
    ),
    non_interactive: bool = typer.Option(False, "--non-interactive"),
) -> None:
    """Optimize a router automatically from the project's completed grounded build.

    Args:
        project: Local project ID below ``<root>/projects``.
        root: Local ``.wmo`` root containing the project and shared model catalog.
        candidate: Repeatable explicit completion candidate aliases.
        candidate_model: Repeatable complete JSON definitions for new candidate aliases.
        incumbent: Explicit quality incumbent among the selected candidates.
        maximum_provider_cost_usd: Shared ceiling for all optimization provider calls.
        maximum_judgments: Maximum rollout judgments admitted by composition.
        preferred_fidelity_overlaps: Bounded preferred real-overlap denominator.
        maximum_model_calls: Candidate turns admitted per simulation episode.
        maximum_router_feature_tokens: Input ceiling for each router feature embedding.
        maximum_retrieval_query_tokens: Input ceiling for each grounded retrieval query.
        simulation_maximum_output_tokens: Candidate and world-model output ceiling per turn.
        maximum_concurrency: Maximum simulation workers.
        yes: Explicit confirmation for an in-budget estimate above the automatic threshold.
        approve_fidelity: Explicit approval for passing measured fidelity evidence.
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
        preferred_fidelity_overlaps=preferred_fidelity_overlaps,
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
        if replay is None and effective_noninteractive and not approve_fidelity:
            raise ValueError("noninteractive router optimization requires --approve-fidelity")

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
    approval = _CliFidelityApproval(
        approve=approve_fidelity,
        non_interactive=effective_noninteractive,
        preferred_overlaps=options.preferred_fidelity_overlaps,
        approved_at=now,
    )
    try:
        with usage_error(OSError, ValueError):
            result = optimize_project_router(
                store,
                candidate_plan,
                RuntimeModelCatalog(catalog),
                options=options,
                provider_spend_consented=True,
                fidelity_approval=approval,
                created_at=now,
                code_revision=producer_revision,
            )
    except ProviderError as exc:
        exit_provider_failure(
            _console,
            exc,
            saved_progress=(
                "completed router artifacts were kept and will be replayed",
                "the failed provider attempt was not recorded as completed evidence",
            ),
            retry_command=router_optimization_retry_command(
                project,
                root=str(root),
                candidates=candidate_plan.selection.candidates,
                candidate_models=tuple(
                    model.model_dump_json() for model in candidate_plan.candidate_models
                ),
                incumbent=candidate_plan.selection.incumbent,
                maximum_provider_cost_usd=options.maximum_provider_cost_usd,
                maximum_judgments=options.maximum_judgments,
                preferred_fidelity_overlaps=options.preferred_fidelity_overlaps,
                maximum_model_calls=options.maximum_model_calls,
                maximum_router_feature_tokens=options.maximum_router_feature_tokens,
                maximum_retrieval_query_tokens=options.maximum_retrieval_query_tokens,
                simulation_maximum_output_tokens=options.simulation_maximum_output_tokens,
                maximum_concurrency=options.maximum_concurrency,
                approve_fidelity=approve_fidelity or effective_noninteractive,
                non_interactive=effective_noninteractive,
            ),
        )
    capture_completion_once(
        "wmo router completed",
        result.composition.optimization.optimization.report.report_id,
        {
            "success": True,
            "candidate_count": len(preflight.candidates),
            "fidelity_overlap_count": preflight.fidelity_overlap_count,
            "duration_seconds": max(time.monotonic() - started, 0.0),
        },
        root=root,
    )
    _console.print(f"policy: {result.composition.optimization.optimization.policy.policy_id}")
    _console.print(f"report: {result.composition.optimization.optimization.report.report_id}")


class _CliFidelityApproval:
    """Interactive or flag-backed approval of the exact measured fidelity denominator."""

    def __init__(
        self,
        *,
        approve: bool,
        non_interactive: bool,
        preferred_overlaps: int,
        approved_at: datetime,
    ) -> None:
        """Configure the separate post-simulation approval boundary.

        Args:
            approve: Explicit non-prompt approval flag.
            non_interactive: Whether prompting is forbidden.
            preferred_overlaps: Preferred denominator used for the low-evidence warning.
            approved_at: Approval time retained in immutable evidence.
        """
        self._approve = approve
        self._non_interactive = non_interactive
        self._preferred = preferred_overlaps
        self._approved_at = approved_at

    def __call__(
        self,
        project: ProjectStore,
        plan: EvaluationPlan,
        evidence: tuple[EvaluationCellEvidence, ...],
        budget: RouterCompositionBudget,
    ) -> FidelityApprovalDecision:
        """Display the exact denominator and obtain separate approval.

        Args:
            project: Project owning immutable evidence.
            plan: Frozen evaluation plan.
            evidence: Completed fidelity-cell evidence selected by composition.
            budget: Finite simulation and judgment ceilings.

        Returns:
            Immutable actor evidence for passing fidelity.

        Raises:
            ValueError: Approval is absent or evidence is empty.
        """
        del project, budget
        if not evidence:
            raise ValueError("fidelity approval requires completed overlap evidence")
        denominator = len(evidence)
        low = denominator < self._preferred
        _console.print(
            f"Fidelity evidence: {denominator}/{self._preferred} preferred real overlaps."
        )
        if low:
            _console.print(
                "[yellow]Low-evidence warning: the exact denominator is below the preferred "
                "target.[/yellow]"
            )
        approved = self._approve
        if not approved:
            if self._non_interactive:
                raise ValueError("fidelity approval requires --approve-fidelity")
            approved = Confirm.ask(
                "Approve this plan-bound fidelity evidence?",
                default=False,
                console=_console,
            )
        if not approved:
            raise ValueError("router optimization stopped without fidelity approval")
        return FidelityApprovalDecision(
            actor_id="cli-operator",
            evidence=(
                f"Approved {denominator} plan-bound fidelity overlaps"
                + (" with the low-evidence warning acknowledged." if low else ".")
            ),
            approved_at=self._approved_at,
        )


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
    _console.print(f"fidelity overlaps: {preflight.fidelity_overlap_count}")
    if preflight.low_fidelity_evidence:
        _console.print("[yellow]Fidelity evidence is below the preferred target.[/yellow]")
