"""Manual local judge setup and calibration commands."""

from __future__ import annotations

import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import typer
from rich.console import Console
from rich.prompt import Confirm

from wmo.cli.consent import can_prompt, require_spend_consent
from wmo.cli.judge_review import build_manual_judge_reviewer
from wmo.cli.judge_rubric import maybe_edit_setup_plan
from wmo.cli.judge_transcript import model_display_name
from wmo.cli.options import ROOT_OPTION, usage_error
from wmo.common.config import resolve_command_budget_usd
from wmo.common.judging import Rubric, RubricDimension, render_rubric_table, score_bounds
from wmo.common.judging.provenance import read_artifact_json
from wmo.common.models import load_model_catalog
from wmo.common.project import ProjectStore
from wmo.common.release_revision import installed_release_revision
from wmo.optimize.router.judging.artifacts import read_audit, require_review_state
from wmo.optimize.router.judging.contracts import (
    JudgePromptTemplate,
    ManualJudgeAxisDecision,
    ManualJudgeCalibrationResult,
    ManualJudgeSetupArtifact,
)
from wmo.optimize.router.judging.labels import (
    calibration_sample_digest,
    read_label_draft,
)
from wmo.optimize.router.judging.review import (
    ManualJudgeReviewer,
    ManualJudgeTraceProposal,
    completed_trace_review_count,
)
from wmo.optimize.router.judging.service import (
    DEFAULT_JUDGE_TEMPLATE,
    ManualJudgeError,
    ManualJudgeSetupPlan,
    calibrate_manual_judge,
    calibration_sample,
    commit_manual_judge_setup,
    estimate_manual_judge_budget,
    manual_judge_calibration_is_complete,
    prepare_manual_judge_calibration,
    prepare_manual_judge_setup,
)
from wmo.runtime.models.registry import RuntimeModelCatalog

judge_app = typer.Typer(help="Set up and manually calibrate a project judge.", no_args_is_help=True)
_console = Console()
_RUBRIC_FILE_OPTION = typer.Option(
    None, "--rubric-file", help="JSON array of rubric axes with IDs, ranges, and score meanings."
)
_TEMPLATE_FILE_OPTION = typer.Option(
    None,
    "--template-file",
    help="JSON judge prompt, variable mapping, and response schema contract.",
)
_LABEL_OPTION = typer.Option(
    None,
    "--label",
    help=(
        "Repeat TRACE_ID:DIMENSION_ID=SCORE, or "
        "TRACE_ID:REFERENCE_TRACE_ID:DIMENSION_ID=winner_a|winner_b|tie. "
        "Use a JSON array target when trace IDs contain ambiguous delimiters. "
        "A value matching the judge proposal accepts it; another value is a correction."
    ),
)
_JUDGMENT_OPTION = typer.Option(
    None,
    "--judgment",
    help=(
        "Repeat TRACE_ID:DIMENSION_ID=TEXT, or include REFERENCE_TRACE_ID for pairwise. "
        "Use a JSON array target when trace IDs contain ambiguous delimiters. "
        "Required with a noninteractive corrected score."
    ),
)


@judge_app.command(
    "setup",
    help="Preview a human-readable rubric and save a confirmed judge contract.",
)
def judge_setup(
    project: str = typer.Argument(..., metavar="PROJECT", help="Configured local project ID."),
    root: Path = ROOT_OPTION,
    judge_alias: str | None = typer.Option(
        None, "--judge-alias", help="Configured completion alias; defaults to roles.judge."
    ),
    rubric_file: Path | None = _RUBRIC_FILE_OPTION,
    template_file: Path | None = _TEMPLATE_FILE_OPTION,
    preview_count: int = typer.Option(3, "--preview-count", min=1),
    approve: bool = typer.Option(
        False, "--approve", help="Confirm the displayed setup contract without prompting."
    ),
    non_interactive: bool = typer.Option(False, "--non-interactive"),
) -> None:
    """Preview and confirm a judge contract without credentials or model calls.

    Args:
        project: Local project ID below ``<root>/projects``.
        root: Local project root containing ``models.toml``.
        judge_alias: Optional configured judge alias override.
        rubric_file: Optional complete human-authored rubric JSON file.
        template_file: Optional versioned prompt contract JSON file.
        preview_count: Maximum number of distinct fit-lineage traces bound into the plan.
        approve: Explicit setup confirmation.
        non_interactive: Refuse prompts and require explicit confirmation flags.

    Raises:
        typer.BadParameter: Local files, build evidence, or confirmation are invalid.
    """
    with usage_error(OSError, ValueError, ManualJudgeError):
        revision = installed_release_revision()
        store = ProjectStore(root, project)
        dimensions = _load_rubric_dimensions(rubric_file)
        template = _load_prompt_template(template_file)
        plan = prepare_manual_judge_setup(
            store,
            load_model_catalog(store.model_catalog_path),
            judge_alias=judge_alias,
            dimensions=dimensions,
            prompt_template=template,
            preview_count=preview_count,
            created_at=datetime.now(UTC),
            code_revision=revision,
        )
        _render_setup(plan)
        if not approve and not non_interactive:
            plan = maybe_edit_setup_plan(plan, console=_console)
        confirmed = approve or _confirm(
            "Save this judge setup and finalize its rubric?",
            default=True,
            non_interactive=non_interactive,
            required_flag="--approve",
        )
        if not confirmed:
            _console.print("Judge setup was not saved.")
            return
        setup = commit_manual_judge_setup(store, plan, confirmed=True)
    _console.print(f"Saved judge setup {setup.setup_id} for {project}.")


@judge_app.command(
    "calibrate",
    help="Judge and review real traces incrementally, then separately approve calibration.",
)
def judge_calibrate(
    project: str = typer.Argument(..., metavar="PROJECT", help="Configured local project ID."),
    root: Path = ROOT_OPTION,
    sample_size: int = typer.Option(
        5,
        "--sample-size",
        min=1,
        help="Distinct trace lineages to review; defaults to the normal sufficient count of five.",
    ),
    label: list[str] | None = _LABEL_OPTION,
    judgment: list[str] | None = _JUDGMENT_OPTION,
    input_price: float | None = typer.Option(
        None,
        "--input-usd-per-million",
        min=0,
        rich_help_panel="Advanced",
        help="Advanced override for judge input price when catalog pricing is unavailable.",
    ),
    output_price: float | None = typer.Option(
        None,
        "--output-usd-per-million",
        min=0,
        rich_help_panel="Advanced",
        help="Advanced override for judge output price when catalog pricing is unavailable.",
    ),
    maximum_input_tokens: int = typer.Option(32_768, "--maximum-input-tokens", min=1),
    maximum_cost_usd: float | None = typer.Option(
        None,
        "--maximum-cost-usd",
        min=0.000001,
        help="Calibration spend ceiling. Defaults to the shared command-budget setting, then $10.",
    ),
    yes: bool = typer.Option(
        False,
        "--yes",
        help="Confirm an in-budget estimate when the shared policy requires it.",
    ),
    approve: bool = typer.Option(
        False, "--approve", help="Approve the report after it is displayed."
    ),
    accept_insufficient_labels: bool = typer.Option(
        False,
        "--accept-insufficient-labels",
        help="Accept valid grouped evidence from fewer than five distinct trace lineages.",
    ),
    non_interactive: bool = typer.Option(False, "--non-interactive"),
    transcript_character_limit: int = typer.Option(
        1_200,
        "--transcript-character-limit",
        min=200,
        help="Maximum characters per transcript field before an exact truncation marker.",
    ),
    page: bool = typer.Option(
        False,
        "--page",
        help=(
            "Page the current full transcript; proposals and decisions remain one trace at a time."
        ),
    ),
) -> None:
    """Run judge-first review, persist each trace, and separately approve evidence.

    Args:
        project: Local project ID below ``<root>/projects``.
        root: Local project root containing ``models.toml``.
        sample_size: Number of distinct fit lineages, with five as the sufficient default.
        label: Optional explicit accepted or corrected score expressions.
        judgment: Optional explicit human-authored corrected judgment expressions.
        input_price: Optional advanced input-price override.
        output_price: Optional advanced output-price override.
        maximum_input_tokens: Conservative input bound for every call attempt.
        maximum_cost_usd: Optional spend ceiling; otherwise the shared command-budget setting.
        yes: Explicit confirmation for an in-budget estimate above the automatic threshold.
        approve: Separate approval of the displayed completed report.
        accept_insufficient_labels: Explicit risk acceptance below five completed reviews.
        non_interactive: Refuse prompts and require complete explicit decisions.
        transcript_character_limit: Maximum displayed characters per transcript field.
        page: Page the untruncated current trace through an interactive terminal.

    Raises:
        typer.BadParameter: Evidence, review inputs, budget, consent, or approval is invalid.
    """
    with usage_error(OSError, ValueError, ManualJudgeError):
        revision = installed_release_revision()
        store = ProjectStore(root, project)
        now = datetime.now(UTC)
        plan = prepare_manual_judge_calibration(store, sample_size=sample_size)
        rubric = _load_setup_rubric(store, plan.setup)
        _console.print(render_rubric_table(rubric, width=_console.width))
        sample_sha256 = calibration_sample_digest(plan.setup, calibration_sample(plan))
        drafted = read_label_draft(store, plan.setup, sample_sha256)
        completed = manual_judge_calibration_is_complete(store)
        reviewed = completed_trace_review_count(store, plan.setup, sample_sha256)
        if completed:
            state = require_review_state(store)
            assert state.audit is not None
            budget = read_audit(store, state.audit).budget
        else:
            catalog = load_model_catalog(store.model_catalog_path)
            shared_ceiling = resolve_command_budget_usd(root, None)
            calibration_ceiling = (
                shared_ceiling
                if maximum_cost_usd is None
                else min(shared_ceiling, maximum_cost_usd)
            )
            budget = estimate_manual_judge_budget(
                plan,
                catalog=catalog,
                input_usd_per_million_tokens=input_price,
                output_usd_per_million_tokens=output_price,
                maximum_input_tokens_per_call=maximum_input_tokens,
                maximum_cost_usd=sys.float_info.max,
                completed_review_count=reviewed,
            )
            if maximum_cost_usd is not None and budget.estimated_cost_usd > maximum_cost_usd:
                raise ValueError(
                    "judge calibration estimate exceeds --maximum-cost-usd; raise the ceiling "
                    "or reduce the labeled sample"
                )
        if page and not completed and budget.call_count and not can_prompt(_console):
            raise ValueError("--page requires an interactive terminal; omit it for wrapped output")
    if completed:
        _console.print(
            "Judge calibration is already complete; replaying immutable evidence with zero "
            "provider calls and no review prompts."
        )
    elif budget.call_count:
        assumptions = (
            f"review progress: {reviewed}/{len(plan.traces)} distinct trace lineages complete",
            f"judge {plan.setup.judge_alias}: {model_display_name(plan.setup.judge_model)}",
            f"pricing source: {budget.pricing_source.value}",
            (
                f"at most {budget.call_count} remaining judge calls with up to "
                f"{budget.maximum_attempts_per_call} attempts each"
            ),
            (
                f"{budget.maximum_input_tokens_per_call} input and "
                f"{budget.maximum_output_tokens_per_call} output tokens per attempt"
            ),
            (
                f"${budget.input_usd_per_million_tokens:.6f} input and "
                f"${budget.output_usd_per_million_tokens:.6f} output per million tokens"
            ),
        )
        if not require_spend_consent(
            _console,
            root=root,
            yes=yes,
            estimated_cost_usd=budget.estimated_cost_usd,
            command=f"wmo config judge calibrate {project}",
            assumptions=assumptions,
            non_interactive=non_interactive,
            previously_confirmed=False,
        ):
            _console.print("Judge calibration was not started. No provider calls or reviews ran.")
            return
        with usage_error(OSError, ValueError, ManualJudgeError):
            budget = estimate_manual_judge_budget(
                plan,
                catalog=catalog,
                input_usd_per_million_tokens=input_price,
                output_usd_per_million_tokens=output_price,
                maximum_input_tokens_per_call=maximum_input_tokens,
                maximum_cost_usd=max(calibration_ceiling, 0.000001),
                completed_review_count=reviewed,
            )
        if drafted:
            _console.print(
                f"Found {len(drafted)} saved human score inputs. They will be applied only "
                "after the configured judge proposals are shown."
            )
    else:
        _console.print(
            f"Review progress: {reviewed}/{len(plan.traces)} distinct trace lineages complete. "
            "Finalizing from immutable reviews with zero provider calls."
        )
    with usage_error(OSError, ValueError, ManualJudgeError):
        reviewer: ManualJudgeReviewer = (
            build_manual_judge_reviewer(
                plan.setup,
                rubric,
                plan.previews,
                drafted_labels=drafted,
                supplied_labels=tuple(label or ()),
                supplied_judgments=tuple(judgment or ()),
                non_interactive=non_interactive,
                character_limit=transcript_character_limit,
                page=page,
                console=_console,
            )
            if not completed and budget.call_count
            else _unexpected_review
        )
        runtime = RuntimeModelCatalog(load_model_catalog(store.model_catalog_path))
        result = calibrate_manual_judge(
            store,
            runtime,
            plan,
            (),
            budget,
            spend_consented=True,
            approve=False,
            accept_insufficient_labels=accept_insufficient_labels,
            created_at=now,
            code_revision=revision,
            reviewer=reviewer,
        )
        _render_report(result)
        should_approve = (
            result.approved_calibration is not None
            or approve
            or _confirm(
                "Approve this immutable judge calibration?",
                non_interactive=non_interactive,
                required_flag="--approve",
            )
        )
        if should_approve and result.approved_calibration is None:
            result = calibrate_manual_judge(
                store,
                runtime,
                plan,
                (),
                budget,
                spend_consented=True,
                approve=True,
                accept_insufficient_labels=accept_insufficient_labels,
                created_at=now,
                code_revision=revision,
                reviewer=reviewer,
            )
    if result.approved_calibration is None:
        _console.print("Calibration evidence saved but not approved.")
    else:
        _console.print(f"Approved judge calibration {result.approved_calibration.artifact_id}.")


def _unexpected_review(
    _proposal: ManualJudgeTraceProposal,
) -> tuple[ManualJudgeAxisDecision, ...]:
    """Fail closed if completed review state requests another decision.

    Args:
        _proposal: Unexpected proposal supplied by the review workflow.

    Raises:
        ManualJudgeError: Always, because completed state cannot request review.
    """
    raise ManualJudgeError("completed calibration unexpectedly requested another human review")


def _load_rubric_dimensions(path: Path | None) -> tuple[RubricDimension, ...] | None:
    """Load an optional complete rubric dimension array from JSON.

    Args:
        path: Optional local JSON file.

    Returns:
        Validated dimensions, or ``None`` to use the editable default.

    Raises:
        ValueError: JSON is malformed or does not contain a nonempty dimension array.
    """
    if path is None:
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list) or not payload:
        raise ValueError("rubric file must contain a nonempty JSON array")
    return tuple(RubricDimension.model_validate(item) for item in payload)


def _load_prompt_template(path: Path | None) -> JudgePromptTemplate:
    """Load an optional exact versioned prompt contract from JSON.

    Args:
        path: Optional local prompt contract file.

    Returns:
        Validated file content or the built-in scalar contract.
    """
    if path is None:
        return DEFAULT_JUDGE_TEMPLATE
    return JudgePromptTemplate.model_validate_json(path.read_text(encoding="utf-8"))


def _render_setup(plan: ManualJudgeSetupPlan) -> None:
    """Display the judge, human-readable rubric table, and representative tasks.

    Args:
        plan: Read-only judge setup awaiting explicit confirmation.
    """
    _console.print("\n[bold]Judge setup[/bold]")
    _console.print(f"Judge name: {plan.judge_alias}", markup=False)
    _console.print(f"Exact model: {model_display_name(plan.judge_model)}", markup=False)
    if plan.prompt_template.response_shape == "pairwise":
        mode = "A/B pairwise comparison"
    else:
        lowest, highest = score_bounds(plan.dimensions)
        mode = f"Integer scoring from {lowest} to {highest}"
    _console.print(f"Calibration mode: {mode}", markup=False)
    _console.print()
    _console.print(render_rubric_table(plan.dimensions, width=_console.width), markup=False)
    _console.print("\n[bold]Representative tasks[/bold]")
    for index, preview in enumerate(plan.previews, start=1):
        _console.print(f"{index}. {preview.task}", markup=False)
        _console.print(f"   Recorded outcome: {preview.outcome}", markup=False)


def _load_setup_rubric(store: ProjectStore, setup: ManualJudgeSetupArtifact) -> Rubric:
    """Load and verify the finalized rubric named by setup.

    Args:
        store: Project-local immutable artifact store.
        setup: Finalized setup naming the exact rubric manifest.

    Returns:
        Verified finalized rubric.

    Raises:
        ValueError: The rubric manifest differs from setup.
    """
    rubric, rubric_input = read_artifact_json(
        store,
        artifact_id=setup.rubric.artifact_id,
        expected_artifact_type="rubric",
        relative_path="rubric.json",
        model_type=Rubric,
    )
    if rubric_input != setup.rubric:
        raise ValueError("manual judge rubric manifest differs from setup")
    return rubric


def _render_report(result: ManualJudgeCalibrationResult) -> None:
    """Display agreement, disagreement, and schema-appropriate positional bias.

    Args:
        result: Completed immutable calibration result.
    """
    report = result.report
    _console.print(
        f"Calibration report {report.report_id}: status={report.status}, "
        f"rollouts={report.eligible_rollout_count}, lineages={report.eligible_lineage_count}"
    )
    for metric in report.dimension_metrics:
        _console.print(
            f"  {metric.dimension_id}: MAE={_metric(metric.mae)}, "
            f"rank agreement={_metric(metric.rank_agreement)}, "
            f"mean optimistic error={_metric(metric.mean_optimistic_error)}"
        )
    for disagreement in report.worst_disagreements:
        prediction = disagreement.prediction
        _console.print(
            f"  {disagreement.direction} disagreement {prediction.dimension_id} "
            f"rollout={prediction.rollout_id} human={prediction.human_score} "
            f"judge={prediction.calibrated_score:.3f}"
        )
    comparisons = result.audit.positional_bias_comparisons
    flips = result.audit.positional_bias_flips
    if comparisons is None or flips is None:
        _console.print("Positional-bias probe: n/a for non-pairwise feedback.")
    else:
        _console.print(
            f"Positional-bias probe: {flips}/{comparisons} order-flip disagreements "
            f"({flips / comparisons:.1%})."
        )


def _metric(value: float | None) -> str:
    """Format one optional calibration metric for concise CLI output.

    Args:
        value: Optional finite calibration metric.

    Returns:
        Three-decimal text or ``n/a`` when evidence is unavailable.
    """
    return "n/a" if value is None else f"{value:.3f}"


def _confirm(
    question: str,
    *,
    default: bool = False,
    non_interactive: bool,
    required_flag: str,
) -> bool:
    """Ask one non-spend confirmation or require its explicit flag.

    Args:
        question: Human-readable confirmation prompt.
        default: Answer selected by a blank response.
        non_interactive: Whether terminal prompting is forbidden.
        required_flag: Explicit flag required without a prompt.

    Returns:
        The operator's explicit answer.

    Raises:
        ValueError: No interactive terminal is available for an omitted flag.
    """
    if non_interactive or not can_prompt(_console):
        raise ValueError(f"noninteractive judge review requires {required_flag}")
    return Confirm.ask(question, default=default)
