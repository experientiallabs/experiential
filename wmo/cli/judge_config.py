"""Manual local judge setup and calibration commands."""

from __future__ import annotations

import json
import sys
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path

import typer
from rich.console import Console
from rich.markup import escape
from rich.prompt import Confirm, IntPrompt, Prompt

from wmo.cli.consent import can_prompt, require_spend_consent
from wmo.cli.judge_rubric import maybe_edit_setup_plan
from wmo.cli.judge_transcript import model_display_name, render_trace
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
    JudgeTracePreview,
    ManualJudgeCalibrationResult,
    ManualJudgeLabel,
    ManualJudgeSetupArtifact,
)
from wmo.optimize.router.judging.labels import (
    calibration_sample_digest,
    read_label_draft,
    save_label_draft,
)
from wmo.optimize.router.judging.service import (
    DEFAULT_JUDGE_TEMPLATE,
    ManualJudgeCalibrationPlan,
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
        "TRACE_ID:REFERENCE_TRACE_ID:DIMENSION_ID=winner_a|winner_b|tie for pairwise labels."
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
    help="Label real traces, authorize bounded judge calls, and separately approve calibration.",
)
def judge_calibrate(
    project: str = typer.Argument(..., metavar="PROJECT", help="Configured local project ID."),
    root: Path = ROOT_OPTION,
    sample_size: int = typer.Option(10, "--sample-size", min=1),
    label: list[str] | None = _LABEL_OPTION,
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
        help="Accept valid grouped evidence from fewer than ten rollouts.",
    ),
    non_interactive: bool = typer.Option(False, "--non-interactive"),
    transcript_character_limit: int = typer.Option(
        1_200,
        "--transcript-character-limit",
        min=200,
        help="Maximum characters shown for each transcript field before a truthful marker.",
    ),
    page: bool = typer.Option(
        False,
        "--page",
        help="Page full transcripts in an interactive terminal instead of truncating them.",
    ),
) -> None:
    """Collect frozen labels, authorize bounded judge calls, and separately approve evidence.

    Args:
        project: Local project ID below ``<root>/projects``.
        root: Local project root containing ``models.toml``.
        sample_size: Maximum number of distinct fit lineages to calibrate.
        label: Repeatable explicit human score inputs.
        input_price: Optional advanced input-price override.
        output_price: Optional advanced output-price override.
        maximum_input_tokens: Conservative input bound for every call attempt.
        maximum_cost_usd: Optional spend ceiling; otherwise the shared command-budget setting.
        yes: Explicit confirmation for an in-budget estimate above the automatic threshold.
        approve: Separate approval of the displayed completed report.
        accept_insufficient_labels: Explicit risk acceptance below ten labeled rollouts.
        non_interactive: Refuse prompts and list missing explicit inputs.
        transcript_character_limit: Maximum displayed characters per transcript field.
        page: Page untruncated transcripts through the interactive terminal.

    Raises:
        typer.BadParameter: Evidence, labels, budget, authorization, or approval is invalid.
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
        if completed:
            state = require_review_state(store)
            assert state.audit is not None
            budget = read_audit(store, state.audit).budget
            _console.print(
                "Judge calibration is already complete; replaying its immutable evidence "
                "without collecting labels."
            )
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
            )
            if (
                maximum_cost_usd is not None
                and budget.estimated_cost_usd > maximum_cost_usd
            ):
                raise ValueError(
                    "judge calibration estimate exceeds --maximum-cost-usd; raise the ceiling "
                    "or reduce the labeled sample"
                )
        if page and not completed and not can_prompt(_console):
            raise ValueError("--page requires an interactive terminal; omit it for wrapped output")

        def persist(collected: tuple[ManualJudgeLabel, ...]) -> None:
            """Save human labels to durable review state before any judge provider work.

            Args:
                collected: Every human label known for the frozen trace sample so far.
            """
            save_label_draft(store, plan.setup, sample_sha256, collected, now)

    estimate = 0.0 if completed else budget.estimated_cost_usd
    assumptions = (
        (
            "verified immutable calibration replay",
            "zero new judge calls",
        )
        if completed
        else (
            f"judge {plan.setup.judge_alias}: {model_display_name(plan.setup.judge_model)}",
            (
                f"{budget.call_count} judge calls with up to "
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
    )
    if not require_spend_consent(
        _console,
        root=root,
        yes=yes,
        estimated_cost_usd=estimate,
        command=f"wmo config judge calibrate {project}",
        assumptions=assumptions,
        non_interactive=non_interactive,
    ):
        _console.print("Judge calibration was not started. No labels or provider calls ran.")
        return
    if not completed:
        with usage_error(OSError, ValueError, ManualJudgeError):
            budget = estimate_manual_judge_budget(
                plan,
                catalog=catalog,
                input_usd_per_million_tokens=input_price,
                output_usd_per_million_tokens=output_price,
                maximum_input_tokens_per_call=maximum_input_tokens,
                maximum_cost_usd=max(calibration_ceiling, 0.000001),
            )
        if drafted:
            _console.print(f"Resuming {len(drafted)} saved human labels for this trace sample.")
        _render_calibration_review(
            plan,
            rubric,
            character_limit=None if page else transcript_character_limit,
            page=page,
        )
        with usage_error(OSError, ValueError, ManualJudgeError):
            labels = _collect_labels(
                plan.setup,
                rubric,
                tuple(label or ()),
                plan.previews,
                drafted,
                persist,
                non_interactive=non_interactive,
            )
    else:
        labels = drafted
    with usage_error(OSError, ValueError, ManualJudgeError):
        runtime = RuntimeModelCatalog(load_model_catalog(store.model_catalog_path))
        result = calibrate_manual_judge(
            store,
            runtime,
            plan,
            labels,
            budget,
            spend_consented=True,
            approve=False,
            accept_insufficient_labels=accept_insufficient_labels,
            created_at=now,
            code_revision=revision,
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
                labels,
                budget,
                spend_consented=True,
                approve=True,
                accept_insufficient_labels=accept_insufficient_labels,
                created_at=now,
                code_revision=revision,
            )
    if result.approved_calibration is None:
        _console.print("Calibration evidence saved but not approved.")
    else:
        _console.print(f"Approved judge calibration {result.approved_calibration.artifact_id}.")


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
        path: Optional local JSON file.

    Returns:
        Validated prompt contract, or the current built-in scalar contract.
    """
    if path is None:
        return DEFAULT_JUDGE_TEMPLATE
    return JudgePromptTemplate.model_validate_json(path.read_text(encoding="utf-8"))


def _render_setup(plan: ManualJudgeSetupPlan) -> None:
    """Display the judge, human-readable rubric table, and representative tasks.

    Args:
        plan: Read-only setup plan awaiting confirmation.
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


def _render_calibration_review(
    plan: ManualJudgeCalibrationPlan,
    rubric: Rubric,
    *,
    character_limit: int | None,
    page: bool,
) -> None:
    """Render readable scalar or A/B transcripts after spend consent.

    Args:
        plan: Frozen traces and optional same-task references in display order.
        rubric: Finalized plain-language scoring rubric.
        character_limit: Per-field limit, or ``None`` for full transcript text.
        page: Whether to send the full review through Rich's terminal pager.
    """

    def render() -> None:
        """Write the complete review into the active console or pager buffer."""
        pairwise = plan.setup.prompt_template.response_shape == "pairwise"
        if pairwise:
            heading = "PAIRWISE A/B CALIBRATION"
        else:
            lowest, highest = score_bounds(rubric.dimensions)
            heading = f"INTEGER {lowest}-{highest} CALIBRATION"
        _console.print(f"\n[bold]{heading}[/bold]")
        _render_rubric(rubric.dimensions)
        for index, (trace, reference) in enumerate(
            zip(plan.traces, plan.reference_traces, strict=True), start=1
        ):
            if pairwise:
                _console.print(f"\n[bold]Pair {index}, candidate A[/bold]")
                render_trace(_console, trace, character_limit=character_limit)
                if reference is None:
                    raise ValueError("pairwise calibration preview is missing candidate B")
                _console.print(f"\n[bold]Pair {index}, candidate B[/bold]")
                render_trace(_console, reference, character_limit=character_limit)
            else:
                _console.print(f"\n[bold]Trace {index}[/bold]")
                render_trace(_console, trace, character_limit=character_limit)

    if page:
        with _console.pager(styles=True):
            render()
    else:
        render()


def _render_rubric(dimensions: tuple[RubricDimension, ...]) -> None:
    """Render complete plain-language rubric axes and score meanings.

    Args:
        dimensions: Finalized rubric dimensions in scoring order.
    """
    _console.print("\n[bold]Rubric[/bold]")
    for dimension in dimensions:
        _console.print(dimension.name, style="bold", markup=False)
        _console.print(dimension.description, markup=False)
        for anchor in dimension.anchors:
            _console.print(f"  {anchor.score}: {anchor.description}", markup=False)


def _collect_labels(
    setup: ManualJudgeSetupArtifact,
    rubric: Rubric,
    supplied: tuple[str, ...],
    previews: tuple[JudgeTracePreview, ...],
    drafted: tuple[ManualJudgeLabel, ...],
    persist: Callable[[tuple[ManualJudgeLabel, ...]], None],
    *,
    non_interactive: bool,
) -> tuple[ManualJudgeLabel, ...]:
    """Resume saved labels, parse explicit ones, and ask only for missing scores.

    Every label is handed to ``persist`` as soon as it exists, so an interrupted or failed
    calibration never discards completed human ratings.

    Args:
        setup: Finalized setup used for stable prompt context.
        rubric: Verified finalized scoring rubric.
        supplied: Repeatable CLI label expressions.
        previews: Frozen ordered calibration trace previews.
        drafted: Labels already persisted for this exact frozen trace sample.
        persist: Durable writer for the labels collected so far.
        non_interactive: Whether all missing inputs must be reported without prompting.

    Returns:
        Complete ordered human label set.

    Raises:
        ValueError: A label is malformed, duplicated, missing, or outside the axis range.
    """
    pairwise = setup.prompt_template.response_shape == "pairwise"
    parsed: dict[tuple[str, str | None, str], ManualJudgeLabel] = {
        (item.trace_id, item.reference_trace_id, item.dimension_id): item for item in drafted
    }
    explicit: set[tuple[str, str | None, str]] = set()
    for item in supplied:
        key = _label_key(item, pairwise=pairwise)
        if key in explicit:
            raise ValueError("duplicate label for " + ":".join(part or "-" for part in key))
        explicit.add(key)
        axis = None if pairwise else rubric.axis(key[2])
        parsed[key] = _label(
            key,
            _label_value(item, pairwise=pairwise, axis=axis),
            pairwise=pairwise,
        )
    expected = tuple(
        (preview.trace_id, preview.reference_trace_id, dimension.dimension_id)
        for preview in previews
        for dimension in rubric.dimensions
    )
    unexpected = sorted(set(parsed).difference(expected))
    if unexpected:
        raise ValueError(
            "unexpected labels: "
            + ", ".join(":".join(part or "-" for part in key) for key in unexpected)
        )
    if explicit:
        persist(tuple(parsed[key] for key in expected if key in parsed))
    missing = tuple(key for key in expected if key not in parsed)
    if missing and non_interactive:
        raise ValueError(
            "missing labels: " + ", ".join(":".join(part or "-" for part in key) for key in missing)
        )
    dimensions = {dimension.dimension_id: dimension for dimension in rubric.dimensions}
    preview_positions = {
        (preview.trace_id, preview.reference_trace_id): index
        for index, preview in enumerate(previews, start=1)
    }
    for key in missing:
        trace_id, reference_id, dimension_id = key
        dimension = dimensions[dimension_id]
        _render_score_prompt(dimension)
        position = preview_positions[(trace_id, reference_id)]
        if pairwise:
            choice = Prompt.ask(
                "Pair "
                f"{position}: choose candidate A, candidate B, or tie for "
                f"{escape(dimension.name)}",
                choices=["A", "B", "tie"],
            )
            value: int | str = {"A": "winner_a", "B": "winner_b", "tie": "tie"}[choice]
        else:
            score = IntPrompt.ask(
                f"Trace {position}: score {escape(dimension.name)} from "
                f"{dimension.min_score} to {dimension.max_score}"
            )
            if not dimension.contains_score(score):
                raise ValueError(
                    f"judge labels for {dimension_id} must be integers from "
                    f"{dimension.min_score} through {dimension.max_score}"
                )
            value = score
        parsed[key] = _label(key, value, pairwise=pairwise)
        persist(tuple(parsed[item] for item in expected if item in parsed))
    return tuple(parsed[key] for key in expected)


def _render_score_prompt(dimension: RubricDimension) -> None:
    """Keep one score question adjacent to its complete plain-language anchors.

    Args:
        dimension: Rubric dimension the next prompt asks the operator to score.
    """
    _console.print()
    _console.print(f"Score prompt: {dimension.name}", style="bold", markup=False)
    _console.print(dimension.description, markup=False)
    for anchor in dimension.anchors:
        _console.print(f"  {anchor.score}: {anchor.description}", markup=False)


def _label(
    key: tuple[str, str | None, str],
    value: int | str,
    *,
    pairwise: bool,
) -> ManualJudgeLabel:
    """Build one validated human label from its key and typed value.

    Args:
        key: Trace, optional reference trace, and dimension identifiers.
        value: Integer axis score, or a typed pairwise winner.
        pairwise: Whether the finalized setup requires a comparison label.

    Returns:
        Validated human label for the calibration sample.
    """
    trace_id, reference_id, dimension_id = key
    return ManualJudgeLabel.model_validate(
        {
            "trace_id": trace_id,
            "reference_trace_id": reference_id,
            "dimension_id": dimension_id,
            **({"winner": value} if pairwise else {"score": value}),
        }
    )


def _label_key(value: str, *, pairwise: bool) -> tuple[str, str | None, str]:
    """Parse the trace and dimension key from one CLI label expression.

    Args:
        value: Scalar or pairwise CLI label expression.
        pairwise: Whether the finalized setup requires a comparison trace.

    Returns:
        Trace, optional reference trace, and dimension identifiers.

    Raises:
        ValueError: The expression does not have the required separators.
    """
    target, separator, _score = value.rpartition("=")
    parts = target.split(":")
    expected_parts = 3 if pairwise else 2
    if not separator or len(parts) != expected_parts or any(not part for part in parts):
        expected = (
            "TRACE_ID:REFERENCE_TRACE_ID:DIMENSION_ID=winner_a|winner_b|tie"
            if pairwise
            else "TRACE_ID:DIMENSION_ID=SCORE"
        )
        raise ValueError(f"labels must use {expected}")
    if pairwise:
        return parts[0], parts[1], parts[2]
    return parts[0], None, parts[1]


def _label_value(
    value: str,
    *,
    pairwise: bool,
    axis: RubricDimension | None = None,
) -> int | str:
    """Parse and validate one CLI label value.

    Args:
        value: Scalar or pairwise CLI label expression.
        pairwise: Whether the finalized setup requires a typed winner.
        axis: Axis that bounds a scalar score.

    Returns:
        Integer score or typed pairwise winner.

    Raises:
        ValueError: The score is not an integer inside the axis range.
    """
    raw = value.rpartition("=")[2]
    if pairwise:
        if raw not in {"winner_a", "winner_b", "tie"}:
            raise ValueError("pairwise judge labels must use winner_a, winner_b, or tie")
        return raw
    try:
        score = int(raw)
    except ValueError as exc:
        raise ValueError("judge labels must use an integer score") from exc
    if axis is not None and not axis.contains_score(score):
        raise ValueError(
            f"judge labels for {axis.dimension_id} must be integers from "
            f"{axis.min_score} through {axis.max_score}"
        )
    return score


def _load_setup_rubric(store: ProjectStore, setup: ManualJudgeSetupArtifact) -> Rubric:
    """Load and verify the finalized rubric named by setup.

    Args:
        store: Project-local immutable artifact store.
        setup: Finalized manual judge setup.

    Returns:
        Verified human-approved rubric.

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
        result: Completed immutable audit and calibration report.
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
        value: Optional finite report metric.

    Returns:
        Three-decimal value or ``n/a`` when grouped evidence is unavailable.
    """
    return "n/a" if value is None else f"{value:.3f}"


def _confirm(question: str, *, non_interactive: bool, required_flag: str) -> bool:
    """Ask one non-spend confirmation or require its explicit noninteractive flag.

    Args:
        question: Human-readable decision prompt.
        non_interactive: Whether prompting is forbidden.
        required_flag: Flag that provides the explicit decision in scripts.

    Returns:
        The operator's explicit answer.

    Raises:
        ValueError: No interactive terminal is available for an omitted flag.
    """
    if non_interactive or not can_prompt(_console):
        raise ValueError(f"noninteractive judge review requires {required_flag}")
    return Confirm.ask(question, default=False)
