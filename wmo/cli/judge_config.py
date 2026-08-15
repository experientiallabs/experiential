"""Manual local judge setup and calibration commands."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import typer
from rich.console import Console
from rich.prompt import Confirm, IntPrompt, Prompt

from wmo.cli.consent import can_prompt, require_spend_consent
from wmo.cli.options import ROOT_OPTION, usage_error
from wmo.common.judging import Rubric, RubricDimension
from wmo.common.judging.provenance import read_artifact_json
from wmo.common.models import load_model_catalog
from wmo.common.project import ProjectStore
from wmo.common.release_revision import installed_release_revision
from wmo.optimize.router.judging.contracts import (
    JudgePromptTemplate,
    JudgeTracePreview,
    ManualJudgeCalibrationResult,
    ManualJudgeLabel,
    ManualJudgeSetupArtifact,
)
from wmo.optimize.router.judging.service import (
    DEFAULT_JUDGE_TEMPLATE,
    ManualJudgeError,
    ManualJudgeSetupPlan,
    calibrate_manual_judge,
    commit_manual_judge_setup,
    estimate_manual_judge_budget,
    prepare_manual_judge_calibration,
    prepare_manual_judge_setup,
)
from wmo.runtime.models.registry import RuntimeModelCatalog

judge_app = typer.Typer(help="Set up and manually calibrate a project judge.", no_args_is_help=True)
_console = Console()
_RUBRIC_FILE_OPTION = typer.Option(
    None, "--rubric-file", help="JSON array of complete zero-to-five rubric dimensions."
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
    help="Preview real traces and save a confirmed judge rubric and prompt contract.",
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
        preview_count: Maximum number of real fit traces to render.
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
    help="Label real traces, run consented judge calls, and separately approve calibration.",
)
def judge_calibrate(
    project: str = typer.Argument(..., metavar="PROJECT", help="Configured local project ID."),
    root: Path = ROOT_OPTION,
    sample_size: int = typer.Option(10, "--sample-size", min=1),
    label: list[str] | None = _LABEL_OPTION,
    input_price: float = typer.Option(..., "--input-usd-per-million", min=0),
    output_price: float = typer.Option(..., "--output-usd-per-million", min=0),
    maximum_input_tokens: int = typer.Option(32_768, "--maximum-input-tokens", min=1),
    maximum_cost_usd: float = typer.Option(10.0, "--maximum-cost-usd", min=0.000001),
    yes: bool = typer.Option(False, "--yes", help="Consent to the displayed judge spend."),
    approve: bool = typer.Option(
        False, "--approve", help="Approve the report after it is displayed."
    ),
    accept_insufficient_labels: bool = typer.Option(
        False,
        "--accept-insufficient-labels",
        help="Accept valid grouped evidence from fewer than ten rollouts.",
    ),
    non_interactive: bool = typer.Option(False, "--non-interactive"),
) -> None:
    """Collect frozen labels, run consented judge calls, and separately approve evidence.

    Args:
        project: Local project ID below ``<root>/projects``.
        root: Local project root containing ``models.toml``.
        sample_size: Maximum number of distinct fit lineages to calibrate.
        label: Repeatable explicit human score inputs.
        input_price: Explicit judge input price per million tokens.
        output_price: Explicit judge output price per million tokens.
        maximum_input_tokens: Conservative input bound for every call attempt.
        maximum_cost_usd: Total calibration spend ceiling.
        yes: Explicit spend consent only.
        approve: Separate approval of the displayed completed report.
        accept_insufficient_labels: Explicit risk acceptance below ten labeled rollouts.
        non_interactive: Refuse prompts and list missing explicit inputs.

    Raises:
        typer.BadParameter: Evidence, labels, budget, consent, or approval is invalid.
    """
    with usage_error(OSError, ValueError, ManualJudgeError):
        revision = installed_release_revision()
        store = ProjectStore(root, project)
        now = datetime.now(UTC)
        plan = prepare_manual_judge_calibration(store, sample_size=sample_size)
        rubric = _load_setup_rubric(store, plan.setup)
        _render_calibration_previews(plan.previews)
        labels = _collect_labels(
            plan.setup,
            rubric,
            tuple(label or ()),
            plan.previews,
            non_interactive=non_interactive,
        )
        budget = estimate_manual_judge_budget(
            plan,
            input_usd_per_million_tokens=input_price,
            output_usd_per_million_tokens=output_price,
            maximum_input_tokens_per_call=maximum_input_tokens,
            maximum_cost_usd=maximum_cost_usd,
        )
    spend = (
        f"at most ${budget.estimated_cost_usd:.4f} across {budget.call_count} judge calls "
        f"with up to {budget.maximum_attempts_per_call} attempts each"
    )
    if not require_spend_consent(
        _console,
        yes=yes,
        spend=spend,
        command="wmo config judge calibrate",
    ):
        _console.print("Judge calibration was not started.")
        return
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
        should_approve = approve or _confirm(
            "Approve this immutable judge calibration?",
            non_interactive=non_interactive,
            required_flag="--approve",
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
    """Display every setup identity and a concise real-trace preview.

    Args:
        plan: Read-only setup plan awaiting confirmation.
    """
    _console.print(f"Judge alias: {plan.judge_alias}")
    _console.print(f"Judge model: {plan.judge_model.provider}/{plan.judge_model.model_id}")
    _console.print(
        f"Prompt: {plan.prompt_template.prompt.prompt_id} ({plan.prompt_template.prompt.sha256})"
    )
    _console.print(f"Response shape: {plan.prompt_template.response_shape}")
    _console.print(f"Variable mapping: {json.dumps(plan.prompt_template.variable_mapping)}")
    _console.print(
        "Structured response schema: "
        + json.dumps(plan.prompt_template.response_schema, sort_keys=True)
    )
    _console.print(
        "Score projection: "
        + json.dumps(plan.prompt_template.score_projection.model_dump(mode="json"), sort_keys=True)
    )
    _console.print("Rubric:")
    for dimension in plan.dimensions:
        _console.print(f"  {dimension.dimension_id}: {dimension.name}")
    _console.print("Real trace preview:")
    for preview in plan.previews:
        _console.print(
            f"  {preview.trace_id} [{preview.outcome}] {preview.task} "
            f"({len(preview.span_names)} spans)"
        )


def _render_calibration_previews(previews: tuple[JudgeTracePreview, ...]) -> None:
    """Display the frozen real-trace sample before requesting human labels.

    Args:
        previews: Ordered real trace previews selected for calibration.
    """
    _console.print("Calibration traces:")
    for preview in previews:
        comparison = (
            f" vs {preview.reference_trace_id}" if preview.reference_trace_id is not None else ""
        )
        _console.print(
            f"  {preview.trace_id}{comparison} [{preview.outcome}] {preview.task} "
            f"spans={', '.join(preview.span_names)}"
        )


def _collect_labels(
    setup: ManualJudgeSetupArtifact,
    rubric: Rubric,
    supplied: tuple[str, ...],
    previews: tuple[JudgeTracePreview, ...],
    *,
    non_interactive: bool,
) -> tuple[ManualJudgeLabel, ...]:
    """Parse explicit labels or ask for every missing trace-dimension score.

    Args:
        setup: Finalized setup used for stable prompt context.
        rubric: Verified finalized scoring rubric.
        supplied: Repeatable CLI label expressions.
        previews: Frozen ordered calibration trace previews.
        non_interactive: Whether all missing inputs must be reported without prompting.

    Returns:
        Complete ordered human label set.

    Raises:
        ValueError: A label is malformed, duplicated, missing, or outside zero through five.
    """
    pairwise = setup.prompt_template.response_shape == "pairwise"
    parsed: dict[tuple[str, str | None, str], int | str] = {}
    for item in supplied:
        key = _label_key(item, pairwise=pairwise)
        if key in parsed:
            raise ValueError("duplicate label for " + ":".join(part or "-" for part in key))
        parsed[key] = _label_value(item, pairwise=pairwise)
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
    missing = tuple(key for key in expected if key not in parsed)
    if missing and non_interactive:
        raise ValueError(
            "missing labels: " + ", ".join(":".join(part or "-" for part in key) for key in missing)
        )
    for trace_id, reference_id, dimension_id in missing:
        if pairwise:
            winner = Prompt.ask(
                f"Winner for {trace_id} vs {reference_id} on {dimension_id}",
                choices=["winner_a", "winner_b", "tie"],
            )
            parsed[(trace_id, reference_id, dimension_id)] = winner
        else:
            score = IntPrompt.ask(f"Score {trace_id} on {dimension_id} (0-5)")
            if score not in range(6):
                raise ValueError("judge labels must be integers from zero through five")
            parsed[(trace_id, reference_id, dimension_id)] = score
    return tuple(
        ManualJudgeLabel.model_validate(
            {
                "trace_id": trace_id,
                "reference_trace_id": reference_id,
                "dimension_id": dimension_id,
                **({"winner": parsed[key]} if pairwise else {"score": parsed[key]}),
            }
        )
        for key in expected
        for trace_id, reference_id, dimension_id in (key,)
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


def _label_value(value: str, *, pairwise: bool) -> int | str:
    """Parse and validate the zero-to-five score from one CLI expression.

    Args:
        value: Scalar or pairwise CLI label expression.
        pairwise: Whether the finalized setup requires a typed winner.

    Returns:
        Integer score or typed pairwise winner.

    Raises:
        ValueError: The score is not an integer from zero through five.
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
    if score not in range(6):
        raise ValueError("judge labels must be integers from zero through five")
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
