"""Explicit local setup and calibration services for one configured LM judge."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime

from wmo.common.core.artifacts import ArtifactInput, stable_id
from wmo.common.judging import (
    HumanLabelSet,
    HumanScoreReview,
    JudgeCalibrationService,
    JudgeScoreObservation,
    RouterLineageAssignment,
    RouterLineageSplit,
    Rubric,
    RubricDimension,
    RubricReview,
    write_router_lineage_split,
)
from wmo.common.judging.evidence import DEFAULT_JUDGE_OUTPUT_TOKENS
from wmo.common.judging.provenance import JudgingProvenanceError, read_artifact_json
from wmo.common.models import ModelCatalog, ModelSnapshot, PricingSource
from wmo.common.project import ProjectStore, artifact_input
from wmo.common.tasks import TaskCase, load_task_set
from wmo.common.traces import Trace, load_trace_dataset
from wmo.optimize.router.judging.artifacts import (
    coordinate_manual_judge_calibration,
    coordinate_manual_judge_setup,
    find_provisional_calibration,
    read_review_state,
    replay_or_approve,
    require_review_state,
    write_audit,
    write_production_rollout,
    write_review_state,
)
from wmo.optimize.router.judging.artifacts import (
    load_build_review as _load_build_review,
)
from wmo.optimize.router.judging.artifacts import (
    require_exact_build_inputs as _require_exact_build_inputs,
)
from wmo.optimize.router.judging.contracts import (
    JudgeCalibrationBudget,
    JudgePromptTemplate,
    JudgeTracePreview,
    ManualJudgeCalibrationResult,
    ManualJudgeError,
    ManualJudgeLabel,
    ManualJudgeReviewState,
    ManualJudgeSetupArtifact,
)
from wmo.optimize.router.judging.labels import calibration_sample_digest, save_label_draft
from wmo.optimize.router.judging.pricing import resolve_manual_judge_prices
from wmo.optimize.router.judging.protocol import positional_bias_count
from wmo.optimize.router.judging.review import (
    ManualJudgeReviewCollection,
    ManualJudgeReviewer,
    collect_trace_reviews,
    labels_from_reviews,
    manual_label_score,
    ordered_completed_reviews,
    read_review_judgment,
    read_trace_reviews,
    review_evidence,
    reviewer_from_labels,
)
from wmo.optimize.router.judging.selection import (
    pairwise_references,
    representative_pairs,
    representative_pairwise_pairs,
    trace_preview,
)
from wmo.optimize.router.judging.setup_store import (
    read_setup_artifact as _read_setup,
)
from wmo.optimize.router.judging.setup_store import (
    write_setup_artifact as _write_setup,
)
from wmo.optimize.router.judging.template_bind import (
    DEFAULT_JUDGE_TEMPLATE,
    bind_prompt_template,
    default_judge_dimensions,
)
from wmo.runtime.models.providers.transport import RetryPolicy
from wmo.runtime.models.registry import RuntimeModelCatalog
from wmo.simulation.build import BuildReviewReadiness


@dataclass(frozen=True)
class ManualJudgeSetupPlan:
    """Read-only setup preview awaiting explicit human confirmation.

    Args:
        project_id: Project whose completed build supplies the preview.
        judge_alias: Configured completion alias selected for judging.
        judge_model: Exact static model snapshot for the alias.
        build: Verified completed-build pointer from review state.
        dimensions: Complete proposed rubric dimensions.
        prompt_template: Versioned prompt, mapping, and response schema.
        previews: Rendered real local traces shown before confirmation.
        created_at: Time used if the plan is committed.
        code_revision: Exact producer revision used if committed.
    """

    project_id: str
    judge_alias: str
    judge_model: ModelSnapshot
    build: BuildReviewReadiness
    dimensions: tuple[RubricDimension, ...]
    prompt_template: JudgePromptTemplate
    previews: tuple[JudgeTracePreview, ...]
    created_at: datetime
    code_revision: str


@dataclass(frozen=True)
class ManualJudgeCalibrationPlan:
    """Read-only real-trace labeling plan for a finalized judge setup.

    Args:
        setup: Verified finalized judge setup.
        tasks: Representative fit tasks selected by unique lineage.
        traces: Real normalized traces matched to the selected tasks.
        reference_traces: Same-task comparison traces for pairwise feedback, otherwise ``None``.
        previews: Human-readable trace summaries in label order.
    """

    setup: ManualJudgeSetupArtifact
    tasks: tuple[TaskCase, ...]
    traces: tuple[Trace, ...]
    reference_traces: tuple[Trace | None, ...]
    previews: tuple[JudgeTracePreview, ...]


def prepare_manual_judge_setup(
    store: ProjectStore,
    catalog: ModelCatalog,
    *,
    judge_alias: str | None = None,
    dimensions: Sequence[RubricDimension] | None = None,
    prompt_template: JudgePromptTemplate = DEFAULT_JUDGE_TEMPLATE,
    preview_count: int = 3,
    created_at: datetime,
    code_revision: str,
) -> ManualJudgeSetupPlan:
    """Prepare a real-trace judge preview without writing or resolving credentials.

    Args:
        store: Existing project with completed deterministic build evidence.
        catalog: Local secret-free model catalog.
        judge_alias: Optional explicit alias, otherwise the configured judge role.
        dimensions: Optional complete rubric replacement.
        prompt_template: Versioned prompt, variable mapping, and response schema.
        preview_count: Maximum number of distinct fit-lineage traces to render.
        created_at: Time to bind if the plan is confirmed.
        code_revision: Exact producer revision to bind if confirmed.

    Returns:
        A fully validated in-memory plan with no persistence or provider side effect.

    Raises:
        ManualJudgeError: Project, build, alias, rubric, or real trace evidence is unavailable.
    """
    if preview_count < 1:
        raise ManualJudgeError("judge setup preview count must be positive")
    project = store.load_project()
    build = _load_build_review(store)
    if build.project_config != project.model_copy(update={"build": None}):
        raise ManualJudgeError("completed build belongs to a different project configuration")
    selected_alias = judge_alias or catalog.roles.judge
    if selected_alias is None:
        raise ManualJudgeError(
            "no judge alias is configured; run wmo config providers with a judge model first"
        )
    try:
        judge_model, _capabilities = RuntimeModelCatalog(catalog).snapshot(selected_alias)
    except ValueError as exc:
        raise ManualJudgeError(str(exc)) from exc
    selected_dimensions = tuple(dimensions or default_judge_dimensions())
    if not selected_dimensions:
        raise ManualJudgeError("judge setup requires at least one rubric axis")
    try:
        prompt_template = bind_prompt_template(prompt_template, selected_dimensions)
    except ValueError as exc:
        raise ManualJudgeError(str(exc)) from exc
    _require_exact_build_inputs(store, build)
    tasks = load_task_set(store.artifacts, build.task_set.artifact_id).tasks
    traces = load_trace_dataset(store.artifacts, build.trace_dataset.artifact_id).traces
    selected = representative_pairs(tasks, traces, preview_count)
    return ManualJudgeSetupPlan(
        project_id=project.project_id,
        judge_alias=selected_alias,
        judge_model=judge_model,
        build=build,
        dimensions=selected_dimensions,
        prompt_template=prompt_template,
        previews=tuple(trace_preview(task, trace) for task, trace in selected),
        created_at=created_at,
        code_revision=code_revision,
    )


@coordinate_manual_judge_setup
def commit_manual_judge_setup(
    store: ProjectStore,
    plan: ManualJudgeSetupPlan,
    *,
    confirmed: bool,
) -> ManualJudgeSetupArtifact:
    """Finalize the reviewed rubric and persist setup after explicit confirmation.

    Args:
        store: Project-local artifact and mutable review store.
        plan: Read-only setup plan previously displayed to the operator.
        confirmed: Explicit confirmation of judge, rubric, template, mapping, and schema.

    Returns:
        Immutable setup artifact, including an idempotent exact replay. When the only
        difference from the saved setup is an advanced template version, the setup is
        replaced and calibration restarts under the current evidence rendering.

    Raises:
        ManualJudgeError: Confirmation is absent, the project/build changed before commit,
            or the plan names a different finalized contract.
    """
    if not confirmed:
        raise ManualJudgeError("judge setup requires explicit confirmation before writing")
    if (
        store.load_project().project_id != plan.project_id
        or _load_build_review(store) != plan.build
    ):
        raise ManualJudgeError("project build changed after judge setup preview")
    _require_exact_build_inputs(store, plan.build)
    existing = read_review_state(store)
    if existing is not None:
        saved = _read_setup(store, existing.setup)
        try:
            saved_rubric, saved_rubric_input = read_artifact_json(
                store,
                artifact_id=saved.rubric.artifact_id,
                expected_artifact_type="rubric",
                relative_path="rubric.json",
                model_type=Rubric,
            )
        except JudgingProvenanceError as exc:
            raise ManualJudgeError("existing manual judge rubric is unavailable") from exc
        same_contract = (
            saved.project_id == plan.project_id
            and saved.judge_alias == plan.judge_alias
            and saved.judge_model == plan.judge_model
            and saved.prompt_template == plan.prompt_template
            and saved.trace_dataset == plan.build.trace_dataset
            and saved.task_set == plan.build.task_set
            and saved.previews == plan.previews
            and saved_rubric_input == saved.rubric
            and saved_rubric.dimensions == plan.dimensions
        )
        if not same_contract:
            if _is_template_version_upgrade(saved, plan, saved_rubric, saved_rubric_input):
                return _persist_setup(store, plan, inputs=saved.inputs, rubric=saved.rubric)
            raise ManualJudgeError("project already has a different finalized judge setup")
        return saved
    review = RubricReview.open(
        store,
        source_task_set_id=plan.build.task_set.artifact_id,
        code_revision=plan.code_revision,
        clock=lambda: plan.created_at,
    )
    review.replace_all(plan.dimensions)
    rubric = review.finalize()
    rubric_input = artifact_input(store.artifacts.read(rubric.rubric_id).manifest)
    inputs = tuple(
        sorted(
            (plan.build.trace_dataset, plan.build.task_set, rubric_input),
            key=lambda item: item.artifact_id,
        )
    )
    return _persist_setup(store, plan, inputs=inputs, rubric=rubric_input)


def _persist_setup(
    store: ProjectStore,
    plan: ManualJudgeSetupPlan,
    *,
    inputs: tuple[ArtifactInput, ...],
    rubric: ArtifactInput,
) -> ManualJudgeSetupArtifact:
    """Write one finalized setup and point fresh review state at it.

    Args:
        store: Project-local artifact and mutable review store.
        plan: Confirmed setup plan to persist.
        inputs: Ordered exact manifest inputs binding the setup.
        rubric: Verified manifest pointer of the finalized rubric.

    Returns:
        The newly persisted immutable setup artifact.
    """
    setup_id = stable_id(
        "manual-judge-setup",
        {
            "project_id": plan.project_id,
            "judge_alias": plan.judge_alias,
            "judge_model": plan.judge_model.model_dump(mode="json"),
            "prompt_template": plan.prompt_template.model_dump(mode="json"),
            "inputs": [item.model_dump(mode="json") for item in inputs],
        },
    )
    setup = ManualJudgeSetupArtifact(
        schema_version=1,
        created_at=plan.created_at,
        inputs=inputs,
        code_revision=plan.code_revision,
        setup_id=setup_id,
        project_id=plan.project_id,
        judge_alias=plan.judge_alias,
        judge_model=plan.judge_model,
        prompt_template=plan.prompt_template,
        trace_dataset=plan.build.trace_dataset,
        task_set=plan.build.task_set,
        rubric=rubric,
        previews=plan.previews,
    )
    setup_input = _write_setup(store, setup)
    write_review_state(store, ManualJudgeReviewState(setup=setup_input))
    return setup


def _is_template_version_upgrade(
    saved: ManualJudgeSetupArtifact,
    plan: ManualJudgeSetupPlan,
    saved_rubric: Rubric,
    saved_rubric_input: ArtifactInput,
) -> bool:
    """Report whether the plan only advances a saved setup to the current template version.

    Previews are intentionally not compared: they are operator-facing renderings whose
    count is a display choice, not part of the judged contract. The saved setup, its
    probes, and any approved audit stay immutable in the artifact store, and review
    state restarts with no drafts, audit, or approval.

    Args:
        saved: Existing finalized setup persisted under an earlier template version.
        plan: Confirmed replacement plan built from the same project evidence.
        saved_rubric: Rubric loaded from the saved setup pointer.
        saved_rubric_input: Verified manifest pointer of the saved rubric.

    Returns:
        True when every judged contract field matches and only the template version
        moves from 2 to 3.
    """
    return (
        saved.prompt_template.template_version == "2"
        and plan.prompt_template
        == saved.prompt_template.model_copy(update={"template_version": "3"})
        and saved.project_id == plan.project_id
        and saved.judge_alias == plan.judge_alias
        and saved.judge_model == plan.judge_model
        and saved.trace_dataset == plan.build.trace_dataset
        and saved.task_set == plan.build.task_set
        and saved_rubric_input == saved.rubric
        and saved_rubric.dimensions == plan.dimensions
    )


def prepare_manual_judge_calibration(
    store: ProjectStore,
    *,
    sample_size: int = 5,
) -> ManualJudgeCalibrationPlan:
    """Select representative real fit-lineage traces for labeling without writes.

    Args:
        store: Project whose judge setup and completed build are immutable.
        sample_size: Maximum number of distinct fit lineages to label.

    Returns:
        Frozen-order trace plan ready for local human labels.

    Raises:
        ManualJudgeError: Setup is absent, sample size invalid, or evidence changed.
    """
    if sample_size < 1:
        raise ManualJudgeError("judge calibration sample size must be positive")
    state = require_review_state(store)
    setup = _read_setup(store, state.setup)
    build = _load_build_review(store)
    if setup.trace_dataset != build.trace_dataset or setup.task_set != build.task_set:
        raise ManualJudgeError("judge setup no longer matches the completed build")
    _require_exact_build_inputs(store, build)
    tasks = load_task_set(store.artifacts, setup.task_set.artifact_id).tasks
    traces = load_trace_dataset(store.artifacts, setup.trace_dataset.artifact_id).traces
    selected = (
        representative_pairwise_pairs(tasks, traces, sample_size)
        if setup.prompt_template.response_shape == "pairwise"
        else representative_pairs(tasks, traces, sample_size)
    )
    reference_traces = pairwise_references(selected, traces, setup.prompt_template.response_shape)
    return ManualJudgeCalibrationPlan(
        setup=setup,
        tasks=tuple(task for task, _trace in selected),
        traces=tuple(trace for _task, trace in selected),
        reference_traces=reference_traces,
        previews=tuple(
            trace_preview(task, trace, reference)
            for (task, trace), reference in zip(selected, reference_traces, strict=True)
        ),
    )


def calibration_sample(
    plan: ManualJudgeCalibrationPlan,
) -> tuple[tuple[str, str | None], ...]:
    """Return the frozen trace identity, with pairwise reference, that labels must cover.

    Args:
        plan: Frozen representative real-trace calibration plan.

    Returns:
        Selected trace IDs with their pairwise reference trace ID in plan order.
    """
    pairwise = plan.setup.prompt_template.response_shape == "pairwise"
    return tuple(
        (trace.trace_id, reference.trace_id if pairwise and reference is not None else None)
        for trace, reference in zip(plan.traces, plan.reference_traces, strict=True)
    )


def manual_judge_calibration_is_complete(store: ProjectStore) -> bool:
    """Report whether a completed audit already fixes this project's calibration.

    Args:
        store: Project-local review store.

    Returns:
        ``True`` when calibration is complete and every further run replays it.

    Raises:
        ManualJudgeError: Setup has not been completed or review state is malformed.
    """
    return require_review_state(store).audit is not None


def estimate_manual_judge_budget(
    plan: ManualJudgeCalibrationPlan,
    *,
    catalog: ModelCatalog | None = None,
    input_usd_per_million_tokens: float | None = None,
    output_usd_per_million_tokens: float | None = None,
    maximum_input_tokens_per_call: int,
    maximum_cost_usd: float,
    retry_policy: RetryPolicy | None = None,
    completed_review_count: int = 0,
) -> JudgeCalibrationBudget:
    """Reserve worst-case judging spend before credentials or provider calls.

    Args:
        plan: Frozen real-trace calibration plan.
        catalog: Local catalog used when advanced overrides are omitted.
        input_usd_per_million_tokens: Optional advanced input-price override.
        output_usd_per_million_tokens: Optional advanced output-price override.
        maximum_input_tokens_per_call: Conservative request-token ceiling.
        maximum_cost_usd: Operator's total calibration spend ceiling.
        retry_policy: Runtime retry bound used by provider clients.
        completed_review_count: Immutable trace reviews that require no further provider calls.

    Returns:
        Finite conservative admission budget for one call per labeled rollout.

    Raises:
        ManualJudgeError: Catalog, identity, or pricing provenance cannot admit the run.
        ValueError: Prices, bounds, or total ceiling cannot admit the complete run.
    """
    if catalog is None:
        if input_usd_per_million_tokens is None or output_usd_per_million_tokens is None:
            raise ManualJudgeError(
                "judge calibration requires the project model catalog to resolve prices"
            )
        input_price = input_usd_per_million_tokens
        output_price = output_usd_per_million_tokens
        source = PricingSource.CONFIGURED
    else:
        input_price, output_price, source = resolve_manual_judge_prices(
            catalog,
            judge_alias=plan.setup.judge_alias,
            expected_model=plan.setup.judge_model,
            input_usd_per_million_tokens=input_usd_per_million_tokens,
            output_usd_per_million_tokens=output_usd_per_million_tokens,
        )
    resolved_retry = retry_policy or RetryPolicy()
    if completed_review_count < 0 or completed_review_count > len(plan.traces):
        raise ValueError("completed judge review count is outside the frozen trace sample")
    calls_per_trace = 2 if plan.setup.prompt_template.response_shape == "pairwise" else 1
    call_count = (len(plan.traces) - completed_review_count) * calls_per_trace
    per_attempt = (
        maximum_input_tokens_per_call * input_price + DEFAULT_JUDGE_OUTPUT_TOKENS * output_price
    ) / 1_000_000
    return JudgeCalibrationBudget(
        input_usd_per_million_tokens=input_price,
        output_usd_per_million_tokens=output_price,
        pricing_source=source,
        maximum_input_tokens_per_call=maximum_input_tokens_per_call,
        maximum_output_tokens_per_call=DEFAULT_JUDGE_OUTPUT_TOKENS,
        maximum_attempts_per_call=resolved_retry.maximum_attempts,
        call_count=call_count,
        estimated_cost_usd=per_attempt * resolved_retry.maximum_attempts * call_count,
        maximum_cost_usd=maximum_cost_usd,
    )


@coordinate_manual_judge_calibration
def calibrate_manual_judge(
    store: ProjectStore,
    runtime_catalog: RuntimeModelCatalog,
    plan: ManualJudgeCalibrationPlan,
    labels: Sequence[ManualJudgeLabel],
    budget: JudgeCalibrationBudget,
    *,
    spend_consented: bool,
    approve: bool,
    accept_insufficient_labels: bool,
    created_at: datetime,
    code_revision: str,
    reviewer: ManualJudgeReviewer | None = None,
) -> ManualJudgeCalibrationResult:
    """Run judge-first trace reviews, report evidence, and optionally approve.

    Args:
        store: Project-local artifact and review store.
        runtime_catalog: Injected resolver for the configured judge alias.
        plan: Frozen representative real-trace calibration plan.
        labels: Complete explicit human scores used when ``reviewer`` is not supplied.
        budget: Explicit conservative spend reservation shown before consent.
        spend_consented: Whether the operator accepted the displayed reservation.
        approve: Separate explicit approval of the completed calibration report.
        accept_insufficient_labels: Explicit risk acceptance below five completed reviews.
        created_at: Time for newly completed artifacts and decisions.
        code_revision: Exact producer revision for new artifacts.
        reviewer: Human decision supplier invoked after each immutable judge proposal.

    Returns:
        Calibration audit, report, optional approved calibration pointer, and call count.

    Raises:
        ManualJudgeError: Consent, labels, identity, evidence, or approval is invalid.
    """
    state = require_review_state(store)
    setup = _read_setup(store, state.setup)
    if setup != plan.setup:
        raise ManualJudgeError("judge calibration plan no longer matches finalized setup")
    if state.audit is not None:
        return replay_or_approve(
            store,
            state,
            approve=approve,
            accept_insufficient_labels=accept_insufficient_labels,
            approved_at=created_at,
        )
    sample_sha256 = calibration_sample_digest(setup, calibration_sample(plan))
    completed_reviews = read_trace_reviews(store, setup, sample_sha256)
    supplied_labels = tuple(labels)
    explicit_label_input = reviewer is None
    if explicit_label_input:
        _validate_labels(store, plan, setup, supplied_labels)
        save_label_draft(store, setup, sample_sha256, supplied_labels, created_at)
        reviewer = reviewer_from_labels(setup, supplied_labels)
    elif supplied_labels:
        raise ManualJudgeError("supply either a review callback or legacy labels, not both")
    calls_per_trace = 2 if setup.prompt_template.response_shape == "pairwise" else 1
    expected_calls = (len(plan.traces) - len(completed_reviews)) * (calls_per_trace)
    total_calls = len(plan.traces) * calls_per_trace
    if budget.call_count not in {expected_calls, total_calls}:
        raise ManualJudgeError("judge budget call count differs from incomplete trace reviews")
    if expected_calls and not spend_consented:
        raise ManualJudgeError("judge calibration requires explicit spend consent before writes")
    rollout_inputs = tuple(
        write_production_rollout(
            store,
            setup,
            task,
            trace,
            created_at,
            code_revision,
            allow_provider_free_source=True,
        )
        for task, trace in zip(plan.tasks, plan.traces, strict=True)
    )
    reference_inputs = tuple(
        (
            write_production_rollout(
                store,
                setup,
                task,
                reference,
                created_at,
                code_revision,
                allow_provider_free_source=True,
            )
            if reference is not None
            else None
        )
        for task, reference in zip(plan.tasks, plan.reference_traces, strict=True)
    )
    split = _write_lineage_split(store, setup, plan, rollout_inputs, created_at, code_revision)
    label_review = HumanScoreReview.open(store)
    provisional = find_provisional_calibration(store, setup, split.split_id)
    if provisional is None:
        empty_labels = label_review.finalize(
            rubric_id=setup.rubric.artifact_id,
            code_revision=code_revision,
            created_at=created_at,
        )
        provisional = JudgeCalibrationService().bootstrap_provisional(
            store,
            rubric_id=setup.rubric.artifact_id,
            label_set_id=empty_labels.label_set_id,
            router_lineage_split_id=split.split_id,
            judge_model=setup.judge_model,
            judge_prompt=setup.prompt_template.prompt,
            created_at=created_at,
            code_revision=code_revision,
        )
    provisional_input = artifact_input(store.artifacts.read(provisional.calibration_id).manifest)
    rubric, rubric_input = read_artifact_json(
        store,
        artifact_id=setup.rubric.artifact_id,
        expected_artifact_type="rubric",
        relative_path="rubric.json",
        model_type=Rubric,
    )
    if rubric_input != setup.rubric:
        raise ManualJudgeError("manual judge rubric manifest differs from setup")
    if len(completed_reviews) < len(plan.traces):
        resolved = runtime_catalog.preflight(setup.judge_alias, role="judge")
        if resolved.snapshot != setup.judge_model:
            raise ManualJudgeError("configured judge identity changed after setup")
        collection = collect_trace_reviews(
            store,
            resolved,
            setup=setup,
            setup_input=state.setup,
            tasks=plan.tasks,
            traces=plan.traces,
            reference_traces=plan.reference_traces,
            rollout_inputs=rollout_inputs,
            reference_inputs=reference_inputs,
            provisional_input=provisional_input,
            rubric=rubric,
            budget=budget,
            sample_sha256=sample_sha256,
            reviewer=reviewer,
            created_at=created_at,
            code_revision=code_revision,
        )
    else:
        collection = ManualJudgeReviewCollection(
            reviews=ordered_completed_reviews(
                completed_reviews,
                setup=setup,
                setup_input=state.setup,
                tasks=plan.tasks,
                traces=plan.traces,
                reference_traces=plan.reference_traces,
                rollout_inputs=rollout_inputs,
                reference_inputs=reference_inputs,
                provisional_input=provisional_input,
                rubric=rubric,
            ),
            provider_calls_made=0,
        )
    accepted_labels = labels_from_reviews(collection.reviews)
    _validate_labels(store, plan, setup, accepted_labels)
    if not explicit_label_input:
        save_label_draft(
            store,
            setup,
            sample_sha256,
            accepted_labels,
            created_at,
        )
    human_labels = _write_labels(
        label_review,
        setup,
        plan,
        rollout_inputs,
        accepted_labels,
        created_at,
        code_revision,
    )
    evidence = review_evidence(collection.reviews)
    observations: list[JudgeScoreObservation] = []
    positional_comparisons = 0
    positional_flips = 0
    for review in collection.reviews:
        judgment = read_review_judgment(store, review)
        if setup.prompt_template.response_shape == "pairwise":
            comparisons, flips = positional_bias_count(store, review.original_judge_response)
            positional_comparisons += comparisons
            positional_flips += flips
        observations.extend(
            JudgeScoreObservation(
                judgment=review.normalized_judgment,
                source_rollout=review.trace_evidence,
                dimension_id=dimension.dimension_id,
                raw_score=dimension.raw_score,
            )
            for dimension in judgment.dimensions
        )
    service = JudgeCalibrationService()
    report = service.build_report(
        store,
        rubric_id=setup.rubric.artifact_id,
        label_set_id=human_labels.label_set_id,
        router_lineage_split_id=split.split_id,
        observations=observations,
        created_at=created_at,
        code_revision=code_revision,
    )
    report = service.write_report(store, report)
    audit = write_audit(
        store,
        setup_input=state.setup,
        label_input=artifact_input(store.artifacts.read(human_labels.label_set_id).manifest),
        split_input=artifact_input(store.artifacts.read(split.split_id).manifest),
        provisional_input=artifact_input(store.artifacts.read(provisional.calibration_id).manifest),
        report_input=artifact_input(store.artifacts.read(report.report_id).manifest),
        budget=budget,
        judgments=evidence,
        trace_reviews=tuple(
            artifact_input(store.artifacts.read(review.review_id).manifest)
            for review in collection.reviews
        ),
        positional_bias=(
            (positional_comparisons, positional_flips)
            if setup.prompt_template.response_shape == "pairwise"
            else None
        ),
        created_at=created_at,
        code_revision=code_revision,
    )
    audit_input = artifact_input(store.artifacts.read(audit.audit_id).manifest)
    next_state = require_review_state(store).model_copy(update={"audit": audit_input})
    write_review_state(store, next_state)
    return replay_or_approve(
        store,
        next_state,
        approve=approve,
        accept_insufficient_labels=accept_insufficient_labels,
        approved_at=created_at,
        provider_calls_made=collection.provider_calls_made,
    )


def _validate_labels(
    store: ProjectStore,
    plan: ManualJudgeCalibrationPlan,
    setup: ManualJudgeSetupArtifact,
    labels: Sequence[ManualJudgeLabel],
) -> None:
    """Require exactly one score per selected trace and rubric dimension.

    Args:
        store: Project-local immutable artifact store.
        plan: Frozen trace selection.
        setup: Finalized setup naming the rubric.
        labels: Operator-supplied scores.

    Raises:
        ManualJudgeError: A score is missing, duplicated, or outside the setup rubric.
    """
    try:
        rubric, rubric_input = read_artifact_json(
            store,
            artifact_id=setup.rubric.artifact_id,
            expected_artifact_type="rubric",
            relative_path="rubric.json",
            model_type=Rubric,
        )
    except JudgingProvenanceError as exc:
        raise ManualJudgeError("manual judge rubric is unavailable") from exc
    if rubric_input != setup.rubric:
        raise ManualJudgeError("manual judge rubric manifest differs from setup")
    pairwise = setup.prompt_template.response_shape == "pairwise"
    expected = {
        (
            trace.trace_id,
            reference.trace_id if pairwise and reference is not None else None,
            dimension.dimension_id,
        )
        for trace, reference in zip(plan.traces, plan.reference_traces, strict=True)
        for dimension in rubric.dimensions
    }
    supplied = [(label.trace_id, label.reference_trace_id, label.dimension_id) for label in labels]
    if pairwise and any(label.reference_trace_id is None for label in labels):
        raise ManualJudgeError("pairwise labels must retain their reference trace")
    if not pairwise and any(
        label.reference_trace_id is not None or label.winner is not None for label in labels
    ):
        raise ManualJudgeError("non-pairwise setups require direct axis-range scores")
    axes = {item.dimension_id: item for item in rubric.dimensions}
    for label in labels:
        if label.score is None:
            continue
        axis = axes[label.dimension_id]
        if not axis.contains_score(label.score):
            raise ManualJudgeError(
                f"human scores for {label.dimension_id} must be integers from "
                f"{axis.min_score} through {axis.max_score}"
            )
    if len(set(supplied)) != len(supplied):
        raise ManualJudgeError("judge calibration labels must not repeat a trace dimension")
    missing = sorted(expected.difference(supplied))
    unexpected = sorted(set(supplied).difference(expected))
    if missing or unexpected:
        details = []
        if missing:
            details.append(
                "missing "
                + ", ".join(
                    f"{trace}:{reference or '-'}:{dimension}"
                    for trace, reference, dimension in missing
                )
            )
        if unexpected:
            details.append(
                "unexpected "
                + ", ".join(
                    f"{trace}:{reference or '-'}:{dimension}"
                    for trace, reference, dimension in unexpected
                )
            )
        raise ManualJudgeError("judge calibration label set is incomplete: " + "; ".join(details))


def _write_lineage_split(
    store: ProjectStore,
    setup: ManualJudgeSetupArtifact,
    plan: ManualJudgeCalibrationPlan,
    rollout_inputs: Sequence[ArtifactInput],
    created_at: datetime,
    code_revision: str,
) -> RouterLineageSplit:
    """Freeze router partitions and selected rollout lineage assignments.

    Args:
        store: Project-local immutable artifact store.
        setup: Finalized setup binding the task set.
        plan: Frozen representative fit trace plan.
        rollout_inputs: Persisted selected production rollouts.
        created_at: Artifact completion time.
        code_revision: Exact producer revision.

    Returns:
        Persisted lineage split used by grouped calibration.
    """
    all_tasks = load_task_set(store.artifacts, setup.task_set.artifact_id).tasks
    fit = tuple(sorted({task.lineage_group_id for task in all_tasks if task.partition == "fit"}))
    held_out = tuple(
        sorted({task.lineage_group_id for task in all_tasks if task.partition == "held_out"})
    )
    assignments = tuple(
        sorted(
            (
                RouterLineageAssignment(
                    rollout_id=rollout.artifact_id,
                    lineage_id=task.lineage_group_id,
                )
                for task, rollout in zip(plan.tasks, rollout_inputs, strict=True)
            ),
            key=lambda item: item.rollout_id,
        )
    )
    split_id = stable_id(
        "router-lineage-split",
        {
            "task_set": setup.task_set.model_dump(mode="json"),
            "fit": list(fit),
            "held_out": list(held_out),
            "assignments": [item.model_dump(mode="json") for item in assignments],
        },
    )
    split = RouterLineageSplit(
        schema_version=1,
        created_at=created_at,
        inputs=(setup.task_set,),
        code_revision=code_revision,
        split_id=split_id,
        source_task_set_id=setup.task_set.artifact_id,
        fit_lineage_ids=fit,
        held_out_lineage_ids=held_out,
        assignments=assignments,
    )
    return write_router_lineage_split(store, split)


def _write_labels(
    review: HumanScoreReview,
    setup: ManualJudgeSetupArtifact,
    plan: ManualJudgeCalibrationPlan,
    rollout_inputs: Sequence[ArtifactInput],
    labels: Sequence[ManualJudgeLabel],
    created_at: datetime,
    code_revision: str,
) -> HumanLabelSet:
    """Append complete human scores and freeze the labeled calibration set.

    Args:
        review: Open score-history service already used to freeze the empty bootstrap set.
        setup: Finalized setup binding the rubric.
        plan: Frozen trace and task selection.
        rollout_inputs: Persisted rollout pointers aligned with the plan.
        labels: Complete validated human scores.
        created_at: Label and artifact completion time.
        code_revision: Exact producer revision.

    Returns:
        Immutable labeled human score set.
    """
    score_by_key = {
        (item.trace_id, item.reference_trace_id, item.dimension_id): manual_label_score(setup, item)
        for item in labels
    }
    for trace, reference, task, rollout in zip(
        plan.traces,
        plan.reference_traces,
        plan.tasks,
        rollout_inputs,
        strict=True,
    ):
        for trace_id, reference_id, dimension_id in sorted(
            key
            for key in score_by_key
            if key[0] == trace.trace_id
            and key[1] == (reference.trace_id if reference is not None else None)
        ):
            review.upsert(
                rubric_id=setup.rubric.artifact_id,
                rollout_id=rollout.artifact_id,
                lineage_id=task.lineage_group_id,
                dimension_id=dimension_id,
                score=score_by_key[(trace_id, reference_id, dimension_id)],
                submission_id=stable_id(
                    "manual-label-submission",
                    {
                        "setup_id": setup.setup_id,
                        "trace_id": trace_id,
                        "reference_trace_id": reference_id,
                        "dimension_id": dimension_id,
                        "score": score_by_key[(trace_id, reference_id, dimension_id)],
                    },
                ),
                created_at=created_at,
            )
    return review.finalize(
        rubric_id=setup.rubric.artifact_id,
        code_revision=code_revision,
        created_at=created_at,
    )
