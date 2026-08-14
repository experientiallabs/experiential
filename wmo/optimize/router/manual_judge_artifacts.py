"""Persistence helpers for explicit local judge calibration evidence."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import datetime
from functools import wraps
from typing import Concatenate, Protocol

from pydantic import JsonValue

from wmo.common.core.artifacts import ArtifactInput, JsonObject, stable_id
from wmo.common.core.locks import FileLockTimeout
from wmo.common.judging import (
    CalibrationReport,
    JudgeCalibration,
    JudgeCalibrationService,
)
from wmo.common.judging.provenance import JudgingProvenanceError, read_artifact_json
from wmo.common.models import AssistantAction, ModelSnapshot, OperationEconomics, Usage
from wmo.common.project import (
    ArtifactAlreadyExistsError,
    ArtifactStoreError,
    ProjectStore,
    ProjectStoreError,
    artifact_input,
)
from wmo.common.rollouts import RolloutArtifact, SimulationMode, StopReason
from wmo.common.rollouts.otel import ProductionSimulatorSnapshot, RolloutEventKind, RolloutSpan
from wmo.common.tasks import TaskCase, load_task_set
from wmo.common.traces import Trace, TraceSpan
from wmo.optimize.router.manual_judge_contracts import (
    JudgeCalibrationBudget,
    JudgeRunEvidence,
    ManualJudgeCalibrationAudit,
    ManualJudgeCalibrationResult,
    ManualJudgeError,
    ManualJudgeReviewState,
    ManualJudgeSetupArtifact,
)
from wmo.simulation.build import BuildReviewReadiness, coordinate_selected_build_review


class _BuildEvidence(Protocol):
    """Structural trace and task binding required by coordinated judge plans."""

    @property
    def trace_dataset(self) -> ArtifactInput:
        """Return the exact trace dataset pointer."""
        ...

    @property
    def task_set(self) -> ArtifactInput:
        """Return the exact task-set pointer."""
        ...


class _SetupMutationPlan(Protocol):
    """Structural setup plan exposing its completed-build evidence."""

    @property
    def build(self) -> _BuildEvidence:
        """Return the reviewed completed-build binding."""
        ...


class _CalibrationMutationPlan(Protocol):
    """Structural calibration plan exposing its finalized setup evidence."""

    @property
    def setup(self) -> _BuildEvidence:
        """Return the finalized setup build binding."""
        ...


_ACTIVE_BUILD_BINDING: ContextVar[tuple[str, ArtifactInput, ArtifactInput] | None] = ContextVar(
    "manual_judge_active_build_binding",
    default=None,
)


def coordinate_manual_judge_setup[SetupPlanT: _SetupMutationPlan, **P, R](
    operation: Callable[Concatenate[ProjectStore, SetupPlanT, P], R],
) -> Callable[Concatenate[ProjectStore, SetupPlanT, P], R]:
    """Wrap a setup mutation in the plan's stable selected-build transaction.

    Args:
        operation: Setup mutation whose first arguments are the project store and setup plan.

    Returns:
        Signature-preserving operation serialized with completed-build replacement.
    """

    @wraps(operation)
    def coordinated(
        store: ProjectStore,
        plan: SetupPlanT,
        *args: P.args,
        **kwargs: P.kwargs,
    ) -> R:
        """Run the setup mutation against one stable selected build."""
        with coordinate_manual_judge_review(
            store,
            trace_dataset=plan.build.trace_dataset,
            task_set=plan.build.task_set,
        ):
            return operation(store, plan, *args, **kwargs)

    return coordinated


def coordinate_manual_judge_calibration[
    CatalogT,
    CalibrationPlanT: _CalibrationMutationPlan,
    **P,
    R,
](
    operation: Callable[
        Concatenate[ProjectStore, CatalogT, CalibrationPlanT, P],
        R,
    ],
) -> Callable[Concatenate[ProjectStore, CatalogT, CalibrationPlanT, P], R]:
    """Wrap calibration in the finalized setup's stable selected-build transaction.

    Args:
        operation: Calibration mutation beginning with store, runtime catalog, and plan.

    Returns:
        Signature-preserving operation serialized with completed-build replacement.
    """

    @wraps(operation)
    def coordinated(
        store: ProjectStore,
        catalog: CatalogT,
        plan: CalibrationPlanT,
        *args: P.args,
        **kwargs: P.kwargs,
    ) -> R:
        """Run calibration against one stable selected build."""
        with coordinate_manual_judge_review(
            store,
            trace_dataset=plan.setup.trace_dataset,
            task_set=plan.setup.task_set,
        ):
            return operation(store, catalog, plan, *args, **kwargs)

    return coordinated


@contextmanager
def coordinate_manual_judge_review(
    store: ProjectStore,
    *,
    trace_dataset: ArtifactInput,
    task_set: ArtifactInput,
) -> Iterator[None]:
    """Keep one manual-judge operation bound to a stable selected build.

    Nested review-state writes reuse the outer operation lock only when they name the same exact
    trace and task inputs. Independent operations contend through the project-local build-review
    coordination lock.

    Args:
        store: Project whose selected build supplies the judge evidence.
        trace_dataset: Exact trace artifact consumed by the judge operation.
        task_set: Exact task artifact consumed by the judge operation.

    Yields:
        None while completed-build replacement is excluded.

    Raises:
        ManualJudgeError: The nested or selected build binding differs from the requested inputs.
    """
    requested = (str(store.paths.project_directory.absolute()), trace_dataset, task_set)
    active = _ACTIVE_BUILD_BINDING.get()
    if active is not None:
        if active != requested:
            raise ManualJudgeError("nested manual judge state names a different project or build")
        yield
        return
    try:
        with coordinate_selected_build_review(
            store,
            trace_dataset=trace_dataset,
            task_set=task_set,
        ):
            token = _ACTIVE_BUILD_BINDING.set(requested)
            try:
                yield
            finally:
                _ACTIVE_BUILD_BINDING.reset(token)
    except (ArtifactStoreError, FileLockTimeout, ProjectStoreError, ValueError) as exc:
        raise ManualJudgeError(str(exc)) from exc


def load_build_review(store: ProjectStore) -> BuildReviewReadiness:
    """Load the exact completed-build handoff from mutable review state.

    Args:
        store: Project-local review store.

    Returns:
        Validated completed-build readiness.

    Raises:
        ManualJudgeError: The build handoff is absent or malformed.
    """
    review = store.read_review()
    if not isinstance(review, dict) or "build_review" not in review:
        raise ManualJudgeError("project has no completed build; run wmo build first")
    try:
        return BuildReviewReadiness.model_validate(review["build_review"])
    except ValueError as exc:
        raise ManualJudgeError("project build review is invalid") from exc


def require_exact_build_inputs(store: ProjectStore, build: BuildReviewReadiness) -> None:
    """Verify build trace and task manifests before preview or persistence.

    Args:
        store: Project-local immutable artifact store.
        build: Build handoff naming exact trace and task inputs.

    Raises:
        ManualJudgeError: Either manifest changed or the task set is not trace-bound.
    """
    trace_input = artifact_input(store.artifacts.read(build.trace_dataset.artifact_id).manifest)
    task_input = artifact_input(store.artifacts.read(build.task_set.artifact_id).manifest)
    if trace_input != build.trace_dataset or task_input != build.task_set:
        raise ManualJudgeError("completed build artifact manifest changed")
    task_set = load_task_set(store.artifacts, build.task_set.artifact_id).task_set
    if task_set.inputs != (trace_input,):
        raise ManualJudgeError("completed task set does not bind the exact trace dataset")


def rollout_id(task: TaskCase, trace: Trace) -> str:
    """Return the stable production-rollout identity for one selected trace.

    Args:
        task: Representative task bound to the trace.
        trace: Normalized production trace.

    Returns:
        Content-derived rollout artifact identifier.
    """
    return stable_id(
        "production-rollout",
        {
            "task_id": task.task_id,
            "lineage_id": task.lineage_group_id,
            "trace": trace.model_dump(mode="json"),
        },
    )


def write_review_state(store: ProjectStore, state: ManualJudgeReviewState) -> None:
    """Update manual judge state only while its selected build remains current.

    Args:
        store: Project-local review store.
        state: Complete replacement manual judge state.

    Raises:
        ManualJudgeError: Setup provenance, selected build, or review state is invalid.
    """
    try:
        setup, setup_input = read_artifact_json(
            store,
            artifact_id=state.setup.artifact_id,
            expected_artifact_type="manual-judge-setup",
            relative_path="setup.json",
            model_type=ManualJudgeSetupArtifact,
        )
    except JudgingProvenanceError as exc:
        raise ManualJudgeError("manual judge setup is unavailable for review update") from exc
    if setup_input != state.setup:
        raise ManualJudgeError("manual judge setup manifest differs from review state")

    def update(current: JsonValue | None) -> JsonObject:
        """Preserve unrelated review namespaces while replacing manual judge state.

        Args:
            current: Current complete review JSON value.

        Returns:
            Updated object preserving every unrelated key.

        Raises:
            ManualJudgeError: The current review value is not an object.
        """
        if current is None:
            root: JsonObject = {}
        elif isinstance(current, dict):
            root = dict(current)
        else:
            raise ManualJudgeError("review.json must be an object")
        root["manual_judge"] = state.model_dump(mode="json")
        return root

    with coordinate_manual_judge_review(
        store,
        trace_dataset=setup.trace_dataset,
        task_set=setup.task_set,
    ):
        store.update_review(update)


def find_provisional_calibration(
    store: ProjectStore,
    setup: ManualJudgeSetupArtifact,
    lineage_split_id: str,
) -> JudgeCalibration | None:
    """Find the unique completed provisional binding for an interrupted calibration.

    Args:
        store: Project-local immutable artifact store.
        setup: Finalized setup defining rubric, model, and prompt identity.
        lineage_split_id: Frozen calibration lineage split.

    Returns:
        Matching provisional calibration, or ``None`` before bootstrap completes.

    Raises:
        ManualJudgeError: Matching evidence is malformed, conflicting, or ambiguous.
    """
    matches: list[JudgeCalibration] = []
    for artifact_id in store.artifacts.list_ids():
        stored = store.artifacts.read(artifact_id)
        if stored.manifest.artifact_type != "judge-calibration":
            continue
        try:
            calibration, _calibration_input = read_artifact_json(
                store,
                artifact_id=artifact_id,
                expected_artifact_type="judge-calibration",
                relative_path="calibration.json",
                model_type=JudgeCalibration,
            )
        except JudgingProvenanceError as exc:
            raise ManualJudgeError("existing provisional calibration cannot be resumed") from exc
        if (
            calibration.status != "provisional"
            or calibration.rubric_id != setup.rubric.artifact_id
            or calibration.judge_model != setup.judge_model
            or calibration.judge_prompt_id != setup.prompt_template.prompt.prompt_id
            or calibration.judge_prompt_sha256 != setup.prompt_template.prompt.sha256
        ):
            continue
        try:
            report, _report_input = read_artifact_json(
                store,
                artifact_id=calibration.out_of_fold_report_id,
                expected_artifact_type="judge-calibration-report",
                relative_path="report.json",
                model_type=CalibrationReport,
            )
        except JudgingProvenanceError as exc:
            raise ManualJudgeError("provisional calibration report cannot be resumed") from exc
        if report.router_lineage_split_id == lineage_split_id:
            matches.append(calibration)
    if len(matches) > 1:
        raise ManualJudgeError("multiple provisional calibrations match the finalized setup")
    return matches[0] if matches else None


def write_production_rollout(
    store: ProjectStore,
    setup: ManualJudgeSetupArtifact,
    task: TaskCase,
    trace: Trace,
    created_at: datetime,
    code_revision: str,
    *,
    attributed_candidate: ModelSnapshot | None = None,
    attribution_input: ArtifactInput | None = None,
) -> ArtifactInput:
    """Persist one real trace as immutable production rollout evidence.

    Args:
        store: Project-local immutable artifact store.
        setup: Finalized judge setup binding the source dataset.
        task: Representative fit task linked to the trace.
        trace: Real normalized trace to preserve.
        created_at: Artifact completion time.
        code_revision: Exact producer revision.
        attributed_candidate: Selected candidate proven by immutable attribution evidence.
        attribution_input: Exact attribution artifact authorizing the selected candidate.

    Returns:
        Exact rollout manifest pointer.

    Raises:
        ManualJudgeError: The trace has no recorded model identity or conflicts on replay.
    """
    if (attributed_candidate is None) != (attribution_input is None):
        raise ManualJudgeError(
            "attributed production rollouts require both candidate and attribution input"
        )
    candidate = attributed_candidate or next(
        (span.model for span in trace.spans if span.model is not None),
        None,
    )
    if candidate is None:
        raise ManualJudgeError(
            f"trace {trace.trace_id!r} has no recorded model identity and cannot be calibrated"
        )
    base_rollout_id = rollout_id(task, trace)
    artifact_id = (
        base_rollout_id
        if attribution_input is None
        else stable_id(
            "attributed-production-rollout",
            {
                "version": "attributed-production-rollout-v1",
                "base_rollout_id": base_rollout_id,
                "candidate": candidate.model_dump(mode="json"),
                "attribution": attribution_input.model_dump(mode="json"),
            },
        )
    )
    inputs = tuple(
        sorted(
            (
                setup.trace_dataset,
                *((attribution_input,) if attribution_input is not None else ()),
            ),
            key=lambda item: item.artifact_id,
        )
    )
    failure = trace.outcome.failure if trace.outcome is not None else None
    rollout = RolloutArtifact(
        schema_version=1,
        created_at=created_at,
        inputs=inputs,
        code_revision=code_revision,
        source=trace.source.identity,
        artifact_id=artifact_id,
        simulation_id=(
            "production-import-v1"
            if attribution_input is None
            else "attributed-production-import-v1"
        ),
        cell_id=stable_id("production-cell", {"rollout_id": artifact_id}),
        mode=SimulationMode.WORLD_MODEL,
        rollout_id=artifact_id,
        trace_id=trace.trace_id,
        evidence_source="production",
        source_run_id=setup.trace_dataset.artifact_id,
        task_id=task.task_id,
        candidate=candidate,
        agent_id=setup.project_id,
        simulator=ProductionSimulatorSnapshot(source=trace.source.identity),
        repeat=0,
        spans=tuple(_rollout_span(span) for span in trace.spans),
        final_output=_trace_final_output(trace),
        stop_reason=StopReason.FAILURE if failure is not None else StopReason.COMPLETED,
        failure=failure,
        candidate_economics=OperationEconomics(usage=_combined_usage(trace)),
    )
    try:
        manifest = store.artifacts.write_json(
            artifact_id=artifact_id,
            artifact_type="rollout",
            envelope=rollout,
            files={"rollout.json": rollout},
        )
    except ArtifactAlreadyExistsError:
        try:
            existing, existing_input = read_artifact_json(
                store,
                artifact_id=artifact_id,
                expected_artifact_type="rollout",
                relative_path="rollout.json",
                model_type=RolloutArtifact,
            )
        except JudgingProvenanceError as exc:
            raise ManualJudgeError("existing production rollout cannot be resumed safely") from exc
        if not _same_rollout_identity(existing, rollout):
            raise ManualJudgeError(
                "existing production rollout conflicts with the selected trace"
            ) from None
        return existing_input
    return artifact_input(manifest)


def _same_rollout_identity(left: RolloutArtifact, right: RolloutArtifact) -> bool:
    """Compare immutable rollout content while excluding materialization time.

    Args:
        left: Persisted rollout from an earlier attempt.
        right: Freshly materialized rollout for the same real trace.

    Returns:
        Whether all semantically immutable fields match.
    """
    return left.model_copy(update={"created_at": right.created_at}) == right


def _rollout_span(span: TraceSpan) -> RolloutSpan:
    """Convert one normalized trace span to immutable rollout evidence.

    Args:
        span: Verified normalized trace span.

    Returns:
        Equivalent rollout span with recorded model, usage, and failure.
    """
    return RolloutSpan(
        span_id=span.span_id,
        parent_span_id=span.parent_span_id,
        kind=(
            RolloutEventKind.AGENT_MODEL_CALL
            if span.model is not None
            else RolloutEventKind.MESSAGE
        ),
        started_at=span.started_at,
        ended_at=span.ended_at,
        payload={"name": span.name, "attributes": span.attributes},
        model=span.model,
        usage=span.usage,
        failure=span.failure,
    )


def _trace_final_output(trace: Trace) -> AssistantAction | None:
    """Extract the final captured text output when normalized attributes contain one.

    Args:
        trace: Normalized real production trace.

    Returns:
        Final assistant text, or ``None`` when no supported text attribute exists.
    """
    for span in reversed(trace.spans):
        for key in ("output", "response", "content"):
            value = span.attributes.get(key)
            if isinstance(value, str) and value:
                return AssistantAction(content=value)
    return None


def _combined_usage(trace: Trace) -> Usage | None:
    """Sum recorded model usage across one normalized production trace.

    Args:
        trace: Normalized real production trace.

    Returns:
        Combined usage, or ``None`` when no span reported usage.
    """
    usages = tuple(span.usage for span in trace.spans if span.usage is not None)
    if not usages:
        return None
    cached = tuple(item.cached_input_tokens for item in usages)
    return Usage(
        input_tokens=sum(item.input_tokens for item in usages),
        output_tokens=sum(item.output_tokens for item in usages),
        cached_input_tokens=(
            sum(item for item in cached if item is not None)
            if all(item is not None for item in cached)
            else None
        ),
    )


def write_audit(
    store: ProjectStore,
    *,
    setup_input: ArtifactInput,
    label_input: ArtifactInput,
    split_input: ArtifactInput,
    provisional_input: ArtifactInput,
    report_input: ArtifactInput,
    budget: JudgeCalibrationBudget,
    judgments: tuple[JudgeRunEvidence, ...],
    positional_bias: tuple[int, int] | None,
    created_at: datetime,
    code_revision: str,
) -> ManualJudgeCalibrationAudit:
    """Persist the exact evidence shown at the separate approval boundary.

    Args:
        store: Project-local immutable artifact store.
        setup_input: Finalized manual setup pointer.
        label_input: Frozen human-label-set pointer.
        split_input: Frozen lineage split pointer.
        provisional_input: Internal provisional calibration pointer.
        report_input: Completed disagreement report pointer.
        budget: Consent-bound conservative spend reservation.
        judgments: Persisted rollout and structured judgment pairs.
        positional_bias: Pairwise comparison and order-flip counts, otherwise ``None``.
        created_at: Artifact completion time.
        code_revision: Exact producer revision.

    Returns:
        Persisted immutable calibration audit.
    """
    inputs = tuple(
        sorted(
            (
                setup_input,
                label_input,
                split_input,
                provisional_input,
                report_input,
                *(item.rollout for item in judgments),
                *(
                    item.reference_rollout
                    for item in judgments
                    if item.reference_rollout is not None
                ),
                *(item.judgment for item in judgments),
                *(probe for item in judgments for probe in item.probes),
            ),
            key=lambda item: item.artifact_id,
        )
    )
    audit_id = stable_id(
        "manual-judge-audit",
        {
            "inputs": [item.model_dump(mode="json") for item in inputs],
            "budget": budget.model_dump(mode="json"),
            "judgments": [item.model_dump(mode="json") for item in judgments],
            "positional_bias": positional_bias,
        },
    )
    audit = ManualJudgeCalibrationAudit(
        schema_version=1,
        created_at=created_at,
        inputs=inputs,
        code_revision=code_revision,
        audit_id=audit_id,
        setup=setup_input,
        human_labels=label_input,
        lineage_split=split_input,
        provisional_calibration=provisional_input,
        report=report_input,
        budget=budget,
        judgments=judgments,
        positional_bias_comparisons=(positional_bias[0] if positional_bias is not None else None),
        positional_bias_flips=(positional_bias[1] if positional_bias is not None else None),
    )
    try:
        store.artifacts.write_json(
            artifact_id=audit_id,
            artifact_type="manual-judge-calibration-audit",
            envelope=audit,
            files={"audit.json": audit},
        )
    except ArtifactAlreadyExistsError:
        existing = read_audit(store, artifact_input(store.artifacts.read(audit_id).manifest))
        if not _same_audit_identity(existing, audit):
            raise ManualJudgeError("existing judge calibration audit conflicts") from None
        return existing
    return audit


def _same_audit_identity(
    existing: ManualJudgeCalibrationAudit,
    replay: ManualJudgeCalibrationAudit,
) -> bool:
    """Compare an existing audit with a retry while ignoring materialization time only.

    Args:
        existing: Manifest-verified immutable audit written by an earlier attempt.
        replay: Audit reconstructed from the current verified calibration evidence.

    Returns:
        Whether every semantic field and input matches after preserving the original timestamp.
    """
    return existing == replay.model_copy(update={"created_at": existing.created_at})


def read_audit(store: ProjectStore, expected: ArtifactInput) -> ManualJudgeCalibrationAudit:
    """Read one audit and require its exact review-state manifest pointer.

    Args:
        store: Project-local immutable artifact store.
        expected: Audit pointer retained by review state.

    Returns:
        Verified calibration audit.

    Raises:
        ManualJudgeError: The audit is unavailable, malformed, or changed.
    """
    try:
        audit, audit_input = read_artifact_json(
            store,
            artifact_id=expected.artifact_id,
            expected_artifact_type="manual-judge-calibration-audit",
            relative_path="audit.json",
            model_type=ManualJudgeCalibrationAudit,
        )
    except JudgingProvenanceError as exc:
        raise ManualJudgeError("completed judge calibration audit is unavailable") from exc
    if audit_input != expected:
        raise ManualJudgeError("judge calibration audit manifest differs from review state")
    return audit


def _read_report(store: ProjectStore, expected: ArtifactInput) -> CalibrationReport:
    """Read one calibration report and require its exact audit pointer.

    Args:
        store: Project-local immutable artifact store.
        expected: Report pointer retained by the audit.

    Returns:
        Verified calibration report.

    Raises:
        ManualJudgeError: The report is unavailable, malformed, or changed.
    """
    try:
        report, report_input = read_artifact_json(
            store,
            artifact_id=expected.artifact_id,
            expected_artifact_type="judge-calibration-report",
            relative_path="report.json",
            model_type=CalibrationReport,
        )
    except JudgingProvenanceError as exc:
        raise ManualJudgeError("completed judge calibration report is unavailable") from exc
    if report_input != expected:
        raise ManualJudgeError("judge calibration report manifest differs from audit")
    return report


def replay_or_approve(
    store: ProjectStore,
    state: ManualJudgeReviewState,
    *,
    approve: bool,
    accept_insufficient_labels: bool,
    approved_at: datetime,
    provider_calls_made: int = 0,
) -> ManualJudgeCalibrationResult:
    """Replay a completed audit and optionally cross the human approval boundary.

    Args:
        store: Project-local immutable artifact and review store.
        state: Review state naming a completed audit.
        approve: Separate explicit report approval decision.
        accept_insufficient_labels: Explicit acceptance below ten eligible rollouts.
        approved_at: Time of the separate approval decision.
        provider_calls_made: Calls performed immediately before this replay step.

    Returns:
        Verified report and audit with an optional final approved calibration pointer.

    Raises:
        ManualJudgeError: Audit state is absent or approval cannot be proven and persisted.
    """
    if state.audit is None:
        raise ManualJudgeError("judge calibration audit is not complete")
    audit = read_audit(store, state.audit)
    report = _read_report(store, audit.report)
    approved_input = state.approved_calibration
    if approve and approved_input is None:
        service = JudgeCalibrationService()
        try:
            calibration = service.approve(
                store,
                report,
                approved_at=approved_at,
                accept_insufficient_labels=accept_insufficient_labels,
            )
            calibration = service.write_calibration(
                store,
                report=report,
                calibration=calibration,
            )
        except ValueError as exc:
            raise ManualJudgeError(str(exc)) from exc
        approved_input = artifact_input(store.artifacts.read(calibration.calibration_id).manifest)
        state = state.model_copy(update={"approved_calibration": approved_input})
        write_review_state(store, state)
    return ManualJudgeCalibrationResult(
        audit=audit,
        report=report,
        approved_calibration=approved_input,
        provider_calls_made=provider_calls_made,
        completed_at=approved_at,
    )
