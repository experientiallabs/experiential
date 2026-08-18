"""Incremental judge-first review persistence for manual calibration."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import datetime

from wmo.common.core.artifacts import ArtifactInput, Sha256, stable_id
from wmo.common.judging import LMJudge, Rubric
from wmo.common.judging.judgment import Judgment
from wmo.common.judging.provenance import JudgingProvenanceError, read_artifact_json
from wmo.common.models import ModelSnapshot, OperationEconomics
from wmo.common.project import ArtifactAlreadyExistsError, ProjectStore, artifact_input
from wmo.common.rollouts import RolloutArtifact
from wmo.common.tasks import TaskCase
from wmo.common.traces import Trace
from wmo.optimize.router.judging.artifacts import (
    require_review_state,
    update_review_state,
)
from wmo.optimize.router.judging.contracts import (
    FinalAcceptedJudgeLabel,
    HumanJudgeCorrection,
    JudgeAxisProposal,
    JudgeCalibrationBudget,
    JudgeRunEvidence,
    ManualJudgeAxisDecision,
    ManualJudgeAxisReview,
    ManualJudgeError,
    ManualJudgeLabel,
    ManualJudgeReviewPricing,
    ManualJudgeReviewProvenance,
    ManualJudgeReviewState,
    ManualJudgeSetupArtifact,
    ManualJudgeTraceReviewArtifact,
)
from wmo.optimize.router.judging.protocol import (
    PairwiseCitationEvidence,
    TemplateJudgeClient,
    pairwise_citation_evidence_from_probes,
)
from wmo.optimize.router.judging.selection import read_rollout
from wmo.runtime.models.registry import ResolvedModel


@dataclass(frozen=True)
class ManualJudgeTraceProposal:
    """One persisted judge proposal awaiting a human decision.

    Args:
        position: One-based trace position in the frozen calibration sample.
        total: Total distinct trace lineages in the sample.
        trace: Original normalized production trace shown to the reviewer.
        reference_trace: Optional same-task comparison trace.
        rubric: Exact finalized rubric revision used by the judge.
        judgment: Immutable normalized configured-judge output.
        pairwise_citations: Separate target and reference citations from both judge orders.
    """

    position: int
    total: int
    trace: Trace
    reference_trace: Trace | None
    rubric: Rubric
    judgment: Judgment
    pairwise_citations: PairwiseCitationEvidence = ()


ManualJudgeReviewer = Callable[[ManualJudgeTraceProposal], Sequence[ManualJudgeAxisDecision]]


@dataclass(frozen=True)
class ManualJudgeReviewCollection:
    """Completed trace reviews and provider work performed during this invocation.

    Args:
        reviews: Immutable completed reviews in frozen sample order.
        provider_calls_made: Provider dispatches made instead of replayed this invocation.
    """

    reviews: tuple[ManualJudgeTraceReviewArtifact, ...]
    provider_calls_made: int


def completed_trace_review_count(
    store: ProjectStore,
    setup: ManualJudgeSetupArtifact,
    sample_sha256: Sha256,
) -> int:
    """Count completed immutable reviews for one frozen calibration sample.

    Args:
        store: Project-local review and artifact store.
        setup: Finalized manual judge setup.
        sample_sha256: Exact frozen trace-sample digest.

    Returns:
        Number of verified completed trace reviews for the sample.
    """
    return len(read_trace_reviews(store, setup, sample_sha256))


def read_trace_reviews(
    store: ProjectStore,
    setup: ManualJudgeSetupArtifact,
    sample_sha256: Sha256,
) -> tuple[ManualJudgeTraceReviewArtifact, ...]:
    """Load verified completed reviews for one setup and sample.

    Args:
        store: Project-local review and artifact store.
        setup: Finalized manual judge setup.
        sample_sha256: Exact frozen trace-sample digest.

    Returns:
        Matching immutable trace reviews in stored pointer order.

    Raises:
        ManualJudgeError: A pointer is corrupt or two reviews claim one trace.
    """
    state = require_review_state(store)
    setup_input = artifact_input(store.artifacts.read(setup.setup_id).manifest)
    matching: list[ManualJudgeTraceReviewArtifact] = []
    keys: set[tuple[str, str | None]] = set()
    for expected in state.trace_reviews:
        review = _read_trace_review(store, expected)
        if review.setup != state.setup:
            raise ManualJudgeError("trace review setup differs from current review state")
        if review.setup != setup_input:
            raise ManualJudgeError("trace review setup differs from finalized judge setup")
        if review.sample_sha256 != sample_sha256:
            continue
        key = (review.trace_id, review.reference_trace_id)
        if key in keys:
            raise ManualJudgeError("multiple completed reviews claim one calibration trace")
        keys.add(key)
        matching.append(review)
    return tuple(matching)


def collect_trace_reviews(
    store: ProjectStore,
    resolved: ResolvedModel,
    *,
    setup: ManualJudgeSetupArtifact,
    setup_input: ArtifactInput,
    tasks: Sequence[TaskCase],
    traces: Sequence[Trace],
    reference_traces: Sequence[Trace | None],
    rollout_inputs: Sequence[ArtifactInput],
    reference_inputs: Sequence[ArtifactInput | None],
    provisional_input: ArtifactInput,
    rubric: Rubric,
    budget: JudgeCalibrationBudget,
    sample_sha256: Sha256,
    reviewer: ManualJudgeReviewer,
    created_at: datetime,
    code_revision: str,
) -> ManualJudgeReviewCollection:
    """Judge and persist each incomplete trace before requesting its human decision.

    Args:
        store: Project-local immutable artifact and review store.
        resolved: Exact configured judge client and model identity.
        setup: Finalized manual judge setup.
        setup_input: Exact setup manifest pointer.
        tasks: Distinct-lineage tasks in frozen sample order.
        traces: Production traces aligned with ``tasks``.
        reference_traces: Optional pairwise traces aligned with ``traces``.
        rollout_inputs: Persisted target rollout pointers.
        reference_inputs: Optional persisted reference rollout pointers.
        provisional_input: Calibration binding used only for raw judge scoring.
        rubric: Exact finalized rubric revision.
        budget: Consent-bound complete-run reservation.
        sample_sha256: Frozen sample digest for resumability.
        reviewer: Human decision supplier called only after judge evidence exists.
        created_at: Artifact completion and review time.
        code_revision: Exact producer revision for immutable outputs.

    Returns:
        Completed reviews in sample order and provider calls made in this invocation.

    Raises:
        ManualJudgeError: Saved progress, judge output, or human decisions violate the contract.
    """
    lengths = {
        len(tasks),
        len(traces),
        len(reference_traces),
        len(rollout_inputs),
        len(reference_inputs),
    }
    if len(lengths) != 1 or not traces:
        raise ManualJudgeError("manual judge review inputs must be nonempty and aligned")
    completed = {
        (item.trace_id, item.reference_trace_id): item
        for item in read_trace_reviews(store, setup, sample_sha256)
    }
    ordered: list[ManualJudgeTraceReviewArtifact] = []
    provider_calls = 0
    total = len(traces)
    for index, values in enumerate(
        zip(
            tasks,
            traces,
            reference_traces,
            rollout_inputs,
            reference_inputs,
            strict=True,
        ),
        start=1,
    ):
        task, trace, reference_trace, rollout_input, reference_input = values
        reference_id = reference_trace.trace_id if reference_trace is not None else None
        saved = completed.get((trace.trace_id, reference_id))
        if saved is not None:
            _require_review_matches_plan(
                saved,
                setup=setup,
                task=task,
                rollout_input=rollout_input,
                reference_input=reference_input,
                provisional_input=provisional_input,
                rubric=rubric,
                judge_model=resolved.snapshot,
            )
            ordered.append(saved)
            continue
        rollout = read_rollout(store, rollout_input)
        reference = read_rollout(store, reference_input) if reference_input is not None else None
        adapter = TemplateJudgeClient(
            resolved.client,
            setup.prompt_template,
            rollout,
            rubric,
            reference,
            store=store,
            setup_input=setup_input,
            rollout_input=rollout_input,
            reference_input=reference_input,
            created_at=created_at,
            code_revision=code_revision,
            maximum_output_tokens=budget.maximum_output_tokens_per_call,
        )
        judge = LMJudge(
            adapter,
            setup.prompt_template.prompt,
            code_revision=code_revision,
            clock=lambda: created_at,
            maximum_output_tokens=budget.maximum_output_tokens_per_call,
        )
        judgment = judge.judge_and_write(
            store,
            rollout_artifact_id=rollout_input.artifact_id,
            rubric_artifact_id=setup.rubric.artifact_id,
            calibration_artifact_id=provisional_input.artifact_id,
        )
        provider_calls += adapter.provider_calls_made
        judgment_input = artifact_input(store.artifacts.read(judgment.judgment_id).manifest)
        proposal = ManualJudgeTraceProposal(
            position=index,
            total=total,
            trace=trace,
            reference_trace=reference_trace,
            rubric=rubric,
            judgment=judgment,
            pairwise_citations=adapter.pairwise_citation_evidence,
        )
        decisions = _validated_decisions(reviewer(proposal), judgment, rubric)
        review = _build_trace_review(
            setup=setup,
            setup_input=setup_input,
            sample_sha256=sample_sha256,
            task=task,
            trace=trace,
            reference_trace=reference_trace,
            rollout=rollout,
            rollout_input=rollout_input,
            reference_input=reference_input,
            provisional_input=provisional_input,
            rubric=rubric,
            judgment=judgment,
            judgment_input=judgment_input,
            probes=adapter.probes,
            pairwise_citations=adapter.pairwise_citation_evidence,
            decisions=decisions,
            budget=budget,
            created_at=created_at,
            code_revision=code_revision,
        )
        saved_input = _write_trace_review(store, review)
        _publish_trace_review(store, setup_input, saved_input, review)
        ordered.append(review)
    return ManualJudgeReviewCollection(
        reviews=tuple(ordered),
        provider_calls_made=provider_calls,
    )


def _validated_decisions(
    supplied: Sequence[ManualJudgeAxisDecision],
    judgment: Judgment,
    rubric: Rubric,
) -> tuple[ManualJudgeAxisDecision, ...]:
    """Validate one explicit decision for every judge-proposed dimension.

    Args:
        supplied: Human accept-or-correct decisions.
        judgment: Immutable normalized configured-judge proposal.
        rubric: Exact axes and inclusive ranges used for the proposal.

    Returns:
        Decisions ordered to match the configured judge dimensions.

    Raises:
        ManualJudgeError: A dimension is missing, unexpected, or repeated.
    """
    decisions = tuple(supplied)
    by_dimension = {item.dimension_id: item for item in decisions}
    expected = tuple(item.dimension_id for item in judgment.dimensions)
    rubric_dimensions = tuple(item.dimension_id for item in rubric.dimensions)
    if expected != rubric_dimensions:
        raise ManualJudgeError("configured judge proposal axes differ from the finalized rubric")
    if len(by_dimension) != len(decisions):
        raise ManualJudgeError("human review must not repeat a rubric axis")
    missing = sorted(set(expected).difference(by_dimension))
    unexpected = sorted(set(by_dimension).difference(expected))
    if missing or unexpected:
        details: list[str] = []
        if missing:
            details.append("missing " + ", ".join(missing))
        if unexpected:
            details.append("unexpected " + ", ".join(unexpected))
        raise ManualJudgeError(
            "human review axes do not match judge proposal: " + "; ".join(details)
        )
    for dimension in judgment.dimensions:
        axis = rubric.axis(dimension.dimension_id)
        if (dimension.min_score, dimension.max_score) != (axis.min_score, axis.max_score):
            raise ManualJudgeError("configured judge proposal range differs from the rubric")
        correction = by_dimension[dimension.dimension_id].correction
        if correction is not None and not axis.contains_score(correction.corrected_score):
            raise ManualJudgeError(
                f"human correction for {axis.dimension_id} must be an integer from "
                f"{axis.min_score} through {axis.max_score}"
            )
    return tuple(by_dimension[item] for item in expected)


def _build_trace_review(
    *,
    setup: ManualJudgeSetupArtifact,
    setup_input: ArtifactInput,
    sample_sha256: Sha256,
    task: TaskCase,
    trace: Trace,
    reference_trace: Trace | None,
    rollout: RolloutArtifact,
    rollout_input: ArtifactInput,
    reference_input: ArtifactInput | None,
    provisional_input: ArtifactInput,
    rubric: Rubric,
    judgment: Judgment,
    judgment_input: ArtifactInput,
    probes: tuple[ArtifactInput, ...],
    pairwise_citations: PairwiseCitationEvidence,
    decisions: Sequence[ManualJudgeAxisDecision],
    budget: JudgeCalibrationBudget,
    created_at: datetime,
    code_revision: str,
) -> ManualJudgeTraceReviewArtifact:
    """Build one immutable authorship-preserving trace review.

    Args:
        setup: Finalized configured-judge setup.
        setup_input: Exact persisted setup pointer.
        sample_sha256: Frozen calibration sample digest.
        task: Canonical task and lineage for the reviewed trace.
        trace: Reviewed production trace.
        reference_trace: Optional same-task comparison trace.
        rollout: Persisted target rollout content.
        rollout_input: Exact target rollout pointer.
        reference_input: Optional comparison rollout pointer.
        provisional_input: Provisional calibration used for raw judging.
        rubric: Exact finalized rubric revision.
        judgment: Normalized immutable configured-judge response.
        judgment_input: Exact normalized judgment pointer.
        probes: Original immutable provider-response pointers.
        pairwise_citations: Separate target and reference citations from both pairwise orders.
        decisions: Human decisions aligned with the judgment axes.
        budget: Consent-bound pricing and retry reservation.
        created_at: Human review completion time.
        code_revision: Exact producer revision.

    Returns:
        Complete immutable review with distinct judge, human, and final fields.
    """
    decision_by_id = {item.dimension_id: item for item in decisions}
    citation_by_id = {
        dimension_id: (target, reference) for dimension_id, target, reference in pairwise_citations
    }
    expected_dimensions = {item.dimension_id for item in judgment.dimensions}
    if reference_trace is None:
        if citation_by_id:
            raise ManualJudgeError("scalar review cannot retain pairwise citations")
    elif set(citation_by_id) != expected_dimensions:
        raise ManualJudgeError("pairwise review citations do not match judge dimensions")
    axes: list[ManualJudgeAxisReview] = []
    for dimension in judgment.dimensions:
        target_evidence, reference_evidence = citation_by_id.get(dimension.dimension_id, ((), ()))
        proposal = JudgeAxisProposal(
            dimension_id=dimension.dimension_id,
            proposed_score=dimension.raw_score,
            proposed_judgment=dimension.rationale or "",
            cited_trace_evidence=target_evidence,
            cited_reference_trace_evidence=reference_evidence,
        )
        correction = decision_by_id[dimension.dimension_id].correction
        final = FinalAcceptedJudgeLabel(
            score=proposal.proposed_score if correction is None else correction.corrected_score,
            judgment=(
                proposal.proposed_judgment
                if correction is None or correction.corrected_judgment is None
                else correction.corrected_judgment
            ),
            cited_trace_evidence=proposal.cited_trace_evidence,
            cited_reference_trace_evidence=proposal.cited_reference_trace_evidence,
            score_source="configured_judge" if correction is None else "human_correction",
            judgment_source=(
                "human_correction"
                if correction is not None and correction.corrected_judgment is not None
                else "configured_judge"
            ),
        )
        axes.append(
            ManualJudgeAxisReview(
                dimension_id=dimension.dimension_id,
                judge_proposal=proposal,
                human_correction=correction,
                final_accepted_label=final,
            )
        )
    calls_per_trace = 2 if setup.prompt_template.response_shape == "pairwise" else 1
    per_attempt = (
        budget.maximum_input_tokens_per_call * budget.input_usd_per_million_tokens
        + budget.maximum_output_tokens_per_call * budget.output_usd_per_million_tokens
    ) / 1_000_000
    pricing = ManualJudgeReviewPricing(
        input_usd_per_million_tokens=budget.input_usd_per_million_tokens,
        output_usd_per_million_tokens=budget.output_usd_per_million_tokens,
        pricing_source=budget.pricing_source,
        maximum_input_tokens_per_call=budget.maximum_input_tokens_per_call,
        maximum_output_tokens_per_call=budget.maximum_output_tokens_per_call,
        maximum_attempts_per_call=budget.maximum_attempts_per_call,
        authorized_call_count=calls_per_trace,
        maximum_reserved_cost_usd=(
            per_attempt * budget.maximum_attempts_per_call * calls_per_trace
        ),
        observed_economics=judgment.judge_economics or OperationEconomics(),
    )
    inputs = tuple(
        sorted(
            (
                setup_input,
                rollout_input,
                *((reference_input,) if reference_input is not None else ()),
                setup.rubric,
                provisional_input,
                *probes,
                judgment_input,
            ),
            key=lambda item: item.artifact_id,
        )
    )
    provenance = ManualJudgeReviewProvenance(
        historical_source=(
            "provider_free_production_trace"
            if rollout.provider_free_source is not None
            else "recorded_model"
        )
    )
    review_id = stable_id(
        "manual-judge-trace-review",
        {
            "setup": setup_input.model_dump(mode="json"),
            "sample_sha256": sample_sha256,
            "trace_id": trace.trace_id,
            "reference_trace_id": (
                reference_trace.trace_id if reference_trace is not None else None
            ),
            "lineage_id": task.lineage_group_id,
            "inputs": [item.model_dump(mode="json") for item in inputs],
            "axes": [item.model_dump(mode="json") for item in axes],
            "judge_model": judgment.judge_model.model_dump(mode="json"),
            "pricing": pricing.model_dump(mode="json"),
            "provenance": provenance.model_dump(mode="json"),
            "code_revision": code_revision,
        },
    )
    return ManualJudgeTraceReviewArtifact(
        schema_version=1,
        created_at=created_at,
        inputs=inputs,
        code_revision=code_revision,
        review_id=review_id,
        setup=setup_input,
        sample_sha256=sample_sha256,
        trace_id=trace.trace_id,
        reference_trace_id=(reference_trace.trace_id if reference_trace is not None else None),
        lineage_id=task.lineage_group_id,
        trace_evidence=rollout_input,
        reference_trace_evidence=reference_input,
        rubric_revision=setup.rubric,
        provisional_calibration=provisional_input,
        original_judge_response=probes,
        normalized_judgment=judgment_input,
        judge_model=judgment.judge_model,
        pricing=pricing,
        axes=tuple(axes),
        provenance=provenance,
        reviewed_at=created_at,
    )


def _write_trace_review(
    store: ProjectStore, review: ManualJudgeTraceReviewArtifact
) -> ArtifactInput:
    """Persist or verify one immutable completed trace review.

    Args:
        store: Project-local immutable artifact store.
        review: Completed trace review to persist.

    Returns:
        Exact manifest pointer for the persisted review.

    Raises:
        ManualJudgeError: An existing artifact conflicts with the review.
    """
    try:
        manifest = store.artifacts.write_json(
            artifact_id=review.review_id,
            artifact_type="manual-judge-trace-review",
            envelope=review,
            files={"review.json": review},
        )
    except ArtifactAlreadyExistsError:
        saved, saved_input = _read_trace_review_with_input(store, review.review_id)
        if saved != review.model_copy(
            update={"created_at": saved.created_at, "reviewed_at": saved.reviewed_at}
        ):
            raise ManualJudgeError(
                "existing trace review conflicts with this human decision"
            ) from None
        return saved_input
    return artifact_input(manifest)


def _publish_trace_review(
    store: ProjectStore,
    setup_input: ArtifactInput,
    review_input: ArtifactInput,
    review: ManualJudgeTraceReviewArtifact,
) -> None:
    """Append one immutable review pointer without replacing concurrent progress.

    Args:
        store: Project-local artifact and mutable review store.
        setup_input: Exact finalized setup pointer.
        review_input: Newly completed immutable review pointer.
        review: Completed review used to detect conflicting decisions.

    Raises:
        ManualJudgeError: The same sample trace already has a different review.
    """

    def mutate(current: ManualJudgeReviewState) -> ManualJudgeReviewState:
        """Retain all reviews and reject another decision for the same sample trace.

        Args:
            current: Latest review state read while holding the project lock.

        Returns:
            State containing the new pointer, or unchanged idempotent state.

        Raises:
            ManualJudgeError: Another review claims the same frozen sample trace.
        """
        for expected in current.trace_reviews:
            saved = _read_trace_review(store, expected)
            same_key = (
                saved.sample_sha256 == review.sample_sha256
                and saved.trace_id == review.trace_id
                and saved.reference_trace_id == review.reference_trace_id
            )
            if same_key and expected != review_input:
                raise ManualJudgeError("calibration trace already has a different human review")
            if expected == review_input:
                return current
        return current.model_copy(update={"trace_reviews": (*current.trace_reviews, review_input)})

    update_review_state(store, setup_input, mutate)


def _read_trace_review(
    store: ProjectStore, expected: ArtifactInput
) -> ManualJudgeTraceReviewArtifact:
    """Read one review and require its exact persisted manifest pointer.

    Args:
        store: Project-local immutable artifact store.
        expected: Manifest pointer retained by review state or an audit.

    Returns:
        Verified immutable trace review.

    Raises:
        ManualJudgeError: The review is missing, malformed, or changed.
    """
    review, _review_input = _read_trace_review_with_input(
        store,
        expected.artifact_id,
        expected=expected,
    )
    return review


def _read_trace_review_with_input(
    store: ProjectStore,
    review_id: str,
    *,
    expected: ArtifactInput | None = None,
) -> tuple[ManualJudgeTraceReviewArtifact, ArtifactInput]:
    """Read one immutable trace review through the shared provenance verifier.

    Args:
        store: Project-local immutable artifact store.
        review_id: Trace review artifact identity.
        expected: Optional exact manifest pointer required by the caller.

    Returns:
        Verified trace review and its canonical manifest pointer.

    Raises:
        ManualJudgeError: The review cannot be verified or has the wrong identity.
    """
    try:
        review, review_input = read_artifact_json(
            store,
            artifact_id=review_id,
            expected_artifact_type="manual-judge-trace-review",
            relative_path="review.json",
            model_type=ManualJudgeTraceReviewArtifact,
            expected_input=expected,
        )
    except JudgingProvenanceError as exc:
        raise ManualJudgeError("completed manual judge trace review is unavailable") from exc
    if review.review_id != review_id:
        raise ManualJudgeError("manual judge trace review has the wrong artifact identity")
    return review, review_input


def _require_review_matches_plan(
    review: ManualJudgeTraceReviewArtifact,
    *,
    setup: ManualJudgeSetupArtifact,
    task: TaskCase,
    rollout_input: ArtifactInput,
    reference_input: ArtifactInput | None,
    provisional_input: ArtifactInput,
    rubric: Rubric,
    judge_model: ModelSnapshot,
) -> None:
    """Reject persisted progress that differs from the current frozen plan.

    Args:
        review: Persisted completed review being resumed.
        setup: Current finalized judge setup.
        task: Expected canonical task and lineage.
        rollout_input: Expected target rollout pointer.
        reference_input: Expected optional comparison rollout pointer.
        provisional_input: Expected provisional calibration pointer.
        rubric: Current finalized rubric revision.
        judge_model: Exact configured judge identity.

    Raises:
        ManualJudgeError: Any immutable plan binding differs.
    """
    expected_dimensions = tuple(item.dimension_id for item in rubric.dimensions)
    reviewed_dimensions = tuple(item.dimension_id for item in review.axes)
    if (
        review.rubric_revision != setup.rubric
        or review.trace_evidence != rollout_input
        or review.reference_trace_evidence != reference_input
        or review.provisional_calibration != provisional_input
        or review.lineage_id != task.lineage_group_id
        or review.judge_model != judge_model
        or reviewed_dimensions != expected_dimensions
    ):
        raise ManualJudgeError("completed trace review differs from the frozen calibration plan")
    for axis_review in review.axes:
        axis = rubric.axis(axis_review.dimension_id)
        scores = [
            axis_review.judge_proposal.proposed_score,
            axis_review.final_accepted_label.score,
        ]
        if axis_review.human_correction is not None:
            scores.append(axis_review.human_correction.corrected_score)
        if any(not axis.contains_score(score) for score in scores):
            raise ManualJudgeError("completed trace review score differs from its rubric range")


def review_evidence(
    reviews: Sequence[ManualJudgeTraceReviewArtifact],
) -> tuple[JudgeRunEvidence, ...]:
    """Project completed trace reviews into the existing audit evidence contract.

    Args:
        reviews: Completed immutable reviews in frozen trace order.

    Returns:
        Existing rollout, judgment, and probe evidence projections.
    """
    return tuple(
        JudgeRunEvidence(
            rollout=review.trace_evidence,
            reference_rollout=review.reference_trace_evidence,
            judgment=review.normalized_judgment,
            probes=review.original_judge_response,
        )
        for review in reviews
    )


def read_review_judgment(store: ProjectStore, review: ManualJudgeTraceReviewArtifact) -> Judgment:
    """Load the normalized immutable judge response named by one completed review.

    Args:
        store: Project-local immutable artifact store.
        review: Completed review naming the normalized judgment.

    Returns:
        Verified normalized configured-judge response.

    Raises:
        ManualJudgeError: The judgment is unavailable or differs from the proposal fields.
    """
    try:
        judgment, _judgment_input = read_artifact_json(
            store,
            artifact_id=review.normalized_judgment.artifact_id,
            expected_artifact_type="judgment",
            relative_path="judgment.json",
            model_type=Judgment,
            expected_input=review.normalized_judgment,
        )
    except JudgingProvenanceError as exc:
        raise ManualJudgeError("trace review normalized judgment is unavailable") from exc
    citation_by_id: dict[str, tuple[tuple[str, ...], tuple[str, ...]]] = {}
    if review.reference_trace_evidence is not None:
        rollout = read_rollout(store, review.trace_evidence)
        reference = read_rollout(store, review.reference_trace_evidence)
        citation_by_id = {
            dimension_id: (target, reference_spans)
            for dimension_id, target, reference_spans in pairwise_citation_evidence_from_probes(
                store,
                review.original_judge_response,
                rollout,
                reference,
            )
        }
        if set(citation_by_id) != {item.dimension_id for item in judgment.dimensions}:
            raise ManualJudgeError("pairwise probe citations differ from normalized judgment")
    proposals: list[JudgeAxisProposal] = []
    for item in judgment.dimensions:
        target_evidence, reference_evidence = citation_by_id.get(item.dimension_id, ((), ()))
        proposals.append(
            JudgeAxisProposal(
                dimension_id=item.dimension_id,
                proposed_score=item.raw_score,
                proposed_judgment=item.rationale or "",
                cited_trace_evidence=target_evidence,
                cited_reference_trace_evidence=reference_evidence,
            )
        )
    if (
        judgment.judge_model != review.judge_model
        or judgment.calibration_id != review.provisional_calibration.artifact_id
        or tuple(item.judge_proposal for item in review.axes) != tuple(proposals)
    ):
        raise ManualJudgeError("trace review does not match its immutable judge response")
    return judgment


def reviewer_from_labels(
    setup: ManualJudgeSetupArtifact,
    labels: Sequence[ManualJudgeLabel],
) -> ManualJudgeReviewer:
    """Adapt explicit scores into truthful judge-first review decisions.

    A score equal to the proposal is an acceptance. A differing score is retained as a human
    correction with no invented human judgment text, so the final judgment remains attributed to
    the configured judge.

    Args:
        setup: Finalized prompt and score projection contract.
        labels: Complete validated explicit human labels.

    Returns:
        Reviewer that consumes the labels only after each judge proposal exists.
    """
    by_key = {(item.trace_id, item.reference_trace_id, item.dimension_id): item for item in labels}

    def review(proposal: ManualJudgeTraceProposal) -> tuple[ManualJudgeAxisDecision, ...]:
        """Return one acceptance or score-only correction per proposed axis.

        Args:
            proposal: Immutable configured-judge proposal for one trace.

        Returns:
            Explicit decisions ordered to match the proposal axes.
        """
        reference_id = (
            proposal.reference_trace.trace_id if proposal.reference_trace is not None else None
        )
        decisions: list[ManualJudgeAxisDecision] = []
        for dimension in proposal.judgment.dimensions:
            key = (proposal.trace.trace_id, reference_id, dimension.dimension_id)
            label = by_key[key]
            score = manual_label_score(setup, label)
            decisions.append(
                ManualJudgeAxisDecision(
                    dimension_id=dimension.dimension_id,
                    accepted=score == dimension.raw_score,
                    correction=(
                        None
                        if score == dimension.raw_score
                        else HumanJudgeCorrection(corrected_score=score)
                    ),
                )
            )
        return tuple(decisions)

    return review


def manual_label_score(setup: ManualJudgeSetupArtifact, label: ManualJudgeLabel) -> int:
    """Project one scalar or pairwise human label onto the finalized axis range.

    Args:
        setup: Finalized prompt and score projection contract.
        label: Validated human label.

    Returns:
        Integer score used by grouped calibration.

    Raises:
        ManualJudgeError: A human label lacks both a direct score and a winner.
    """
    if label.score is not None:
        return label.score
    if label.winner is None:
        raise ManualJudgeError("human labels require an axis-range score or pairwise winner")
    return setup.prompt_template.score_projection.pairwise_scores[label.winner]


def labels_from_reviews(
    reviews: Sequence[ManualJudgeTraceReviewArtifact],
) -> tuple[ManualJudgeLabel, ...]:
    """Project human-authorized review results into the existing label-set boundary.

    Args:
        reviews: Completed immutable reviews in frozen trace order.

    Returns:
        Final accepted axis-range scores for grouped calibration.
    """
    return tuple(
        ManualJudgeLabel(
            trace_id=review.trace_id,
            reference_trace_id=review.reference_trace_id,
            dimension_id=axis.dimension_id,
            score=axis.final_accepted_label.score,
        )
        for review in reviews
        for axis in review.axes
    )


def ordered_completed_reviews(
    reviews: Sequence[ManualJudgeTraceReviewArtifact],
    *,
    setup: ManualJudgeSetupArtifact,
    setup_input: ArtifactInput,
    tasks: Sequence[TaskCase],
    traces: Sequence[Trace],
    reference_traces: Sequence[Trace | None],
    rollout_inputs: Sequence[ArtifactInput],
    reference_inputs: Sequence[ArtifactInput | None],
    provisional_input: ArtifactInput,
    rubric: Rubric,
) -> tuple[ManualJudgeTraceReviewArtifact, ...]:
    """Order and verify a completed sample without resolving a provider.

    Args:
        reviews: Verified trace reviews loaded from resumable state.
        setup: Finalized judge setup.
        setup_input: Exact finalized setup pointer.
        tasks: Frozen distinct-lineage tasks.
        traces: Frozen target traces.
        reference_traces: Optional pairwise traces.
        rollout_inputs: Persisted target rollout pointers.
        reference_inputs: Optional pairwise rollout pointers.
        provisional_input: Exact scoring calibration pointer.
        rubric: Exact finalized rubric revision.

    Returns:
        Reviews in frozen trace order.

    Raises:
        ManualJudgeError: A frozen trace is missing, unexpected, or bound differently.
    """
    by_key = {(item.trace_id, item.reference_trace_id): item for item in reviews}
    ordered: list[ManualJudgeTraceReviewArtifact] = []
    for task, trace, reference, rollout_input, reference_input in zip(
        tasks,
        traces,
        reference_traces,
        rollout_inputs,
        reference_inputs,
        strict=True,
    ):
        reference_id = reference.trace_id if reference is not None else None
        review = by_key.get((trace.trace_id, reference_id))
        if review is None:
            raise ManualJudgeError("completed review state is missing a frozen calibration trace")
        if review.setup != setup_input:
            raise ManualJudgeError("completed trace review differs from finalized setup")
        _require_review_matches_plan(
            review,
            setup=setup,
            task=task,
            rollout_input=rollout_input,
            reference_input=reference_input,
            provisional_input=provisional_input,
            rubric=rubric,
            judge_model=setup.judge_model,
        )
        ordered.append(review)
    if len(ordered) != len(reviews):
        raise ManualJudgeError("completed review state contains traces outside the frozen sample")
    return tuple(ordered)
