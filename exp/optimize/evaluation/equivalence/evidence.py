"""Project saved executions into paired evidence without inventing state or reference truth."""

from typing import Literal

from exp.common.core.artifacts import ContractModel, JsonObject, JsonValue, Sha256, sha256_json
from exp.common.evaluations.actor_equivalence import (
    RUBRIC_DIMENSIONS,
    ActorComparison,
    ActorEquivalenceReport,
    ActorExecutionAssessment,
)
from exp.common.evaluations.reconstruction import WorkflowEvidence, WorkflowStep
from exp.common.evaluations.uncertainty import (
    missing_outcome_bounds,
    unit_difference_summary,
    unit_mean_summary,
)
from exp.common.models import ModelRequest
from exp.optimize.evaluation.continuations.study import ContinuationBranch
from exp.optimize.evaluation.equivalence.dimensions import PairDimension, pair_dimensions
from exp.runtime.environments.interface import Observation
from exp.simulation.continuations import ContinuationCheckpoint
from exp.simulation.evaluation.tasks import TaskVerifier, grade
from exp.simulation.reconstruction.models import resolve


class PairMeasurement(ContractModel):
    """Separate mechanical agreement from semantic review and task competence.

    Attributes:
        case: Registered actor/task pair index.
        repeat: Registered branch index.
        condition: Simulator comparison or repeated-reference variability control.
        comparison: Exact observed pair and per-side task checks.
        behavior_assessable: Both executions completed with only valid steps from the same
            first actor request and equal declared initial probes, so their behavior can be
            judged even when declared outcomes or final state are unavailable.
        material_agreement: Equality on declared outcomes and consequences, or unavailable.
        differences: Failed or missing prerequisites and differing material properties.
        judgment: Optional semantic ratings, never an independent truth certificate.
        dimensions: Every admission dimension observed on both sides, in policy order. These
            are measurements; only a registered policy turns them into calibration verdicts.
    """

    case: int
    repeat: int
    condition: Literal["candidate", "reference_control"]
    comparison: ActorComparison
    behavior_assessable: bool
    material_agreement: Literal["match", "different", "unassessable"]
    differences: tuple[str, ...] = ()
    judgment: ActorEquivalenceReport | None = None
    dimensions: tuple[PairDimension, ...] = ()


def _object(value: JsonObject, key: str) -> JsonObject | None:
    """Retain missing state explicitly instead of manufacturing an empty observed state."""
    item = value.get(key)
    return item if isinstance(item, dict) else None


def workflow(
    branch: ContinuationBranch, checkpoint: ContinuationCheckpoint, identity: str
) -> WorkflowEvidence:
    """Use actual actions, observations and private probes from the saved chronology."""
    raw_steps = branch.outcome.get("steps", [])
    if not isinstance(raw_steps, list):
        raise ValueError("paired execution has a malformed step inventory")
    before = _object(branch.outcome, "initial_state")
    steps = []
    audit: JsonObject = {}
    for index, row in enumerate(raw_steps):
        if not isinstance(row, dict) or not isinstance(row.get("action"), dict):
            raise ValueError("paired execution has a malformed action")
        after = _object(row, "after")
        observed = None if row.get("reply") is None else Observation.model_validate(row["reply"])
        audit[str(index)] = {
            "observation_metadata": None if observed is None else observed.metadata,
            "execution": row.get("execution"),
        }
        steps.append(
            WorkflowStep.model_validate(
                {
                    "step": index,
                    "action": row["action"],
                    "status": row.get("status", "unknown"),
                    "reply": None
                    if observed is None
                    else {"content": observed.content, "is_error": observed.is_error},
                    "before": before,
                    "after": after,
                }
            )
        )
        before = after
    return WorkflowEvidence(
        workflow_id=identity,
        task=checkpoint.task.instruction,
        task_context={
            "messages": [row.model_dump(mode="json") for row in checkpoint.actor.messages],
            "tools": [row.model_dump(mode="json") for row in checkpoint.task.tools],
        },
        initial_state=_object(branch.outcome, "initial_state") or {},
        steps=tuple(steps),
        audit_metadata={"steps": audit},
        final_answer=branch.outcome.get("final_answer"),
        stop_reason=str(branch.grading_outcome().get("stop_reason", "unknown")),
    )


def assessment(branch: ContinuationBranch, verifier: TaskVerifier) -> ActorExecutionAssessment:
    """Recompute goal and policy from evidence; never trust a stored success flag."""
    checks = grade(verifier, branch.grading_outcome())
    return ActorExecutionAssessment(
        execution="completed"
        if branch.status == "completed" and checks.complete
        else "unknown"
        if branch.status == "outcome_unknown"
        else "incomplete",
        task_result=checks.status,
        checks=checks.model_dump(mode="json"),
    )


def _probe(state: JsonObject | None, path: str) -> tuple[JsonValue] | None:
    """Resolve one declared probe, keeping an unavailable value distinct from JSON null."""
    if state is None:
        return None
    try:
        return (resolve(state, path),)
    except ValueError:
        return None


def _initial_request(branch: ContinuationBranch) -> ModelRequest | None:
    """Return the first saved actor request, or None when the execution never made one."""
    calls = branch.outcome.get("model_calls")
    if not isinstance(calls, list) or not calls or not isinstance(calls[0], dict):
        return None
    return ModelRequest.model_validate(calls[0].get("request"))


def measure_pair(
    reference: ContinuationBranch,
    candidate: ContinuationBranch,
    *,
    checkpoint: ContinuationCheckpoint,
    actor_sha256: Sha256,
    verifier: TaskVerifier,
    initial_paths: tuple[str, ...],
    final_paths: tuple[str, ...],
    case: int,
    repeat: int,
    condition: Literal["candidate", "reference_control"],
    identity: str,
) -> PairMeasurement:
    """Require matching initial evidence and compare declared consequences without a judge.

    Missing reference evidence, mismatched initial evidence or an outcome-unknown candidate
    leaves the pair unassessable. A known candidate execution failure against a completed
    reference is a material difference: the candidate evidence it could not produce is
    explained by that failure, so it cannot hide the pair among unknowns.

    Behavior stays assessable without final-state probes or outcome predicates: only an
    unavailable or different initial probe, a different first actor request, an incomplete
    execution or a non-valid step makes the paired behavior unassessable.
    """
    left, right = assessment(reference, verifier), assessment(candidate, verifier)
    differences: list[str] = []
    incomparable = candidate_missing = False
    behavior_assessable = left.execution == "completed" and right.execution == "completed"
    for field, paths in (("initial_state", initial_paths), ("final_state", final_paths)):
        a, b = _object(reference.outcome, field), _object(candidate.outcome, field)
        for path in paths:
            x, y = _probe(a, path), _probe(b, path)
            if x is None or y is None:
                differences.append(f"unavailable:{field}{path}")
                incomparable = incomparable or x is None
                candidate_missing = candidate_missing or y is None
                behavior_assessable = behavior_assessable and field != "initial_state"
            elif sha256_json(x[0]) != sha256_json(y[0]):
                differences.append(f"different:{field}{path}")
                incomparable = incomparable or field == "initial_state"
                behavior_assessable = behavior_assessable and field != "initial_state"
    requests = [_initial_request(branch) for branch in (reference, candidate)]
    if requests[0] is None or requests[0] != requests[1]:
        behavior_assessable = False
        differences.append("unavailable:matching_initial_actor_requests")
        incomparable = incomparable or requests[0] is None or requests[1] is not None
        candidate_missing = candidate_missing or requests[1] is None
    failed = left.execution == "completed" and candidate.status == "failed"
    if failed:
        differences.append("different:completed_execution")
    elif left.execution != "completed" or right.execution != "completed":
        differences.append("unavailable:completed_execution")
        incomparable = True
    for key in ("goal", "policy"):
        a, b = left.checks.get(key), right.checks.get(key)
        if not isinstance(a, dict) or a.get("status") == "unassessable":
            differences.append("unavailable:" + key)
            incomparable = True
        elif not isinstance(b, dict) or b.get("status") == "unassessable":
            differences.append("unavailable:" + key)
            candidate_missing = True
        elif a.get("status") != b.get("status"):
            differences.append("different:" + key)
    comparison = ActorComparison(
        comparison_id=identity,
        actor_sha256=actor_sha256,
        reference=workflow(reference, checkpoint, identity + "-reference"),
        candidate=workflow(candidate, checkpoint, identity + "-candidate"),
        reference_assessment=left,
        candidate_assessment=right,
    )
    behavior_assessable = behavior_assessable and all(
        step.status == "valid"
        for trace in (comparison.reference, comparison.candidate)
        for step in trace.steps
    )
    agreement: Literal["match", "different", "unassessable"] = (
        "unassessable"
        if incomparable or (candidate_missing and not failed)
        else "different"
        if failed or differences
        else "match"
    )
    return PairMeasurement(
        case=case,
        repeat=repeat,
        condition=condition,
        comparison=comparison,
        behavior_assessable=behavior_assessable,
        material_agreement=agreement,
        differences=tuple(differences),
        dimensions=pair_dimensions(
            reference,
            candidate,
            verifier=verifier,
            tools=checkpoint.task.tools,
            material_agreement=agreement,
        ),
    )


def equivalence_metrics(measurements: tuple[PairMeasurement, ...]) -> JsonObject:
    """Average repeats within each case first, then weigh every case equally.

    The headline per condition is the equal-weight case mean of known material agreement with a
    Wilson interval at the case count, beside case-averaged missing-outcome bounds. The candidate
    is read against the repeated-reference control: their per-case difference measures how far
    the candidate falls short of the reference's own repeat agreement. Pair counts are retained
    as an inventory, never as the estimate.
    """
    cases = sorted({row.case for row in measurements})
    known: dict[str, dict[str, list[float]]] = {}
    result: JsonObject = {
        "unit": "case",
        "aggregation": "repeats averaged within each case first; every case weighs equally",
    }
    for condition in ("candidate", "reference_control"):
        rows = tuple(row for row in measurements if row.condition == condition)
        known[condition] = {
            str(case): [
                float(row.material_agreement == "match")
                for row in rows
                if row.case == case and row.material_agreement != "unassessable"
            ]
            for case in cases
        }
        bounds = [
            _bounds(tuple(row for row in rows if row.case == case))
            for case in cases
            if any(row.case == case for row in rows)
        ]
        result[condition] = {
            **_counts(rows),
            "agreement": unit_mean_summary(known[condition]),
            "agreement_bounds": [
                sum(low for low, _ in bounds) / len(bounds),
                sum(high for _, high in bounds) / len(bounds),
            ]
            if bounds
            else None,
        }
    result["candidate_minus_control"] = unit_difference_summary(
        known["candidate"], known["reference_control"]
    )
    result["by_case"] = {
        str(case): _case_metrics(tuple(row for row in measurements if row.case == case))
        for case in cases
    }
    result["interpretation"] = (
        "declared material outcomes only; agreement intervals use known pairs at the case count; "
        "bounds assign unavailable pairs both mismatch and match; candidate_minus_control "
        "subtracts repeated-reference agreement within each case; not population equivalence "
        "or training admission; judge ratings separate"
    )
    return result


def _bounds(rows: tuple[PairMeasurement, ...]) -> tuple[float, float]:
    """Bound one case's agreement under each possible outcome of its unavailable pairs."""
    matches = sum(row.material_agreement == "match" for row in rows)
    unknown = sum(row.material_agreement == "unassessable" for row in rows)
    bounds = missing_outcome_bounds(matches, unknown, len(rows))
    if bounds is None:
        raise ValueError("a measured case has at least one planned pair")
    return bounds[0], bounds[1]


def _case_metrics(rows: tuple[PairMeasurement, ...]) -> JsonObject:
    """Keep one case's pair inventory, bounds and candidate-minus-control agreement."""
    result: JsonObject = {}
    means: dict[str, float | None] = {}
    for condition in ("candidate", "reference_control"):
        group = tuple(row for row in rows if row.condition == condition)
        known = [
            row.material_agreement == "match"
            for row in group
            if row.material_agreement != "unassessable"
        ]
        means[condition] = sum(known) / len(known) if known else None
        result[condition] = {
            **_counts(group),
            "agreement": means[condition],
            "agreement_bounds": list(_bounds(group)) if group else None,
        }
    candidate, control = means["candidate"], means["reference_control"]
    result["candidate_minus_control"] = (
        None if candidate is None or control is None else candidate - control
    )
    return result


def _counts(rows: tuple[PairMeasurement, ...]) -> JsonObject:
    """Count every planned pair and distinguish matching failures from matching successes."""
    unknown = sum(row.material_agreement == "unassessable" for row in rows)
    matches = sum(row.material_agreement == "match" for row in rows)
    matrix: dict[str, int] = {}
    for row in rows:
        pair = row.comparison
        key = pair.reference_assessment.task_result + "/" + pair.candidate_assessment.task_result
        matrix[key] = matrix.get(key, 0) + 1
    return {
        "planned_pairs": len(rows),
        "behavior_assessable_pairs": sum(row.behavior_assessable for row in rows),
        "behavior_unassessable_pairs": sum(not row.behavior_assessable for row in rows),
        "rubric_dimensions": {
            dimension: _dimension_metrics(rows, dimension) for dimension in RUBRIC_DIMENSIONS
        },
        "material_matches": matches,
        "material_differences": len(rows) - matches - unknown,
        "unassessable": unknown,
        "task_result_matrix_reference_then_other": matrix,
        "both_task_pass": sum(
            row.material_agreement != "unassessable"
            and row.comparison.reference_assessment.task_result == "pass"
            and row.comparison.candidate_assessment.task_result == "pass"
            for row in rows
        ),
    }


def _dimension_metrics(rows: tuple[PairMeasurement, ...], dimension: str) -> JsonObject:
    """Report one judged criterion, keeping unavailable and failed judgments in its counts.

    The mean is a 1 to 5 rating, not an accuracy: repeats are averaged within each case first
    and every case with an assessed rating weighs equally. No interval is reported because a
    bounded rating mean is not a proportion.

    Args:
        rows: Planned pairs of one condition, or of one case.
        dimension: Frozen rubric reporting dimension.

    Returns:
        Assessed, unassessable, judge-failure and unjudged counts and the case-weighted mean.
    """
    axes = [
        (row.case, axis)
        for row in rows
        if row.judgment is not None
        for axis in row.judgment.axes
        if axis.dimension_id == dimension
    ]
    scores: dict[int, list[int]] = {}
    for case, axis in axes:
        if axis.status == "assessed" and axis.score is not None:
            scores.setdefault(case, []).append(axis.score)
    means = [sum(values) / len(values) for values in scores.values()]
    return {
        "planned_pairs": len(rows),
        "assessed": sum(len(values) for values in scores.values()),
        "unassessable": sum(axis.status == "unassessable" for _, axis in axes),
        "judge_failure": sum(axis.status == "judge_failure" for _, axis in axes),
        "not_judged": len(rows) - len(axes),
        "cases_assessed": len(means),
        "case_mean_score_1_to_5_not_accuracy": sum(means) / len(means) if means else None,
    }
