"""Project saved executions into paired evidence without inventing state or reference truth."""

from typing import Literal

from exp.common.core.artifacts import ContractModel, JsonObject, Sha256, sha256_json
from exp.common.evaluations.actor_equivalence import (
    ActorComparison,
    ActorEquivalenceReport,
    ActorExecutionAssessment,
)
from exp.common.evaluations.reconstruction import WorkflowEvidence, WorkflowStep
from exp.common.models import ModelRequest
from exp.optimize.evaluation.continuation_study import ContinuationBranch
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
        material_agreement: Equality on declared outcomes and consequences, or unavailable.
        differences: Failed or missing prerequisites and differing material properties.
        judgment: Optional semantic ratings, never an independent truth certificate.
    """

    case: int
    repeat: int
    condition: Literal["candidate", "reference_control"]
    comparison: ActorComparison
    material_agreement: Literal["match", "different", "unassessable"]
    differences: tuple[str, ...] = ()
    judgment: ActorEquivalenceReport | None = None


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
        stop_reason=str(branch.outcome.get("stop_reason", "unknown")),
    )


def assessment(branch: ContinuationBranch, verifier: TaskVerifier) -> ActorExecutionAssessment:
    """Recompute goal and policy from evidence; never trust a stored success flag."""
    checks = grade(verifier, branch.outcome)
    return ActorExecutionAssessment(
        execution="completed"
        if branch.status == "completed" and checks.complete
        else "unknown"
        if branch.status == "outcome_unknown"
        else "incomplete",
        task_result=checks.status,
        checks=checks.model_dump(mode="json"),
    )


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
    """Require matching initial evidence and compare declared consequences without a judge."""
    left, right = assessment(reference, verifier), assessment(candidate, verifier)
    differences, missing = [], False
    for field, paths in (("initial_state", initial_paths), ("final_state", final_paths)):
        for path in paths:
            a, b = _object(reference.outcome, field), _object(candidate.outcome, field)
            try:
                if a is None or b is None:
                    raise ValueError("state unavailable")
                x, y = resolve(a, path), resolve(b, path)
            except ValueError:
                differences.append(f"unavailable:{field}{path}")
                missing = True
            else:
                if sha256_json(x) != sha256_json(y):
                    differences.append(f"different:{field}{path}")
                    if field == "initial_state":
                        missing = True
    requests = []
    for branch in (reference, candidate):
        calls = branch.outcome.get("model_calls")
        if not isinstance(calls, list) or not calls or not isinstance(calls[0], dict):
            requests.append(None)
        else:
            requests.append(ModelRequest.model_validate(calls[0].get("request")))
    if requests[0] is None or requests[0] != requests[1]:
        differences.append("unavailable:matching_initial_actor_requests")
        missing = True
    if left.execution != "completed" or right.execution != "completed":
        differences.append("unavailable:completed_execution")
        missing = True
    for key in ("goal", "policy"):
        a, b = left.checks.get(key), right.checks.get(key)
        if (
            not isinstance(a, dict)
            or not isinstance(b, dict)
            or "unassessable" in (a.get("status"), b.get("status"))
        ):
            differences.append("unavailable:" + key)
            missing = True
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
    return PairMeasurement(
        case=case,
        repeat=repeat,
        condition=condition,
        comparison=comparison,
        material_agreement="unassessable" if missing else "different" if differences else "match",
        differences=tuple(differences),
    )


def equivalence_metrics(measurements: tuple[PairMeasurement, ...]) -> JsonObject:
    """Keep every planned pair and distinguish matching failures from matching successes."""
    result: JsonObject = {}
    for condition in ("candidate", "reference_control"):
        rows = tuple(row for row in measurements if row.condition == condition)
        total = len(rows)
        unknown = sum(row.material_agreement == "unassessable" for row in rows)
        matches = sum(row.material_agreement == "match" for row in rows)
        matrix: dict[str, int] = {}
        for row in rows:
            pair = row.comparison
            key = (
                pair.reference_assessment.task_result + "/" + pair.candidate_assessment.task_result
            )
            matrix[key] = matrix.get(key, 0) + 1
        result[condition] = {
            "planned_pairs": total,
            "material_matches": matches,
            "material_differences": total - matches - unknown,
            "unassessable": unknown,
            "agreement_bounds": [matches / total, (matches + unknown) / total] if total else None,
            "task_result_matrix_reference_then_other": matrix,
            "both_task_pass": sum(
                row.material_agreement != "unassessable"
                and row.comparison.reference_assessment.task_result == "pass"
                and row.comparison.candidate_assessment.task_result == "pass"
                for row in rows
            ),
        }
    result["interpretation"] = (
        "declared material outcomes only; bounds assign unavailable pairs both mismatch and match; "
        "not confidence intervals, population equivalence or training admission; "
        "judge ratings separate"
    )
    return result
