"""Mechanical failure localization from immutable continuation evidence.

A failed goal does not identify its cause. Tool errors can follow invalid actor requests or
broken environments; observed state violations can originate in a simulated transition. This
report preserves those distinctions and supplies saved actor-turn boundaries for subsequent
controlled experiments, without awarding blame or admitting training data.
"""

from datetime import datetime
from typing import Literal, TypedDict

from exp.common.core.artifacts import (
    ArtifactEnvelope,
    ArtifactInput,
    ContractModel,
    JsonObject,
    sha256_json,
    stable_id,
)
from exp.common.models import ModelRequest, ModelResponse, ToolCall
from exp.common.project import ArtifactStore, artifact_input
from exp.optimize.evaluation.continuations.records import load_study
from exp.optimize.evaluation.continuations.study import ContinuationBranch, ContinuationStudy
from exp.runtime.environments.world_outcomes import WorldExecutionReceipt
from exp.simulation.continuations import load_continuation_checkpoint
from exp.simulation.development_scope import require_development_task
from exp.simulation.evaluation.tasks import VerifierGrade, grade


class _Location(TypedDict):
    """Typed location shared by every symptom of one attempted action."""

    step: int
    actor_call: int | None
    preceding_tools: int | None
    checkpoint: ArtifactInput | None


class FailureSignal(ContractModel):
    """Observed symptom with its available rewind boundary, never inferred causal blame.

    Attributes:
        kind: Mechanical evidence category, independent of the branch's final verdict.
        reason: Meaning and limitations of this observation.
        step: Zero-based executed tool position, absent for a terminal/operational symptom.
        actor_call: Zero-based model-call position within this continuation, when recoverable.
        checkpoint: Exact saved pre-response boundary, or None if unavailable.
        preceding_tools: Earlier tools in the same response; rewinding also redoes these tools.
        repair_destination: Mechanical repair routing when execution evidence establishes it,
            otherwise unresolved. This does not establish a mistaken actor decision.
    """

    kind: Literal[
        "policy_violation",
        "tool_error",
        "execution_incomplete",
        "goal_failed",
        "unassessable",
        "simulator_execution_failure",
        "dispatch_refused",
        "execution_rolled_back",
    ]
    reason: str
    repair_destination: Literal["simulator", "interface", "execution_recovery", "unresolved"] = (
        "unresolved"
    )
    step: int | None = None
    actor_call: int | None = None
    checkpoint: ArtifactInput | None = None
    preceding_tools: int | None = None


class BranchFailureAnalysis(ContractModel):
    """One planned branch, including passing and unavailable outcomes.

    Attributes:
        index: Original branch position.
        assessment: Recomputed declared goal and policy, not the cached verdict.
        signals: All observed symptoms, without selecting a convenient first failure.
        attribution: Causal attribution is not established by these observations.
    """

    index: int
    assessment: VerifierGrade
    signals: tuple[FailureSignal, ...]
    attribution: Literal["unresolved"] = "unresolved"


class FailureAnalysis(ContractModel):
    """Source-bound analysis that can seed experiments but cannot authorize training.

    Attributes:
        study: Complete source result, retaining its registered denominator.
        branches: Every planned branch in original order.
        scope: Explicit interpretation of checkpoint granularity and attribution.
        training_admitted: Always false; independent admission remains a separate operation.
    """

    study: ArtifactInput
    branches: tuple[BranchFailureAnalysis, ...]
    scope: str = (
        "development observations; rewind before actor response, "
        "not between batched tools; no causal blame"
    )
    training_admitted: Literal[False] = False


def analyse_failures(store: ArtifactStore, reference: ArtifactInput) -> FailureAnalysis:
    """Localize symptoms from verified inventory without runtime imports or model calls."""
    study = load_study(store, reference)
    root = load_continuation_checkpoint(store, study.spec.checkpoint)
    require_development_task(store, root.task)
    return FailureAnalysis(
        study=reference,
        branches=tuple(_analyse_branch(store, study, branch) for branch in study.branches),
    )


def _analyse_branch(
    store: ArtifactStore, study: ContinuationStudy, branch: ContinuationBranch
) -> BranchFailureAnalysis:
    """Keep final failure separate from observed per-step violations and operational symptoms."""
    assessment = grade(study.spec.verifier, branch.grading_outcome())
    points = (study.spec.checkpoint, *branch.checkpoints)
    capsules = {ref: load_continuation_checkpoint(store, ref) for ref in points}
    root_calls = capsules[study.spec.checkpoint].actor.calls
    calls = branch.outcome.get("model_calls", [])
    if not isinstance(calls, list):
        raise ValueError("saved model-call inventory is malformed; inspect the source study")
    decisions: list[tuple[ToolCall, int, int, ArtifactInput | None]] = []
    terminal: tuple[int, ArtifactInput | None] | None = None
    for index, call in enumerate(calls):
        if not isinstance(call, dict):
            raise ValueError("saved model-call receipt is malformed; inspect the source study")
        request = ModelRequest.model_validate(call.get("request"))
        matching = [
            ref
            for ref, point in capsules.items()
            if point.actor.calls == root_calls + index and point.actor.messages == request.messages
        ]
        boundary = matching[0] if len(matching) == 1 else None
        terminal = (index, boundary)
        if call.get("response") is None:
            continue
        response = ModelResponse.model_validate(call["response"])
        if response.model != study.spec.actor_model:
            continue
        decisions.extend(
            (action, index, offset, boundary)
            for offset, action in enumerate(response.output.tool_calls)
        )
    signals: list[FailureSignal] = []
    steps = branch.outcome.get("steps", [])
    if not isinstance(steps, list):
        raise ValueError("saved action inventory is malformed; inspect the source study")
    for index, step in enumerate(steps):
        if not isinstance(step, dict):
            raise ValueError("saved action observation is malformed; inspect the source study")
        action = ToolCall.model_validate(step.get("action"))
        # Execution is an ordered prefix of the response's tools. Never guess a boundary when
        # receipts disagree, including when an earlier call was not executed.
        decision = decisions[index] if index < len(decisions) else None
        if decision is not None and decision[0] != action:
            decision = None
        context: _Location = {
            "step": index,
            "actor_call": None if decision is None else decision[1],
            "preceding_tools": None if decision is None else decision[2],
            "checkpoint": None if decision is None else decision[3],
        }
        observed: JsonObject = {
            "initial_state": branch.outcome.get("initial_state"),
            "final_state": step.get("after"),
            "steps": [step],
            "stop_reason": "failure",
        }
        policy = grade(study.spec.verifier, observed).policy
        if policy.status == "fail":
            signals.append(FailureSignal(kind="policy_violation", reason=policy.reason, **context))
        execution = step.get("execution")
        if isinstance(execution, dict):
            receipt = WorldExecutionReceipt.model_validate(execution)
            if receipt.action_sha256 != sha256_json(action):
                raise ValueError("execution receipt belongs to a different attempted action")
            if receipt.error_type is not None:
                if receipt.owner == "simulated" and receipt.dispatched:
                    signals.append(
                        FailureSignal(
                            kind="simulator_execution_failure",
                            repair_destination="simulator",
                            reason="Simulated backend failed; repair the simulator "
                            "before using this step to assess actor competence.",
                            **context,
                        )
                    )
                elif receipt.effects == "not_started":
                    signals.append(
                        FailureSignal(
                            kind="dispatch_refused",
                            repair_destination="interface",
                            reason="Backend was not dispatched; inspect declared capabilities and "
                            "the attempted arguments before assigning actor blame.",
                            **context,
                        )
                    )
                elif receipt.effects == "rolled_back":
                    signals.append(
                        FailureSignal(
                            kind="execution_rolled_back",
                            repair_destination="execution_recovery",
                            reason="Tool effects were rolled back; scheduled events and provider "
                            "exposure are separate. Inspect execution before another attempt.",
                            **context,
                        )
                    )
        reply = step.get("reply")
        if isinstance(reply, dict) and reply.get("is_error") is True:
            signals.append(
                FailureSignal(
                    kind="tool_error",
                    reason="Tool returned an error; actor versus environment cause unresolved.",
                    **context,
                )
            )
        if step.get("status") != "valid":
            signals.append(
                FailureSignal(
                    kind="execution_incomplete",
                    reason="Tool execution or observation incomplete; effect may be unknown.",
                    **context,
                )
            )
    if assessment.goal.status == "fail":
        signals.append(
            FailureSignal(
                kind="goal_failed",
                reason="Final goal failed; this does not locate the first mistaken action.",
                actor_call=None if terminal is None else terminal[0],
                checkpoint=None if terminal is None else terminal[1],
            )
        )
    if branch.status != "completed" or not assessment.complete:
        signals.append(
            FailureSignal(
                kind="execution_incomplete",
                reason="Branch did not complete; no failed-skill label is justified.",
                actor_call=None if terminal is None else terminal[0],
                checkpoint=None if terminal is None else terminal[1],
            )
        )
    if assessment.status == "unassessable":
        signals.append(FailureSignal(kind="unassessable", reason=assessment.reason))
    # Some policies constrain a final probe not present in any tool step. Preserve that finding
    # without assigning it to the last tool or inventing an unavailable checkpoint.
    if assessment.policy.status == "fail" and not any(
        row.kind == "policy_violation" for row in signals
    ):
        signals.append(FailureSignal(kind="policy_violation", reason=assessment.policy.reason))
    return BranchFailureAnalysis(index=branch.index, assessment=assessment, signals=tuple(signals))


def persist_failure_analysis(
    store: ArtifactStore, reference: ArtifactInput, *, created_at: datetime, code_revision: str
) -> ArtifactInput:
    """Reproduce the full report on replay and bind it to the immutable source inventory."""
    report = analyse_failures(store, reference)
    identity = stable_id(
        "failure-analysis", {"study": reference.model_dump(mode="json"), "code": code_revision}
    )
    if store.exists(identity):
        saved = FailureAnalysis.model_validate_json(store.read_bytes(identity, "analysis.json"))
        if saved != report or set(store.read(identity).manifest.inputs) != {reference}:
            raise ValueError("saved failure analysis differs; inspect its frozen source evidence")
    else:
        store.write_json(
            artifact_id=identity,
            artifact_type="failure-analysis",
            envelope=ArtifactEnvelope(
                schema_version=1,
                created_at=created_at,
                code_revision=code_revision,
                inputs=(reference,),
            ),
            files={"analysis.json": report},
        )
    return artifact_input(store.read(identity).manifest)
