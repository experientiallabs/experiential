"""Execute one carried workload and always retain its required dual-evaluation status."""

from collections.abc import Callable
from datetime import datetime

from pydantic import Field

from exp.common.core.artifacts import ArtifactId, ContractModel, JsonObject, strict_equal
from exp.common.evaluations.reconstruction import (
    DualReport,
    EvaluationReferences,
    JudgeConfig,
    WorkflowEvidence,
    WorkflowStep,
    evaluate,
)
from exp.common.models import ToolCall
from exp.common.models.client import RequestAdmissionRefused
from exp.common.models.unknown_outcome import DispatchOutcomeUnknown
from exp.common.progress import ProgressHook, report
from exp.common.project import ArtifactStore
from exp.runtime.models.budget import SpendLimitReached
from exp.runtime.models.providers.transport import ProviderTransportError
from exp.runtime.sessions.dispatch_ledger import DispatchCapExhaustedError
from exp.simulation.engines.transition import Completion
from exp.simulation.world_model.durable import DurableWorldModelSession

# Failures of the execution machinery, not of the simulator, stop a workflow unscored. A scored
# report would count the failed step and its blocked suffix as misses, so these are raised
# instead: no report is saved, the run stays incomplete and visible, and only an explicit
# in-place resumption can continue it. Every other step failure is simulator behavior and scores.
INFRASTRUCTURE_STOPS: tuple[type[BaseException], ...] = (
    ProviderTransportError,
    # Deadlines and refusals before dispatch: ProviderDeadlineExceeded, the admitted transport's
    # and the structured completion's deadline errors, and CompletionNotDispatched.
    TimeoutError,
    DispatchOutcomeUnknown,
    DispatchCapExhaustedError,
    # Local stops before dispatch, such as a dispatch guard's refusal. A step-level spend limit
    # is caught first and scored as budget exhaustion.
    RequestAdmissionRefused,
)

# Gateway error codes of a definite response that refuses the request itself, rather than an
# outage: the provider's stream for this exact request could not be encoded as a valid reply.
# Resending the identical request repeats it, so a learning pass may retire the trace after a
# bounded number of identical failures. Transport outages, timeouts, overload and unknown
# outcomes are never listed here: they stay resumable.
DETERMINISTIC_REFUSALS: frozenset[str] = frozenset({"invalid_provider_stream"})


def deterministic_refusal(error: BaseException) -> ProviderTransportError | None:
    """The error, when it is a registered refusal of an identifiable request; else None.

    Args:
        error: A stop raised by a workflow.

    Returns:
        The provider error when it carries an HTTP status, a code in
        ``DETERMINISTIC_REFUSALS`` and the digest of the request it refused.
    """
    if (
        isinstance(error, ProviderTransportError)
        and error.status_code is not None
        and error.error_code in DETERMINISTIC_REFUSALS
        and error.request_sha256 is not None
    ):
        return error
    return None


class FixedWorkflow(ContractModel):
    """Prospectively declared actions and task, independent of native reference outcomes.

    Attributes:
        id: Logical workflow identity used by all receipts.
        task: Public task instruction.
        actions: Ordered single-tool actions with distinct call identities.
    """

    id: ArtifactId
    task: str = Field(min_length=1)
    actions: tuple[ToolCall, ...] = Field(min_length=1)


def evaluate_workflow(
    session: DurableWorldModelSession,
    workload: FixedWorkflow,
    *,
    references: EvaluationReferences,
    judge: JudgeConfig,
    judge_complete: Completion,
    artifacts: ArtifactStore,
    clock: Callable[[], datetime],
    code_revision: str,
    evaluation_id: ArtifactId | None = None,
    reply_feedback_source: str | None = None,
    progress: ProgressHook | None = None,
    feedback_after_predictions: bool = False,
    resume: bool = False,
) -> DualReport:
    """Run without native rescue; retain blocked slots and a separate receipt identity.

    With ``feedback_after_predictions`` (pinned memory delivery), every step is predicted
    first and replies are disclosed and investigated only afterwards, together, so a failed
    investigation never prevents a prediction.

    Matched parent/child evaluations share a public workload ID but must use distinct
    evaluation IDs for their immutable request and result artifacts. ``progress`` observes
    each simulated turn as an exact completed/total count; after a hard stop it reports
    the turns actually simulated, qualified as stopped, never the blocked remainder.

    ``resume`` continues an interrupted evaluation of the same workload: the session's
    committed steps must be a prefix of the workload, and they replay from the session; an
    interrupted judge intent is completed instead of being reported as unknown.
    """
    committed = session.transitions
    if committed and not resume:
        raise ValueError("a scored workflow requires a fresh initialized session")
    if [row.get("operation_id") for row in committed] != [
        action.call_id for action in workload.actions[: len(committed)]
    ]:
        raise ValueError("resumed session steps differ from the workload's leading actions")
    judge = JudgeConfig.model_validate(judge.model_dump(mode="json"))
    if len({action.call_id for action in workload.actions}) != len(workload.actions):
        raise ValueError("workflow actions need distinct logical call identities")
    actions: list[JsonObject] = [
        {"name": a.name, "arguments": a.arguments} for a in workload.actions
    ]
    if references.steps and (
        len(references.steps) != len(actions)
        or any(
            not strict_equal(r.action, a) for r, a in zip(references.steps, actions, strict=True)
        )
    ):
        raise ValueError("reference inventory differs from the public workload before dispatch")
    if reply_feedback_source is not None and (
        len(references.steps) != len(actions)
        or any(not row.reply_available for row in references.steps)
    ):
        raise ValueError("online feedback requires a disclosed reply for every supplied action")
    initial = committed[0]["before"] if committed else session.state
    if not isinstance(initial, dict):
        raise ValueError("committed transition lacks its prior state")
    steps = []
    stopped = False
    feedback_receipts: list[JsonObject] = []
    feedback_error: str | None = None
    for index, action in enumerate(workload.actions):
        if not stopped:
            report(progress, "Simulating turns", completed=index, total=len(workload.actions))
        before = session.state
        if stopped:
            steps.append(WorkflowStep(step=index, action=actions[index], status="blocked"))
            continue
        try:
            session.execute(action)
        except SpendLimitReached:
            stopped = True
            steps.append(
                WorkflowStep(
                    step=index,
                    action=actions[index],
                    status="failure",
                    before=before,
                    after=session.state,
                    error_type="budget_exhausted",
                )
            )
        except INFRASTRUCTURE_STOPS:
            # Infrastructure, not simulator behavior: no result is saved, so the run stays
            # incomplete (shown, exit non-zero) instead of scoring the rest as misses.
            raise
        except Exception as error:  # noqa: BLE001 - execution failures retain planned suffixes.
            stopped = True
            steps.append(
                WorkflowStep(
                    step=index,
                    action=actions[index],
                    status="failure",
                    before=before,
                    after=session.state,
                    error_type=type(error).__name__,
                )
            )
        else:
            transition = _transition(session, action.call_id)
            replies = transition["replies"]
            if not isinstance(replies, list) or len(replies) != 1:
                raise ValueError("fixed single-tool workflow received a different reply inventory")
            steps.append(
                WorkflowStep.model_validate(
                    {
                        "step": index,
                        "action": actions[index],
                        "status": "valid",
                        "before": transition["before"],
                        "reply": replies[0],
                        "after": transition["after"],
                    }
                )
            )
            if reply_feedback_source is not None and not feedback_after_predictions:
                try:
                    feedback_receipts.append(
                        session.observe_reply_feedback(
                            action.call_id,
                            references.steps[index].reply,
                            source=reply_feedback_source,
                        )
                    )
                except INFRASTRUCTURE_STOPS:
                    raise
                except Exception as error:  # noqa: BLE001 - failed adaptation blocks future slots.
                    feedback_error = type(error).__name__
                    stopped = True
    if reply_feedback_source is not None and feedback_after_predictions and not stopped:
        # Whole-trace boundary: every step is predicted before any reply is disclosed, so an
        # investigation can never decide which predictions exist.
        report(progress, "Investigating misses")
        settled = session.observe_all_reply_feedback(
            [
                (action.call_id, references.steps[i].reply)
                for i, action in enumerate(workload.actions)
            ],
            source=reply_feedback_source,
        )
        feedback_receipts = [row for row in settled if not isinstance(row, Exception)]
        failures = [row for row in settled if isinstance(row, Exception)]
        for failure in failures:
            if isinstance(failure, INFRASTRUCTURE_STOPS):
                # Predictions and successful feedback are durable; no terminal report may
                # hide an operational interruption from explicit in-place recovery.
                raise failure
        feedback_error = type(failures[0]).__name__ if failures else None
    simulated = sum(step.status != "blocked" for step in steps)
    report(
        progress,
        "Simulating turns",
        completed=simulated,
        total=len(workload.actions),
        detail="stopped" if simulated < len(workload.actions) else None,
    )
    context = session.evaluation_context
    deliveries = []
    for index, transition in enumerate(session.transitions):
        evidence = transition.get("evidence")
        if isinstance(evidence, dict) and "memory_claims" in evidence:
            deliveries.append(
                {
                    "step": index,
                    "operation_id": transition.get("operation_id"),
                    "request_sha256": evidence.get("request_sha256"),
                    "claims": evidence["memory_claims"],
                }
            )
    if deliveries:
        metadata = context.get("audit_metadata", {})
        if not isinstance(metadata, dict):
            raise ValueError("evaluation audit metadata must be an object")
        context["audit_metadata"] = {**metadata, "memory_claim_deliveries": deliveries}
    if reply_feedback_source is not None:
        sources = context.get("sources", {})
        if not isinstance(sources, dict):
            raise ValueError("evaluation sources must be an object")
        context["sources"] = {
            **sources,
            "online_feedback": {
                "source": reply_feedback_source,
                "receipts": feedback_receipts,
                "error_type": feedback_error,
                "state_rescue": False,
            },
        }
    evidence = WorkflowEvidence.model_validate(
        {
            **context,
            "workflow_id": workload.id,
            "task": workload.task,
            "initial_state": initial,
            "steps": [step.model_dump(mode="json") for step in steps],
            "stop_reason": "research_update_failure"
            if feedback_error
            else ("failure" if stopped else "completed"),
        }
    )

    report(progress, "Judging")
    return evaluate(
        evidence,
        references,
        judge=judge,
        complete=judge_complete,
        artifacts=artifacts,
        artifact_id=evaluation_id or workload.id + "-evaluation",
        clock=clock,
        code_revision=code_revision,
        resume=resume,
    )


def _transition(session: DurableWorldModelSession, operation_id: str) -> JsonObject:
    """The committed transition of one operation, replayed or new."""
    for row in reversed(session.transitions):
        if row.get("operation_id") == operation_id:
            return row
    raise ValueError(f"operation {operation_id} has no committed transition")
