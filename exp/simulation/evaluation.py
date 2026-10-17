"""Execute one carried workload and always retain its required dual-evaluation status."""

from collections.abc import Callable
from datetime import datetime

from pydantic import Field

from exp.common.core.artifacts import ArtifactId, ContractModel, JsonObject
from exp.common.evaluations.reconstruction import (
    DualReport,
    EvaluationReferences,
    JudgeConfig,
    WorkflowEvidence,
    WorkflowStep,
    evaluate,
    strict_equal,
)
from exp.common.models import ToolCall
from exp.common.progress import ProgressHook, report
from exp.common.project import ArtifactStore
from exp.runtime.models.budget import SpendLimitReached
from exp.simulation.engines.transition import Completion
from exp.simulation.world_model.durable import DurableWorldModelSession


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
