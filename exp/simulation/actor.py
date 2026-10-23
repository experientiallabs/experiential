"""Capture a maintained actor's dynamic tool workflow without exposing evaluator state."""

from collections.abc import Callable
from datetime import datetime

from exp.common.core.artifacts import ArtifactEnvelope, ContractModel, JsonObject
from exp.common.evaluations.reconstruction import WorkflowEvidence, WorkflowStep
from exp.common.models import ModelClient, ToolCall
from exp.common.project import ArtifactStore
from exp.common.tasks import TaskCase
from exp.runtime.agents import AgentEpisode, AgentRuntime
from exp.runtime.environments import Observation
from exp.simulation.world_model.durable import DurableWorldModelSession


class ActorWorkflow(ContractModel):
    """An actor outcome and its candidate-owned chronology for independent evaluation.

    Attributes:
        episode: Actual agent result, absent if the adapter raised.
        evidence: Every emitted tool slot, including an execution failure if present.
        error_type: Sanitized adapter or execution exception class, if raised.
    """

    episode: AgentEpisode | None
    evidence: WorkflowEvidence
    error_type: str | None = None


class _RecordingEnvironment:
    """Expose tool execution while keeping candidate state and references out of actor inputs."""

    def __init__(self, session: DurableWorldModelSession) -> None:
        """Bind a fresh session and accumulate only actually emitted actions."""
        self._session = session
        self._steps: list[WorkflowStep] = []
        self._failed = False

    def execute(self, action: ToolCall) -> Observation:
        """Capture one tool consequence through the existing guarded commit path."""
        if self._failed:
            raise RuntimeError("actor environment stopped after an execution failure")
        index = len(self._steps)
        before = self._session.state
        public_action: JsonObject = {"name": action.name, "arguments": action.arguments}
        try:
            # Provider call IDs may repeat across turns; logical operations are episode slots.
            observation = self._session.step((action,), operation_id=f"actor-step-{index}")[0]
        except Exception as error:  # noqa: BLE001 - failures remain evaluation evidence.
            self._failed = True
            self._steps.append(
                WorkflowStep(
                    step=index,
                    action=public_action,
                    status="failure",
                    before=before,
                    after=self._session.state,
                    error_type=type(error).__name__,
                )
            )
            raise
        transition = self._session.transitions[-1]
        replies = transition["replies"]
        if not isinstance(replies, list) or len(replies) != 1:
            raise ValueError("single-tool actor received a different reply inventory")
        self._steps.append(
            WorkflowStep(
                step=index,
                action=public_action,
                status="valid",
                before=before,
                reply=replies[0],
                after=self._session.state,
            )
        )
        return observation


def run_actor_workflow(
    session: DurableWorldModelSession,
    task: TaskCase,
    *,
    agent: AgentRuntime,
    model: ModelClient,
    artifacts: ArtifactStore,
    clock: Callable[[], datetime],
    code_revision: str,
) -> ActorWorkflow:
    """Run one bounded actor and preserve its chronology before independent native replay.

    The supplied agent and session own model-call and tool-operation limits. No native
    reference or judge is available during interaction. Unlike a fixed workflow, the
    tool inventory is determined by the actor; unused capacity is not invented as steps.
    A no-action or failed actor remains an explicit outcome, never an exact empty workflow.

    Args:
        session: Fresh, initialized candidate session with frozen behavior and policy.
        task: Actor-visible task and tools; its private reference answer is removed.
        agent: Maintained whole-episode actor implementation.
        model: Metered actor client through the existing provider owner.
        artifacts: Immutable project store for the resulting episode and evidence.
        clock: Aware clock used for artifact publication.
        code_revision: Implementation identity associated with the execution.

    Returns:
        The actual actor outcome and evidence suitable for the common dual evaluator.
    """
    if session.transitions:
        raise ValueError("actor workflow requires a fresh initialized session")
    initial = session.state
    environment = _RecordingEnvironment(session)
    episode = None
    error_type = None
    try:
        episode = agent.run(
            task.model_copy(update={"reference_answer": None}),
            model=model,
            environment=environment,
        )
    except Exception as error:  # noqa: BLE001 - preserve failed actors without resampling.
        error_type = type(error).__name__
    if error_type is None:
        error_type = next(
            (step.error_type for step in environment._steps if step.status == "failure"), None
        )
    context = session.evaluation_context
    task_context: JsonObject = {"actor_context": task.initial_context}
    initialization_context = context.get("task_context")
    if isinstance(initialization_context, dict):
        task_context.update(initialization_context)
    evidence = WorkflowEvidence.model_validate(
        {
            **context,
            "workflow_id": task.task_id,
            "task": task.instruction,
            "task_context": task_context,
            "initial_state": initial,
            "steps": [step.model_dump(mode="json") for step in environment._steps],
            "final_answer": episode.final_action.model_dump(mode="json")
            if episode is not None and episode.final_action is not None
            else None,
            "stop_reason": "actor_failure"
            if error_type is not None
            else episode.stop_reason.value
            if episode is not None
            else "actor_failure",
        }
    )
    result = ActorWorkflow(episode=episode, evidence=evidence, error_type=error_type)
    artifacts.write_json(
        artifact_id=task.task_id + "-actor",
        artifact_type="actor-workflow",
        envelope=ArtifactEnvelope(
            schema_version=1,
            created_at=clock(),
            inputs=(),
            code_revision=code_revision,
        ),
        files={"actor.json": result.model_dump(mode="json")},
    )
    return result
