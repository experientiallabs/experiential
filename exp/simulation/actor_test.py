"""Actor capture uses real durable state while preserving reference and failure boundaries."""

from pathlib import Path

from jsonschema import ValidationError

from exp.common.evaluations.reconstruction import EvaluationReferences, mechanical_checks
from exp.common.models import AssistantAction, ModelClient, ModelRequest, ModelResponse, ToolCall
from exp.common.rollouts import StopReason
from exp.common.tasks import TaskCase
from exp.runtime.agents import AgentEpisode
from exp.runtime.environments import EnvironmentSession
from exp.simulation.actor import run_actor_workflow
from exp.simulation.world_model.durable_test import NOW, setup


class _UnusedModel:
    """Fail if the deterministic boundary fixture accidentally dispatches inference."""

    def complete(self, request: ModelRequest) -> ModelResponse:
        """Reject any unexpected model request."""
        raise AssertionError("this actor fixture does not use a model")


class _Actor:
    """Issue a declared fixture sequence through the actual environment capability."""

    def __init__(self, calls: tuple[ToolCall, ...]) -> None:
        """Retain fixture actions independently of candidate state."""
        self.calls = calls

    def run(
        self, task: TaskCase, *, model: ModelClient, environment: EnvironmentSession
    ) -> AgentEpisode:
        """Verify reference redaction and use only the actor's execution surface."""
        assert task.reference_answer is None
        assert not hasattr(environment, "state")
        assert not hasattr(environment, "steps")
        for call in self.calls:
            environment.execute(call)
        return AgentEpisode(
            final_action=AssistantAction(content="Finished."),
            stop_reason=StopReason.COMPLETED,
        )


def _task() -> TaskCase:
    """Declare a private reference that must never reach the actor."""
    return TaskCase(
        task_id="actor-check",
        lineage_group_id="fixture",
        partition="fit",
        instruction="Append the requested values.",
        workload_weight=1.0,
        origin="authored",
        reference_answer="Evaluator-private expected answer.",
    )


def test_repeated_provider_call_ids_are_distinct_logical_operations(tmp_path: Path) -> None:
    """Distinct actor turns commit twice even if the provider reuses an external ID."""
    session, project, _, _ = setup(tmp_path)
    result = run_actor_workflow(
        session,
        _task(),
        agent=_Actor(
            (
                ToolCall(call_id="reused", name="append", arguments={"item": "a"}),
                ToolCall(call_id="reused", name="append", arguments={"item": "b"}),
            )
        ),
        model=_UnusedModel(),
        artifacts=project.artifacts,
        clock=lambda: NOW,
        code_revision="fixture",
    )
    assert result.error_type is None
    assert session.state == {"items": ["existing", "a", "b"], "owner": "current-user"}
    assert len(result.evidence.steps) == 2
    assert result.evidence.steps[1].before == result.evidence.steps[0].after
    assert project.artifacts.read("actor-check-actor")


def test_failed_tool_is_retained_without_mutation_or_synthetic_suffix(tmp_path: Path) -> None:
    """A rejected argument records the emitted failure and stops the actor."""
    session, project, _, _ = setup(tmp_path)
    initial = session.state
    result = run_actor_workflow(
        session,
        _task(),
        agent=_Actor((ToolCall(call_id="bad", name="append", arguments={}),)),
        model=_UnusedModel(),
        artifacts=project.artifacts,
        clock=lambda: NOW,
        code_revision="fixture",
    )
    assert result.error_type is not None
    assert result.evidence.stop_reason == "actor_failure"
    assert len(result.evidence.steps) == 1
    assert result.evidence.steps[0].status == "failure"
    assert result.evidence.steps[0].before == result.evidence.steps[0].after == initial
    assert session.transitions == ()


def test_no_action_actor_does_not_receive_a_vacuous_exact_workflow(tmp_path: Path) -> None:
    """An immediate final answer remains assessable without fabricating a tool transition."""
    session, project, _, _ = setup(tmp_path)
    result = run_actor_workflow(
        session,
        _task(),
        agent=_Actor(()),
        model=_UnusedModel(),
        artifacts=project.artifacts,
        clock=lambda: NOW,
        code_revision="fixture",
    )
    assert result.evidence.steps == ()
    checks = mechanical_checks(result.evidence, EvaluationReferences(capability="description_only"))
    assert checks["planned_steps"] == 0
    assert checks["workflow_joint_exact"] is None


class _ErrorCatchingActor:
    """Return a final answer after catching an environment failure."""

    def run(
        self, task: TaskCase, *, model: ModelClient, environment: EnvironmentSession
    ) -> AgentEpisode:
        """Exercise an adapter that hides the exception from its caller."""
        try:
            environment.execute(ToolCall(call_id="bad", name="append", arguments={}))
        except ValidationError:
            pass
        return AgentEpisode(
            final_action=AssistantAction(content="Finished."),
            stop_reason=StopReason.COMPLETED,
        )


def test_caught_execution_failure_cannot_be_reported_as_completed(tmp_path: Path) -> None:
    """A final answer cannot erase the failed transition recorded by the coordinator."""
    session, project, _, _ = setup(tmp_path)
    result = run_actor_workflow(
        session,
        _task(),
        agent=_ErrorCatchingActor(),
        model=_UnusedModel(),
        artifacts=project.artifacts,
        clock=lambda: NOW,
        code_revision="fixture",
    )
    assert result.episode is not None
    assert result.episode.stop_reason == StopReason.COMPLETED
    assert result.error_type is not None
    assert result.evidence.stop_reason == "actor_failure"
    assert result.evidence.steps[0].status == "failure"
