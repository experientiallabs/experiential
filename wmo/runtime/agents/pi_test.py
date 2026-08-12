"""Tests for the installed-Pi adapter seam."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from wmo.common.core.artifacts import JsonObject
from wmo.common.models import ToolCall
from wmo.common.rollouts import RolloutEventKind, RolloutSpan, StopReason
from wmo.common.tasks import TaskCase
from wmo.runtime.agents.pi import PiAgentRuntime, PiRuntimePreflightError, PiTranscriptError
from wmo.runtime.environments import EnvironmentSession, Observation


def test_pi_runner_receives_injected_model_and_normalizes_transcript() -> None:
    model = _Model()
    environment = _Environment()
    runner = _TranscriptRunner()
    runtime = PiAgentRuntime(transcript_runner=runner)

    episode = runtime.run(_task(), model=model, environment=environment)

    assert runner.model is model
    assert runner.observation == Observation(content="tool response")
    assert episode.stop_reason == StopReason.COMPLETED
    assert episode.final_action is not None
    assert episode.final_action.content == "Pi completed the task"
    assert len(episode.events) == 1
    assert episode.events[0].kind == RolloutEventKind.TOOL_CALL
    assert episode.events[0].payload == {"source": "installed-pi"}


def test_pi_preflight_names_the_external_install_and_injected_bridge() -> None:
    runtime = PiAgentRuntime(executable="wmo-pi-not-installed")

    with pytest.raises(PiRuntimePreflightError, match="Install Pi outside WMO"):
        runtime.preflight()


def test_pi_rejects_an_invalid_bridge_transcript() -> None:
    runtime = PiAgentRuntime(transcript_runner=_invalid_transcript)

    with pytest.raises(PiTranscriptError, match="AgentEpisode"):
        runtime.run(_task(), model=_Model(), environment=_Environment())


class _Model:
    """A deterministic stand-in for W3's pending ModelClient contract."""


class _Environment:
    """A fake execute-only session accepted by the installed Pi bridge."""

    def execute(self, action: ToolCall) -> Observation:
        return Observation(content="tool response")


class _TranscriptRunner:
    """Captures injected dependencies and emits a representative Pi JSON transcript."""

    def __init__(self) -> None:
        self.model: object | None = None
        self.observation: Observation | None = None

    def __call__(
        self,
        task: TaskCase,
        model: object,
        environment: EnvironmentSession,
    ) -> JsonObject:
        self.model = model
        self.observation = environment.execute(
            ToolCall(call_id="pi-call", name="bash", arguments={})
        )
        event = RolloutSpan(
            span_id="pi-tool-1",
            kind=RolloutEventKind.TOOL_CALL,
            started_at=datetime(2026, 8, 11, tzinfo=UTC),
            ended_at=datetime(2026, 8, 11, tzinfo=UTC),
            payload={"source": "installed-pi"},
        )
        return {
            "events": [event.model_dump(mode="json")],
            "final_action": {"content": "Pi completed the task"},
            "stop_reason": StopReason.COMPLETED.value,
        }


def _invalid_transcript(
    task: TaskCase,
    model: object,
    environment: EnvironmentSession,
) -> JsonObject:
    return {"events": [], "stop_reason": StopReason.FAILURE.value}


def _task() -> TaskCase:
    return TaskCase(
        task_id="task-1",
        lineage_group_id="lineage-1",
        partition="held_out",
        instruction="Complete the deterministic Pi fixture.",
        workload_weight=1.0,
        source_trace_ids=("trace-1",),
    )
