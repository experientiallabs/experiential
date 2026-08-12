"""Tests for the installed-Pi adapter seam."""

from __future__ import annotations

import sys
from datetime import UTC, datetime
from pathlib import Path
from stat import S_IXUSR

import pytest

from wmo.common.core.artifacts import JsonObject
from wmo.common.models import AssistantAction, ToolCall
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


def test_default_pi_runtime_invokes_the_installed_json_executable_and_translates_events(
    tmp_path: Path,
) -> None:
    runtime = PiAgentRuntime(executable=str(_write_pi_json_fixture(tmp_path)))

    episode = runtime.run(_task(), model=_Model(), environment=_Environment())

    assert episode.stop_reason == StopReason.COMPLETED
    assert episode.final_action == AssistantAction(
        content="Pi completed the task",
        tool_calls=(ToolCall(call_id="pi-call", name="lookup", arguments={"id": "record-1"}),),
    )
    assert [(event.kind, event.tool_name) for event in episode.events] == [
        (RolloutEventKind.MESSAGE, None),
        (RolloutEventKind.TOOL_CALL, "lookup"),
        (RolloutEventKind.OBSERVATION, "lookup"),
    ]
    assert episode.events[0].payload == {
        "source": "installed-pi",
        "event": "message_end",
        "role": "assistant",
        "content": "Pi completed the task",
    }
    assert episode.events[2].payload == {
        "source": "installed-pi",
        "event": "tool_execution_end",
        "is_error": False,
    }


def test_pi_runtime_names_the_missing_external_install() -> None:
    runtime = PiAgentRuntime(executable="wmo-pi-not-installed")

    with pytest.raises(PiRuntimePreflightError, match="Install Pi outside WMO"):
        runtime.run(_task(), model=_Model(), environment=_Environment())


def test_pi_rejects_an_invalid_bridge_transcript() -> None:
    runtime = PiAgentRuntime(transcript_runner=_invalid_transcript)

    with pytest.raises(PiTranscriptError, match="AgentEpisode"):
        runtime.run(_task(), model=_Model(), environment=_Environment())


class _Model:
    """A temporary stand-in that must conform to ModelClient during the W3 restack."""


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


def _write_pi_json_fixture(tmp_path: Path) -> Path:
    """Write a local executable that mimics Pi's documented JSON event stream."""
    executable = tmp_path / "pi-json-fixture"
    executable.write_text(
        f"""#!{sys.executable}
import json
import sys

expected = [
    "--mode",
    "json",
    "--no-session",
    "--no-context-files",
    "--no-extensions",
    "--no-skills",
    "--no-prompt-templates",
    "--no-themes",
    "--offline",
    "--no-tools",
    "Complete the deterministic Pi fixture.",
]
if sys.argv[1:] != expected:
    raise SystemExit(17)
events = [
    {{"type": "session", "version": 3}},
    {{"type": "agent_start"}},
    {{
        "type": "message_end",
        "message": {{
            "role": "assistant",
            "timestamp": "2026-08-11T00:00:00+00:00",
            "content": [
                {{"type": "text", "text": "Pi completed the task"}},
                {{
                    "type": "toolCall",
                    "id": "pi-call",
                    "name": "lookup",
                    "arguments": {{"id": "record-1"}},
                }},
            ],
        }},
    }},
    {{
        "type": "tool_execution_start",
        "timestamp": "2026-08-11T00:00:01+00:00",
        "toolName": "lookup",
    }},
    {{
        "type": "tool_execution_end",
        "timestamp": "2026-08-11T00:00:02+00:00",
        "toolName": "lookup",
        "isError": False,
    }},
    {{"type": "agent_end"}},
]
sys.stdout.write("\\n".join(json.dumps(event) for event in events) + "\\n")
""",
        encoding="utf-8",
    )
    executable.chmod(executable.stat().st_mode | S_IXUSR)
    return executable
