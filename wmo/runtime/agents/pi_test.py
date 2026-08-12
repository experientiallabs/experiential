"""Tests for the installed-Pi adapter's injected model and environment bridge."""

from __future__ import annotations

import json
import sys
from contextlib import AbstractContextManager
from datetime import UTC, datetime, timedelta
from pathlib import Path
from stat import S_IXUSR
from types import TracebackType

import pytest

from wmo.common.core.artifacts import FailureAttribution, FailureCode
from wmo.common.models import (
    AssistantAction,
    ModelRequest,
    ModelResponse,
    ModelSnapshot,
    OperationEconomics,
    ToolCall,
)
from wmo.common.rollouts import RolloutEventKind, StopReason
from wmo.common.tasks import TaskCase, ToolSchema
from wmo.runtime.agents import execute_agent_episode
from wmo.runtime.agents.pi import (
    PiAgentRuntime,
    PiRuntimePreflightError,
    PiTranscriptError,
    _episode_from_pi_events,
)
from wmo.runtime.environments import EnvironmentSession, Observation

_DETERMINISTIC_EVENT_EPOCH = datetime(1970, 1, 1, tzinfo=UTC)
_DIGEST = "a" * 64


def test_default_pi_runtime_binds_the_injected_model_and_environment(
    tmp_path: Path,
) -> None:
    """The deterministic executable exercises the same process binding as installed Pi."""
    model = _Model()
    environment_runtime = _EnvironmentRuntime()
    runtime = PiAgentRuntime(executable=str(_write_pi_binding_fixture(tmp_path)))

    episode = execute_agent_episode(runtime, environment_runtime, _task(), model)

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
    assert model.requests[0].messages[-1].content == _task().instruction
    assert model.requests[0].tools == _task().tools
    assert environment_runtime.actions == [
        ToolCall(call_id="pi-call", name="lookup", arguments={"id": "record-1"})
    ]
    assert environment_runtime.close_calls == 1
    assert episode.events[2].payload == {
        "source": "installed-pi",
        "event": "tool_execution_end",
        "is_error": False,
    }


@pytest.mark.parametrize("instruction", ("@candidate", "--candidate"))
def test_installed_pi_keeps_special_task_text_out_of_argv(
    tmp_path: Path,
    instruction: str,
) -> None:
    """Literal task text reaches the injected model through stdin rather than Pi argv."""
    model = _Model()
    task = _task(instruction)
    runtime = PiAgentRuntime(executable=str(_write_pi_binding_fixture(tmp_path)))

    episode = execute_agent_episode(runtime, _EnvironmentRuntime(), task, model)

    assert episode.stop_reason == StopReason.COMPLETED
    assert [message.content for message in model.requests[0].messages] == [instruction]


def test_installed_pi_uses_private_cwd_without_caller_pi_system_files(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Caller project Pi prompts cannot affect an isolated installed-Pi invocation."""
    caller_cwd = tmp_path / "caller"
    pi_directory = caller_cwd / ".pi"
    pi_directory.mkdir(parents=True)
    (pi_directory / "SYSTEM.md").write_text("hostile system prompt", encoding="utf-8")
    (pi_directory / "APPEND_SYSTEM.md").write_text("hostile appended prompt", encoding="utf-8")
    monkeypatch.chdir(caller_cwd)
    monkeypatch.setenv("WMO_TEST_CALLER_CWD", str(caller_cwd))
    model = _Model()
    task = _task()
    runtime = PiAgentRuntime(executable=str(_write_pi_binding_fixture(tmp_path)))

    episode = execute_agent_episode(runtime, _EnvironmentRuntime(), task, model)

    assert episode.stop_reason == StopReason.COMPLETED
    assert Path.cwd() == caller_cwd
    assert [message.content for message in model.requests[0].messages] == [task.instruction]


def test_installed_pi_parses_a_0731_numeric_millisecond_timestamp(tmp_path: Path) -> None:
    """Pi 0.73.1 JSON mode preserves numeric assistant message timestamps through WMO."""
    runtime = PiAgentRuntime(executable=str(_write_pi_0731_timestamp_fixture(tmp_path)))

    episode = execute_agent_episode(runtime, _EnvironmentRuntime(), _task(), _Model())

    expected_timestamp = datetime(2024, 12, 3, 14, 2, 47, 890000, tzinfo=UTC)
    assert episode.events[0].started_at == expected_timestamp
    assert episode.events[0].ended_at == expected_timestamp


def test_pi_distinguishes_numeric_second_timestamps_from_milliseconds() -> None:
    """Unix seconds keep their wall-clock value instead of collapsing near the epoch."""
    episode = _episode_from_pi_events(
        _pi_events(
            [
                {
                    "type": "message_end",
                    "message": {
                        "role": "assistant",
                        "timestamp": 1_733_234_567.89,
                        "content": [{"type": "text", "text": "Complete."}],
                    },
                },
                {"type": "agent_end"},
            ]
        )
    )

    expected_timestamp = datetime(2024, 12, 3, 14, 2, 47, 890000, tzinfo=UTC)
    assert episode.events[0].started_at == expected_timestamp
    assert episode.events[0].ended_at == expected_timestamp


def test_pi_rejects_an_absurd_numeric_millisecond_timestamp() -> None:
    """Unrepresentable integer timestamps are transcript errors, not leaked overflows."""
    with pytest.raises(PiTranscriptError, match="invalid timestamp"):
        _episode_from_pi_events(
            _pi_events(
                [
                    {
                        "type": "message_end",
                        "message": {
                            "role": "assistant",
                            "timestamp": 10**100,
                            "content": [],
                        },
                    },
                    {"type": "agent_end"},
                ]
            )
        )


@pytest.mark.parametrize(
    ("pi_stop_reason", "code", "attribution"),
    [
        ("error", FailureCode.PROVIDER, FailureAttribution.MODEL),
        ("aborted", FailureCode.CANCELLED, FailureAttribution.AGENT),
    ],
)
def test_unsuccessful_assistant_stop_reason_is_a_failed_episode_with_partial_evidence(
    pi_stop_reason: str,
    code: FailureCode,
    attribution: FailureAttribution,
) -> None:
    """Pi error and aborted message ends retain all parsed evidence before failing terminally."""
    episode = _episode_from_pi_events(
        _pi_events(
            [
                {
                    "type": "message_end",
                    "message": {
                        "role": "assistant",
                        "timestamp": "2026-08-11T00:00:00+00:00",
                        "content": [
                            {"type": "text", "text": "Partial response."},
                            {
                                "type": "toolCall",
                                "id": "pi-call",
                                "name": "lookup",
                                "arguments": {"id": "record-1"},
                            },
                        ],
                    },
                },
                {
                    "type": "tool_execution_start",
                    "timestamp": "2026-08-11T00:00:01+00:00",
                    "toolName": "lookup",
                },
                {
                    "type": "tool_execution_end",
                    "timestamp": "2026-08-11T00:00:02+00:00",
                    "toolName": "lookup",
                    "isError": False,
                },
                {
                    "type": "message_end",
                    "message": {
                        "role": "assistant",
                        "timestamp": "2026-08-11T00:00:03+00:00",
                        "stopReason": pi_stop_reason,
                        "content": [{"type": "text", "text": "Last partial output."}],
                    },
                },
                {"type": "agent_end"},
            ]
        )
    )

    assert episode.stop_reason == StopReason.FAILURE
    assert episode.failure is not None
    assert episode.failure.code == code
    assert episode.failure.attribution == attribution
    assert episode.failure.details == {"pi_stop_reason": pi_stop_reason}
    assert [event.span_id for event in episode.events] == [
        "pi-message-1",
        "pi-tool-2",
        "pi-tool-3",
        "pi-message-4",
    ]
    assert episode.events[-1].failure == episode.failure
    assert episode.final_action == AssistantAction(content="Last partial output.")


def test_missing_pi_timestamps_use_deterministic_jsonl_ordering() -> None:
    """Timestamp-less Pi events retain deterministic ordering without using the process clock."""
    episode = _episode_from_pi_events(
        _pi_events(
            [
                {
                    "type": "message_end",
                    "message": {
                        "role": "assistant",
                        "content": [{"type": "text", "text": "First response."}],
                    },
                },
                {"type": "tool_execution_start", "toolName": "lookup"},
                {"type": "tool_execution_end", "toolName": "lookup", "isError": False},
                {"type": "agent_end"},
            ]
        )
    )

    assert [event.started_at for event in episode.events] == [
        _DETERMINISTIC_EVENT_EPOCH + timedelta(microseconds=1),
        _DETERMINISTIC_EVENT_EPOCH + timedelta(microseconds=2),
        _DETERMINISTIC_EVENT_EPOCH + timedelta(microseconds=3),
    ]


def test_installed_pi_timeout_becomes_lifecycle_timeout_and_closes_environment(
    tmp_path: Path,
) -> None:
    """The bounded Pi process reaches the existing lifecycle timeout and cleanup outcome."""
    environment_runtime = _EnvironmentRuntime()
    agent = PiAgentRuntime(
        executable=str(_write_sleeping_pi_fixture(tmp_path)),
        timeout_seconds=0.01,
    )

    episode = execute_agent_episode(agent, environment_runtime, _task(), _Model())

    assert episode.stop_reason == StopReason.FAILURE
    assert episode.failure is not None
    assert episode.failure.code == FailureCode.TIMEOUT
    assert episode.failure.attribution == FailureAttribution.AGENT
    assert episode.failure.exception_type == "PiInvocationTimeoutError"
    assert episode.events[0].payload == {"phase": "agent timeout"}
    assert environment_runtime.close_calls == 1


def test_pi_runtime_names_the_missing_external_install() -> None:
    """Missing executable errors retain an actionable external-install remedy."""
    runtime = PiAgentRuntime(executable="wmo-pi-not-installed")

    with pytest.raises(PiRuntimePreflightError, match="Install Pi outside WMO"):
        runtime.run(_task(), model=_Model(), environment=_Environment())


def test_pi_rejects_a_transcript_that_ends_before_agent_end() -> None:
    """Pi transcripts must contain an explicit terminal agent event."""
    with pytest.raises(PiTranscriptError, match="agent_end"):
        _episode_from_pi_events(_pi_events([]))


class _Model:
    """A deterministic canonical model client used by the Pi binding fixture."""

    def __init__(self) -> None:
        self.requests: list[ModelRequest] = []

    def complete(self, request: ModelRequest) -> ModelResponse:
        """Record the canonical request and return a deterministic injected completion."""
        self.requests.append(request)
        return ModelResponse(
            output=AssistantAction(
                content="Injected model output.",
                tool_calls=(
                    ToolCall(call_id="pi-call", name="lookup", arguments={"id": "record-1"}),
                ),
            ),
            model=ModelSnapshot(
                provider="fixture",
                model_id="fixture-model",
                capabilities_sha256=_DIGEST,
                connection_sha256=_DIGEST,
            ),
            economics=OperationEconomics(),
        )


class _Environment:
    """A fake execute-only session that records tools crossing the Pi binding."""

    def __init__(self) -> None:
        self.actions: list[ToolCall] = []

    def execute(self, action: ToolCall) -> Observation:
        """Record one routed tool call and return a deterministic observation."""
        self.actions.append(action)
        return Observation(content="tool response")


class _EnvironmentRuntime:
    """A cleanup-owning fake used to prove subprocess timeout lifecycle behavior."""

    def __init__(self) -> None:
        self.close_calls = 0
        self._session = _Environment()

    def open(self, task: TaskCase) -> AbstractContextManager[EnvironmentSession]:
        """Open the one deterministic execute-only session."""
        return _EnvironmentContext(self)

    @property
    def actions(self) -> list[ToolCall]:
        """Return tool calls recorded by the session owned for this lifecycle fixture."""
        return self._session.actions


class _EnvironmentContext(AbstractContextManager[EnvironmentSession]):
    """Context manager that makes lifecycle cleanup observable in the timeout regression test."""

    def __init__(self, runtime: _EnvironmentRuntime) -> None:
        self._runtime = runtime

    def __enter__(self) -> EnvironmentSession:
        """Return the fake session that the timeout fixture never reaches."""
        return self._runtime._session

    def __exit__(
        self,
        exception_type: type[BaseException] | None,
        exception: BaseException | None,
        traceback: TracebackType | None,
    ) -> bool:
        """Record environment cleanup after the agent's bounded subprocess fails."""
        self._runtime.close_calls += 1
        return False


def _task(instruction: str = "Complete the deterministic Pi fixture.") -> TaskCase:
    """Return one task whose explicit tool schema the local binding fixture verifies."""
    return TaskCase(
        task_id="task-1",
        lineage_group_id="lineage-1",
        partition="held_out",
        instruction=instruction,
        tools=(
            ToolSchema(
                name="lookup",
                description="Look up one deterministic record.",
                input_schema={
                    "type": "object",
                    "properties": {"id": {"type": "string"}},
                    "required": ["id"],
                },
            ),
        ),
        workload_weight=1.0,
        source_trace_ids=("trace-1",),
    )


def _pi_events(events: list[dict[str, object]]) -> str:
    """Serialize a deterministic Pi JSONL stream with no unrelated process behavior."""
    return "\n".join(json.dumps(event) for event in events)


def _write_pi_binding_fixture(tmp_path: Path) -> Path:
    """Write a local executable that exercises the installed-Pi invocation binding end to end."""
    executable = tmp_path / "pi-binding-fixture"
    executable.write_text(
        f"""#!{sys.executable}
import json
import os
import sys
from pathlib import Path
from urllib.request import Request, urlopen

instruction = sys.stdin.read()
arguments = sys.argv[1:]
if "--no-tools" in arguments:
    raise SystemExit(10)
if arguments[arguments.index("--provider") + 1] != "wmo-injected":
    raise SystemExit(11)
if arguments[arguments.index("--model") + 1] != "wmo-injected-model":
    raise SystemExit(12)
if "--no-builtin-tools" not in arguments or "--no-extensions" not in arguments:
    raise SystemExit(13)
extension = Path(arguments[arguments.index("-e") + 1])
extension_source = extension.read_text(encoding="utf-8")
if (
    'import {{ Type }} from "typebox";' not in extension_source
    or "registerTool" not in extension_source
    or "WMO_PI_BRIDGE_URL" not in extension_source
):
    raise SystemExit(14)
if instruction in arguments:
    raise SystemExit(15)
if "@candidate" in arguments:
    raise SystemExit(21)
if "--candidate" in arguments:
    raise SystemExit(22)
agent_dir = Path(os.environ["PI_CODING_AGENT_DIR"])
caller_cwd = os.environ.get("WMO_TEST_CALLER_CWD")
if caller_cwd is not None and (
    Path.cwd().resolve() != agent_dir.parent.resolve()
    or Path.cwd().resolve() == Path(caller_cwd).resolve()
):
    raise SystemExit(23)
model_config = json.loads((agent_dir / "models.json").read_text(encoding="utf-8"))
provider = model_config["providers"]["wmo-injected"]
if provider["models"][0]["id"] != "wmo-injected-model":
    raise SystemExit(16)
tools = json.loads(Path(os.environ["WMO_PI_TOOLS_PATH"]).read_text(encoding="utf-8"))
if tools["tools"][0]["name"] != "lookup":
    raise SystemExit(17)
bridge_url = os.environ["WMO_PI_BRIDGE_URL"]
system_messages = []
for system_filename in ("SYSTEM.md", "APPEND_SYSTEM.md"):
    system_path = Path(".pi") / system_filename
    if system_path.exists():
        system_messages.append(
            {{"role": "system", "content": system_path.read_text(encoding="utf-8")}}
        )
def post(path, payload):
    request = Request(
        bridge_url + path,
        data=json.dumps(payload).encode("utf-8"),
        headers={{"content-type": "application/json"}},
        method="POST",
    )
    with urlopen(request) as response:
        return response.read()
completion = post("/v1/chat/completions", {{
    "messages": [*system_messages, {{"role": "user", "content": instruction}}],
    "stream": True,
}})
if b"Injected model output." not in completion or b'"name":"lookup"' not in completion:
    raise SystemExit(18)
if b"data: [DONE]" not in completion:
    raise SystemExit(20)
tool_result = post("/tools/lookup", {{"call_id": "pi-call", "arguments": {{"id": "record-1"}}}})
tool_result = json.loads(tool_result)
if tool_result["content"] != "tool response" or tool_result["is_error"]:
    raise SystemExit(19)
events = [
    {{"type": "message_end", "message": {{
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
    }}}},
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


def _write_pi_0731_timestamp_fixture(tmp_path: Path) -> Path:
    """Write a static Pi 0.73.1 JSON-mode transcript with a numeric message timestamp."""
    executable = tmp_path / "pi-0731-timestamp-fixture"
    executable.write_text(
        f"""#!{sys.executable}
import json
import sys

message = {{
    "role": "assistant",
    "content": [{{"type": "text", "text": "Pi completed the timestamp fixture."}}],
    "api": "openai-responses",
    "provider": "wmo-injected",
    "model": "wmo-injected-model",
    "usage": {{
        "input": 0,
        "output": 0,
        "cacheRead": 0,
        "cacheWrite": 0,
        "totalTokens": 0,
        "cost": {{"input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0, "total": 0}},
    }},
    "stopReason": "stop",
    "timestamp": 1733234567890,
}}
events = [
    {{"type": "message_end", "message": message}},
    {{"type": "agent_end", "messages": [message]}},
]
sys.stdout.write("\\n".join(json.dumps(event) for event in events) + "\\n")
""",
        encoding="utf-8",
    )
    executable.chmod(executable.stat().st_mode | S_IXUSR)
    return executable


def _write_sleeping_pi_fixture(tmp_path: Path) -> Path:
    """Write a deterministic local executable that exceeds the supplied subprocess deadline."""
    executable = tmp_path / "pi-sleeping-fixture"
    executable.write_text(
        f"#!{sys.executable}\nimport time\ntime.sleep(60)\n",
        encoding="utf-8",
    )
    executable.chmod(executable.stat().st_mode | S_IXUSR)
    return executable
