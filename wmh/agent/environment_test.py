"""Tests for the environment seam: action->command rendering and the world-model backend."""

from __future__ import annotations

from wmh.agent.environment import (
    WorldModelEnvironment,
    _render_command,
    _result_to_observation,
    is_env_action,
)
from wmh.agent.tools import ToolCall, to_action
from wmh.core.types import Action, ActionKind
from wmh.engine.world_model import WorldModel
from wmh.providers.base import Completion, Message, ProviderConfig, ProviderKind
from wmh.retrieval import EmbeddingRetriever, HashingEmbedder


def _tc(tool: str, **args: str) -> Action:
    return to_action(ToolCall(tool=tool, arguments=dict(args)))


def test_render_bash_command() -> None:
    assert _render_command(_tc("bash", command="echo hi")) == "echo hi"


def test_render_read_file_is_cat() -> None:
    assert _render_command(_tc("read_file", path="/etc/hosts")) == "cat '/etc/hosts'"


def test_render_write_file_uses_heredoc() -> None:
    cmd = _render_command(_tc("write_file", path="/a/b.txt", content="line1\nline2"))
    assert cmd is not None
    assert "cat > '/a/b.txt'" in cmd
    assert "line1\nline2" in cmd


def test_non_env_tool_renders_none() -> None:
    assert _render_command(_tc("submit", answer="x")) is None
    assert not is_env_action(_tc("submit", answer="x"))
    assert is_env_action(_tc("bash", command="ls"))


class _FakeResult:
    """Stands in for an E2B CommandResult / CommandExitException (same duck-typed fields)."""

    def __init__(self, stdout: str, stderr: str, exit_code: int) -> None:
        self.stdout = stdout
        self.stderr = stderr
        self.exit_code = exit_code


def test_result_to_observation_success() -> None:
    obs = _result_to_observation(_FakeResult("hello\n", "", 0))
    assert obs.content == "hello"
    assert not obs.is_error
    assert obs.metadata["exit_code"] == 0


def test_result_to_observation_failure_flags_error() -> None:
    # A raised CommandExitException carries the same fields; a non-zero exit must become is_error.
    obs = _result_to_observation(_FakeResult("", "cat: /nope: No such file", 3))
    assert obs.is_error
    assert obs.metadata["exit_code"] == 3
    assert "No such file" in obs.content


class _WMProvider:
    """World-model provider that always returns a canned observation JSON."""

    def __init__(self) -> None:
        self.config = ProviderConfig(kind=ProviderKind.BEDROCK, model="m")

    def complete(
        self,
        system: str,
        messages: list[Message],
        *,
        temperature: float = 0.7,
        max_tokens: int = 2048,
    ) -> Completion:
        return Completion(text='{"output": "simulated", "is_error": false}')

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [[0.0] for _ in texts]

    def verify(self):  # noqa: ANN202
        raise NotImplementedError


def test_world_model_environment_executes_via_step() -> None:
    wm = WorldModel(_WMProvider(), EmbeddingRetriever(HashingEmbedder(dim=16)))
    env = WorldModelEnvironment(wm, task="do a thing")
    obs = env.execute(Action(kind=ActionKind.TOOL_CALL, name="bash", arguments={"command": "ls"}))
    assert obs.content == "simulated"
    env.close()
