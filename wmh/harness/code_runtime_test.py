"""CodeRuntime tests: the kit's budget/recording/crash-isolation guarantees, offline."""

from __future__ import annotations

import pytest

from wmh.core.types import Action, Observation
from wmh.harness.code_runtime import (
    DEFAULT_RUNTIME_CODE,
    CodeRuntime,
    RunBudget,
    compile_harness_code,
)
from wmh.harness.doc import CODE_RUNTIME_ID, HarnessDoc, Surface, SurfaceKind, code_baseline
from wmh.harness.runtime import Runtime, StopReason
from wmh.harness.tools import SUBMIT, TOOL_REGISTRY
from wmh.providers.base import Completion, Message, ProviderConfig, ProviderKind


class ScriptedProvider:
    """Replays canned completions in order; records calls."""

    def __init__(self, replies: list[str]) -> None:
        self.config = ProviderConfig(kind=ProviderKind.BEDROCK, model="m")
        self._replies = replies
        self.calls = 0

    def complete(
        self,
        system: str,
        messages: list[Message],
        *,
        temperature: float = 0.7,
        max_tokens: int = 2048,
    ) -> Completion:
        reply = self._replies[min(self.calls, len(self._replies) - 1)]
        self.calls += 1
        return Completion(text=reply)

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [[0.0] for _ in texts]

    def verify(self):  # noqa: ANN201 - test fake never calls it
        raise NotImplementedError


class FakeEnv:
    """Echo environment; records executed actions."""

    def __init__(self) -> None:
        self.actions: list[Action] = []
        self.closed = False

    def execute(self, action: Action) -> Observation:
        self.actions.append(action)
        return Observation(content=f"ran {action.name}")

    def close(self) -> None:
        self.closed = True


def _runtime(code: str, replies: list[str] | None = None, **kwargs) -> CodeRuntime:  # noqa: ANN003
    provider = ScriptedProvider(replies or ['{"tool": "submit", "arguments": {"answer": "ok"}}'])
    tools = [TOOL_REGISTRY["bash"], SUBMIT]
    return CodeRuntime(provider, code=code, tools=tools, **kwargs)


# -- compile validation -------------------------------------------------------------------------


def test_compile_rejects_syntax_errors_and_missing_run() -> None:
    with pytest.raises(ValueError, match="does not compile"):
        compile_harness_code("def run(kit:\n")
    with pytest.raises(ValueError, match="must define"):
        compile_harness_code("x = 1\n")
    compile_harness_code(DEFAULT_RUNTIME_CODE)  # the seed passes its own gate


def test_run_must_be_callable() -> None:
    with pytest.raises(ValueError, match="not callable"):
        _runtime("run = 42\n")


# -- the default loop, end to end ---------------------------------------------------------------


def test_default_code_runs_the_baseline_loop() -> None:
    runtime = _runtime(
        DEFAULT_RUNTIME_CODE,
        replies=[
            '{"tool": "bash", "arguments": {"command": "ls"}}',
            '{"tool": "submit", "arguments": {"answer": "two files"}}',
        ],
        system_prompt="You are a capable agent.",
    )
    env = FakeEnv()
    result = runtime.run("t1", "list the files", env)
    assert result.stop_reason is StopReason.SUBMITTED
    assert result.answer == "two files"
    assert [a.name for a in env.actions] == ["bash"]
    assert len(result.steps) == 1  # the bash step, kit-recorded
    assert isinstance(runtime, Runtime)


# -- kit guarantees -----------------------------------------------------------------------------


def test_budget_exhaustion_stops_the_episode() -> None:
    infinite = 'def run(kit):\n    while True:\n        kit.execute("bash", {"command": "true"})\n'
    runtime = _runtime(infinite, budget=RunBudget(max_llm_calls=5, max_env_actions=3))
    result = runtime.run("t1", "loop forever", FakeEnv())
    assert result.stop_reason is StopReason.BUDGET
    # 3 recorded env steps + the terminal error step explaining why the episode ended.
    assert len(result.steps) == 4
    assert result.steps[-1].observation.is_error
    assert "budget exhausted" in result.steps[-1].observation.content


def test_llm_budget_is_enforced_too() -> None:
    chatty = 'def run(kit):\n    while True:\n        kit.complete("s", [("user", "hi")])\n'
    runtime = _runtime(chatty, budget=RunBudget(max_llm_calls=2, max_env_actions=5))
    result = runtime.run("t1", "chat forever", FakeEnv())
    assert result.stop_reason is StopReason.BUDGET


def test_crash_is_isolated_and_transcript_survives() -> None:
    crashy = (
        "def run(kit):\n"
        '    kit.execute("bash", {"command": "ls"})\n'
        '    raise RuntimeError("boom")\n'
    )
    result = _runtime(crashy).run("t1", "crash", FakeEnv())
    assert result.stop_reason is StopReason.ERROR
    assert result.answer == ""
    assert len(result.steps) == 2  # the real step survives, plus the error note
    assert "RuntimeError: boom" in result.steps[-1].observation.content


def test_transcript_is_kit_recorded_not_code_claimed() -> None:
    # Code that claims success without acting produces an EMPTY transcript: the judge sees
    # exactly what the kit recorded, nothing else.
    liar = 'def run(kit):\n    return "I did everything"\n'
    result = _runtime(liar).run("t1", "do things", FakeEnv())
    assert result.stop_reason is StopReason.SUBMITTED
    assert result.steps == []


def test_unavailable_tool_is_an_error_observation_not_an_env_call() -> None:
    curious = (
        "def run(kit):\n"
        '    obs = kit.execute("rm_rf", {})\n'
        '    return "err" if obs.is_error else "ok"\n'
    )
    env = FakeEnv()
    result = _runtime(curious).run("t1", "poke", env)
    assert result.answer == "err"
    assert env.actions == []  # never reached the environment
    assert result.steps[0].observation.is_error


def test_tuple_messages_convert_and_bad_roles_raise() -> None:
    code = (
        "def run(kit):\n"
        '    return kit.complete("s", [("user", "hi"), ("assistant", "yo"), ("user", "go")])\n'
    )
    result = _runtime(code, replies=["fine"]).run("t1", "chat", FakeEnv())
    assert result.answer == "fine"
    bad = 'def run(kit):\n    return kit.complete("s", [("system", "no")])\n'
    result = _runtime(bad).run("t1", "chat", FakeEnv())
    assert result.stop_reason is StopReason.ERROR
    assert "role" in result.steps[-1].observation.content


def test_non_string_return_is_empty_answer() -> None:
    result = _runtime("def run(kit):\n    return 42\n").run("t1", "x", FakeEnv())
    assert result.stop_reason is StopReason.SUBMITTED
    assert result.answer == ""


# -- HarnessDoc integration ----------------------------------------------------------------------


def test_code_baseline_validates_and_dispatches_code_runtime() -> None:
    doc = code_baseline("seed")
    assert doc.surface(CODE_RUNTIME_ID) is not None
    provider = ScriptedProvider(['{"tool": "submit", "arguments": {"answer": "done"}}'])
    runtime = doc.runtime(provider)
    assert isinstance(runtime, CodeRuntime)
    result = runtime.run("t1", "do it", FakeEnv())
    assert result.stop_reason is StopReason.SUBMITTED
    assert result.answer == "done"


def test_doc_rejects_bad_code_surface_at_construction() -> None:
    core = Surface(id="prompt:core", kind=SurfaceKind.PROMPT, content="p")
    bad = Surface(id=CODE_RUNTIME_ID, kind=SurfaceKind.CODE, content="x = (")
    with pytest.raises(ValueError, match="does not compile"):
        HarnessDoc(name="x", surfaces=[core, bad])
    misnamed = Surface(id="code:other", kind=SurfaceKind.CODE, content="def run(kit):\n    pass\n")
    with pytest.raises(ValueError, match="path-less code surface"):
        HarnessDoc(name="x", surfaces=[core, misnamed])


def test_doc_without_code_surface_keeps_the_fixed_loop() -> None:
    from wmh.harness.runtime import AgentRuntime

    doc = HarnessDoc.baseline("plain")
    provider = ScriptedProvider(['{"tool": "submit", "arguments": {"answer": "ok"}}'])
    assert isinstance(doc.runtime(provider), AgentRuntime)
