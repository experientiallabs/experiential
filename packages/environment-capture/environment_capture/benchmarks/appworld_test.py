"""Tests for the AppWorld adapter, env, and capture agent.

The real ``appworld`` engine lives in a separate venv, so these tests never touch it: a tiny
stdlib ``fake_backend.py`` written into ``tmp_path`` speaks the same stdio JSON protocol as
``backend/world_backend.py`` (proving statefulness and the serve/grade wiring), and a stub converse
client drives the agent through a scripted tool-use sequence.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from environment_capture.adapter import ExecResult
from environment_capture.benchmarks.appworld import (
    AppWorldAdapter,
    AppWorldAgent,
    AppWorldError,
)
from environment_capture.trajectory import JsonValue, Task

# A stand-in for backend/world_backend.py: stdlib-only, keeps world state across execute calls
# (returns a growing step count), marks completion when the code contains COMPLETE, and grades from
# an env-name-encoded reward, so the adapter's serve/grade wiring can be checked without appworld.
_FAKE_BACKEND = """\
import json, sys
def main():
    op = sys.argv[1]
    if op == "serve":
        print(json.dumps({"ready": True}), flush=True)
        history = []
        for line in sys.stdin:
            req = json.loads(line)
            if req.get("op") == "close":
                break
            code = req.get("code", "")
            history.append(code)
            if code == "BOOM":
                print(json.dumps({"output": "Execution failed. boom", "error": True,
                                  "completed": False}), flush=True)
            else:
                print(json.dumps({"output": "steps=%d" % len(history),
                                  "error": False, "completed": "COMPLETE" in code}), flush=True)
    elif op == "grade":
        experiment_name = sys.argv[3]
        reward = float(experiment_name.split("--", 1)[0].rsplit("-", 1)[-1])
        print(json.dumps({"reward": reward, "success": reward >= 1.0, "num_tests": 2}))
main()
"""


@pytest.fixture()
def root(tmp_path: Path) -> Path:
    backend = tmp_path / "backend"
    backend.mkdir()
    (backend / "fake_backend.py").write_text(_FAKE_BACKEND, encoding="utf-8")
    data = tmp_path / "data"
    data.mkdir()
    tasks = [
        {"task_id": "aw-train-0", "prompt": "Do a thing.", "data": {"appworld_id": "abc_1"}},
        {"task_id": "aw-train-1", "prompt": "Do another.", "data": {"appworld_id": "abc_2"}},
    ]
    (data / "train.jsonl").write_text(
        "\n".join(json.dumps(t) for t in tasks) + "\n", encoding="utf-8"
    )
    return tmp_path


def _adapter(root: Path, *, experiment_prefix: str = "reward-1.0") -> AppWorldAdapter:
    return AppWorldAdapter(
        root,
        experiment_prefix=experiment_prefix,
        venv_python=Path("python3"),
        backend=root / "backend" / "fake_backend.py",
    )


def test_tasks_parses_split(root: Path) -> None:
    tasks = _adapter(root).tasks("train")
    assert [t.task_id for t in tasks] == ["aw-train-0", "aw-train-1"]
    assert tasks[0].data["appworld_id"] == "abc_1"


def test_open_env_state_persists_across_calls(root: Path) -> None:
    adapter = _adapter(root)
    env = adapter.open_env(adapter.tasks("train")[0])
    try:
        # The growing step count proves the world is the SAME process across execute calls.
        assert env.execute("x = 1").output == "steps=1"
        assert env.execute("y = 2").output == "steps=2"
        assert not env.completed
        result = env.execute("apis.supervisor.complete_task()  # COMPLETE")
        assert result.returncode == 0
        assert env.completed
    finally:
        env.close()


def test_execute_flags_errors(root: Path) -> None:
    adapter = _adapter(root)
    env = adapter.open_env(adapter.tasks("train")[0])
    try:
        result = env.execute("BOOM")
        assert result.returncode == 1
        assert "Execution failed" in result.output
    finally:
        env.close()


def test_grade_reads_backend_reward(root: Path) -> None:
    adapter = _adapter(root, experiment_prefix="reward-1.0")
    assert adapter.grade(adapter.tasks("train")[0], "ignored") == 1.0
    half = _adapter(root, experiment_prefix="reward-0.5")
    assert half.grade(half.tasks("train")[0], "ignored") == 0.5


def test_open_env_raises_on_missing_appworld_id(root: Path) -> None:
    adapter = _adapter(root)
    with pytest.raises(ValueError, match="appworld_id"):
        adapter.open_env(Task(task_id="x", prompt="p", data={}))


def test_env_raises_when_backend_cannot_start(root: Path) -> None:
    adapter = AppWorldAdapter(
        root, venv_python=Path("python3"), backend=root / "backend" / "does_not_exist.py"
    )
    with pytest.raises((AppWorldError, OSError)):
        adapter.open_env(adapter.tasks("train")[0])


# --------------------------------------------------------------------------- agent


class _ScriptedClient:
    """A ConverseClient stub that replays a fixed list of assistant messages, in order."""

    def __init__(self, messages: list[dict[str, JsonValue]]) -> None:
        self._messages = messages
        self.calls = 0

    def converse(
        self,
        *,
        modelId: str,  # noqa: N803 - matches the boto3 converse signature
        messages: list[JsonValue],
        system: list[JsonValue],
        toolConfig: JsonValue,  # noqa: N803
        inferenceConfig: JsonValue,  # noqa: N803
    ) -> dict[str, JsonValue]:
        message = self._messages[self.calls]
        self.calls += 1
        return {"output": {"message": message}}


class _FakeEnv:
    """Minimal AppWorldEnv stand-in recording the code the agent runs."""

    def __init__(self) -> None:
        self.executed: list[str] = []
        self.completed = False

    def execute(self, command: str) -> ExecResult:
        self.executed.append(command)
        self.completed = "complete_task" in command
        return ExecResult(output=f"ran: {command}", returncode=0)

    def close(self) -> None:
        pass


def _tool_use_msg(name: str, arguments: dict[str, JsonValue]) -> dict[str, JsonValue]:
    return {"content": [{"toolUse": {"toolUseId": f"tu-{name}", "name": name, "input": arguments}}]}


def test_agent_runs_python_then_finishes() -> None:
    client = _ScriptedClient(
        [
            _tool_use_msg("execute_python", {"code": "apis.supervisor.complete_task(answer='42')"}),
            _tool_use_msg("finish", {"answer": "42"}),
        ]
    )
    agent = AppWorldAgent("test-model", client=client)
    env = _FakeEnv()
    run = agent.run(Task(task_id="aw-train-0", prompt="Answer 42."), env)

    assert env.executed == ["apis.supervisor.complete_task(answer='42')"]
    assert run.final_answer == "42"
    assert run.model == "test-model"
    assert len(run.steps) == 1
    assert run.steps[0].action.name == "execute_python"
    assert run.steps[0].action.arguments["code"] == "apis.supervisor.complete_task(answer='42')"


def test_agent_stops_at_max_steps() -> None:
    looping = [_tool_use_msg("execute_python", {"code": f"step_{i}"}) for i in range(10)]
    agent = AppWorldAgent("test-model", client=_ScriptedClient(looping), max_steps=3)
    env = _FakeEnv()
    run = agent.run(Task(task_id="aw-train-0", prompt="Loop."), env)
    assert len(run.steps) == 3
    assert [s.action.arguments["code"] for s in run.steps] == ["step_0", "step_1", "step_2"]


def test_agent_ends_on_plain_text() -> None:
    agent = AppWorldAgent(
        "test-model", client=_ScriptedClient([{"content": [{"text": "I give up."}]}])
    )
    run = agent.run(Task(task_id="aw-train-0", prompt="x"), _FakeEnv())
    assert run.final_answer == "I give up."
    assert run.steps == []


def test_retry_boot_gets_fresh_experiment_name(root: Path) -> None:
    """run_capture's retry re-opens the env; the second boot must NOT reuse the experiment dir
    the crashed first attempt dirtied (AppWorld state is per-experiment and persists). The first
    boot keeps the legacy un-suffixed name so committed corpora stay reproducible."""
    adapter = _adapter(root)
    task = adapter.tasks("train")[0]

    env1 = adapter.open_env(task)
    name1 = adapter._experiment(task)
    env1.close()
    env2 = adapter.open_env(task)
    name2 = adapter._experiment(task)
    env2.close()

    assert name1 == "reward-1.0--abc_1"  # first boot: unchanged naming
    assert name2 == "reward-1.0--abc_1--a2"  # retry: disjoint experiment dir
    assert name1 != name2


def test_boot_serials_are_per_task(root: Path) -> None:
    adapter = _adapter(root)
    task_a, task_b = adapter.tasks("train")
    adapter.open_env(task_a).close()
    adapter.open_env(task_b).close()
    # each task saw exactly one boot, so both keep the un-suffixed name
    assert adapter._experiment(task_a).endswith("--abc_1")
    assert adapter._experiment(task_b).endswith("--abc_2")
