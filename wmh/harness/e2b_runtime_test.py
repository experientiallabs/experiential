"""E2BSandboxRuntime tests: offline via a fake sandbox that plays the stdio broker protocol.

No E2B key, no microVM. The fake sandbox, on `commands.run`, emits scripted `__WMH_TOOL__` /
`__WMH_DONE__` lines to the `on_stdout` callback and reads back the response files the host writes
via `files.write` — exercising the whole broker: tool routing to the in-process world model, the
env-action budget, response-file round trips, and submit/done handling.
"""

from __future__ import annotations

import json

from wmh.core.types import Action, Observation
from wmh.harness.e2b_runtime import E2BSandboxRuntime
from wmh.harness.runtime import StopReason
from wmh.harness.tools import SUBMIT, TOOL_REGISTRY


class _Env:
    def __init__(self) -> None:
        self.actions: list[Action] = []

    def execute(self, action: Action) -> Observation:
        self.actions.append(action)
        return Observation(content=f"ran {action.name}")

    def close(self) -> None:
        pass


class _Result:
    exit_code = 0
    stderr = ""


class _Files:
    def __init__(self, store: dict[str, str]) -> None:
        self._store = store

    def write(self, path: str, data: str) -> None:
        self._store[path] = data


class _Commands:
    def __init__(self, play) -> None:  # noqa: ANN001
        self._play = play

    def run(self, cmd, timeout=None, on_stdout=None) -> _Result:  # noqa: ANN001
        return self._play(cmd, on_stdout)


class _FakeSandbox:
    """Plays a scripted harness: emits tool/done lines, then reads host-written response files."""

    def __init__(self, script: list[dict], workdir: str = "/home/user/harness") -> None:
        self.uploaded: dict[str, str] = {}
        self._script = script
        self._workdir = workdir
        self.killed = False
        self.files = _Files(self.uploaded)
        self.commands = _Commands(self._play)

    def _play(self, cmd: str, on_stdout):  # noqa: ANN001, ANN202
        if on_stdout is None:  # bootstrap call
            return _Result()
        for step in self._script:
            if step["kind"] == "tool":
                cid = step["id"]
                on_stdout(
                    "__WMH_TOOL__"
                    + json.dumps(
                        {"id": cid, "name": step["name"], "arguments": step.get("args", {})}
                    )
                    + "\n"
                )
                # the host writes the response file synchronously in the callback; read it back
                path = f"{self._workdir}/resp/{cid}.json"
                if path in self.uploaded:
                    step["_resp"] = json.loads(self.uploaded[path])
            elif step["kind"] == "done":
                on_stdout("__WMH_DONE__" + json.dumps({"answer": step["answer"]}) + "\n")
        return _Result()

    def kill(self) -> None:
        self.killed = True


class _P:
    from wmh.providers.base import ProviderConfig, ProviderKind

    config = ProviderConfig(kind=ProviderKind.BEDROCK, model="m")

    def complete(self, *a, **k) -> object:  # noqa: ANN002, ANN003
        raise NotImplementedError

    def embed(self, texts) -> list:  # noqa: ANN001
        return [[0.0] for _ in texts]

    def verify(self) -> object:
        raise NotImplementedError


def _runtime(sandbox, **kw):  # noqa: ANN001, ANN003, ANN202
    from typing import cast

    from wmh.providers.base import Provider

    return E2BSandboxRuntime(
        cast("Provider", _P()),
        files={"src/agent.ts": "// agent"},
        tools=[TOOL_REGISTRY["bash"], SUBMIT],
        sandbox_factory=lambda: sandbox,
        bootstrap=False,
        **kw,
    )


def test_uploads_harness_task_and_entry() -> None:
    sb = _FakeSandbox([{"kind": "done", "answer": "x"}])
    _runtime(sb).run("t1", "do it", _Env())
    assert "/home/user/harness/entry_e2b.ts" in sb.uploaded
    assert sb.uploaded["/home/user/harness/src/agent.ts"] == "// agent"
    task = json.loads(sb.uploaded["/home/user/harness/wm_task.json"])
    assert task["instruction"] == "do it"
    assert {t["name"] for t in task["tools"]} >= {"bash", "submit"}
    assert sb.killed


def test_tool_calls_route_to_world_model_over_the_broker() -> None:
    env = _Env()
    script = [
        {"kind": "tool", "id": "t0", "name": "bash", "args": {"command": "ls"}},
        {"kind": "done", "answer": "done-42"},
    ]
    result = _runtime(_FakeSandbox(script)).run("t1", "do it", env)
    assert result.stop_reason is StopReason.SUBMITTED
    assert result.answer == "done-42"
    assert [a.name for a in env.actions] == ["bash"]  # the host WM answered the tool call
    assert script[0]["_resp"] == {"content": "ran bash", "is_error": False}  # bridged back
    assert len(result.steps) == 1


def test_env_action_budget_enforced() -> None:
    env = _Env()
    script = [{"kind": "tool", "id": f"t{i}", "name": "bash", "args": {}} for i in range(4)]
    script.append({"kind": "done", "answer": "ok"})
    result = _runtime(_FakeSandbox(script), max_env_actions=2).run("t1", "x", env)
    assert len(env.actions) == 2  # only the budgeted calls reached the world model
    over = script[3]["_resp"]  # over-budget call got an error observation
    assert isinstance(over, dict) and over["is_error"] is True
    assert result.stop_reason is StopReason.SUBMITTED


def test_no_submit_reports_error_and_kills_sandbox() -> None:
    sb = _FakeSandbox([{"kind": "tool", "id": "t0", "name": "bash", "args": {}}])  # never submits
    result = _runtime(sb).run("t1", "x", _Env())
    assert result.stop_reason in (StopReason.MAX_TURNS, StopReason.ERROR)
    assert sb.killed
