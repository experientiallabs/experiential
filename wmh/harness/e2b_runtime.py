"""`E2BSandboxRuntime`: run the harness in an E2B microVM; the host's world model answers tools.

The isolation goal without the credential or tunnel problems: only the harness (pi's source + a
node runtime) goes into the sandbox. The world model, its artifact, and every credential stay on
the control host, which never opens an inbound connection — communication rides E2B's own
host-driven channel:

- the sandboxed harness (`pi_entry/entry_e2b.ts`) emits each tool call on **stdout** as
  `__WMH_TOOL__<json>`; E2B streams that to the host live,
- the host answers it with the in-process `AgentEnvironment` (the world model) and writes the
  observation **back into the sandbox** as a response file (`sandbox.files.write` works mid-run),
- the harness polls that file and continues; `submit` emits `__WMH_DONE__<json>`.

The agent's own LLM calls go straight out from the sandbox to the model provider with the user's
key (`PI_AGENT_*`), so no host secret ever enters the VM. The env-action budget and the recorded
transcript (the tool-call `Step`s the host brokered) are enforced host-side, exactly like the
in-process runtimes; `run()` returns the same `RunResult`.

`sandbox_factory` is injected so the broker is testable without a live sandbox or an E2B key.
"""

from __future__ import annotations

import json
import os
import threading
from collections.abc import Callable
from typing import Protocol

from wmh.core.types import Action, ActionKind, EnvState, JsonObject, Observation, Step
from wmh.harness.environment import AgentEnvironment, is_env_action
from wmh.harness.runtime import RunResult, StopReason
from wmh.harness.skills import SkillLibrary
from wmh.harness.tools import ToolSpec
from wmh.providers.base import Provider

E2B_API_KEY_ENV = "E2B_API_KEY"
_ENTRY_TS = os.path.join(os.path.dirname(__file__), "pi_entry", "entry_e2b.ts")

_TOOL_TAG = "__WMH_TOOL__"
_DONE_TAG = "__WMH_DONE__"


class Sandbox(Protocol):
    """The slice of the E2B v2 SDK this runtime uses (see `e2b.Sandbox`)."""

    @property
    def files(self) -> _Files: ...
    @property
    def commands(self) -> _Commands: ...
    def kill(self) -> None: ...


class _Files(Protocol):
    def write(self, path: str, data: str) -> object: ...


class _Commands(Protocol):
    def run(
        self, cmd: str, timeout: int | None = ..., on_stdout: Callable[[str], None] | None = ...
    ) -> object: ...


SandboxFactory = Callable[[], Sandbox]


def _default_sandbox_factory() -> Sandbox:
    from e2b import Sandbox as _E2BSandbox  # ty: ignore[unresolved-import]  # optional dep

    key = os.environ.get(E2B_API_KEY_ENV)
    if not key:
        raise RuntimeError(f"set ${E2B_API_KEY_ENV} to run the harness in an E2B sandbox")
    return _E2BSandbox.create(timeout=900, api_key=key)  # type: ignore[return-value]


class E2BSandboxRuntime:
    """Runs one harness episode in an E2B sandbox; the host world model answers every tool call."""

    def __init__(
        self,
        provider: Provider,
        *,
        files: dict[str, str],
        tools: list[ToolSpec],
        temperature: float = 0.7,
        skills: SkillLibrary | None = None,
        system_prompt: str = "",
        workdir: str = "/home/user/harness",
        max_env_actions: int = 40,
        max_turns: int = 20,
        agent_base_url: str = "https://api.deepseek.com/v1",
        agent_model: str = "deepseek-chat",
        agent_key_env: str = "DEEPSEEK_API_KEY",
        sandbox_factory: SandboxFactory | None = None,
        bootstrap: bool = True,
    ) -> None:
        self._files = files
        self._tools = tools
        self._system_prompt = system_prompt
        self._workdir = workdir
        self._max_env_actions = max_env_actions
        self._max_turns = max_turns
        self._agent_base_url = agent_base_url
        self._agent_model = agent_model
        self._agent_key = os.environ.get(agent_key_env, "")
        self._sandbox_factory = sandbox_factory or _default_sandbox_factory
        self._bootstrap = bootstrap

    def run(self, task_id: str, instruction: str, environment: AgentEnvironment) -> RunResult:
        steps: list[Step] = []
        answer = ""
        env_calls = 0
        done = threading.Event()
        sandbox = self._sandbox_factory()

        def broker(line: str) -> None:
            nonlocal answer, env_calls
            line = line.strip()
            if line.startswith(_DONE_TAG):
                payload = _parse(line[len(_DONE_TAG) :])
                answer = str(payload.get("answer", "")) if payload else ""
                done.set()
                return
            if not line.startswith(_TOOL_TAG):
                return
            payload = _parse(line[len(_TOOL_TAG) :])
            if payload is None:
                return
            call_id = str(payload.get("id", "t"))
            name = str(payload.get("name", ""))
            args = payload.get("arguments")
            action = Action(
                kind=ActionKind.TOOL_CALL,
                name=name,
                arguments=args if isinstance(args, dict) else {},
            )
            if env_calls >= self._max_env_actions:
                obs = Observation(content="environment action budget exhausted", is_error=True)
            elif name not in {t.name for t in self._tools} or not is_env_action(action):
                obs = Observation(content=f"tool {name!r} not available", is_error=True)
            else:
                env_calls += 1
                obs = environment.execute(action)
            steps.append(
                Step(action=action, observation=obs, state_before=EnvState(), task=instruction)
            )
            sandbox.files.write(
                f"{self._workdir}/resp/{call_id}.json",
                json.dumps({"content": obs.content, "is_error": obs.is_error}),
            )

        note = ""
        code = 0
        try:
            self._upload(sandbox, instruction)
            if self._bootstrap:
                self._run_bootstrap(sandbox)
            code, note = self._exec(sandbox, broker)
        except Exception as exc:  # noqa: BLE001 - a sandbox/transport failure fails the episode
            code, note = 1, f"{type(exc).__name__}: {exc}"
        finally:
            try:
                sandbox.kill()
            except Exception:  # noqa: BLE001 - best-effort cleanup
                pass

        if done.is_set():
            return RunResult(
                task_id=task_id,
                steps=steps,
                stop_reason=StopReason.SUBMITTED,
                answer=answer,
                turns=len(steps),
            )
        stop = StopReason.ERROR if code != 0 else StopReason.MAX_TURNS
        steps.append(
            Step(
                action=Action(kind=ActionKind.MESSAGE, content="(e2b runtime)"),
                observation=Observation(
                    content=note or "episode ended without submit", is_error=True
                ),
                state_before=EnvState(),
                task=instruction,
            )
        )
        return RunResult(
            task_id=task_id, steps=steps, stop_reason=stop, answer="", turns=len(steps)
        )

    def _task_json(self, instruction: str) -> str:
        return json.dumps(
            {
                "instruction": instruction,
                "system": self._system_prompt,
                "tools": [
                    {"name": t.name, "description": t.description, "parameters": _schema(t)}
                    for t in self._tools
                ],
            }
        )

    def _upload(self, sandbox: Sandbox, instruction: str) -> None:
        sandbox.files.write(f"{self._workdir}/entry_e2b.ts", _read(_ENTRY_TS))
        sandbox.files.write(f"{self._workdir}/wm_task.json", self._task_json(instruction))
        sandbox.files.write(f"{self._workdir}/resp/.keep", "")
        for rel, content in self._files.items():
            sandbox.files.write(f"{self._workdir}/{rel}", content)

    def _run_bootstrap(self, sandbox: Sandbox) -> None:
        # node 22 (for --experimental-strip-types) + pi's npm deps. Templated away for campaigns.
        sandbox.commands.run(
            "npm install -g n >/dev/null 2>&1 && n 22 >/dev/null 2>&1; "
            f"cd {self._workdir} && npm init -y >/dev/null 2>&1 && "
            "npm install @earendil-works/pi-ai@0.80.3 typebox ignore yaml >/dev/null 2>&1",
            timeout=420,
        )

    def _exec(self, sandbox: Sandbox, broker: Callable[[str], None]) -> tuple[int, str]:
        env = (
            f"PI_TASK_FILE={self._workdir}/wm_task.json PI_RESP_DIR={self._workdir}/resp "
            f"PI_AGENT_BASE_URL={self._agent_base_url} PI_AGENT_MODEL={self._agent_model} "
            f"PI_AGENT_KEY={self._agent_key} PI_MAX_TURNS={self._max_turns}"
        )
        cmd = f"cd {self._workdir} && {env} node --experimental-strip-types entry_e2b.ts"
        result = sandbox.commands.run(cmd, timeout=600, on_stdout=broker)
        tail = str(getattr(result, "stderr", "") or "")[-500:]
        return int(getattr(result, "exit_code", 0) or 0), tail


def _schema(tool: ToolSpec) -> JsonObject:
    props: JsonObject = {
        name: {"type": "string", "description": desc} for name, desc in tool.arguments.items()
    }
    return {"type": "object", "properties": props, "required": list(tool.arguments)}


def _parse(text: str) -> JsonObject | None:
    try:
        obj = json.loads(text)
    except json.JSONDecodeError:
        return None
    return obj if isinstance(obj, dict) else None


def _read(path: str) -> str:
    with open(path, encoding="utf-8") as fh:
        return fh.read()
