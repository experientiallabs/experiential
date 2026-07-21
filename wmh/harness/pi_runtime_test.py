"""PiRuntime unit tests: schema + tool routing/budget, offline (no ssh/node)."""

from __future__ import annotations

import threading
import time
from socket import socket
from typing import cast

import pytest
from llm_waterfall import ChatRequest, ChatResponse

from wmh.core.types import Action, Observation
from wmh.harness.doc import (
    MAX_OUTPUT_TOKENS_ID,
    MAX_TURNS_ID,
    RUNTIME_KIND_ID,
    TEMPERATURE_ID,
    TOOL_POLICY_ID,
    HarnessDoc,
    Surface,
    SurfaceKind,
)
from wmh.harness.pi_runtime import (
    PiRuntime,
    _Episode,
    _params_schema,
    _ShimHandler,
    _ShimServer,
)
from wmh.harness.skills import Skill, SkillLibrary
from wmh.harness.tools import SUBMIT, TOOL_REGISTRY


class _Env:
    def __init__(self) -> None:
        self.actions: list[Action] = []

    def execute(self, action: Action) -> Observation:
        self.actions.append(action)
        return Observation(content=f"ran {action.name}")

    def close(self) -> None:
        pass


class _Provider:
    def complete_chat(self, request: ChatRequest) -> ChatResponse:
        del request
        return ChatResponse.model_validate(
            {
                "choices": [
                    {
                        "message": {"role": "assistant", "content": "ok"},
                        "finish_reason": "stop",
                    }
                ]
            }
        )


def _episode(
    env: _Env, *, budget: int = 40, skills: SkillLibrary | None = None, temperature: float = 0.7
) -> _Episode:
    return _Episode(
        instruction="do it",
        system_prompt="sys",
        tools=[TOOL_REGISTRY["bash"], SUBMIT],
        provider=_Provider(),
        environment=env,
        temperature=temperature,
        skills=skills if skills is not None else SkillLibrary(),
        max_env_actions=budget,
        max_turns=7,
        max_output_tokens=16384,
    )


def test_task_json_shape() -> None:
    ep = _episode(_Env())
    import json

    tj = json.loads(json.dumps(ep.task_json()))
    assert tj["instruction"] == "do it" and tj["system"] == "sys"
    assert tj["max_turns"] == 7
    assert tj["max_output_tokens"] == 16384
    names = {t["name"] for t in tj["tools"]}
    assert "bash" in names and "submit" in names
    schema = json.loads(json.dumps(_params_schema(TOOL_REGISTRY["bash"])))
    assert schema["type"] == "object" and "command" in schema["properties"]


def test_tool_routing_records_steps_and_rejects_unknown() -> None:
    env = _Env()
    ep = _episode(env)
    ok = ep.run_tool("bash", {"command": "ls"})
    assert ok == {"content": "ran bash", "is_error": False}
    bad = ep.run_tool("rm_rf", {})
    assert bad["is_error"] is True and env.actions == [ep.steps[0].action]
    assert len(ep.steps) == 2  # both recorded (the transcript the judge sees)


def test_env_action_budget_is_enforced() -> None:
    env = _Env()
    ep = _episode(env, budget=2)
    for _ in range(4):
        ep.run_tool("bash", {"command": "true"})
    assert len(env.actions) == 2  # only the budgeted calls reached the environment
    assert "budget exhausted" in ep.steps[-1].observation.content


def test_worker_request_uses_document_temperature() -> None:
    request = _episode(_Env(), temperature=0.35).worker_request(
        {"messages": [], "temperature": 1.75}
    )
    assert request.temperature == 0.35


def test_read_skill_is_runtime_local_and_does_not_consume_environment_budget() -> None:
    env = _Env()
    skills = SkillLibrary(
        [Skill(name="count-words", description="count words", body="wc -w <path>")]
    )
    ep = _episode(env, budget=0, skills=skills)
    ep.tools.append(TOOL_REGISTRY["read_skill"])

    found = ep.run_tool("read_skill", {"name": "count-words"})
    missing = ep.run_tool("read_skill", {"name": "ghost"})

    assert found == {"content": "wc -w <path>", "is_error": False}
    assert missing == {"content": "no skill named 'ghost'", "is_error": True}
    assert env.actions == []


def test_local_shim_close_waits_for_active_environment_handler(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    handler_started = threading.Event()
    release_handler = threading.Event()
    close_finished = threading.Event()
    server = _ShimServer(("127.0.0.1", 0), _ShimHandler, bind_and_activate=False)

    def blocking_request(_request: object, _client_address: object) -> None:
        handler_started.set()
        assert release_handler.wait(timeout=2)

    monkeypatch.setattr(server, "process_request_thread", blocking_request)
    server.process_request(cast("socket", object()), ("127.0.0.1", 1))
    assert handler_started.wait(timeout=1)

    def close_server() -> None:
        server.server_close()
        close_finished.set()

    close_thread = threading.Thread(target=close_server)
    close_thread.start()
    try:
        time.sleep(0.05)
        assert _ShimServer.daemon_threads is False
        assert close_finished.is_set() is False
        assert close_thread.is_alive()
    finally:
        release_handler.set()
        close_thread.join(timeout=2)

    assert close_thread.is_alive() is False
    assert close_finished.is_set()


def test_doc_dispatches_pi_runtime_for_pi_node_kind() -> None:
    from wmh.providers.base import ProviderConfig, ProviderKind

    class _P:
        config = ProviderConfig(kind=ProviderKind.BEDROCK, model="m")

        def complete(self, *a, **k) -> object:  # noqa: ANN002, ANN003
            raise NotImplementedError

        def complete_chat(self, request: ChatRequest) -> ChatResponse:
            return _Provider().complete_chat(request)

        def embed(self, texts) -> list:  # noqa: ANN001
            return [[0.0] for _ in texts]

        def verify(self) -> object:
            raise NotImplementedError

    doc = HarnessDoc(
        name="pi",
        surfaces=[
            Surface(id="prompt:core", kind=SurfaceKind.PROMPT, content="p"),
            Surface(id=TOOL_POLICY_ID, kind=SurfaceKind.TOOL_POLICY, content="bash\nsubmit"),
            Surface(id=RUNTIME_KIND_ID, kind=SurfaceKind.PARAM, content="pi-node"),
            Surface(id=MAX_TURNS_ID, kind=SurfaceKind.PARAM, content="7"),
            Surface(id=MAX_OUTPUT_TOKENS_ID, kind=SurfaceKind.PARAM, content="16384"),
            Surface(id=TEMPERATURE_ID, kind=SurfaceKind.PARAM, content="0.35"),
            Surface(
                id="skill:count-words",
                kind=SurfaceKind.SKILL,
                content=Skill(
                    name="count-words", description="count words", body="wc -w <path>"
                ).to_markdown(),
            ),
            Surface(
                id="code:src-agent-ts",
                kind=SurfaceKind.CODE,
                path="src/agent.ts",
                content="// agent",
            ),
        ],
    )
    assert doc.runtime_kind() == "pi-node"
    assert [s.path for s in doc.code_files()] == ["src/agent.ts"]
    from typing import cast

    from wmh.providers.base import Provider

    runtime = doc.runtime(cast("Provider", _P()))
    assert isinstance(runtime, PiRuntime)
    assert runtime._max_turns == 7  # noqa: SLF001 - document parameter reaches entry.ts
    assert runtime._max_output_tokens == 16384  # noqa: SLF001 - same agent model contract
    assert runtime._temperature == 0.35  # noqa: SLF001 - same worker sampling contract
    assert {tool.name for tool in runtime._tools} >= {  # noqa: SLF001 - runtime plumbing
        "bash",
        "submit",
        "read_skill",
    }
