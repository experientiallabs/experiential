"""PiRuntime unit tests: schema + tool routing/budget, offline (no ssh/node)."""

from __future__ import annotations

from llm_waterfall import ChatRequest, ChatResponse

from wmh.core.types import Action, Observation
from wmh.harness.doc import RUNTIME_KIND_ID, TOOL_POLICY_ID, HarnessDoc, Surface, SurfaceKind
from wmh.harness.pi_runtime import PiRuntime, _Episode, _params_schema
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


def _episode(env: _Env, *, budget: int = 40) -> _Episode:
    return _Episode(
        instruction="do it",
        system_prompt="sys",
        tools=[TOOL_REGISTRY["bash"], SUBMIT],
        provider=_Provider(),
        environment=env,
        max_env_actions=budget,
    )


def test_task_json_shape() -> None:
    ep = _episode(_Env())
    import json

    tj = json.loads(json.dumps(ep.task_json()))
    assert tj["instruction"] == "do it" and tj["system"] == "sys"
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

    assert isinstance(doc.runtime(cast("Provider", _P())), PiRuntime)
