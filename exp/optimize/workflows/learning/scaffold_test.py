"""Real local HTTP and official SDK tool episodes using the existing customer-agent loop."""

from __future__ import annotations

import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from exp.common.claas.generation import GenerationRequest, GenerationResult
from exp.common.models import AssistantAction, ToolCall
from exp.common.rollouts import StopReason
from exp.common.tasks import TaskCase, ToolSchema
from exp.optimize.claas.buffer.store_test import item
from exp.optimize.claas.service.configuration import RunConfiguration
from exp.optimize.claas.service.controller import LearningController
from exp.optimize.claas.service.controller_test import Runtime
from exp.optimize.claas.service.http.app import create_app
from exp.optimize.claas.service.http.app_test import serve
from exp.optimize.claas.training_contracts_test import spec
from exp.optimize.workflows.learning.scaffold import run_learning_episode
from exp.runtime.agents.chat import ChatAgentRuntime
from exp.runtime.claas.client import LearningClient
from exp.runtime.environments.interface import EnvironmentSession, Observation


class ToolRuntime(Runtime):
    """Generate deterministic tool actions without replacing HTTP, queue, or agent orchestration."""

    async def generate(self, request: GenerationRequest) -> GenerationResult:
        """Emit a function call, then a final answer after seeing the actual tool observation."""
        self.generate_count += 1
        observed = request.messages and request.messages[-1].role == "tool"
        action = (
            AssistantAction(content="The tool returned value.")
            if observed
            else AssistantAction(
                tool_calls=(
                    ToolCall(call_id="call-1", name="lookup", arguments={"key": "example"}),
                )
            )
        )
        tokens = item(policy=self.policy_revision).experience.exact_tokens
        assert tokens is not None
        return GenerationResult(
            response_id=request.request_id,
            action=action,
            exact_tokens=tokens,
            raw_text=action.content or "tool call",
        )


class Environment:
    """An external executable environment with explicit reset and cleanup ownership."""

    def __init__(self) -> None:
        """Record the test's real tool execution and cleanup count."""
        self.calls: list[ToolCall] = []
        self.closed = 0

    @contextmanager
    def open(self, task: TaskCase) -> Iterator[EnvironmentSession]:
        """Reset a fresh task and release it after the existing agent finishes."""
        try:
            yield self
        finally:
            self.closed += 1

    def execute(self, action: ToolCall) -> Observation:
        """Return an ordinary observation to the existing ChatAgentRuntime loop."""
        self.calls.append(action)
        return Observation(content="value")


def test_existing_agent_uses_real_http_tools_and_later_feedback(tmp_path: Path) -> None:
    """Drive SDK, HTTP service, controller, SQLite, agent tools, and explicit feedback together."""
    runtime = ToolRuntime()
    controller = LearningController(tmp_path, spec(), runtime, RunConfiguration())
    app = create_app(controller, api_key="local-roundtrip-test-key")
    task = TaskCase(
        task_id="example-task",
        lineage_group_id="example-group",
        partition="fit",
        instruction="Use lookup to find the value.",
        workload_weight=1,
        source_trace_ids=("authored-example",),
        tools=(
            ToolSchema(
                name="lookup",
                description="Read one fixture key.",
                input_schema={
                    "type": "object",
                    "properties": {"key": {"type": "string"}},
                    "required": ["key"],
                },
            ),
        ),
    )
    environment = Environment()
    with (
        serve(app) as base_url,
        LearningClient(
            base_url=base_url, api_key="local-roundtrip-test-key", model="tiny-model"
        ) as client,
    ):
        collected = run_learning_episode(
            client=client, agent=ChatAgentRuntime(), environment=environment, task=task
        )
        assert collected.episode.stop_reason == StopReason.COMPLETED
        assert collected.episode.final_action == AssistantAction(content="The tool returned value.")
        assert len(collected.response_ids) == 2
        assert environment.calls[0].call_id == "call-1"
        assert environment.closed == 1
        feedback = client.submit_feedback(collected.response_ids[-1], text="Correct use of lookup.")
        queue = feedback["buffer"]
        assert isinstance(queue, dict)
        assert queue["pending_feedback"] == 1
        assert queue["ready"] == 1
        client.trigger_train()
        deadline = time.monotonic() + 3
        while client.status()["policy_revision"] != "policy-1" and time.monotonic() < deadline:
            time.sleep(0.005)
        trained = client.status()
        consumed = trained["buffer"]
        assert isinstance(consumed, dict)
        assert consumed["consumed"] == 1
        assert trained["policy_revision"] == "policy-1"
    assert runtime.open_count == runtime.close_count == runtime.optimizations == 1
