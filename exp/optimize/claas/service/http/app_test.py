"""Authenticated HTTP, official SDK, replay, and asynchronous learner integration."""

from __future__ import annotations

import asyncio
import socket
import threading
import time
from collections.abc import AsyncIterator, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import cast

import httpx
import pytest
import uvicorn
from fastapi import FastAPI
from fastapi.testclient import TestClient
from openai import OpenAI
from openai.types.chat import ChatCompletionAssistantMessageParam
from openai.types.responses import ResponseFunctionToolCall, ResponseInputParam
from openai.types.responses.function_tool_param import FunctionToolParam

from exp.common.claas.generation import GenerationRequest, GenerationResult
from exp.common.core.artifacts import JsonObject
from exp.common.models import AssistantAction, ToolCall
from exp.optimize.claas.service.configuration import RunConfiguration
from exp.optimize.claas.service.controller import LearningController
from exp.optimize.claas.service.controller_test import Runtime
from exp.optimize.claas.service.http.app import create_app
from exp.optimize.claas.training_contracts import TrainingBatch, TrainingResult
from exp.optimize.claas.training_contracts_test import spec

_KEY = "private-local-test-key"
_HEADERS = {"Authorization": f"Bearer {_KEY}"}
_TOOL: FunctionToolParam = {
    "type": "function",
    "name": "lookup",
    "parameters": {},
    "strict": False,
}


@contextmanager
def serve(app: FastAPI) -> Iterator[str]:
    """Own a bounded real loopback server and shut down its application lifespan."""
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        listener.listen(128)
        port = listener.getsockname()[1]
        server = uvicorn.Server(uvicorn.Config(app, log_config=None, log_level="critical"))
        thread = threading.Thread(target=server.run, kwargs={"sockets": [listener]}, daemon=True)
        thread.start()
        try:
            deadline = time.monotonic() + 5
            while not server.started and thread.is_alive() and time.monotonic() < deadline:
                time.sleep(0.005)
            assert server.started, "local learner did not start"
            yield f"http://127.0.0.1:{port}/v1"
        finally:
            server.should_exit = True
            thread.join(timeout=5)
            assert not thread.is_alive(), "local learner did not release its runtime"


class WireRuntime(Runtime):
    """Expose deterministic text and tools through the actual service and SQLite queue."""

    async def generate(self, request: GenerationRequest) -> GenerationResult:
        """Return a tool call until its caller supplies an explicit observation."""
        result = await super().generate(request)
        action = result.action
        if request.tools and request.messages[-1].role != "tool":
            action = AssistantAction(
                tool_calls=(
                    ToolCall(
                        call_id="call-1",
                        name="lookup",
                        arguments={"key": "value"},
                        raw_arguments='{ "key" : "value" }',
                    ),
                )
            )
        return result.model_copy(
            update={
                "response_id": request.request_id,
                "action": action,
                "finish_reason": "length" if request.prompt is not None else "stop",
            }
        )


def test_official_sdk_all_surfaces_retry_and_tool_history(tmp_path: Path) -> None:
    """Exercise real sockets and SDK output replay without replacing the API or queue."""
    runtime = WireRuntime()
    controller = LearningController(tmp_path, spec(), runtime, RunConfiguration())
    app = create_app(controller, api_key=_KEY)
    with serve(app) as base_url, OpenAI(base_url=base_url, api_key=_KEY, max_retries=0) as sdk:
        first = sdk.chat.completions.create(
            model="adapter-1",
            messages=[{"role": "user", "content": "hello"}],
            extra_headers={"Idempotency-Key": "chat-retry"},
        )
        replay = sdk.chat.completions.create(
            model="adapter-1",
            messages=[{"role": "user", "content": "hello"}],
            extra_headers={"Idempotency-Key": "chat-retry"},
        )
        assert replay.model_dump() == first.model_dump()
        assert runtime.generate_count == 1
        tool = sdk.chat.completions.create(
            model="tiny-model",
            messages=[{"role": "user", "content": "lookup"}],
            tools=[{"type": "function", "function": {"name": "lookup", "parameters": {}}}],
            extra_headers={"Idempotency-Key": "tool-retry"},
        )
        tool_replay = sdk.chat.completions.create(
            model="tiny-model",
            messages=[{"role": "user", "content": "lookup"}],
            tools=[{"type": "function", "function": {"name": "lookup", "parameters": {}}}],
            extra_headers={"Idempotency-Key": "tool-retry"},
        )
        assert tool_replay.model_dump() == tool.model_dump()
        continued = sdk.chat.completions.create(
            model="tiny-model",
            messages=[
                {"role": "user", "content": "lookup"},
                cast(
                    ChatCompletionAssistantMessageParam,
                    tool.choices[0].message.model_dump(exclude_unset=True),
                ),
                {"role": "tool", "tool_call_id": "call-1", "content": "found"},
            ],
        )
        assert continued.choices[0].message.content == "answer"
        response = sdk.responses.create(
            model="tiny-model",
            input="lookup",
            tools=[_TOOL],
            extra_headers={"Idempotency-Key": "responses-retry"},
        )
        response_replay = sdk.responses.create(
            model="tiny-model",
            input="lookup",
            tools=[_TOOL],
            extra_headers={"Idempotency-Key": "responses-retry"},
        )
        assert response_replay.model_dump() == response.model_dump()
        assert isinstance(response.output[0], ResponseFunctionToolCall)
        assert response.output[0].arguments == '{ "key" : "value" }'
        followup = sdk.responses.create(
            model="tiny-model",
            input=cast(
                ResponseInputParam,
                [
                    {"role": "user", "content": "lookup"},
                    *response.output,
                    {"type": "function_call_output", "call_id": "call-1", "output": "found"},
                ],
            ),
        )
        assert followup.output_text == "answer"
        completion = sdk.completions.create(
            model="tiny-model",
            prompt="hello",
            max_tokens=2,
            extra_headers={"Idempotency-Key": "text-retry"},
        )
        completion_replay = sdk.completions.create(
            model="tiny-model",
            prompt="hello",
            max_tokens=2,
            extra_headers={"Idempotency-Key": "text-retry"},
        )
        assert completion_replay.model_dump() == completion.model_dump()
        assert completion.choices[0].finish_reason == "length"
        assert completion.usage is not None and completion.usage.total_tokens == 4
    assert runtime.generate_count == 6
    assert runtime.open_count == runtime.close_count == 1


@pytest.mark.parametrize(
    "body",
    [
        {"model": "tiny-model", "messages": [{"role": [], "content": "secret-prompt"}]},
        {
            "model": "tiny-model",
            "messages": [{"role": "user", "content": [{"type": [], "text": "x"}]}],
        },
        {
            "model": "tiny-model",
            "messages": [{"role": "user", "content": "hello"}],
            "temperature": True,
        },
        {
            "model": "tiny-model",
            "messages": [{"role": "user", "content": "hello"}],
            "max_tokens": True,
        },
        {"model": "tiny-model", "messages": [{"role": "user", "content": "hello"}], "stream": True},
        {"model": "tiny-model", "messages": [{"role": "user", "content": "hello"}], "seed": 1},
        {
            "model": "tiny-model",
            "messages": [{"role": "tool", "content": "x", "tool_call_id": "missing"}],
        },
    ],
)
def test_invalid_requests_do_not_call_runtime(tmp_path: Path, body: JsonObject) -> None:
    """Reject unsupported or malformed nested values before acquiring student compute."""
    runtime = Runtime()
    controller = LearningController(tmp_path, spec(), runtime, RunConfiguration())
    with TestClient(create_app(controller, api_key=_KEY), headers=_HEADERS) as client:
        response = client.post("/v1/chat/completions", json=body)
        assert response.status_code == 400, response.text
        assert "secret-prompt" not in response.text
        assert runtime.generate_count == 0


def test_authorization_size_and_strict_json_precede_generation(tmp_path: Path) -> None:
    """Neither fixed nor streamed over-limit bodies can reach generation or alter the queue."""
    runtime = Runtime()
    controller = LearningController(tmp_path, spec(), runtime, RunConfiguration())
    with TestClient(create_app(controller, api_key=_KEY, maximum_request_bytes=1024)) as client:
        assert client.post("/v1/completions", content=b"x" * 2048).status_code == 401
        assert (
            client.post("/v1/completions", content=b"x" * 2048, headers=_HEADERS).status_code == 413
        )
        for body in (b'{"model":"one","model":"two"}', b'{"value":1e999}', b'{"broken":'):
            assert client.post("/v1/completions", content=body, headers=_HEADERS).status_code == 400
        assert runtime.generate_count == 0


def test_feedback_readiness_and_train_202_remain_responsive(tmp_path: Path) -> None:
    """Sparse labels combine before lease and train acceptance does not wait for the optimizer."""

    async def run() -> None:
        runtime = Runtime()
        runtime.train_gate = asyncio.Event()
        controller = LearningController(
            tmp_path, spec(), runtime, RunConfiguration(minimum_ready_examples=4)
        )
        await controller.start()
        app = create_app(
            controller, api_key=_KEY, manage_lifecycle=False, maximum_request_bytes=1024
        )
        try:
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app), base_url="http://local", headers=_HEADERS
            ) as client:
                generated = await client.post(
                    "/v1/completions",
                    json={"model": "tiny-model", "prompt": "hello"},
                    headers={"Idempotency-Key": "request"},
                )
                response_id = generated.json()["id"]
                changed = await client.post(
                    "/v1/completions",
                    json={"model": "tiny-model", "prompt": "changed"},
                    headers={"Idempotency-Key": "request"},
                )
                assert changed.status_code == 400
                assert runtime.generate_count == 1
                binary = await client.post(
                    "/v1/feedback", json={"response_id": response_id, "success": True}
                )
                assert binary.json()["buffer"]["pending_feedback"] == 1
                assert binary.json()["buffer"]["ready"] == 0
                text = await client.post(
                    "/v1/feedback", json={"response_id": response_id, "text": "Correct"}
                )
                assert text.json()["buffer"]["ready"] == 1
                accepted = await asyncio.wait_for(client.post("/v1/train"), 0.25)
                assert accepted.status_code == 202
                await asyncio.wait_for(runtime.train_entered.wait(), 1)
                assert (await asyncio.wait_for(client.get("/v1/status"), 0.25)).json()["buffer"][
                    "inflight"
                ] == 1
                retry = await client.post(
                    "/v1/feedback", json={"response_id": response_id, "text": "Correct"}
                )
                assert retry.status_code == 200
                conflicting = await client.post(
                    "/v1/feedback", json={"response_id": response_id, "success": False}
                )
                assert conflicting.status_code == 400

                async def chunks() -> AsyncIterator[bytes]:
                    yield b'{"prompt":"'
                    yield b"x" * 1024
                    yield b'"}'

                assert (await client.post("/v1/completions", content=chunks())).status_code == 413
                runtime.train_gate.set()
        finally:
            runtime.train_gate.set()
            await controller.close()
        assert runtime.optimizations == 1

    asyncio.run(run())


class IntegrityFailureRuntime(Runtime):
    """Return internally inconsistent runtime evidence to test HTTP failure classification."""

    async def generate(self, request: GenerationRequest) -> GenerationResult:
        result = await super().generate(request)
        return result.model_copy(
            update={
                "exact_tokens": result.exact_tokens.model_copy(
                    update={"model_id": "wrong-student"}
                ),
            }
        )


class RevisionFailureRuntime(Runtime):
    """Return an optimizer receipt without selecting its promised runtime revision."""

    async def train(self, batch: TrainingBatch) -> TrainingResult:
        result = await super().train(batch)
        self.policy_revision = "incorrect-runtime-revision"
        return result


def test_runtime_generation_integrity_failure_returns_unavailable(tmp_path: Path) -> None:
    """A valid caller request cannot become HTTP400 because the runtime sampled another model."""
    runtime = IntegrityFailureRuntime()
    controller = LearningController(tmp_path, spec(), runtime, RunConfiguration())
    with TestClient(create_app(controller, api_key=_KEY), headers=_HEADERS) as client:
        response = client.post("/v1/completions", json={"model": "tiny-model", "prompt": "hello"})
        assert response.status_code == 503
        assert response.json()["error"]["type"] == "learner_unavailable"
        assert runtime.generate_count == 1
        assert client.get("/v1/status").json()["buffer"]["pending_feedback"] == 0


def test_runtime_optimizer_integrity_failure_returns_unavailable(tmp_path: Path) -> None:
    """A bad optimizer acknowledgement returns HTTP503 and keeps the immutable lease recoverable."""
    runtime = RevisionFailureRuntime()
    controller = LearningController(tmp_path, spec(), runtime, RunConfiguration())
    with TestClient(create_app(controller, api_key=_KEY), headers=_HEADERS) as client:
        response = client.post("/v1/completions", json={"model": "tiny-model", "prompt": "hello"})
        feedback = client.post(
            "/v1/feedback", json={"response_id": response.json()["id"], "text": "Correct"}
        )
        assert feedback.status_code == 200
        drained = client.post("/v1/drain")
        assert drained.status_code == 503
        assert drained.json()["error"]["type"] == "learner_unavailable"
        status = client.get("/v1/status").json()
        assert status["state"] == "failed"
        assert status["buffer"]["inflight"] == 1
        assert status["buffer"]["consumed"] == 0
