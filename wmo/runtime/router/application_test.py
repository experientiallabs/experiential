"""Public official OpenAI Python client over a loaded project router."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Protocol, cast, runtime_checkable

import pytest
from fastapi import APIRouter, FastAPI
from fastapi.responses import Response as FastAPIResponse
from fastapi.routing import APIRoute
from fastapi.testclient import TestClient
from openai import OpenAI
from openai.types.chat import ChatCompletion
from openai.types.responses import Response

from wmo.common.project import ProjectStore
from wmo.runtime.router.application import (
    RouterApplicationError,
    create_project_completion_service,
    create_project_router_app,
    load_router,
)
from wmo.runtime.router.endpoint import HttpChatRequest, HttpResponseRequest
from wmo.runtime.router.journal import (
    RuntimeAcceptedEvent,
    RuntimeAttemptFailedEvent,
    RuntimeCompletedEvent,
    RuntimeInteractionJournal,
)
from wmo.runtime.router.runtime import RouterRuntime
from wmo.runtime.router.runtime_test import _runtime


@runtime_checkable
class _IncludedRouterLike(Protocol):
    """FastAPI include wrapper exposing the original router for direct route tests."""

    original_router: APIRouter


def _journaled_application(
    root: Path,
    runtime: RouterRuntime,
) -> tuple[FastAPI, RuntimeInteractionJournal]:
    """Compose one application and expose its canonical project journal.

    Args:
        root: Pytest-owned local artifact root.
        runtime: Frozen test router to wrap durably.

    Returns:
        Integrated application and the same project's readable journal.
    """
    store = ProjectStore(root, "support-agent")
    policy_directory = store.paths.artifact_directory(runtime.policy.policy_id)
    if not policy_directory.exists():
        store.artifacts.write_json(
            artifact_id=runtime.policy.policy_id,
            artifact_type="router-policy",
            envelope=runtime.policy,
            files={"policy.json": runtime.policy},
        )
    service = create_project_completion_service(store, runtime)
    return (
        create_project_router_app(
            "support-agent",
            runtime,
            completion_service=service,
        ),
        RuntimeInteractionJournal(store.paths),
    )


def _post_handler(
    application: FastAPI,
    path: str,
) -> Callable[[HttpChatRequest | HttpResponseRequest, str | None], FastAPIResponse]:
    """Return one generated POST route handler for pre-serialization assertions.

    Args:
        application: Composed FastAPI application.
        path: Exact OpenAI route path.

    Returns:
        Synchronous route function accepting a request and optional standard key.
    """
    included_router = next(
        route.original_router
        for route in application.routes
        if isinstance(route, _IncludedRouterLike)
    )
    route = cast(
        APIRoute,
        next(route for route in included_router.routes if getattr(route, "path", None) == path),
    )
    assert route.methods is not None and "POST" in route.methods
    return cast(
        Callable[[HttpChatRequest | HttpResponseRequest, str | None], FastAPIResponse],
        route.endpoint,
    )


def _payload(path: str, content: str, *, stream: bool = False) -> dict[str, object]:
    """Build one supported Chat or Responses payload.

    Args:
        path: Exact OpenAI route path.
        content: User content sent to the routed candidate.
        stream: Whether to request buffered SSE delivery.

    Returns:
        Official request-shaped JSON mapping.
    """
    if path.endswith("responses"):
        return {"model": "support-agent", "input": content, "stream": stream}
    return {
        "model": "support-agent",
        "messages": [{"role": "user", "content": content}],
        "stream": stream,
    }


def test_load_router_exposes_official_chat_and_responses_resources(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The happy-path Python API needs no WMO request or message types."""
    runtime, model_client = _runtime()
    monkeypatch.setattr(
        "wmo.runtime.router.application.load_project_router",
        lambda project, root, **kwargs: runtime,
    )

    with load_router("support-agent") as router:
        assert isinstance(router, OpenAI)
        chat = router.chat.completions.create(
            model="support-agent",
            messages=[{"role": "user", "content": "Help me"}],
        )
        response = router.responses.create(model="support-agent", input="Help me")

    assert isinstance(chat, ChatCompletion)
    assert isinstance(response, Response)
    assert model_client.complete_calls == 2


def test_injected_project_service_journals_every_unkeyed_openai_call(tmp_path: Path) -> None:
    """Give each ordinary Chat and Responses call a distinct durable internal key.

    Args:
        tmp_path: Pytest-owned local artifact root.
    """
    runtime, model_client = _runtime()
    application, journal = _journaled_application(tmp_path / ".wmo", runtime)
    http = TestClient(application)

    chat = http.post("/v1/chat/completions", json=_payload("/v1/chat/completions", "chat"))
    response = http.post("/v1/responses", json=_payload("/v1/responses", "response"))

    assert chat.status_code == response.status_code == 200
    events = journal.read_events()
    accepted = tuple(event for event in events if isinstance(event, RuntimeAcceptedEvent))
    completed = tuple(event for event in events if isinstance(event, RuntimeCompletedEvent))
    assert len(accepted) == len(completed) == 2
    assert accepted[0].idempotency_key_sha256 != accepted[1].idempotency_key_sha256
    assert model_client.embed_calls == model_client.complete_calls == 2


def test_identical_unkeyed_chat_calls_remain_distinct_interactions(tmp_path: Path) -> None:
    """Never join unrelated stateless Chat callers by shared transcript content.

    Args:
        tmp_path: Pytest-owned local artifact root.
    """
    runtime, model_client = _runtime()
    application, journal = _journaled_application(tmp_path / ".wmo", runtime)
    http = TestClient(application)
    payload = _payload("/v1/chat/completions", "same transcript")

    assert http.post("/v1/chat/completions", json=payload).status_code == 200
    assert http.post("/v1/chat/completions", json=payload).status_code == 200

    accepted = tuple(
        event for event in journal.read_events() if isinstance(event, RuntimeAcceptedEvent)
    )
    assert len({event.interaction_id for event in accepted}) == 2
    assert len({event.lineage_id for event in accepted}) == 2
    assert model_client.embed_calls == model_client.complete_calls == 2


def test_responses_previous_response_id_preserves_journal_lineage(tmp_path: Path) -> None:
    """Keep official Responses continuation turns in one durable routing lineage.

    Args:
        tmp_path: Pytest-owned local artifact root.
    """
    runtime, model_client = _runtime()
    application, journal = _journaled_application(tmp_path / ".wmo", runtime)
    http = TestClient(application)

    first = http.post("/v1/responses", json=_payload("/v1/responses", "first"))
    second = http.post(
        "/v1/responses",
        json={
            "model": "support-agent",
            "input": "second",
            "previous_response_id": first.json()["id"],
        },
    )

    assert first.status_code == second.status_code == 200
    accepted = tuple(
        event for event in journal.read_events() if isinstance(event, RuntimeAcceptedEvent)
    )
    assert len(accepted) == 2
    assert accepted[0].lineage_id == accepted[1].lineage_id
    assert model_client.embed_calls == 1
    assert model_client.complete_calls == 2


def test_project_service_rejects_a_store_without_the_runtime_policy(tmp_path: Path) -> None:
    """Refuse cross-project journaling before model selection or provider dispatch.

    Args:
        tmp_path: Pytest-owned local artifact root.
    """
    runtime, model_client = _runtime()
    wrong_store = ProjectStore(tmp_path / ".wmo", "another-project")

    with pytest.raises(RouterApplicationError, match="verified router policy"):
        create_project_completion_service(wrong_store, runtime)

    assert model_client.embed_calls == model_client.complete_calls == 0
    assert not wrong_store.paths.runtime_journal.exists()


@pytest.mark.parametrize("path", ["/v1/chat/completions", "/v1/responses"])
def test_standard_idempotency_key_replays_after_application_restart(
    tmp_path: Path,
    path: str,
) -> None:
    """Replay keyed Chat and Responses results from the project journal after restart.

    Args:
        tmp_path: Pytest-owned local artifact root.
        path: OpenAI endpoint path exercised by the parameterized regression.
    """
    root = tmp_path / ".wmo"
    first_runtime, first_client = _runtime()
    first_application, journal = _journaled_application(root, first_runtime)
    headers = {"Idempotency-Key": "durable-request"}
    payload = _payload(path, "same")
    first = TestClient(first_application).post(path, json=payload, headers=headers)

    restarted_runtime, restarted_client = _runtime()
    restarted_application, _ = _journaled_application(root, restarted_runtime)
    replay = TestClient(restarted_application).post(path, json=payload, headers=headers)

    assert first.status_code == replay.status_code == 200
    assert first.content == replay.content
    assert first_client.complete_calls == 1
    assert restarted_client.embed_calls == restarted_client.complete_calls == 0
    assert sum(isinstance(event, RuntimeCompletedEvent) for event in journal.read_events()) == 1


@pytest.mark.parametrize("path", ["/v1/chat/completions", "/v1/responses"])
def test_standard_idempotency_key_rejects_changed_request(
    tmp_path: Path,
    path: str,
) -> None:
    """Return 409 when one durable caller key names different request content.

    Args:
        tmp_path: Pytest-owned local artifact root.
        path: OpenAI endpoint path exercised by the parameterized regression.
    """
    runtime, model_client = _runtime()
    application, journal = _journaled_application(tmp_path / ".wmo", runtime)
    http = TestClient(application)
    headers = {"Idempotency-Key": "one-request"}

    assert http.post(path, json=_payload(path, "first"), headers=headers).status_code == 200
    conflict = http.post(path, json=_payload(path, "changed"), headers=headers)

    assert conflict.status_code == 409
    assert model_client.embed_calls == model_client.complete_calls == 1
    assert sum(isinstance(event, RuntimeCompletedEvent) for event in journal.read_events()) == 1


def test_provider_failure_has_no_completed_sft_target(tmp_path: Path) -> None:
    """Persist a failed attempt without creating a completed runtime target.

    Args:
        tmp_path: Pytest-owned local artifact root.
    """
    runtime, model_client = _runtime()
    model_client.completion_error = RuntimeError("provider secret")
    application, journal = _journaled_application(tmp_path / ".wmo", runtime)
    http = TestClient(application, raise_server_exceptions=False)
    headers = {"Idempotency-Key": "failed-request"}
    payload = _payload("/v1/chat/completions", "fail")

    first = http.post("/v1/chat/completions", json=payload, headers=headers)
    replay = http.post("/v1/chat/completions", json=payload, headers=headers)

    assert first.status_code == replay.status_code == 502
    events = journal.read_events()
    assert sum(isinstance(event, RuntimeAttemptFailedEvent) for event in events) == 1
    assert not any(isinstance(event, RuntimeCompletedEvent) for event in events)
    assert model_client.complete_calls == 1
    assert "provider secret" not in first.text + replay.text


@pytest.mark.parametrize("path", ["/v1/chat/completions", "/v1/responses"])
def test_stream_is_completed_before_delivery_and_keyed_replay_does_not_dispatch(
    tmp_path: Path,
    path: str,
) -> None:
    """Commit provider success before an unconsumed stream and replay without dispatch.

    Args:
        tmp_path: Pytest-owned local artifact root.
        path: Buffered OpenAI stream route exercised without consuming its iterator.
    """
    runtime, model_client = _runtime()
    application, journal = _journaled_application(tmp_path / ".wmo", runtime)
    handler = _post_handler(application, path)
    request: HttpChatRequest | HttpResponseRequest
    if path.endswith("responses"):
        request = HttpResponseRequest.model_validate(_payload(path, "stream", stream=True))
    else:
        request = HttpChatRequest.model_validate(_payload(path, "stream", stream=True))

    first = handler(request, "stream-request")
    completed_before_delivery = tuple(
        event for event in journal.read_events() if isinstance(event, RuntimeCompletedEvent)
    )
    replay = handler(request, "stream-request")

    assert first.media_type == replay.media_type == "text/event-stream"
    assert len(completed_before_delivery) == 1
    assert sum(isinstance(event, RuntimeCompletedEvent) for event in journal.read_events()) == 1
    assert model_client.embed_calls == model_client.complete_calls == 1
