"""Public official OpenAI Python client over a loaded project router."""

from __future__ import annotations

import threading
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Literal, Protocol, cast, runtime_checkable

import pytest
from fastapi import APIRouter, FastAPI
from fastapi.responses import Response as FastAPIResponse
from fastapi.routing import APIRoute
from fastapi.testclient import TestClient
from openai import ConflictError, InternalServerError, OpenAI
from openai.types.chat import ChatCompletion
from openai.types.responses import Response

from wmo.common.core.artifacts import ArtifactId, FailureCode
from wmo.common.project import ProjectStore
from wmo.common.routing import RoutingDecision
from wmo.runtime.models import RuntimeModelCatalog
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
from wmo.runtime.router.runtime import DecisionSink, RouterRuntime
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
    store = _store_with_policy(root, "support-agent", runtime)
    service = create_project_completion_service(store, runtime)
    return (
        create_project_router_app(
            "support-agent",
            runtime,
            completion_service=service,
        ),
        RuntimeInteractionJournal(store.paths),
    )


def _store_with_policy(
    root: Path,
    project: str,
    runtime: RouterRuntime,
) -> ProjectStore:
    """Persist the exact runtime policy under one test project when absent.

    Args:
        root: Pytest-owned local artifact root.
        project: Project identity that must own the runtime policy.
        runtime: Frozen runtime whose policy file is persisted exactly.

    Returns:
        Project store that passes journal composition integrity checks.
    """
    store = ProjectStore(root, project)
    policy_directory = store.paths.artifact_directory(runtime.policy.policy_id)
    if not policy_directory.exists():
        store.artifacts.write_json(
            artifact_id=runtime.policy.policy_id,
            artifact_type="router-policy",
            envelope=runtime.policy,
            files={"policy.json": runtime.policy},
        )
    return store


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
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The journaling Python API needs no WMO request or message types.

    Args:
        tmp_path: Pytest-owned local artifact root.
        monkeypatch: Scoped loaded-runtime replacement.
    """
    runtime, model_client = _runtime()
    root = tmp_path / ".wmo"
    store = _store_with_policy(root, "support-agent", runtime)
    monkeypatch.setattr(
        "wmo.runtime.router.application.load_project_router",
        lambda project, root, **kwargs: runtime,
    )

    with load_router("support-agent", root=root) as router:
        assert isinstance(router, OpenAI)
        chat = router.chat.completions.create(
            model="support-agent",
            messages=[{"role": "user", "content": "Help me"}],
        )
        response = router.responses.create(model="support-agent", input="Help me")

    assert isinstance(chat, ChatCompletion)
    assert isinstance(response, Response)
    assert model_client.complete_calls == 2
    events = RuntimeInteractionJournal(store.paths).read_events()
    assert sum(isinstance(event, RuntimeCompletedEvent) for event in events) == 2


@pytest.mark.parametrize("api", ["chat", "responses"])
def test_ghost_router_dispatches_without_durable_traffic(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    api: Literal["chat", "responses"],
) -> None:
    """Ghost calls accept caller keys but never create replay or training state.

    Args:
        tmp_path: Pytest-owned local artifact root.
        monkeypatch: Scoped loaded-runtime replacement.
        api: Official OpenAI resource exercised by the parameterized regression.
    """
    runtime, model_client = _runtime()
    root = tmp_path / ".wmo"
    store = _store_with_policy(root, "support-agent", runtime)

    def load_selected_runtime(
        _project: str,
        _root: Path,
        *,
        policy_id: ArtifactId | None = None,
        environment: Mapping[str, str] | None = None,
        runtime_catalog: RuntimeModelCatalog | None = None,
        decision_sink: DecisionSink | None = None,
    ) -> RouterRuntime:
        """Return the already verified runtime without reconstructing providers."""
        del policy_id, environment, runtime_catalog, decision_sink
        return runtime

    monkeypatch.setattr(
        "wmo.runtime.router.application.load_project_router",
        load_selected_runtime,
    )
    before = {
        path.relative_to(store.paths.project_directory): path.read_bytes()
        for path in store.paths.project_directory.rglob("*")
        if path.is_file()
    }
    headers = {"Idempotency-Key": "ghost-request"}

    with load_router("support-agent", root=root, ghost=True) as router:
        for content in ("first", "changed"):
            if api == "responses":
                router.responses.create(
                    model="support-agent",
                    input=content,
                    extra_headers=headers,
                )
            else:
                router.chat.completions.create(
                    model="support-agent",
                    messages=[{"role": "user", "content": content}],
                    extra_headers=headers,
                )

    assert model_client.complete_calls == 2
    assert not store.paths.runtime_journal.exists()
    after = {
        path.relative_to(store.paths.project_directory): path.read_bytes()
        for path in store.paths.project_directory.rglob("*")
        if path.is_file()
    }
    assert after == before


def test_ghost_router_rejects_a_persistent_decision_sink(
    tmp_path: Path,
) -> None:
    """Fail before activation when ghost mode is paired with a decision recorder.

    Args:
        tmp_path: Pytest-owned local artifact root.
    """

    def decision_sink(_decision: RoutingDecision) -> None:
        """Accept a routing decision for the rejected option combination."""

    runtime, model_client = _runtime()
    store = _store_with_policy(tmp_path / ".wmo", "support-agent", runtime)
    runtime._decision_sink = decision_sink  # noqa: SLF001 - adversarial composition regression

    with pytest.raises(RouterApplicationError, match="ghost mode cannot use"):
        create_project_completion_service(store, runtime, ghost=True)

    with pytest.raises(RouterApplicationError, match="ghost mode cannot use"):
        load_router(
            "support-agent",
            root=tmp_path / ".wmo",
            ghost=True,
            decision_sink=decision_sink,
        )
    assert model_client.complete_calls == 0


def test_load_and_close_are_provider_and_journal_idle(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Load and close the official client without routing or journal state.

    Args:
        tmp_path: Pytest-owned local artifact root.
        monkeypatch: Scoped loaded-runtime replacement.
    """
    runtime, model_client = _runtime()
    root = tmp_path / ".wmo"
    store = _store_with_policy(root, "support-agent", runtime)
    monkeypatch.setattr(
        "wmo.runtime.router.application.load_project_router",
        lambda project, selected_root, **kwargs: runtime,
    )

    router = load_router("support-agent", root=root)
    router.close()

    assert model_client.embed_calls == model_client.complete_calls == 0
    assert not store.paths.runtime_journal.exists()


def test_load_router_preserves_runtime_selection_parameters(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Forward policy, environment, catalog, and decision sink through journal composition.

    Args:
        tmp_path: Pytest-owned local artifact root.
        monkeypatch: Scoped runtime-loader capture.
    """
    runtime, model_client = _runtime()
    root = tmp_path / ".wmo"
    _store_with_policy(root, "support-agent", runtime)
    environment = {"TEST_API_KEY": "value"}

    def decision_sink(_decision: object) -> None:
        """Accept a routing decision without external side effects."""

    calls: list[
        tuple[
            str,
            Path,
            ArtifactId | None,
            Mapping[str, str] | None,
            RuntimeModelCatalog | None,
            DecisionSink | None,
        ]
    ] = []

    def load(
        project: str,
        selected_root: Path,
        *,
        policy_id: ArtifactId | None,
        environment: Mapping[str, str] | None,
        runtime_catalog: RuntimeModelCatalog | None,
        decision_sink: DecisionSink | None,
    ) -> RouterRuntime:
        """Capture the complete runtime selection call.

        Args:
            project: Requested project identifier.
            selected_root: Requested artifact root.
            policy_id: Explicit frozen policy selection.
            environment: Explicit credential environment mapping.
            runtime_catalog: Explicit runtime catalog override.
            decision_sink: Explicit routing-decision sink.

        Returns:
            Provider-idle runtime fixture.
        """
        calls.append(
            (project, selected_root, policy_id, environment, runtime_catalog, decision_sink)
        )
        return runtime

    monkeypatch.setattr("wmo.runtime.router.application.load_project_router", load)

    router = load_router(
        "support-agent",
        root=root,
        policy_id="policy-a",
        environment=environment,
        runtime_catalog=runtime.catalog,
        decision_sink=decision_sink,
    )
    router.close()

    assert calls == [
        (
            "support-agent",
            root,
            "policy-a",
            environment,
            runtime.catalog,
            decision_sink,
        )
    ]
    assert model_client.embed_calls == model_client.complete_calls == 0


def test_loaded_router_keeps_identical_unkeyed_chat_calls_distinct(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Assign separate durable identities to identical stateless Chat calls.

    Args:
        tmp_path: Pytest-owned local artifact root.
        monkeypatch: Scoped loaded-runtime replacement.
    """
    runtime, model_client = _runtime()
    root = tmp_path / ".wmo"
    store = _store_with_policy(root, "support-agent", runtime)
    monkeypatch.setattr(
        "wmo.runtime.router.application.load_project_router",
        lambda project, selected_root, **kwargs: runtime,
    )

    with load_router("support-agent", root=root) as router:
        for _ in range(2):
            router.chat.completions.create(
                model="support-agent",
                messages=[{"role": "user", "content": "same transcript"}],
            )

    accepted = tuple(
        event
        for event in RuntimeInteractionJournal(store.paths).read_events()
        if isinstance(event, RuntimeAcceptedEvent)
    )
    assert len({event.interaction_id for event in accepted}) == 2
    assert len({event.lineage_id for event in accepted}) == 2
    assert model_client.embed_calls == model_client.complete_calls == 2


def test_loaded_router_responses_continuation_keeps_one_lineage(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Preserve Responses routing lineage through the official previous response ID.

    Args:
        tmp_path: Pytest-owned local artifact root.
        monkeypatch: Scoped loaded-runtime replacement.
    """
    runtime, model_client = _runtime()
    root = tmp_path / ".wmo"
    store = _store_with_policy(root, "support-agent", runtime)
    monkeypatch.setattr(
        "wmo.runtime.router.application.load_project_router",
        lambda project, selected_root, **kwargs: runtime,
    )

    with load_router("support-agent", root=root) as router:
        first = router.responses.create(model="support-agent", input="first")
        router.responses.create(
            model="support-agent",
            input="second",
            previous_response_id=first.id,
        )

    accepted = tuple(
        event
        for event in RuntimeInteractionJournal(store.paths).read_events()
        if isinstance(event, RuntimeAcceptedEvent)
    )
    assert len(accepted) == 2
    assert accepted[0].lineage_id == accepted[1].lineage_id
    assert model_client.embed_calls == 1
    assert model_client.complete_calls == 2


@pytest.mark.parametrize("api", ["chat", "responses"])
def test_loaded_router_key_replays_across_separate_clients(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    api: Literal["chat", "responses"],
) -> None:
    """Replay one standard caller key after a process-style runtime reload.

    Args:
        tmp_path: Pytest-owned local artifact root.
        monkeypatch: Scoped runtime sequence replacement.
        api: Official OpenAI resource exercised by the parameterized regression.
    """
    root = tmp_path / ".wmo"
    first_runtime, first_client = _runtime()
    restarted_runtime, restarted_client = _runtime()
    _store_with_policy(root, "support-agent", first_runtime)
    runtimes = iter((first_runtime, restarted_runtime))
    monkeypatch.setattr(
        "wmo.runtime.router.application.load_project_router",
        lambda project, selected_root, **kwargs: next(runtimes),
    )
    headers = {"Idempotency-Key": "official-replay"}

    with load_router("support-agent", root=root) as first_router:
        first = (
            first_router.responses.create(
                model="support-agent",
                input="same",
                extra_headers=headers,
            )
            if api == "responses"
            else first_router.chat.completions.create(
                model="support-agent",
                messages=[{"role": "user", "content": "same"}],
                extra_headers=headers,
            )
        )
    with load_router("support-agent", root=root) as restarted_router:
        replay = (
            restarted_router.responses.create(
                model="support-agent",
                input="same",
                extra_headers=headers,
            )
            if api == "responses"
            else restarted_router.chat.completions.create(
                model="support-agent",
                messages=[{"role": "user", "content": "same"}],
                extra_headers=headers,
            )
        )

    assert first.id == replay.id
    assert first_client.complete_calls == 1
    assert restarted_client.embed_calls == restarted_client.complete_calls == 0


@pytest.mark.parametrize("api", ["chat", "responses"])
def test_loaded_router_key_rejects_changed_content_with_official_409(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    api: Literal["chat", "responses"],
) -> None:
    """Expose durable request conflict through the official SDK status exception.

    Args:
        tmp_path: Pytest-owned local artifact root.
        monkeypatch: Scoped loaded-runtime replacement.
        api: Official OpenAI resource exercised by the parameterized regression.
    """
    runtime, model_client = _runtime()
    root = tmp_path / ".wmo"
    _store_with_policy(root, "support-agent", runtime)
    monkeypatch.setattr(
        "wmo.runtime.router.application.load_project_router",
        lambda project, selected_root, **kwargs: runtime,
    )
    headers = {"Idempotency-Key": "official-conflict"}

    with load_router("support-agent", root=root) as router:
        if api == "responses":
            router.responses.create(
                model="support-agent",
                input="first",
                extra_headers=headers,
            )
            with pytest.raises(ConflictError) as caught:
                router.responses.create(
                    model="support-agent",
                    input="changed",
                    extra_headers=headers,
                )
        else:
            router.chat.completions.create(
                model="support-agent",
                messages=[{"role": "user", "content": "first"}],
                extra_headers=headers,
            )
            with pytest.raises(ConflictError) as caught:
                router.chat.completions.create(
                    model="support-agent",
                    messages=[{"role": "user", "content": "changed"}],
                    extra_headers=headers,
                )

    assert caught.value.status_code == 409
    assert model_client.embed_calls == model_client.complete_calls == 1


def test_loaded_router_provider_failure_has_no_completed_target(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Persist a provider failure without producing an automatic SFT target.

    Args:
        tmp_path: Pytest-owned local artifact root.
        monkeypatch: Scoped loaded-runtime replacement.
    """
    runtime, model_client = _runtime()
    model_client.completion_error = RuntimeError("provider secret")
    root = tmp_path / ".wmo"
    store = _store_with_policy(root, "support-agent", runtime)
    monkeypatch.setattr(
        "wmo.runtime.router.application.load_project_router",
        lambda project, selected_root, **kwargs: runtime,
    )

    with load_router("support-agent", root=root) as router:
        with pytest.raises(InternalServerError) as caught:
            router.chat.completions.create(
                model="support-agent",
                messages=[{"role": "user", "content": "fail"}],
                extra_headers={"Idempotency-Key": "official-failure"},
            )

    assert caught.value.status_code == 502
    events = RuntimeInteractionJournal(store.paths).read_events()
    assert sum(isinstance(event, RuntimeAttemptFailedEvent) for event in events) == 1
    assert not any(isinstance(event, RuntimeCompletedEvent) for event in events)
    assert model_client.complete_calls == 1


@pytest.mark.parametrize("api", ["chat", "responses"])
def test_loaded_router_stream_is_completed_before_consumption(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    api: Literal["chat", "responses"],
) -> None:
    """Commit buffered provider success before the official stream is consumed.

    Args:
        tmp_path: Pytest-owned local artifact root.
        monkeypatch: Scoped loaded-runtime replacement.
        api: Official OpenAI streaming resource exercised by the regression.
    """
    runtime, model_client = _runtime()
    root = tmp_path / ".wmo"
    store = _store_with_policy(root, "support-agent", runtime)
    monkeypatch.setattr(
        "wmo.runtime.router.application.load_project_router",
        lambda project, selected_root, **kwargs: runtime,
    )
    headers = {"Idempotency-Key": "official-stream"}

    with load_router("support-agent", root=root) as router:
        stream = (
            router.responses.create(
                model="support-agent",
                input="stream",
                stream=True,
                extra_headers=headers,
            )
            if api == "responses"
            else router.chat.completions.create(
                model="support-agent",
                messages=[{"role": "user", "content": "stream"}],
                stream=True,
                extra_headers=headers,
            )
        )
        completed = tuple(
            event
            for event in RuntimeInteractionJournal(store.paths).read_events()
            if isinstance(event, RuntimeCompletedEvent)
        )
        stream.close()

    assert len(completed) == 1
    assert model_client.embed_calls == model_client.complete_calls == 1


def test_two_loaded_clients_share_one_project_journal_safely(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Serialize one concurrent keyed interaction across separately loaded clients.

    Args:
        tmp_path: Pytest-owned local artifact root.
        monkeypatch: Scoped runtime sequence replacement.
    """
    root = tmp_path / ".wmo"
    first_runtime, first_client = _runtime()
    second_runtime, second_client = _runtime()
    store = _store_with_policy(root, "support-agent", first_runtime)
    runtimes = iter((first_runtime, second_runtime))
    monkeypatch.setattr(
        "wmo.runtime.router.application.load_project_router",
        lambda project, selected_root, **kwargs: next(runtimes),
    )
    first_router = load_router("support-agent", root=root)
    second_router = load_router("support-agent", root=root)
    barrier = threading.Barrier(3)
    response_ids: list[str] = []
    errors: list[Exception] = []

    def complete(router: OpenAI) -> None:
        """Start together and capture one keyed official completion.

        Args:
            router: Separately loaded client sharing the project journal.
        """
        try:
            barrier.wait()
            response = router.chat.completions.create(
                model="support-agent",
                messages=[{"role": "user", "content": "concurrent"}],
                extra_headers={"Idempotency-Key": "concurrent-request"},
            )
            response_ids.append(response.id)
        except Exception as exc:  # noqa: BLE001 - capture thread failures for the main assertion
            errors.append(exc)

    threads = (
        threading.Thread(target=complete, args=(first_router,), daemon=True),
        threading.Thread(target=complete, args=(second_router,), daemon=True),
    )
    for thread in threads:
        thread.start()
    barrier.wait()
    for thread in threads:
        thread.join(timeout=2.0)
    assert all(not thread.is_alive() for thread in threads), "loaded clients deadlocked"
    first_router.close()
    second_router.close()

    assert errors == []
    assert len(response_ids) == 2 and len(set(response_ids)) == 1
    assert first_client.complete_calls + second_client.complete_calls == 1
    assert first_client.embed_calls + second_client.embed_calls == 1
    assert (
        sum(
            isinstance(event, RuntimeCompletedEvent)
            for event in RuntimeInteractionJournal(store.paths).read_events()
        )
        == 1
    )


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


@pytest.mark.parametrize("path", ["/v1/chat/completions", "/v1/responses"])
@pytest.mark.parametrize("capability", ["tools", "output_capacity"])
def test_capability_rejection_replays_exactly_after_application_restart(
    tmp_path: Path,
    path: str,
    capability: str,
) -> None:
    """Replay a durable pre-dispatch capability rejection with its original 501 meaning.

    Args:
        tmp_path: Pytest-owned local artifact root.
        path: OpenAI endpoint path exercised by the parameterized regression.
        capability: Request capability that the selected model cannot prove.
    """
    root = tmp_path / ".wmo"
    first_runtime, first_client = _runtime(candidate_tools=False)
    first_application, journal = _journaled_application(root, first_runtime)
    payload = _payload(path, "unsupported")
    if capability == "tools":
        payload["tools"] = (
            [{"type": "function", "name": "read", "parameters": {}}]
            if path.endswith("responses")
            else [
                {
                    "type": "function",
                    "function": {"name": "read", "parameters": {}},
                }
            ]
        )
    elif path.endswith("responses"):
        payload["max_output_tokens"] = 20_000
    else:
        payload["max_completion_tokens"] = 20_000
    headers = {"Idempotency-Key": f"unsupported-{capability}"}

    first = TestClient(first_application).post(path, json=payload, headers=headers)
    restarted_runtime, restarted_client = _runtime(candidate_tools=False)
    restarted_application, _ = _journaled_application(root, restarted_runtime)
    replay = TestClient(restarted_application).post(path, json=payload, headers=headers)

    assert first.status_code == replay.status_code == 501
    assert first.content == replay.content
    assert first_client.complete_calls == restarted_client.complete_calls == 0
    assert restarted_client.embed_calls == 0
    events = journal.read_events()
    assert [event.event for event in events] == ["accepted", "attempt_failed"]
    failure = cast(RuntimeAttemptFailedEvent, events[-1])
    assert failure.failure.code == FailureCode.UNSUPPORTED
    assert failure.failure.exception_type == "RouterModelCapabilityError"
    assert not failure.retryable
    assert not any(isinstance(event, RuntimeCompletedEvent) for event in events)


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
