"""Decision evidence is immutable and idempotent within its run identity."""

import json
import threading
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest

from exp.common.core.artifacts import JsonObject
from exp.common.project import ArtifactStore, ProjectStore
from exp.optimize.evaluation.fidelity import decision_access
from exp.optimize.evaluation.fidelity.decision_access import decision_caller, receipt_recorder
from exp.runtime.gateway.decisions_contracts import DecisionRequest
from exp.runtime.models.providers.transport import JsonHttpResponse, ProviderTransportError


def test_receipts_replay_within_run_but_do_not_collapse_other_runs(tmp_path: Path) -> None:
    """Identical reader outputs remain attributable to the exact run that inspected them."""
    project = ProjectStore(tmp_path, "receipts")
    record = receipt_recorder(project, "first", "test", lambda: datetime.now(UTC))
    record({"id": "source", "response": {"support": False}})
    record({"id": "source", "response": {"support": False}})
    receipt_recorder(project, "second", "test", lambda: datetime.now(UTC))(
        {"id": "source", "response": {"support": False}}
    )
    ids = project.artifacts.list_ids(artifact_type="jev-receipt")
    assert len(ids) == 2
    assert {
        json.loads(project.artifacts.read_bytes(key, "receipt.json"))["run"] for key in ids
    } == {
        "first",
        "second",
    }


class _Gateway:
    """An admitted-transport stand-in answering from a script of (status, Retry-After)."""

    def __init__(self, script: list[tuple[int, float | None]]) -> None:
        self.script, self.posts = script, 0

    def priced_as(self, alias: str) -> "_Gateway":
        del alias
        return self

    def post(self, url: str, **kwargs: object) -> JsonHttpResponse:
        del url, kwargs
        status, after = self.script[self.posts]
        self.posts += 1
        body = {"selected": ["a"]} if status == 200 else {"error": {"code": "rate_limited"}}
        return JsonHttpResponse(status_code=status, body=body, retry_after_seconds=after)


def _decide(gateway: _Gateway) -> Callable[[DecisionRequest], JsonObject]:
    profile = SimpleNamespace(
        dialect="typesafe_systemone", decisions_url="https://gw/decisions", headers={}, model_id="m"
    )
    resolved = SimpleNamespace(
        alias="clef", client=SimpleNamespace(gateway_wire_profile=lambda: profile)
    )
    deadline = datetime.now(UTC) + timedelta(minutes=5)
    return decision_caller(resolved, gateway, deadline, lambda: datetime.now(UTC))  # type: ignore[arg-type]


def test_a_rate_limited_decision_is_retried_after_retry_after_instead_of_aborting(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A 429 (never processed) and a 503 are retried, honoring Retry-After; a 400 is not.

    One 429 from the gateway under sustained concurrency killed the whole state-rag phase.
    """
    waits: list[float] = []
    monkeypatch.setattr(decision_access, "_sleep", waits.append)
    monkeypatch.setattr(decision_access, "_sample", lambda: 0.5)
    request = SimpleNamespace(provider_body=lambda model: {"model": model})
    gateway = _Gateway([(429, 7.0), (503, None), (429, None), (200, None)])

    assert _decide(gateway)(request) == {"selected": ["a"]}  # type: ignore[arg-type]
    assert gateway.posts == 4
    assert waits == [7.0, 1.0, 2.0]

    refused = _Gateway([(400, None)])
    with pytest.raises(ProviderTransportError, match="HTTP 400: code=rate_limited"):
        _decide(refused)(request)  # type: ignore[arg-type]
    assert refused.posts == 1

    endless = _Gateway([(429, 1.0)] * 20)
    with pytest.raises(ProviderTransportError, match="HTTP 429"):
        _decide(endless)(request)  # type: ignore[arg-type]
    assert endless.posts == decision_access.MAXIMUM_DECISION_RETRIES + 1


def test_parallel_receipt_recorders_publish_one_immutable_receipt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Independent rollouts can publish the same recall receipt without aborting either.

    Both writers miss the existence check; the second write meets the first's immutable artifact
    and is absorbed as the same content-addressed receipt rather than failing its rollout.
    """
    project = ProjectStore(tmp_path, "parallel-receipts")
    original_exists = project.artifacts.exists
    rendezvous = threading.Barrier(2)

    def simultaneous_misses(store: ArtifactStore, identity: str) -> bool:
        """Expose both unsynchronized writers to the same absent receipt."""
        del store
        present = original_exists(identity)
        if not present and identity.startswith("jev-receipt-"):
            try:
                rendezvous.wait(timeout=0.5)
            except threading.BrokenBarrierError:
                pass
        return present

    monkeypatch.setattr(ArtifactStore, "exists", simultaneous_misses)
    recorders = [
        receipt_recorder(project, "same-run", "test", lambda: datetime.now(UTC)) for _ in range(2)
    ]
    receipt: JsonObject = {"id": "source", "response": {"support": False}}
    with ThreadPoolExecutor(max_workers=2) as pool:
        list(pool.map(lambda record: record(receipt), recorders))
    ids = project.artifacts.list_ids(artifact_type="jev-receipt")
    assert len(ids) == 1
    assert json.loads(project.artifacts.read_bytes(ids[0], "receipt.json")) == {
        "run": "same-run",
        **receipt,
    }


def test_a_refused_decision_names_the_providers_reason_but_never_the_request() -> None:
    """A 400 carries the gateway's error type, code and message, bounded; the state never."""

    class _Refusing(_Gateway):
        def post(self, url: str, **kwargs: object) -> JsonHttpResponse:
            del url, kwargs
            self.posts += 1
            message = "The model's provider rejected the request without giving a reason. " * 20
            error = {"message": message, "type": "invalid_request_error", "code": "invalid_request"}
            body = {"error": error}
            return JsonHttpResponse(status_code=400, body=body, retry_after_seconds=None)

    request = SimpleNamespace(provider_body=lambda model: {"model": model, "state": "SECRET-STATE"})
    with pytest.raises(ProviderTransportError) as caught:
        _decide(_Refusing([]))(request)  # type: ignore[arg-type]
    text = str(caught.value)
    assert caught.value.status_code == 400
    assert "type=invalid_request_error; code=invalid_request; message=The model's provider" in text
    assert "SECRET-STATE" not in text and len(text) < 500
    assert decision_access.refusal_reason("not json") == "no error object in the response"
