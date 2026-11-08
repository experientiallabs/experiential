"""Decision evidence is immutable and idempotent within its run identity."""

import json
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest

from exp.common.core.artifacts import JsonObject
from exp.common.project import ProjectStore
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
    with pytest.raises(ProviderTransportError, match="HTTP 400"):
        _decide(refused)(request)  # type: ignore[arg-type]
    assert refused.posts == 1

    endless = _Gateway([(429, 1.0)] * 20)
    with pytest.raises(ProviderTransportError, match="HTTP 429"):
        _decide(endless)(request)  # type: ignore[arg-type]
    assert endless.posts == decision_access.MAXIMUM_DECISION_RETRIES + 1
