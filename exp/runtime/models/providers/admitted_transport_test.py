"""Physical receipts survive unusable output while credentials stay outside artifacts."""

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from exp.common.core.artifacts import JsonObject
from exp.common.project import ArtifactStoreError, ProjectConfig, ProjectStore
from exp.runtime.models.providers.admitted_transport import AdmittedJsonTransport
from exp.runtime.models.providers.transport import JsonHttpResponse, ScriptedJsonTransport
from exp.runtime.sessions.dispatch_ledger import DispatchCapExhaustedError, DispatchLedger


def test_raw_response_is_retained_before_model_parsing_and_cap_blocks_second_send(
    tmp_path: Path,
) -> None:
    """Even an empty assistant response is saved exactly, without authenticated headers."""
    project = ProjectStore(tmp_path, "attempt")
    project.initialize(ProjectConfig(project_id="attempt"))
    body: JsonObject = {"choices": [{"message": {"content": None}}]}
    inner = ScriptedJsonTransport([JsonHttpResponse(status_code=200, body=body)])
    ledger = DispatchLedger(tmp_path / "physical.sqlite", 1, "a" * 64)
    transport = AdmittedJsonTransport(
        inner,
        ledger=ledger,
        artifacts=project.artifacts,
        code_revision="fixture",
        deadline=datetime.now(UTC) + timedelta(minutes=1),
    )
    headers = {"Authorization": "Bearer never-retain-this"}
    payload: JsonObject = {"model": "fixture"}
    assert (
        transport.post(
            "https://fixture.invalid",
            headers=headers,
            payload=payload,
            timeout_seconds=30,
        ).body
        == body
    )
    with pytest.raises(DispatchCapExhaustedError):
        transport.post(
            "https://fixture.invalid",
            headers=headers,
            payload=payload,
            timeout_seconds=30,
        )
    assert ledger.used() == 1
    artifacts = project.artifacts.list_ids()
    assert len(artifacts) == 2
    response = next(identity for identity in artifacts if identity.endswith("-response"))
    receipt = project.artifacts.read_bytes(response, "receipt.json").decode()
    assert '"content":null' in receipt
    assert "never-retain-this" not in payload and "Authorization" not in receipt


def test_credential_response_is_rejected_with_a_hash_only_receipt(tmp_path: Path) -> None:
    """Receipt recording does not bypass the shared provider-output credential boundary."""
    project = ProjectStore(tmp_path, "attempt")
    project.initialize(ProjectConfig(project_id="attempt"))
    inner = ScriptedJsonTransport(
        [
            JsonHttpResponse(status_code=200, body={"api_key": "do-not-retain"}),
        ]
    )
    transport = AdmittedJsonTransport(
        inner,
        ledger=DispatchLedger(tmp_path / "physical.sqlite", 1, "a" * 64),
        artifacts=project.artifacts,
        code_revision="fixture",
        deadline=datetime.now(UTC) + timedelta(minutes=1),
    )
    with pytest.raises(ArtifactStoreError, match="secret boundary"):
        transport.post("https://fixture.invalid", headers={}, payload={}, timeout_seconds=30)
    ids = project.artifacts.list_ids()
    assert len(ids) == 2 and any(identity.endswith("-rejected") for identity in ids)
    assert all(
        b"do-not-retain" not in project.artifacts.read_bytes(identity, "receipt.json")
        for identity in ids
    )


def test_intent_persistence_cannot_extend_dispatch_deadline(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A slow durable intent may consume a slot, but must not start late network I/O."""
    from collections.abc import Mapping

    from exp.common.core.artifacts import ArtifactEnvelope, JsonValue
    from exp.common.project import ArtifactManifest

    project = ProjectStore(tmp_path, "attempt")
    project.initialize(ProjectConfig(project_id="attempt"))
    now = [datetime(2030, 1, 1, tzinfo=UTC)]
    inner = ScriptedJsonTransport([JsonHttpResponse(status_code=200, body={})])
    ledger = DispatchLedger(tmp_path / "physical.sqlite", 1, "a" * 64)
    original = project.artifacts.write_json

    def slow_intent(
        *,
        artifact_id: str,
        artifact_type: str,
        envelope: ArtifactEnvelope,
        files: Mapping[str, JsonValue],
    ) -> ArtifactManifest:
        """Advance the clock after the intent was saved, before the transport can send."""
        result = original(
            artifact_id=artifact_id,
            artifact_type=artifact_type,
            envelope=envelope,
            files=files,
        )
        if artifact_id.endswith("-intent"):
            now[0] += timedelta(seconds=2)
        return result

    monkeypatch.setattr(project.artifacts, "write_json", slow_intent)
    transport = AdmittedJsonTransport(
        inner,
        ledger=ledger,
        artifacts=project.artifacts,
        code_revision="fixture",
        deadline=now[0] + timedelta(seconds=1),
        clock=lambda: now[0],
    )
    with pytest.raises(TimeoutError, match="persisting intent"):
        transport.post("https://fixture.invalid", headers={}, payload={}, timeout_seconds=30)
    assert ledger.used() == 1 and not inner.requests
    assert any(name.endswith("-not-dispatched") for name in project.artifacts.list_ids())


def test_a_refusing_guard_runs_before_every_claim_and_stops_the_send(tmp_path: Path) -> None:
    """The guard sees each claim first; once it refuses, no slot is claimed and nothing is sent."""
    project = ProjectStore(tmp_path, "attempt")
    project.initialize(ProjectConfig(project_id="attempt"))
    body: JsonObject = {"choices": []}
    inner = ScriptedJsonTransport([JsonHttpResponse(status_code=200, body=body)] * 2)
    ledger = DispatchLedger(tmp_path / "physical.sqlite", 5, "a" * 64)
    allowed = [True]
    checks: list[int] = []

    def guard() -> None:
        """Record the check and refuse once dispatch is no longer allowed."""
        checks.append(ledger.used())
        if not allowed[0]:
            raise RuntimeError("reservation released")

    transport = AdmittedJsonTransport(
        inner,
        ledger=ledger,
        artifacts=project.artifacts,
        code_revision="fixture",
        deadline=datetime.now(UTC) + timedelta(minutes=1),
        before_claim=guard,
    )
    transport.post("https://fixture.invalid", headers={}, payload={}, timeout_seconds=30)
    allowed[0] = False
    with pytest.raises(RuntimeError, match="reservation released"):
        transport.post("https://fixture.invalid", headers={}, payload={}, timeout_seconds=30)
    assert checks == [0, 1] and ledger.used() == 1 and len(inner.requests) == 1


def test_final_transport_body_is_saved_without_headers_or_endpoint_secrets(tmp_path: Path) -> None:
    """The receipt describes the actual provider-protocol input, not a reconstructed prompt."""
    import json

    from exp.common.core.artifacts import sha256_json
    from exp.runtime.models.call_evidence import call_consumer

    project = ProjectStore(tmp_path, "attempt")
    inner = ScriptedJsonTransport([JsonHttpResponse(status_code=200, body={})])
    transport = AdmittedJsonTransport(
        inner,
        ledger=DispatchLedger(tmp_path / "calls.sqlite", 1, "b" * 64),
        artifacts=project.artifacts,
        code_revision="test",
        deadline=datetime.now(UTC) + timedelta(minutes=1),
    )
    payload: JsonObject = {
        "messages": [{"role": "user", "content": "post-hook input"}],
        "model": "x",
    }
    with call_consumer("actor"):
        transport.post(
            "https://fixture.invalid/?key=inert-url-secret",
            headers={"Authorization": "inert-header-secret"},
            payload=payload,
            timeout_seconds=5,
        )
    identity = next(x for x in project.artifacts.list_ids() if x.endswith("-intent"))
    encoded = project.artifacts.read_bytes(identity, "receipt.json")
    saved = json.loads(encoded)
    assert saved["request_body"] == inner.requests[0].payload == payload
    assert saved["request_sha256"] == sha256_json(saved["request_body"])
    assert saved["consumer"] == "actor"
    assert b"inert-url-secret" not in encoded and b"inert-header-secret" not in encoded


def test_credential_bearing_request_refuses_before_claim_and_io(tmp_path: Path) -> None:
    """A body that cannot be safely retained cannot be silently dispatched with hash-only proof."""
    project = ProjectStore(tmp_path, "attempt")
    inner = ScriptedJsonTransport([JsonHttpResponse(status_code=200, body={})])
    ledger = DispatchLedger(tmp_path / "calls.sqlite", 1, "b" * 64)
    transport = AdmittedJsonTransport(
        inner,
        ledger=ledger,
        artifacts=project.artifacts,
        code_revision="test",
        deadline=datetime.now(UTC) + timedelta(minutes=1),
    )
    with pytest.raises(ArtifactStoreError, match="request"):
        transport.post(
            "https://fixture.invalid",
            headers={},
            payload={"messages": [{"role": "user", "content": '{"api_key": "inert-body-secret"}'}]},
            timeout_seconds=5,
        )
    assert not inner.requests and ledger.used() == 0
    assert not project.artifacts.list_ids()


def test_late_paid_response_is_retained_but_cannot_authorize_an_action(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A late response is evidence, not permission to continue mutating the world."""
    import json
    from collections.abc import Mapping

    now = [datetime(2030, 1, 1, tzinfo=UTC)]
    body: JsonObject = {"choices": [{"message": {"content": "perform the write"}}]}
    inner = ScriptedJsonTransport([JsonHttpResponse(status_code=200, body=body)])
    original = inner.post

    def late(
        url: str, *, headers: Mapping[str, str], payload: JsonObject, timeout_seconds: float
    ) -> JsonHttpResponse:
        """Return after the declared boundary despite the requested transport timeout."""
        now[0] += timedelta(seconds=2)
        return original(url, headers=headers, payload=payload, timeout_seconds=timeout_seconds)

    monkeypatch.setattr(inner, "post", late)
    project = ProjectStore(tmp_path, "attempt")
    ledger = DispatchLedger(tmp_path / "calls.sqlite", 1, "c" * 64)
    transport = AdmittedJsonTransport(
        inner,
        ledger=ledger,
        artifacts=project.artifacts,
        code_revision="test",
        deadline=now[0] + timedelta(seconds=1),
        clock=lambda: now[0],
    )
    with pytest.raises(TimeoutError, match="response.*deadline"):
        transport.post("https://fixture.invalid", headers={}, payload={}, timeout_seconds=30)
    assert len(inner.requests) == ledger.used() == 1
    identity = next(x for x in project.artifacts.list_ids() if x.endswith("-response"))
    saved = json.loads(project.artifacts.read_bytes(identity, "receipt.json"))
    assert saved["response_body"] == body and saved["returned_after_deadline"] is True
    with pytest.raises(TimeoutError, match="before physical dispatch"):
        transport.post("https://fixture.invalid", headers={}, payload={}, timeout_seconds=30)
    assert len(inner.requests) == ledger.used() == 1
