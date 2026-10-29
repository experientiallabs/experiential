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
