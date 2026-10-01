"""Request accounting, response integrity and atomic settlement across project restarts."""

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from exp.common.core.artifacts import ArtifactEnvelope
from exp.common.project import ArtifactCorruptionError, ArtifactStoreError, ProjectStore
from exp.common.project.database import project_connection
from exp.common.project.manifests import artifact_input
from exp.common.project.request_budget import RequestBudgetStore, RequestReceipt


@pytest.mark.parametrize(
    "payload",
    [
        '{"output": "The company sells password management and authorization software."}',
        '{"output": {"text": "The company sells password management software."}}',
        '{"output": "Set OPENAI_API_KEY in the environment. Do not share the secret."}',
        '{"output": "The product includes password management: resets and vaults."}',
        json.dumps({"output": json.dumps({"text": "password management software"})}),
    ],
)
def test_response_prose_replays_without_treating_credential_words_as_values(
    tmp_path: Path, payload: str
) -> None:
    """Ordinary research prose survives settlement and a process restart byte for byte."""
    ledger = RequestBudgetStore(ProjectStore(tmp_path, "project-a"), "run-a")
    ledger.write("request-a", RequestReceipt(fingerprint="request", charge=2, state="pending"))
    ledger.complete("request-a", 0.4, payload)

    restored = RequestBudgetStore(ProjectStore(tmp_path, "project-a"), "run-a")
    receipt = restored.read("request-a")
    assert receipt is not None and receipt.state == "complete"
    assert restored.response(receipt) == payload
    assert restored.total() == pytest.approx(0.4)


@pytest.mark.parametrize(
    "payload",
    [
        "sk-" + "sensitive" * 4,
        '{"output": "sk-' + "sensitive" * 4 + '"}',
        '{"output": "\\u0073k-' + "sensitive" * 4 + '"}',
        '{"configuration": {"credential_ref": "local-credential"}}',
        '{"api_key": "a-value-without-a-known-prefix"}',
        '{"OPENAI_API_KEY": "a-value-without-a-known-prefix"}',
        '{"provider": [{"CUSTOM_AUTH_TOKEN": "a-value-without-a-known-prefix"}]}',
        '{"\\u004fPENAI_API_KEY": "a-value-without-a-known-prefix"}',
        "credential_ref=local-credential",
        "OPENAI_API_KEY=a-value-without-a-known-prefix",
        "password=a-value-without-a-known-prefix",
        '{"api_key": "malformed-json"',
        '{"output": {"content": "OPENAI_API_KEY=arbitrary-value"}}',
        '{"output": ["password = arbitrary-value"]}',
        '{"output": "api key: arbitrary-value"}',
        json.dumps({"output": '"CUSTOM_AUTH_TOKEN": "arbitrary-value"'}),
        json.dumps(json.dumps({"api_key": "arbitrary-value"})),
        json.dumps({"output": '{"\\u0061pi_key":"arbitrary-value"}'}),
        '{"output":{"api_key":"arbitrary-value"},"output":"safe"}',
        '{"output":"sk-' + "sensitive" * 4 + '","output":"safe"}',
        '{"output":[{"text":"first","text":"last"}]}',
        '{"output":"first","\\u006futput":"last"}',
    ],
)
def test_secret_response_rejected_without_settling_or_persisting(
    tmp_path: Path, payload: str
) -> None:
    """Actual secrets and credential configuration fields cannot become replayable evidence."""
    project = ProjectStore(tmp_path, "project-a")
    ledger = RequestBudgetStore(project, "run-a")
    pending = RequestReceipt(fingerprint="request", charge=2, state="pending")
    ledger.write("request-a", pending)

    with pytest.raises(ArtifactStoreError, match="secret boundary"):
        ledger.complete("request-a", 0.4, payload)

    assert ledger.read("request-a") == pending
    assert ledger.total() == 2
    assert project.artifacts.list_ids() == ()


def test_saved_text_receipt_replays_without_mutation(tmp_path: Path) -> None:
    """Previously paid version-one receipts remain readable without rewriting their hashes."""
    project = ProjectStore(tmp_path, "project-a")
    manifest = project.artifacts.write(
        artifact_id="saved-response",
        artifact_type="provider-response",
        envelope=ArtifactEnvelope(
            schema_version=1, created_at=datetime.now(UTC), code_revision="fixture"
        ),
        files={"response.txt": b'{"text": "already paid"}\n'},
    )
    ledger = RequestBudgetStore(project, "run-a")
    receipt = RequestReceipt(
        fingerprint="request", charge=0.4, state="complete", response=artifact_input(manifest)
    )
    ledger.write("request-a", receipt)

    restored = RequestBudgetStore(ProjectStore(tmp_path, "project-a"), "run-a")
    assert restored.response(receipt) == '{"text": "already paid"}\n'
    assert project.artifacts.read(manifest.artifact_id).manifest == manifest
    assert restored.total() == 0.4


def test_large_response_is_referenced_and_replays_with_exact_scope(tmp_path: Path) -> None:
    """Restarted requests retain paid bytes without sharing charges across projects or runs."""
    project = ProjectStore(tmp_path, "project-a")
    ledger = RequestBudgetStore(project, "run-a")
    ledger.write("request-a", RequestReceipt(fingerprint="request", charge=2, state="pending"))
    payload = "response " * 200_000
    ledger.complete("request-a", 0.4, payload)

    restored = RequestBudgetStore(ProjectStore(tmp_path, "project-a"), "run-a")
    receipt = restored.read("request-a")
    assert receipt is not None and receipt.state == "complete"
    assert restored.response(receipt) == payload
    assert restored.total() == pytest.approx(0.4)
    assert RequestBudgetStore(project, "run-b").total() == 0
    assert RequestBudgetStore(ProjectStore(tmp_path, "project-b"), "run-a").total() == 0
    with project_connection(tmp_path) as connection:
        row = connection.execute("SELECT payload, blob_path FROM project_artifact_files").fetchone()
    assert row is not None and row[0] is None
    (tmp_path / row[1]).write_bytes(b"tampered")
    with pytest.raises(ArtifactCorruptionError):
        restored.response(receipt)


def test_failed_settlement_rolls_back_receipt_total_and_response(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A failed metadata commit retains the full reservation and publishes no response record."""
    project = ProjectStore(tmp_path, "project-a")
    ledger = RequestBudgetStore(project, "run-a")
    pending = RequestReceipt(fingerprint="request", charge=2, state="pending")
    ledger.write("request-a", pending)
    write = ledger._records.write

    def fail_total(record_id: str, payload: bytes, *, exclusive: bool = False) -> None:
        """Interrupt settlement after the response and receipt writes but before the total."""
        if record_id == "total":
            raise OSError("interrupted settlement")
        write(record_id, payload, exclusive=exclusive)

    monkeypatch.setattr(ledger._records, "write", fail_total)
    with pytest.raises(OSError, match="interrupted settlement"):
        ledger.complete("request-a", 0.4, "paid response")
    restored = RequestBudgetStore(ProjectStore(tmp_path, "project-a"), "run-a")
    assert restored.read("request-a") == pending
    assert restored.total() == 2
    assert project.artifacts.list_ids() == ()
