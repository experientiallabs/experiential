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


def test_unbilled_failure_receipt_requires_explicit_zero_charge_attempt_proof() -> None:
    """Zero cost alone never certifies a saved failure, and existing receipt bytes stay stable."""
    for state in ("pending", "complete", "unknown"):
        receipt = RequestReceipt(fingerprint="request", charge=0, state=state)
        assert "unbilled_attempts" not in receipt.model_dump()
    with pytest.raises(ValueError, match="positive attempt proof"):
        RequestReceipt(fingerprint="request", charge=0, state="unbilled")
    with pytest.raises(ValueError, match="zero charge"):
        RequestReceipt(fingerprint="request", charge=1, state="unbilled", unbilled_attempts=2)
    with pytest.raises(ValueError, match="terminal unbilled receipt"):
        RequestReceipt(fingerprint="request", charge=0, state="unknown", unbilled_attempts=2)
    certified = RequestReceipt(
        fingerprint="request", charge=0, state="unbilled", unbilled_attempts=5
    )
    assert RequestReceipt.model_validate_json(certified.model_dump_json()) == certified


@pytest.mark.parametrize("invalid", [True, "5", 1.0, 1.5])
def test_unbilled_attempt_proof_rejects_coerced_counts(invalid: bool | str | float) -> None:
    """Persisted proof requires actual integer counts, never bool, string, or float coercion."""
    with pytest.raises(ValueError, match="valid integer"):
        RequestReceipt.model_validate(
            {
                "fingerprint": "request",
                "charge": 0,
                "state": "unbilled",
                "unbilled_attempts": invalid,
            }
        )


@pytest.mark.parametrize(
    "payload",
    [
        '{"output": "The company sells password management and authorization software."}',
        '{"output": {"text": "The company sells password management software."}}',
        '{"output": "Set OPENAI_API_KEY in the environment. Do not share the secret."}',
        '{"output": "The product includes password management: resets and vaults."}',
        json.dumps({"output": json.dumps({"text": "password management software"})}),
        '{"output": {"password-policy": "rotated", "authorization status": "granted"}}',
        pytest.param(json.dumps({"output": "9" * 10_000}), id="long-numeric-prose"),
        pytest.param('{"output":' + "9" * 10_000 + "}", id="long-json-integer"),
        pytest.param(
            "[" * 300 + '{"text":"password management software"}' + "]" * 300,
            id="deep-json-prose",
        ),
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
        '{"output": "The password is arbitrary-value"}',
        '{"output": "The API key was arbitrary-value"}',
        '{"output": "The credential equals arbitrary-value"}',
        '{"output": "The token env is set to arbitrary-value"}',
        '{"OPENAI_API_KEY=arbitrary-value": "output"}',
        '{" api_key ": "arbitrary-value"}',
        '{"openai_api_key": "arbitrary-value"}',
        '{"provider.auth_token": "arbitrary-value"}',
        '{"provider/password": "arbitrary-value"}',
        pytest.param(
            '{"number":' + "9" * 10_000 + ',"\\u0061pi_key":"arbitrary-value"}',
            id="long-integer-cannot-hide-escaped-key",
        ),
        pytest.param(
            "[" * 300 + '{"\\u0061pi_key":"arbitrary-value"}' + "]" * 300,
            id="deep-json-cannot-hide-escaped-key",
        ),
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


@pytest.mark.parametrize(
    ("payload", "reason"),
    [
        ('{"output": "OPENAI_API_KEY=arbitrary-value"}', "credential-like content"),
        ('{"output": "first", "output": "last"}', "duplicate JSON keys"),
    ],
)
def test_response_rejection_explains_safe_reason_and_recovery(
    tmp_path: Path, payload: str, reason: str
) -> None:
    """Diagnostics identify the repair without echoing rejected response data."""
    ledger = RequestBudgetStore(ProjectStore(tmp_path, "project-a"), "run-a")
    ledger.write("request-a", RequestReceipt(fingerprint="request", charge=2, state="pending"))
    with pytest.raises(ArtifactStoreError, match=reason) as rejected:
        ledger.complete("request-a", 0.4, payload)
    assert "fresh evaluation" in str(rejected.value)
    assert "arbitrary-value" not in str(rejected.value)
    assert "OPENAI_API_KEY" not in str(rejected.value)


@pytest.mark.parametrize("encoded_in_string", [False, True])
def test_excessive_response_nesting_fails_with_safe_recovery_guidance(
    tmp_path: Path, encoded_in_string: bool
) -> None:
    """Decoder depth failures do not leak raw interpreter errors or bypass validation."""
    project = ProjectStore(tmp_path, "project-a")
    ledger = RequestBudgetStore(project, "run-a")
    pending = RequestReceipt(fingerprint="request", charge=2, state="pending")
    ledger.write("request-a", pending)
    depth = 10_000
    payload = "[" * depth + '{"\\u0061pi_key":"arbitrary-value"}' + "]" * depth
    if encoded_in_string:
        payload = json.dumps({"output": payload})
    with pytest.raises(ArtifactStoreError, match="JSON nesting") as rejected:
        ledger.complete("request-a", 0.4, payload)
    assert "fresh evaluation" in str(rejected.value)
    assert "arbitrary-value" not in str(rejected.value)
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


@pytest.mark.parametrize("known", [False, True])
def test_failed_settlement_rolls_back_receipt_total_and_response(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, known: bool
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
        ledger.complete("request-a", 0.4 if known else None, "paid response")
    restored = RequestBudgetStore(ProjectStore(tmp_path, "project-a"), "run-a")
    assert restored.read("request-a") == pending
    assert restored.total() == 2
    assert not restored.has_unbounded_liability()
    assert project.artifacts.list_ids() == ()
