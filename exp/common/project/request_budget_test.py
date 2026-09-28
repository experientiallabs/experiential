"""Request accounting, response integrity and atomic settlement across project restarts."""

from pathlib import Path

import pytest

from exp.common.project import ArtifactCorruptionError, ProjectStore
from exp.common.project.database import project_connection
from exp.common.project.request_budget import RequestBudgetStore, RequestReceipt


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
