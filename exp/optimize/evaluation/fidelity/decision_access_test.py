"""Decision evidence is immutable and idempotent within its run identity."""

import json
from datetime import UTC, datetime
from pathlib import Path

from exp.common.project import ProjectStore
from exp.optimize.evaluation.fidelity.decision_access import receipt_recorder


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
