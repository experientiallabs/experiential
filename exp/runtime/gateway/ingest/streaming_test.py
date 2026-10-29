"""Scoped streaming imports preserve gateway evidence, exclusions and retention independence."""

import sqlite3
from pathlib import Path

from exp.common.traces.ingest.persistence import read_ingested_traces
from exp.common.traces.sqlite import SQLiteTraceStore
from exp.common.traces.sqlite_schema import trace_database_path
from exp.runtime.gateway.ingest.conversion import load_gateway_capture
from exp.runtime.gateway.ingest.conversion_test import _database, _experience
from exp.runtime.gateway.ingest.streaming import ingest_gateway_capture


def test_gateway_snapshot_exceeds_one_page_and_survives_retention(tmp_path: Path) -> None:
    """All 1001 scoped rows import into the same database without inheriting capture expiry."""
    path = trace_database_path(tmp_path)
    path.parent.mkdir()
    base = _experience()
    captures = tuple(
        base.model_copy(
            update={
                "experience_id": f"experience-{index}",
                "response_id": f"response-{index}",
            }
        )
        for index in range(1001)
    )
    _database(path, (*captures, _experience("other")))
    expected = load_gateway_capture(path, identity_id="developer")
    result, receipt = ingest_gateway_capture(
        "powerset", root=tmp_path, path=path, identity_id="developer"
    )
    assert result.trace_count == 1001 and not result.issues
    assert receipt is not None and receipt.new_records == 1001
    assert all(trace.initial_context["identity_id"] == "developer" for trace in expected.traces)
    with sqlite3.connect(path) as connection:
        assert connection.execute("SELECT COUNT(*) FROM gateway_captures").fetchone() == (1002,)
        connection.execute("DELETE FROM gateway_captures WHERE user_id='developer'")
    assert read_ingested_traces(tmp_path, receipt.import_id) == expected
    assert SQLiteTraceStore(path).list_imports("powerset") == (receipt.import_id,)


def test_gateway_conversion_exclusions_preserve_the_original_reason(tmp_path: Path) -> None:
    """A captured exchange lacking a user task retains the canonical conversion diagnostic."""
    experience = _experience()
    context = experience.request["exp_context"]
    assert isinstance(context, dict)
    request = context["request"]
    assert isinstance(request, dict)
    request["messages"] = []
    path = tmp_path / "traffic.db"
    _database(path, (experience,))
    expected = load_gateway_capture(path, identity_id="developer")
    assert not expected.traces and expected.issues
    _, receipt = ingest_gateway_capture(
        "powerset",
        path=path,
        root=tmp_path / "state",
        identity_id="developer",
    )
    assert receipt is not None
    assert read_ingested_traces(tmp_path / "state", receipt.import_id) == expected
