"""Scoped streaming imports preserve gateway evidence, exclusions and retention independence."""

import sqlite3
from pathlib import Path

import pytest

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


@pytest.mark.parametrize("winner", ["worker", "child-model", None])
@pytest.mark.parametrize("truncated", [True, False, None])
def test_shared_import_retains_typed_winner_and_omission_after_capture_retention(
    tmp_path: Path, winner: str | None, truncated: bool | None
) -> None:
    """The disk-backed consumer retains distinct root/winner facts through import and replay."""
    path = trace_database_path(tmp_path)
    path.parent.mkdir()
    experience = _experience()
    output = {"canonical_model_id": winner, "gemini_thought_parts_truncated": truncated}
    captured = experience.model_copy(
        update={"request": {**experience.request, "exp_capture_output": output}}
    )
    _database(path, (captured, _experience("other")))
    expected = load_gateway_capture(path, identity_id="developer")
    summary, receipt = ingest_gateway_capture(
        "powerset", root=tmp_path, path=path, identity_id="developer"
    )
    assert summary.trace_count == 1 and not summary.issues and receipt is not None
    persisted = read_ingested_traces(tmp_path, receipt.import_id)
    assert persisted == expected
    context = persisted.traces[0].initial_context
    assert context["model_id"] == "worker"
    assert context["canonical_model_id"] == winner
    assert context["capture_output"] == output
    _, repeated = ingest_gateway_capture(
        "powerset", root=tmp_path, path=path, identity_id="developer"
    )
    assert repeated is not None and repeated.import_id == receipt.import_id
    with sqlite3.connect(path) as connection:
        connection.execute("DELETE FROM gateway_captures WHERE user_id='developer'")
    assert read_ingested_traces(tmp_path, receipt.import_id) == persisted


@pytest.mark.parametrize(
    "output",
    [{"canonical_model_id": 1}, {"gemini_thought_parts_truncated": "false"}],
)
def test_shared_import_keeps_malformed_optional_evidence_as_exclusions(
    tmp_path: Path, output: dict[str, str | int]
) -> None:
    """Streaming and materialized consumers reject identical untyped output facts."""
    experience = _experience()
    path = tmp_path / "traffic.db"
    _database(
        path,
        (
            experience.model_copy(
                update={"request": {**experience.request, "exp_capture_output": output}}
            ),
        ),
    )
    expected = load_gateway_capture(path, identity_id="developer")
    assert not expected.traces and expected.issues
    _, receipt = ingest_gateway_capture(
        "powerset", root=tmp_path / "state", path=path, identity_id="developer"
    )
    assert receipt is not None
    assert read_ingested_traces(tmp_path / "state", receipt.import_id) == expected


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
