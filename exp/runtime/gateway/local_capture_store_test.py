"""Scoped durable gateway capture reader regressions."""

import sqlite3
from datetime import UTC, datetime
from pathlib import Path

from exp.runtime.gateway.local_capture_contracts import (
    CapturedExchange,
    CaptureProvenance,
    LocalCaptureScope,
)
from exp.runtime.gateway.local_capture_store import LocalCaptureStore


def test_reader_filters_scope_and_retention_and_resumes_cursor(tmp_path: Path) -> None:
    """A resumed consumer cannot read another application or expired content."""
    path = tmp_path / "capture.db"
    scope = LocalCaptureScope(user_id="user", application_id="app")
    experience = CapturedExchange(
        experience_id="exp-one",
        response_id="response-one",
        scope=scope,
        protocol="chat_completions",
        captured_at=datetime.now(UTC),
        request={"messages": []},
        response={"choices": []},
        provenance=CaptureProvenance(source_id="one", model_id="model"),
    )
    connection = sqlite3.connect(path)
    connection.execute(
        "CREATE TABLE gateway_captures (sequence INTEGER PRIMARY KEY, "
        "user_id TEXT, application_id TEXT, expires_at INTEGER, payload TEXT)"
    )
    for sequence, application, expires in [
        (1, "app", 9999999999),
        (2, "other", 9999999999),
        (3, "app", 1),
    ]:
        connection.execute(
            "INSERT INTO gateway_captures VALUES (?, ?, ?, ?, ?)",
            (sequence, "user", application, expires, experience.model_dump_json()),
        )
    connection.commit()
    connection.close()
    store = LocalCaptureStore(path, scope)
    assert [row.sequence for row in store.read_after()] == [1]
    assert store.read_after(1) == ()
