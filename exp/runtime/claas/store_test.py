"""Scoped durable experience reader regressions."""

import sqlite3
from datetime import UTC, datetime
from pathlib import Path

from exp.common.claas import ClaasScope, Experience, ExperienceProvenance
from exp.runtime.claas.store import ExperienceStore


def test_reader_filters_scope_and_retention_and_resumes_cursor(tmp_path: Path) -> None:
    """A resumed consumer cannot read another application or expired content."""
    path = tmp_path / "capture.db"
    scope = ClaasScope(user_id="user", application_id="app")
    experience = Experience(
        experience_id="exp-one",
        response_id="response-one",
        scope=scope,
        protocol="chat_completions",
        captured_at=datetime.now(UTC),
        request={"messages": []},
        response={"choices": []},
        provenance=ExperienceProvenance(source_kind="traffic", source_id="one", model_id="model"),
    )
    connection = sqlite3.connect(path)
    connection.execute(
        "CREATE TABLE claas_experiences (sequence INTEGER PRIMARY KEY, "
        "user_id TEXT, application_id TEXT, expires_at INTEGER, payload TEXT)"
    )
    for sequence, application, expires in [
        (1, "app", 9999999999),
        (2, "other", 9999999999),
        (3, "app", 1),
    ]:
        connection.execute(
            "INSERT INTO claas_experiences VALUES (?, ?, ?, ?, ?)",
            (sequence, "user", application, expires, experience.model_dump_json()),
        )
    connection.commit()
    connection.close()
    store = ExperienceStore(path, scope)
    assert [row.sequence for row in store.read_after()] == [1]
    assert store.read_after(1) == ()
