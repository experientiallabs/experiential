"""Scoped durable gateway capture reader regressions."""

import json
import sqlite3
from datetime import UTC, datetime
from pathlib import Path

import pytest

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


@pytest.mark.parametrize("with_evidence", [False, True])
def test_native_versions_keep_local_envelope_strict_without_inventing_evidence(
    tmp_path: Path, with_evidence: bool
) -> None:
    """Local schema1 permits new output-map facts without relaxing outer or provenance fields."""
    path = tmp_path / "capture.db"
    scope = LocalCaptureScope(user_id="user", application_id="app")
    output = (
        {"canonical_model_id": "child-model", "gemini_thought_parts_truncated": True}
        if with_evidence
        else {}
    )
    payload = {
        "schema_version": 1,
        "experience_id": "experience",
        "response_id": "response",
        "scope": scope.model_dump(),
        "protocol": "chat_completions",
        "captured_at": datetime.now(UTC).isoformat(),
        "request": {"exp_capture_output": output},
        "response": {"choices": []},
        "provenance": {"source_id": "request", "model_id": "root", "deployment_id": "child"},
    }
    encoded = json.dumps(payload)
    parsed = CapturedExchange.model_validate_json(encoded)
    assert CapturedExchange.model_validate_json(parsed.model_dump_json()) == parsed
    with pytest.raises(ValueError):
        CapturedExchange.model_validate({**payload, "canonical_model_id": "child-model"})
    with pytest.raises(ValueError):
        CapturedExchange.model_validate(
            {
                **payload,
                "provenance": {**payload["provenance"], "canonical_model_id": "child-model"},
            }
        )
    with sqlite3.connect(path) as connection:
        connection.execute(
            "CREATE TABLE gateway_captures (sequence INTEGER PRIMARY KEY, "
            "user_id TEXT, application_id TEXT, expires_at INTEGER, payload TEXT)"
        )
        connection.execute(
            "INSERT INTO gateway_captures VALUES (1, 'user', 'app', 9999999999, ?)", (encoded,)
        )
    (row,) = LocalCaptureStore(path, scope).read_after()
    assert row.experience == parsed
    assert row.experience.provenance.model_id == "root"
    retained = row.experience.request["exp_capture_output"]
    assert retained == output
    assert isinstance(retained, dict)
    assert retained.get("canonical_model_id") == ("child-model" if with_evidence else None)
    assert retained.get("gemini_thought_parts_truncated") is (True if with_evidence else None)
