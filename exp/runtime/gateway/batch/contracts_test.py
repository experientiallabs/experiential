"""Contract tests: input JSONL parsing and public object rendering."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from exp.runtime.gateway.batch.contracts import (
    MAXIMUM_BATCH_LINES,
    BatchCounts,
    BatchFile,
    BatchJob,
    BatchLineResult,
    BatchStatus,
    BatchSubmitError,
    parse_input_jsonl,
)


def _job(**overrides: object) -> BatchJob:
    """Build one minimal valid job for rendering assertions."""
    created = datetime(2026, 9, 1, tzinfo=UTC)
    fields: dict[str, object] = {
        "batch_id": "batch_a",
        "organization_id": "org_a",
        "identity_id": "id_a",
        "surface": "/v1/chat/completions",
        "provider": "openrouter",
        "credential_reference": "secret://fixture",
        "input_file_id": "file_a",
        "counts": BatchCounts(total=2),
        "created_at": created,
        "expires_at": created + timedelta(hours=24),
    }
    fields.update(overrides)
    return BatchJob.model_validate(fields)


def test_parse_input_jsonl_returns_numbered_objects() -> None:
    """Non-empty lines parse in order with 1-indexed numbering."""
    payload = b'{"custom_id": "a"}\n\n{"custom_id": "b"}\n'
    parsed = parse_input_jsonl(payload)
    assert [number for number, _ in parsed] == [1, 3]
    assert parsed[0][1]["custom_id"] == "a"


def test_parse_input_jsonl_rejects_non_object_lines() -> None:
    """A JSON array line is refused with its line number."""
    with pytest.raises(BatchSubmitError, match="line 2"):
        parse_input_jsonl(b'{"custom_id": "a"}\n[1, 2]\n')


def test_parse_input_jsonl_rejects_invalid_json() -> None:
    """A malformed line is refused with its line number."""
    with pytest.raises(BatchSubmitError, match="line 1"):
        parse_input_jsonl(b"{nope}\n")


def test_parse_input_jsonl_rejects_empty_payload() -> None:
    """A payload without any request line is refused."""
    with pytest.raises(BatchSubmitError, match="no request lines"):
        parse_input_jsonl(b"\n\n")


def test_parse_input_jsonl_enforces_the_line_budget() -> None:
    """One line above the product cap refuses the whole payload."""
    payload = b"\n".join(b'{"custom_id": "x"}' for _ in range(MAXIMUM_BATCH_LINES + 1))
    with pytest.raises(BatchSubmitError, match="limit"):
        parse_input_jsonl(payload)


def test_line_result_requires_exactly_one_payload() -> None:
    """Both-set and neither-set results are invalid."""
    with pytest.raises(ValueError, match="exactly one"):
        BatchLineResult(custom_id="a", status_code=200)
    with pytest.raises(ValueError, match="exactly one"):
        BatchLineResult(custom_id="a", status_code=200, response={}, error={})


def test_line_result_output_object_shapes_success_and_error() -> None:
    """The rendered output line matches the OpenAI batch output schema."""
    success = BatchLineResult(custom_id="a", status_code=200, response={"ok": True})
    rendered = success.output_jsonl_object(line_id="batch_a_line_0")
    assert rendered["response"] == {"status_code": 200, "body": {"ok": True}}
    assert rendered["error"] is None
    failure = BatchLineResult(custom_id="b", status_code=429, error={"message": "slow down"})
    assert failure.output_jsonl_object(line_id="x")["response"] is None


def test_job_public_object_is_openai_batch_shaped() -> None:
    """The public batch object carries the compatibility fields."""
    rendered = _job(status=BatchStatus.IN_PROGRESS).public_object()
    assert rendered["object"] == "batch"
    assert rendered["endpoint"] == "/v1/chat/completions"
    assert rendered["status"] == "in_progress"
    assert rendered["completion_window"] == "24h"
    assert rendered["request_counts"] == {"total": 2, "completed": 0, "failed": 0}
    assert isinstance(rendered["created_at"], int)


def test_job_public_object_reports_the_failure_reason_and_failed_at() -> None:
    """A failed job lists its failure under errors and stamps failed_at only."""
    finalized = datetime(2026, 9, 1, 1, tzinfo=UTC)
    rendered = _job(
        status=BatchStatus.FAILED,
        failure_message="provider rejected the batch submission: status 400: bad model",
        finalized_at=finalized,
    ).public_object()
    assert rendered["errors"] == {
        "object": "list",
        "data": [
            {
                "code": "failed",
                "message": "provider rejected the batch submission: status 400: bad model",
                "line": None,
                "custom_id": None,
            }
        ],
    }
    assert rendered["failed_at"] == int(finalized.timestamp())
    assert rendered["completed_at"] is None
    assert rendered["expired_at"] is None
    assert rendered["cancelled_at"] is None


def test_job_public_object_stamps_the_terminal_timestamp_by_status() -> None:
    """Each terminal status owns exactly one ``*_at`` field."""
    finalized = datetime(2026, 9, 1, 1, tzinfo=UTC)
    stamp = int(finalized.timestamp())
    by_status = {
        BatchStatus.COMPLETED: "completed_at",
        BatchStatus.EXPIRED: "expired_at",
        BatchStatus.CANCELLED: "cancelled_at",
    }
    for status, field in by_status.items():
        rendered = _job(status=status, finalized_at=finalized).public_object()
        assert rendered[field] == stamp, status
        others = {"completed_at", "failed_at", "expired_at", "cancelled_at"} - {field}
        assert all(rendered[other] is None for other in others), status
    assert _job(status=BatchStatus.IN_PROGRESS).public_object()["errors"] is None


def test_file_public_object_is_openai_file_shaped() -> None:
    """The public file object carries the compatibility fields."""
    record = BatchFile(
        file_id="file_a",
        organization_id="org_a",
        filename="input.jsonl",
        size_bytes=10,
        created_at=datetime(2026, 9, 1, tzinfo=UTC),
    )
    rendered = record.public_object()
    assert rendered["object"] == "file"
    assert rendered["purpose"] == "batch"
    assert rendered["bytes"] == 10
