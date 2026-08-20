"""Tests for shared vendor export transport and record helpers."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from exp.simulation.ingest.vendor_records import (
    VendorTraceFormatError,
    first_text,
    first_user_text,
    flatten_records,
    json_text,
    message_text,
    read_vendor_export,
    required_text,
    source_timestamp,
    vendor_w3c_id,
)


def test_read_vendor_export_hashes_the_exact_source_bytes(tmp_path: Path) -> None:
    """The source identity records the vendor label and the exact file digest."""
    path = tmp_path / "export.json"
    raw = json.dumps({"spans": [{"id": "span-1"}]}).encode("utf-8")
    path.write_bytes(raw)

    export = read_vendor_export(path, vendor="langfuse")

    assert export.source.kind == "file"
    assert export.source.source_id == f"langfuse:{path}"
    assert export.source.sha256 == hashlib.sha256(raw).hexdigest()
    assert export.issues == ()
    assert export.payloads == ({"spans": [{"id": "span-1"}]},)


def test_read_vendor_export_retains_malformed_jsonl_lines(tmp_path: Path) -> None:
    """A corrupt JSONL line is excluded with its line number instead of being skipped."""
    path = tmp_path / "export.jsonl"
    path.write_text('{"id": "span-1"}\n{oops\n{"id": "span-2"}\n', encoding="utf-8")

    export = read_vendor_export(path, vendor="braintrust")

    assert export.payloads == ({"id": "span-1"}, {"id": "span-2"})
    assert [issue.source_record for issue in export.issues] == ["line-2"]


def test_read_vendor_export_rejects_unreadable_files(tmp_path: Path) -> None:
    """A missing file and an undecodable file both fail loudly."""
    with pytest.raises(VendorTraceFormatError):
        read_vendor_export(tmp_path / "missing.json", vendor="mastra")

    binary = tmp_path / "export.json"
    binary.write_bytes(b"\xff\xfe\x00")
    with pytest.raises(VendorTraceFormatError):
        read_vendor_export(binary, vendor="mastra")


def test_flatten_records_reads_supported_wrappers() -> None:
    """Declared wrapper and record keys are unwrapped, and other shapes are rejected."""
    payload = {"data": [[{"id": "a"}], {"id": "b"}]}

    records = flatten_records(
        payload, vendor="langfuse", wrapper_keys=("data",), record_keys=("id",)
    )

    assert [record["id"] for record in records] == ["a", "b"]
    with pytest.raises(VendorTraceFormatError, match="record objects"):
        flatten_records(3, vendor="langfuse", wrapper_keys=("data",), record_keys=("id",))
    with pytest.raises(VendorTraceFormatError, match="expected keys"):
        flatten_records(
            {"other": []}, vendor="langfuse", wrapper_keys=("data",), record_keys=("id",)
        )


def test_vendor_w3c_id_preserves_valid_ids_and_hashes_opaque_keys() -> None:
    """Vendor keys map deterministically without discarding already valid W3C ids."""
    valid_trace = "a" * 32
    assert vendor_w3c_id(valid_trace, vendor="phoenix", kind="trace", namespace="trace") == (
        valid_trace
    )

    first = vendor_w3c_id("opaque-key", vendor="phoenix", kind="span", namespace="span")
    again = vendor_w3c_id("opaque-key", vendor="phoenix", kind="span", namespace="span")
    other = vendor_w3c_id("another-key", vendor="phoenix", kind="span", namespace="span")

    assert first == again
    assert first != other
    assert len(first) == 16
    assert first != vendor_w3c_id("opaque-key", vendor="phoenix", kind="span", namespace="tool")


def test_record_readers_read_declared_values_only() -> None:
    """Text, message, and timestamp readers refuse to invent missing values."""
    assert required_text("value", "label") == "value"
    with pytest.raises(VendorTraceFormatError):
        required_text(None, "label")

    assert first_text({"model": "gpt-4o"}, ("model_id", "model")) == "gpt-4o"
    assert first_text({"model": "  "}, ("model",)) is None

    assert message_text([{"type": "text", "text": "hello"}]) == "hello"
    assert message_text(None) == ""
    assert json_text({"city": "Paris"}) == '{"city":"Paris"}'
    assert first_user_text([{"role": "assistant", "content": "hi"}]) is None
    assert first_user_text([{"role": "user", "content": "why"}]) == "why"


def test_source_timestamp_reads_declared_instants() -> None:
    """ISO text and epoch numbers are read as instants, and other values are rejected."""
    expected = datetime(2024, 5, 1, tzinfo=UTC)

    assert source_timestamp("2024-05-01T00:00:00Z", "start") == expected
    assert source_timestamp(expected.timestamp(), "start") == expected
    with pytest.raises(VendorTraceFormatError):
        source_timestamp(None, "start")
    with pytest.raises(VendorTraceFormatError):
        source_timestamp("not-a-time", "start")
