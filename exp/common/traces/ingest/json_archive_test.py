"""Incremental JSON parsing preserves exact record shapes and excludes malformed lines."""

import json
import sqlite3
from pathlib import Path

import pytest

from exp.common.traces.ingest.json_archive import JsonArchive
from exp.common.traces.ingest.vendor_records import VendorTraceFormatError


@pytest.mark.parametrize(
    "document",
    [
        '{"duplicate":1,"duplicate":2,"values":[null,true,false,-0.2,123456789012345678901234567890,"☃"]}',
        '[{"messages":[{"role":"user","content":"hello"}]}]',
        '{"nested":{"array":[1,2,3]}}',
    ],
)
def test_disk_json_matches_standard_decoder(tmp_path: Path, document: str) -> None:
    """Container indexing preserves duplicate keys, Unicode and arbitrary JSON integers."""
    path = tmp_path / "input.json"
    path.write_text(document)
    directory = tmp_path / "scratch"
    directory.mkdir()
    with sqlite3.connect(directory / "archive.db") as connection:
        archive = JsonArchive(path, directory, connection, "chat-json")
        assert [node.value() for node in archive.documents()] == [json.loads(document)]
        assert not archive.issues


def test_partial_bad_line_cannot_leak_nodes_into_next_record(tmp_path: Path) -> None:
    """Malformed JSONL is rolled back completely before parsing the next line."""
    path = tmp_path / "input.jsonl"
    path.write_text('{"good":1}\n{"bad":[1,\n{"good":2}\n')
    directory = tmp_path / "scratch"
    directory.mkdir()
    with sqlite3.connect(directory / "archive.db") as connection:
        archive = JsonArchive(path, directory, connection, "chat-json")
        assert [node.value() for node in archive.documents()] == [{"good": 1}, {"good": 2}]
        assert len(archive.issues) == 1 and archive.issues[0].source_record == "line-2"


def test_utf8_failure_is_not_silently_replaced(tmp_path: Path) -> None:
    """Unrecoverable transport encoding fails before any normalized evidence is accepted."""
    path = tmp_path / "input.jsonl"
    path.write_bytes(b'{"a": "\xff"}')
    directory = tmp_path / "scratch"
    directory.mkdir()
    with (
        sqlite3.connect(directory / "archive.db") as connection,
        pytest.raises(VendorTraceFormatError, match="UTF-8"),
    ):
        JsonArchive(path, directory, connection, "chat-json")
