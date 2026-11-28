"""Incremental JSON parsing preserves exact record shapes and excludes malformed lines."""

import hashlib
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


def test_bom_prefixed_json_is_parsed_and_original_bytes_are_hashed(
    tmp_path: Path,
) -> None:
    """A UTF-8 BOM is ignored for parsing but retained in the source digest."""
    original = b'\xef\xbb\xbf{"good":1}'
    path = tmp_path / "input.json"
    path.write_bytes(original)
    directory = tmp_path / "scratch"
    directory.mkdir()

    with sqlite3.connect(directory / "archive.db") as connection:
        archive = JsonArchive(path, directory, connection, "chat-json")

        assert [node.value() for node in archive.documents()] == [{"good": 1}]
        assert not archive.issues
        assert archive.source.sha256 == hashlib.sha256(original).hexdigest()


def test_bom_prefixed_jsonl_preserves_first_record_and_original_hash(
    tmp_path: Path,
) -> None:
    """A BOM on the first JSONL line does not drop its record."""
    original = b'\xef\xbb\xbf{"good":1}\n{"good":2}\n'
    path = tmp_path / "input.jsonl"
    path.write_bytes(original)
    directory = tmp_path / "scratch"
    directory.mkdir()

    with sqlite3.connect(directory / "archive.db") as connection:
        archive = JsonArchive(path, directory, connection, "chat-json")

        assert [node.value() for node in archive.documents()] == [
            {"good": 1},
            {"good": 2},
        ]
        assert not archive.issues
        assert archive.source.sha256 == hashlib.sha256(original).hexdigest()
