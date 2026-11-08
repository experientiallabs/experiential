"""The front door's trace source digest over a file or a directory of session files."""

from __future__ import annotations

import hashlib
from pathlib import Path

from exp.optimize.evaluation.variance.frontdoor_source import TraceSource


def test_a_session_directory_has_a_content_digest(tmp_path: Path) -> None:
    """A directory hashes its sorted relative paths and bytes: stable, and moved by any change."""
    pool = tmp_path / "pool"
    (pool / "b").mkdir(parents=True)
    (pool / "a.jsonl").write_text('{"type": "session"}\n')
    (pool / "b" / "c.jsonl").write_text('{"type": "user"}\n')
    source = TraceSource(path=pool, source="agent-sessions")
    first = source.sha256()
    assert len(first) == 64 and source.sha256() == first

    (pool / "b" / "c.jsonl").write_text('{"type": "user!"}\n')
    edited = source.sha256()
    assert edited != first
    (pool / "b" / "c.jsonl").rename(pool / "b" / "d.jsonl")
    assert source.sha256() not in (first, edited)
    (pool / "e.jsonl").write_text("")
    assert source.sha256() != edited


def test_a_trace_file_digest_is_its_bytes(tmp_path: Path) -> None:
    """A single file still hashes as its exact bytes, so saved source records stay valid."""
    path = tmp_path / "traces.jsonl"
    path.write_bytes(b'{"x": 1}\n')
    assert TraceSource(path=path).sha256() == hashlib.sha256(b'{"x": 1}\n').hexdigest()
