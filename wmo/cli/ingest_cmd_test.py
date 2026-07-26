"""Tests for the `wmo ingest` CLI command (typer runner; no network, no database)."""

from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from wmo.cli.app import app

runner = CliRunner()

_CONVERSATION = {
    "messages": [
        {"role": "user", "content": "hi"},
        {
            "role": "assistant",
            "tool_calls": [{"id": "c1", "function": {"name": "get", "arguments": "{}"}}],
        },
        {"role": "tool", "tool_call_id": "c1", "content": "ok"},
    ]
}


def _corpus(tmp_path: Path) -> Path:
    src = tmp_path / "chat.json"
    src.write_text(json.dumps(_CONVERSATION), encoding="utf-8")
    return src


def test_ingest_file_json_events_end_to_end(tmp_path: Path) -> None:
    """`wmo ingest --file x --json` streams the pinned event vocabulary and writes the corpus."""
    src = _corpus(tmp_path)
    out = tmp_path / "normalized.otel.jsonl"
    result = runner.invoke(app, ["ingest", "--file", str(src), "--out", str(out), "--json"])
    assert result.exit_code == 0, result.output
    events = [json.loads(line) for line in result.output.strip().splitlines()]
    assert [e["type"] for e in events][0] == "detected"
    assert events[0] == {"type": "detected", "format": "chat-json", "traces": 1}
    assert events[-1]["type"] == "done"
    assert events[-1]["otel_object"] == str(out)
    assert all(e["type"] == "progress" for e in events[1:-1])
    assert all("total" in e for e in events if e["type"] == "progress")
    assert out.exists()


def test_ingest_rich_output_and_build_hint(tmp_path: Path) -> None:
    src = _corpus(tmp_path)
    out = tmp_path / "n.otel.jsonl"
    result = runner.invoke(app, ["ingest", "--file", str(src), "--out", str(out)])
    assert result.exit_code == 0, result.output
    assert "detected" in result.output
    assert "chat-json" in result.output
    assert "wmo build" in result.output  # tells the user the next step


def test_ingest_error_exits_nonzero(tmp_path: Path) -> None:
    src = tmp_path / "junk.json"
    src.write_text('{"nothing": true}', encoding="utf-8")
    result = runner.invoke(app, ["ingest", "--file", str(src), "--json"])
    assert result.exit_code == 1
    events = [json.loads(line) for line in result.output.strip().splitlines()]
    assert events[-1]["type"] == "error"
    assert events[-1]["code"] == "bad_format"


def test_ingest_requires_a_transport() -> None:
    result = runner.invoke(app, ["ingest"])
    assert result.exit_code == 2
    assert "--file" in result.output


def test_ingest_rejects_file_plus_pull(tmp_path: Path) -> None:
    src = _corpus(tmp_path)
    result = runner.invoke(app, ["ingest", "--file", str(src), "--pull"])
    assert result.exit_code == 2


def test_ingest_dsn_implies_postgres_source(tmp_path: Path) -> None:
    """`--dsn/--table` alone select the postgres adapter; unreachable db = error event, exit 1."""
    result = runner.invoke(
        app,
        [
            "ingest",
            "--dsn",
            "postgresql://nobody@127.0.0.1:1/none?connect_timeout=1",
            "--table",
            "agent_traces",
            "--json",
            "--out",
            str(tmp_path / "o.jsonl"),
        ],
    )
    assert result.exit_code == 1
    events = [json.loads(line) for line in result.output.strip().splitlines()]
    assert events[-1]["type"] == "error"
    assert events[-1]["code"] in ("unreachable", "bad_credentials")
