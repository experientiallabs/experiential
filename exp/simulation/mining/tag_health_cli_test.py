"""The read-only tag-health command over saved matrix files."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from exp.simulation.mining.tag_health_cli import main, read_excluded
from exp.simulation.mining.tagging import TagMatrix, TagRow


def test_the_command_reads_matrices_and_writes_only_the_requested_output(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Inputs stay unchanged; the statistics land in ``--json``; the summary names flags."""
    matrix = TagMatrix(
        bank_id="shape",
        bank_version=1,
        question_ids=("always", "split"),
        view="task",
        rows=tuple(
            TagRow(trace_id=f"t{i}", unit_hash=f"h{i}", values=(0.95, 0.9 if i % 2 else 0.1))
            for i in range(8)
        ),
    )
    source = tmp_path / "shape.task.json"
    source.write_text(matrix.model_dump_json(), encoding="utf-8")
    before = source.read_bytes()
    excluded = tmp_path / "excluded.json"
    excluded.write_text(json.dumps([{"view": "task"}, {"view": "task"}, "x"]), encoding="utf-8")
    output = tmp_path / "health.json"
    assert main([str(source), "--excluded", str(excluded), "--json", str(output)]) == 0
    assert source.read_bytes() == before
    health = json.loads(output.read_text(encoding="utf-8"))
    assert health["banks"][0]["excluded"] == 2
    assert "always_yes: 1: always (1.000)" in capsys.readouterr().out
    assert read_excluded(excluded) == {"task": 2, "*": 1}
