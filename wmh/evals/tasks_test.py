"""Tests for task-spec loading."""

from __future__ import annotations

from pathlib import Path

import pytest

from wmh.evals.tasks import TaskSpec, load_tasks


def test_load_tasks_reads_jsonl_and_skips_blanks(tmp_path: Path) -> None:
    path = tmp_path / "tasks.jsonl"
    path.write_text(
        '{"task_id": "t1", "instruction": "x", "gold": ["did x"]}\n'
        "\n"
        '{"task_id": "t2", "instruction": "y"}\n',
        encoding="utf-8",
    )
    tasks = load_tasks(path)
    assert [t.task_id for t in tasks] == ["t1", "t2"]
    assert tasks[0] == TaskSpec(task_id="t1", instruction="x", gold=["did x"])
    assert tasks[1].gold == []


def test_load_tasks_empty_raises(tmp_path: Path) -> None:
    path = tmp_path / "empty.jsonl"
    path.write_text("\n\n", encoding="utf-8")
    with pytest.raises(ValueError, match="no tasks"):
        load_tasks(path)


def test_setup_defaults_empty_and_roundtrips_through_load_tasks(tmp_path: Path) -> None:
    path = tmp_path / "tasks.jsonl"
    path.write_text(
        '{"task_id": "t1", "instruction": "x", "setup": ["apt-get install -y jq", "mkdir /data"]}\n'
        '{"task_id": "t2", "instruction": "y"}\n',
        encoding="utf-8",
    )
    tasks = load_tasks(path)
    assert tasks[0].setup == ["apt-get install -y jq", "mkdir /data"]
    assert tasks[1].setup == []  # absent in JSONL -> default, sim path can ignore it
    # And the field survives a serialize/parse round trip unchanged.
    assert TaskSpec.model_validate_json(tasks[0].model_dump_json()) == tasks[0]


def test_load_tasks_duplicate_ids_raise(tmp_path: Path) -> None:
    path = tmp_path / "dup.jsonl"
    path.write_text(
        '{"task_id": "t1", "instruction": "x"}\n{"task_id": "t1", "instruction": "y"}\n',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="duplicate task_id"):
        load_tasks(path)
