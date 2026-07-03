"""Tests for task-spec loading/saving."""

from __future__ import annotations

from pathlib import Path

import pytest

from wmh.agent.tasks import TaskSpec, load_tasks, save_tasks


def test_tasks_roundtrip(tmp_path: Path) -> None:
    tasks = [
        TaskSpec(task_id="t1", instruction="do a", gold=["a done"], setup=["mkdir /w"]),
        TaskSpec(task_id="t2", instruction="do b"),
    ]
    path = tmp_path / "tasks.jsonl"
    save_tasks(tasks, path)
    assert load_tasks(path) == tasks


def test_load_tasks_skips_blank_lines(tmp_path: Path) -> None:
    path = tmp_path / "tasks.jsonl"
    path.write_text(
        '{"task_id": "t1", "instruction": "x"}\n\n{"task_id": "t2", "instruction": "y"}\n',
        encoding="utf-8",
    )
    assert [t.task_id for t in load_tasks(path)] == ["t1", "t2"]


def test_load_tasks_empty_raises(tmp_path: Path) -> None:
    path = tmp_path / "empty.jsonl"
    path.write_text("\n\n", encoding="utf-8")
    with pytest.raises(ValueError):
        load_tasks(path)
