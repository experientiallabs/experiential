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


def test_load_tasks_duplicate_ids_raise(tmp_path: Path) -> None:
    path = tmp_path / "dup.jsonl"
    path.write_text(
        '{"task_id": "t1", "instruction": "x"}\n{"task_id": "t1", "instruction": "y"}\n',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="duplicate task_id"):
        load_tasks(path)


def test_task_text_is_canonicalized_before_evaluation_and_persistence() -> None:
    task = TaskSpec(
        task_id="before\x00after",
        instruction="before\ud800after",
        gold=["before\udcffafter", "emoji \ud83d\ude00"],
    )

    replacement = "\N{REPLACEMENT CHARACTER}"
    assert task.task_id == f"before{replacement}after"
    assert task.instruction == f"before{replacement}after"
    assert task.gold == [f"before{replacement}after", "emoji 😀"]
