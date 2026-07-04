"""Task specs for closed-loop evaluation: an instruction plus gold assertions that define success.

Gold assertions are semantic post-conditions the `GoldJudge` checks against the run transcript —
conditions on the final state, made robust to wording by an LLM judge instead of brittle exact
matching. Tasks are typically derived from the same benchmark the world model's traces came from —
`Trace.metadata` already carries gold assertions for traces captured with them.
"""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, Field


class TaskSpec(BaseModel):
    """One task: what the agent must do, and the assertions that must hold afterwards."""

    task_id: str
    instruction: str
    gold: list[str] = Field(default_factory=list)  # assertions that define success


def load_tasks(path: str | Path) -> list[TaskSpec]:
    """Read a JSONL task file (one TaskSpec per line; blank lines ignored).

    Duplicate `task_id`s are an error: reports key outcomes by task_id, so a duplicate would run
    (and cost) k passes twice while silently keeping only the last outcome.
    """
    tasks: list[TaskSpec] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if stripped:
            tasks.append(TaskSpec.model_validate_json(stripped))
    if not tasks:
        raise ValueError(f"no tasks in {path}")
    ids = [t.task_id for t in tasks]
    duplicates = sorted({i for i in ids if ids.count(i) > 1})
    if duplicates:
        raise ValueError(f"duplicate task_id(s) in {path}: {duplicates}")
    return tasks
