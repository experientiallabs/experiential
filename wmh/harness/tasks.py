"""Task specs for closed-loop evaluation: an instruction plus gold assertions that define success.

Gold assertions are semantic post-conditions the `GoldJudge` checks against the run transcript
(WebArena/OSWorld-style "programmatic post-condition" evals, made robust to wording by an LLM
judge). Tasks are typically derived from the same benchmark the world model's traces came from —
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
    """Read a JSONL task file (one TaskSpec per line; blank lines ignored)."""
    tasks: list[TaskSpec] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if stripped:
            tasks.append(TaskSpec.model_validate_json(stripped))
    if not tasks:
        raise ValueError(f"no tasks in {path}")
    return tasks
