"""Task specs the managed agent runs — the same task drives a real run and a simulated one.

A task is (instruction, gold assertions, optional real-env setup). Gold assertions are semantic
post-conditions the `GoldJudge` checks against the run transcript (WebArena/OSWorld-style
"programmatic post-condition" evals, made robust to wording by an LLM judge). `setup` seeds the
*real* sandbox before the run; closed-loop runs against the world model ignore it (the world model
has already internalized environment state from traces).
"""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, Field


class TaskSpec(BaseModel):
    """One task: what to do, how to seed the real env, and what must hold afterwards."""

    task_id: str
    instruction: str
    gold: list[str] = Field(default_factory=list)  # assertions that define success
    setup: list[str] = Field(default_factory=list)  # bash run in the real sandbox before the agent


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


def save_tasks(tasks: list[TaskSpec], path: str | Path) -> None:
    """Write tasks as JSONL (the inverse of `load_tasks`)."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    lines = "".join(f"{task.model_dump_json()}\n" for task in tasks)
    Path(path).write_text(lines, encoding="utf-8")
