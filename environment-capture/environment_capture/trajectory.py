"""Core record types: one real benchmark run = one Trajectory of (action -> observation) steps.

Stdlib-only (dataclasses, no pydantic) so the package stays dependency-free and shareable with
non-wmh consumers.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TypeAlias

JsonValue: TypeAlias = "str | int | float | bool | None | list[JsonValue] | dict[str, JsonValue]"


@dataclass(frozen=True)
class Task:
    """One benchmark task as the agent sees it (gold answers live elsewhere)."""

    task_id: str
    prompt: str
    data: dict[str, JsonValue] = field(default_factory=dict)


@dataclass(frozen=True)
class ToolCall:
    """The agent's action: a named tool invocation with JSON arguments."""

    name: str
    arguments: dict[str, JsonValue]


@dataclass(frozen=True)
class StepRecord:
    """One real transition: the action taken and the observation the environment returned."""

    action: ToolCall
    output: str
    is_error: bool = False


@dataclass(frozen=True)
class Trajectory:
    """A full agent run on one task: ordered steps plus the graded outcome."""

    task: Task
    steps: list[StepRecord]
    final_answer: str = ""
    reward: float | None = None
    model: str = ""
    split: str = ""
    metadata: dict[str, JsonValue] = field(default_factory=dict)
