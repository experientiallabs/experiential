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
    """A full agent run on one task: ordered steps plus the graded outcome.

    Beyond the transitions, a trajectory records the *contract the agent ran under* so the corpus
    preserves everything the agent was shown — essential for faithful world modeling:
      - ``system_prompt``: the system instructions that framed the agent.
      - ``tools``: the tool definitions the agent could call (name + schema).
      - ``harness``: how the agent was driven (provider, model, step/token budgets, inference cfg).
    All default empty, so a producer that doesn't set them emits exactly as before.
    """

    task: Task
    steps: list[StepRecord]
    final_answer: str = ""
    reward: float | None = None
    model: str = ""
    split: str = ""
    metadata: dict[str, JsonValue] = field(default_factory=dict)
    system_prompt: str = ""
    tools: list[JsonValue] = field(default_factory=list)
    harness: dict[str, JsonValue] = field(default_factory=dict)
