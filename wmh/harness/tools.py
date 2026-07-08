"""The rollout agent's tool surface: a deliberately minimal set, rendered compactly.

The agent gets four tools: `bash` + file read/write against the environment, and `submit` to end
the run. Everything else the agent needs it composes with bash — a small always-loaded tool schema
keeps the prompt cheap, and capability comes from composition rather than tool count.

Env tools become `Action`s the environment answers (the world model, in closed-loop eval); `submit`
is handled by the runtime itself.
"""

from __future__ import annotations

from pydantic import BaseModel, Field, ValidationError

from wmh.core.parsing import extract_json_object
from wmh.core.types import Action, ActionKind, JsonObject


class ToolSpec(BaseModel):
    """One tool the agent may call: a name, what it does, and its named arguments."""

    name: str
    description: str
    arguments: dict[str, str] = Field(default_factory=dict)  # arg name -> one-line description


BASH = ToolSpec(
    name="bash",
    description="Run a shell command in the environment; returns stdout+stderr and exit status.",
    arguments={"command": "the shell command to run"},
)
READ_FILE = ToolSpec(
    name="read_file",
    description="Read a file from the environment.",
    arguments={"path": "absolute path of the file to read"},
)
WRITE_FILE = ToolSpec(
    name="write_file",
    description="Write a file in the environment (parent dirs are created).",
    arguments={"path": "absolute path to write", "content": "full file content"},
)
SUBMIT = ToolSpec(
    name="submit",
    description="Finish the task and submit your answer/result summary. This ends the run.",
    arguments={"answer": "your final answer or a summary of what you did"},
)
READ_SKILL = ToolSpec(
    name="read_skill",
    description="Read the full body of a skill from your library (the prompt lists only names).",
    arguments={"name": "the skill name to read"},
)

TOOL_REGISTRY: dict[str, ToolSpec] = {
    t.name: t for t in (BASH, READ_FILE, WRITE_FILE, SUBMIT, READ_SKILL)
}

# `read_skill` is only useful when a harness ships skills, so it is not a default tool; the runtime
# adds it automatically when constructed with a non-empty skill library.
DEFAULT_TOOLS = [t.name for t in (BASH, READ_FILE, WRITE_FILE, SUBMIT)]


def resolve_tools(names: list[str]) -> list[ToolSpec]:
    """Map tool names to specs, raising on unknown names. `submit` is mandatory (ends the run)."""
    unknown = [n for n in names if n not in TOOL_REGISTRY]
    if unknown:
        raise ValueError(f"unknown tools {unknown}; registry has {sorted(TOOL_REGISTRY)}")
    if SUBMIT.name not in names:
        raise ValueError(f"the {SUBMIT.name!r} tool is required (without it a run cannot end)")
    return [TOOL_REGISTRY[n] for n in names]


def render_tools(tools: list[ToolSpec]) -> str:
    """Render the tool list for the system prompt, one compact block per tool."""
    lines: list[str] = []
    for tool in tools:
        args = ", ".join(f'"{k}": <{v}>' for k, v in tool.arguments.items())
        lines.append(f"- {tool.name}: {tool.description}\n  arguments: {{{args}}}")
    return "\n".join(lines)


class ToolCall(BaseModel):
    """One parsed tool call from the agent's reply."""

    tool: str
    arguments: JsonObject = Field(default_factory=dict)


def parse_tool_call(text: str) -> ToolCall | None:
    """Parse the agent's reply into a ToolCall (`{"tool": ..., "arguments": {...}}`), or None.

    Lenient about surrounding prose/fences (the JSON is extracted, not matched), strict about
    shape: a reply whose JSON has no string `tool` field is not a call.
    """
    raw = extract_json_object(text)
    if raw is None:
        return None
    try:
        return ToolCall.model_validate_json(raw)
    except ValidationError:
        return None


def to_action(call: ToolCall) -> Action:
    """An env tool call as the normalized Action the environment answers."""
    return Action(kind=ActionKind.TOOL_CALL, name=call.tool, arguments=call.arguments)
