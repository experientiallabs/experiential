"""The managed agent's tool surface: a pi-style minimal set, rendered compactly.

Following pi (badlogic/pi-mono: "four tools and a ~180-word system prompt"), the always-loaded tool
surface is deliberately tiny and priced in tokens: `bash` + file read/write against the environment,
plus three harness-side tools (`save_skill`/`read_skill` for the Voyager-style skill library, and
`submit` to end the run). Everything else the agent needs it builds *in* the environment with bash —
progressive disclosure instead of a wide tool schema.

Env tools become `Action`s the environment executes; harness tools (`HARNESS_TOOLS`) are handled by
the runtime itself so they behave identically whether the environment is a real E2B sandbox or the
world model.
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
SAVE_SKILL = ToolSpec(
    name="save_skill",
    description=(
        "Save a reusable skill to your persistent library for future tasks. Save a skill whenever "
        "you work out a non-obvious, repeatable way to do something."
    ),
    arguments={
        "name": "short kebab-case skill name",
        "description": "one line: when to use this skill",
        "body": "the skill itself: steps, commands, or a script, ready to reuse",
    },
)
READ_SKILL = ToolSpec(
    name="read_skill",
    description="Read the full body of a skill from your library (the prompt lists only names).",
    arguments={"name": "the skill name to read"},
)
SUBMIT = ToolSpec(
    name="submit",
    description="Finish the task and submit your answer/result summary. This ends the run.",
    arguments={"answer": "your final answer or a summary of what you did"},
)

TOOL_REGISTRY: dict[str, ToolSpec] = {
    t.name: t for t in (BASH, READ_FILE, WRITE_FILE, SAVE_SKILL, READ_SKILL, SUBMIT)
}

# Tools the runtime handles itself (never routed to the environment). `submit` must always be
# available or the agent cannot end a run; the spec validator enforces that.
HARNESS_TOOLS = frozenset({SAVE_SKILL.name, READ_SKILL.name, SUBMIT.name})

DEFAULT_TOOLS = [t.name for t in (BASH, READ_FILE, WRITE_FILE, SAVE_SKILL, READ_SKILL, SUBMIT)]


def resolve_tools(names: list[str]) -> list[ToolSpec]:
    """Map tool names to specs, raising on unknown names (a mutated spec must not invent tools)."""
    unknown = [n for n in names if n not in TOOL_REGISTRY]
    if unknown:
        raise ValueError(f"unknown tools {unknown}; registry has {sorted(TOOL_REGISTRY)}")
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
    """An env tool call as the normalized Action the environment (real or simulated) executes."""
    return Action(kind=ActionKind.TOOL_CALL, name=call.tool, arguments=call.arguments)
