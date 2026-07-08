"""Core data types shared across the harness.

These are the normalized, vendor-agnostic representations that ingestion produces and that the
WorldModel, retriever, optimizer, and providers all operate on.
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, Field, JsonValue

# Tool arguments, env config, and span metadata are user-defined JSON. `JsonValue` is pydantic's
# concrete recursive JSON type — honest about the shape without falling back to `Any`.
JsonObject = dict[str, JsonValue]


class ActionKind(StrEnum):
    TOOL_CALL = "tool_call"
    MESSAGE = "message"


class Action(BaseModel):
    """What the agent did this step. Either a tool call or a free-text message."""

    kind: ActionKind
    name: str | None = None  # tool name, when kind == tool_call
    arguments: JsonObject = Field(default_factory=dict)
    content: str | None = None  # message text, when kind == message


class Observation(BaseModel):
    """What the environment returned in response to an action.

    `reward` is optional and exists to support RL-style use (DreamGym assigns r at terminal steps).
    """

    content: str
    is_error: bool = False
    reward: float | None = None
    metadata: JsonObject = Field(default_factory=dict)


class EnvState(BaseModel):
    """A snapshot of the environment as seen by the agent.

    `structured` holds machine-readable env config (cwd, open files, cart contents, ...).
    `scratchpad` is the free-text "database" the world model writes to itself to stay consistent
    across a session (e.g. "user created foo.txt", "logged in as alice").
    """

    structured: JsonObject = Field(default_factory=dict)
    scratchpad: str = ""


class Step(BaseModel):
    """One (state, action) -> observation transition. The unit of retrieval and scoring."""

    action: Action
    observation: Observation
    state_before: EnvState = Field(default_factory=EnvState)
    task: str | None = None  # originating instruction (tau in DreamGym Eq. 4)
    raw_span_ids: list[str] = Field(default_factory=list)


class ToolDefinition(BaseModel):
    """One tool the agent's harness exposes: its name, description, and JSON-schema parameters."""

    name: str
    description: str = ""
    parameters: JsonObject = Field(default_factory=dict)


class HarnessSource(StrEnum):
    """Where a harness context came from: recorded in traces, or predicted from their behavior."""

    CAPTURED = "captured"
    INFERRED = "inferred"


class HarnessContext(BaseModel):
    """The agent-side harness: the system prompt and the tool definitions the agent ran under.

    `CAPTURED` contexts come verbatim from traces (`gen_ai.system_instructions` /
    `gen_ai.tool.definitions`) — what the original agent actually received. `INFERRED` contexts
    are predicted from sparse trace evidence (observed calls, argument shapes, validation
    errors) when the capture recorded none; a captured context always takes precedence.

    Consumed strictly AGENT-side: rendering the token-realistic messages a fresh agent would
    receive for a new task (`wmh scenarios create`) and running the baseline `LLMAgent` under
    the real system prompt. It is never part of the world model's env prompt — the world model
    simulates the environment's response to an action, not the agent's context assembly (which
    is a function of the harness's full per-step state, a different and harder object).
    """

    system_prompt: str = ""
    tools: list[ToolDefinition] = Field(default_factory=list)
    source: HarnessSource = HarnessSource.CAPTURED

    def __bool__(self) -> bool:
        """Empty contexts are falsy, so callers can gate sections on `if harness:`."""
        return bool(self.system_prompt or self.tools)


class Trace(BaseModel):
    """One full agent session: an ordered list of steps, plus provenance."""

    trace_id: str
    steps: list[Step] = Field(default_factory=list)
    source: str = "unknown"  # vendor name or file path
    metadata: JsonObject = Field(default_factory=dict)
    harness: HarnessContext | None = None  # agent-side system prompt + tools, when captured


class Session(BaseModel):
    """A live interaction the WorldModel maintains while an agent steps against it."""

    id: str
    task: str | None = None
    state: EnvState = Field(default_factory=EnvState)
    history: list[Step] = Field(default_factory=list)  # {(s_i, a_i)} fed back into the prompt
