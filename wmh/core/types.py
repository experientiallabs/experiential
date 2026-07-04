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


class Trace(BaseModel):
    """One full agent session: an ordered list of steps, plus provenance."""

    trace_id: str
    steps: list[Step] = Field(default_factory=list)
    source: str = "unknown"  # vendor name or file path
    metadata: JsonObject = Field(default_factory=dict)


class Session(BaseModel):
    """A live interaction the WorldModel maintains while an agent steps against it."""

    id: str
    task: str | None = None
    state: EnvState = Field(default_factory=EnvState)
    history: list[Step] = Field(default_factory=list)  # {(s_i, a_i)} fed back into the prompt
    # Whether this session's steps enrich the shared retrieval buffer. Serve-time default is True
    # (DreamGym-style online enrichment). Evaluation sessions set False: a closed-loop rollout's
    # PREDICTED steps must not become retrieval demos for later rollouts, or scores become
    # order-dependent and self-reinforcing.
    enrich: bool = True
