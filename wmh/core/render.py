"""Canonical text rendering of states, actions, and steps.

These are the *single* source of truth for turning the core types into prompt/embedding text. The
retriever embeds `encode_state_action` (phi in DreamGym Eq. 4), the world-model engine and the GEPA
optimizer both render the env prompt from the same helpers, and demos render via `render_demo`.

Keeping this in `wmh.core` (which depends on nothing) lets engine, optimize, and retrieval all share
one rendering without an import cycle — so a step embedded for retrieval and the same step shown to
the model as a demo are described identically.
"""

from __future__ import annotations

import json

from wmh.core.types import Action, EnvState, HarnessContext, HarnessSource, JsonObject, Step


def render_json(value: JsonObject) -> str:
    """Stable, compact one-liner for a JSON object: sorted keys, no whitespace churn.

    Sorting makes semantically equal objects render byte-identically regardless of insertion order,
    which is what keeps cosine similarity (and cross-run prompt text) meaningful.
    """
    if not value:
        return "{}"
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def render_action(action: Action) -> str:
    """One-line rendering of an action: tool call (name + args) or a free-text message."""
    if action.kind.value == "tool_call":
        name = action.name or "(unnamed)"
        return f"tool_call {name}({render_json(action.arguments)})"
    return f"message: {action.content or ''}"


def encode_state_action(state: EnvState, action: Action) -> str:
    """Render (state, action) into the text embedded for phi(s, a) and reused in prompts.

    A labelled, line-oriented structured summary: env state (structured config + scratchpad
    "database") then the action (kind, tool name, arguments, message). Empty fields are omitted so
    equal steps render identically.
    """
    lines = ["STATE:", f"  structured: {render_json(state.structured)}"]
    if state.scratchpad:
        lines.append(f"  scratchpad: {state.scratchpad}")
    lines.append(f"ACTION kind={action.kind.value}")
    if action.name is not None:
        lines.append(f"  tool: {action.name}")
    if action.arguments:
        lines.append(f"  arguments: {render_json(action.arguments)}")
    if action.content is not None:
        lines.append(f"  message: {action.content}")
    return "\n".join(lines)


def render_demo(step: Step) -> str:
    """Render a retrieved past step as a (state, action) -> observation few-shot example."""
    obs = step.observation
    return (
        f"{encode_state_action(step.state_before, step.action)}\n"
        f"OBSERVATION (is_error={obs.is_error}): {obs.content}"
    )


def render_harness(harness: HarnessContext) -> str:
    """Render an agent harness (system prompt + tool definitions) as an evidence block.

    Used inside the env prompt: the world model reads the harness the agent runs under so its
    predictions honor the real contract — tool argument schemas (what a malformed call's
    validation error looks like), the tools that exist at all, and the conventions the system
    prompt establishes.
    """
    lines: list[str] = []
    if harness.source is HarnessSource.INFERRED:
        lines.append(
            "(reconstructed from the corpus' observed behavior — approximate, not a verbatim "
            "capture)"
        )
    if harness.system_prompt:
        lines.append("AGENT SYSTEM PROMPT:")
        lines.append(harness.system_prompt)
    if harness.tools:
        lines.append("TOOLS THE AGENT CAN CALL:")
        for tool in harness.tools:
            suffix = f": {tool.description}" if tool.description else ""
            lines.append(f"- {tool.name}{suffix}")
            if tool.parameters:
                lines.append(f"  parameters: {render_json(tool.parameters)}")
    return "\n".join(lines)


def render_agent_messages(harness: HarnessContext, task: str) -> tuple[str, str]:
    """The (system, user) messages a fresh agent would receive for `task` under `harness`.

    Token-realistic by construction: the system text is the captured system prompt verbatim,
    followed by the tool definitions serialized as the JSON schemas a harness advertises; the
    user message is the task exactly as the user would type it. (A real harness passes tools via
    the API's tools parameter rather than in-text; this is the closest single-text equivalent.)
    """
    system = harness.system_prompt
    if harness.tools:
        tools_json = json.dumps(
            [tool.model_dump() for tool in harness.tools], indent=2, sort_keys=True
        )
        system = f"{system}\n\n# Tools\n{tools_json}" if system else f"# Tools\n{tools_json}"
    return system, task


def build_env_prompt(
    base_prompt: str,
    task: str | None,
    state: EnvState,
    action: Action,
    *,
    history: list[Step] | None = None,
    demos: list[Step] | None = None,
    harness: HarnessContext | None = None,
) -> tuple[str, str]:
    """Assemble the (system, user) world-model completion that predicts the next observation.

    Mirrors DreamGym Eq. 4 ``M_exp(R_t | {(s_i,a_i)}, {d_j}, tau)``: the base/optimized prompt is
    the system message; the task, current state, recent history, retrieved demos, and the incoming
    action form the user message. This is the *single* assembly used by both the serving engine
    (`wmh.engine.prompts`) and the GEPA optimizer, so prompts are evolved against exactly what the
    world model serves. A non-empty `harness` (the agent's captured system prompt + tools) leads
    the user message: it is evidence about the environment's contract, not instructions.
    """
    system = base_prompt
    demo_block = (
        "\n\n".join(render_demo(d) for d in demos) if demos else "(no similar past examples)"
    )
    history_block = (
        "\n".join(
            f"{encode_state_action(h.state_before, h.action)}\n"
            f"OBSERVATION (is_error={h.observation.is_error}): {h.observation.content}"
            for h in history
        )
        if history
        else "(start of session)"
    )
    harness_block = (
        f"AGENT HARNESS (the system prompt and tools the agent operates under):\n"
        f"{render_harness(harness)}\n\n"
        if harness
        else ""
    )
    user = (
        f"{harness_block}"
        f"TASK:\n{task or '(none)'}\n\n"
        f"INTERACTION HISTORY:\n{history_block}\n\n"
        f"SIMILAR PAST EXAMPLES:\n{demo_block}\n\n"
        f"CURRENT ENV STATE:\n  structured: {render_json(state.structured)}\n"
        f"  scratchpad: {state.scratchpad or '(empty)'}\n\n"
        f"AGENT ACTION:\n{render_action(action)}\n\n"
        f"{OUTPUT_CONTRACT}"
    )
    return system, user


# The world-model output contract. Parsed by `wmh.core.parsing.parse_observation`; kept next to the
# prompt assembly so the instruction and the parser never drift.
OUTPUT_CONTRACT = (
    "Respond with ONLY a JSON object describing the environment's response to this action:\n"
    '{"output": "<exactly what the environment returns to the agent>", '
    '"is_error": <true if the action failed/was invalid>, '
    '"state_note": "<one short fact to remember about the new env state, or empty>"}'
)
