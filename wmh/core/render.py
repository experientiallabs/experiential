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

from wmh.core.types import Action, EnvState, JsonObject, Step


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


def encode_action(action: Action) -> str:
    """Command-only retrieval key: the action itself (tool + arguments, or message) with none of the
    `STATE:` / `ACTION kind=` scaffolding `encode_state_action` adds.

    For stateless traces (empty env state) that scaffolding is constant across every step, so it
    dominates the embedding and dilutes the part that actually varies — the command. Embedding just
    the action concentrates the signal, which helps a semantic embedder find same-intent neighbours.
    """
    # Truthy checks (not `is not None`): a blank name/content contributes no retrieval signal, and
    # an all-blank action would otherwise embed as an empty string — indistinguishable from every
    # other blank step. Fall back to `render_action` (labelled, never empty) in that case.
    parts: list[str] = []
    if action.name:
        parts.append(action.name)
    if action.arguments:
        parts.append(render_json(action.arguments))
    if action.content:
        parts.append(action.content)
    return " ".join(parts) if parts else render_action(action)


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


def render_demo(step: Step, *, max_observation_chars: int | None = None) -> str:
    """Render a retrieved past step as a (state, action) -> observation few-shot example.

    `max_observation_chars`, when set, keeps only the first N characters of the observation and
    appends a "… [+N chars]" marker. Retrieval keys on (state, action), so the *format* and salient
    head of a past observation carry the signal; capping the tail bounds prompt growth when many or
    large past examples are retrieved (a big `top_k` over verbose shell/log output crowds context).
    """
    obs = step.observation
    content = obs.content
    if max_observation_chars is not None and len(content) > max_observation_chars:
        dropped = len(content) - max_observation_chars
        content = f"{content[:max_observation_chars]}… [+{dropped} chars]"
    return (
        f"{encode_state_action(step.state_before, step.action)}\n"
        f"OBSERVATION (is_error={obs.is_error}): {content}"
    )


def build_env_prompt(
    base_prompt: str,
    task: str | None,
    state: EnvState,
    action: Action,
    *,
    history: list[Step] | None = None,
    demos: list[Step] | None = None,
    max_retrieved_observation_chars: int | None = None,
) -> tuple[str, str]:
    """Assemble the (system, user) world-model completion that predicts the next observation.

    Mirrors DreamGym Eq. 4 ``M_exp(R_t | {(s_i,a_i)}, {d_j}, tau)``: the base/optimized prompt is
    the system message; the task, current state, recent history, retrieved demos, and the incoming
    action form the user message. This is the *single* assembly used by both the serving engine
    (`wmh.engine.prompts`) and the GEPA optimizer, so prompts are evolved against exactly what the
    world model serves.
    """
    system = base_prompt
    demo_block = (
        "\n\n".join(
            render_demo(d, max_observation_chars=max_retrieved_observation_chars) for d in demos
        )
        if demos
        else "(no similar past examples)"
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
    user = (
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
