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
    knowledge: str | None = None,
    reasoning: bool = False,
    grounding: bool = False,
    confidence: bool = False,
    confidence_why: bool = False,
    max_retrieved_observation_chars: int | None = None,
) -> tuple[str, str]:
    """Assemble the (system, user) world-model completion that predicts the next observation.

    Mirrors DreamGym Eq. 4 ``M_exp(R_t | {(s_i,a_i)}, {d_j}, tau)``: the base/optimized prompt is
    the system message; the task, current state, recent history, retrieved demos, and the incoming
    action form the user message. This is the *single* assembly used by both the serving engine
    (`wmh.engine.prompts`) and the GEPA optimizer, so prompts are evolved against exactly what the
    world model serves.

    `knowledge`/`reasoning`/`grounding`/`confidence` are the opt-in agentic-mode extensions; at
    their defaults the rendering is byte-identical to the pre-knowledge shape (pinned in
    render_test), so prebuilt models keep serving unchanged. `knowledge` (the cross-session
    knowledge base, rendered by `wmh.engine.knowledge`) becomes an authoritative facts section;
    `reasoning` switches the output contract to deliberate-then-answer; `grounding` offers the
    `ground_query` escape hatch (pass it only when a live grounder will actually serve it).
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
    knowledge_block = (
        "KNOWLEDGE BASE (canonical facts about this environment — authoritative over your priors;"
        f" entities, rules, schemas, and state-dependent gates):\n{knowledge}\n\n"
        if knowledge
        else ""
    )
    user = (
        f"TASK:\n{task or '(none)'}\n\n"
        f"{knowledge_block}"
        f"INTERACTION HISTORY:\n{history_block}\n\n"
        f"SIMILAR PAST EXAMPLES:\n{demo_block}\n\n"
        f"CURRENT ENV STATE:\n  structured: {render_json(state.structured)}\n"
        f"  scratchpad: {state.scratchpad or '(empty)'}\n\n"
        f"AGENT ACTION:\n{render_action(action)}\n\n"
        + output_contract(
            reasoning=reasoning,
            grounding=grounding,
            confidence=confidence,
            confidence_why=confidence_why,
        )
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

# Reasoning-mode contract pieces. `reasoning` MUST be the first key: decoding is ordered, so
# putting the deliberation before `output` is what makes it an actual deliberation rather than a
# post-hoc rationalization. `kb_note` is the cross-session counterpart of `state_note` (persisted
# to the knowledge base by the engine); `ground_query` is offered only when a grounder is active.
# Deliberation instruction, tuned on observed live failures (.agents/docs/research/
# agentic_results inspection):
# unbounded deliberations blew the token budget and truncated the output (hence "1-4 short
# sentences"); agent-side policy was mistaken for an env gate (a cancel the policy forbids still
# EXECUTES — tools are mechanical); unobserved state was assumed ("already installed");
# exploratory greps/finds were assumed to hit because the task narrative implied the target
# exists; and exact computations missed edge cases (fold/wc off-by-one on a trailing newline).
# NOTE the search guidance is deliberately evidence-NEUTRAL: a first draft said "searches miss
# often — predict empty", derived from a corpus whose empties turned out to be capture junk
# (D24), and it measurably hurt on the clean corpus. Bias neither way; follow the evidence.
_REASONING_FIELD = (
    '{"reasoning": "<1-4 short sentences BEFORE deciding the output: what would this action'
    " really do given the current state? Check the gates the ENVIRONMENT itself enforces (auth"
    " checks, availability, preconditions, timeouts) against the knowledge base, history, and"
    " examples — policy the AGENT is merely supposed to follow does not make a tool refuse, so"
    " predict refusal only where the environment demonstrably enforces it. Never assume"
    " unobserved state (installed packages, existing files) — default to a fresh environment."
    " Searches and reads (grep/find for guessed strings, line ranges that may exceed the file)"
    " can genuinely miss: decide hit vs. miss from the evidence and from how similar commands"
    " behaved in the examples, not from what the task narrative hopes is there. Work exact"
    ' computations carefully (counts, off-by-one, trailing newlines)>", '
)
_OUTPUT_IS_ERROR_FIELDS = (
    '"output": "<exactly what the environment returns to the agent>", '
    '"is_error": <true if the action failed/was invalid>'
)
_STATE_NOTE_FIELD = (
    ', "state_note": "<one short fact to remember about the new env state, or empty>"'
)
# Verbalized-confidence fields (WS-A6, D75). `confidence` sits AFTER output/is_error so ordered
# decoding conditions it on the answer actually emitted (the post-hoc p(true) framing, better
# calibrated than prospective confidence). One decimal = 11 levels: enough resolution for a
# risk-coverage sweep, coarse enough not to invite false precision. The optional `confidence_why`
# one-liner comes BEFORE the number (justify-then-rate) so the rating conditions on the
# articulated reason. Stated confidence is analysis-only: the judge and GEPA never see it
# (guarded by tests in judge_test/gepa_test).
_CONFIDENCE_WHY_FIELD = (
    ', "confidence_why": "<one short sentence: the strongest reason to trust or doubt the'
    ' "output" above>"'
)
_CONFIDENCE_FIELD = (
    ', "confidence": <your probability, 0.0-1.0 with ONE decimal, that the "output" above'
    " matches what the real environment would return for this action>"
)
_KB_NOTE_FIELD = (
    ', "kb_note": "<one canonical fact about this environment worth remembering across ALL'
    ' future sessions (an entity that exists, a rule, a schema), or empty>"'
    ', "state_update": "<the REVISED environment profile: what is running/installed/existing'
    " right now, with beliefs this step contradicted removed — the FULL replacement profile,"
    ' or empty to keep the previous one>"'
)
_GROUND_QUERY_FIELD = (
    ', "ground_query": "<a web search query IF the action references a real-world entity (API,'
    " package, flight, product) you cannot ground in the knowledge base, examples, or history —"
    ' the search runs and you answer again with the results; else empty>"'
)


def output_contract(
    *,
    reasoning: bool = False,
    grounding: bool = False,
    confidence: bool = False,
    confidence_why: bool = False,
) -> str:
    """Return the output-contract instruction for the requested mode.

    The base contract (all flags off) is exactly `OUTPUT_CONTRACT` — the shape every existing
    model was built against. All variants are parsed by the one lenient
    `wmh.core.parsing.parse_observation`. `confidence` inserts the verbalized-confidence field
    after `is_error`; `confidence_why` (a no-op without `confidence`) prepends its one-line
    justification.
    """
    if not reasoning and not grounding and not confidence:
        return OUTPUT_CONTRACT
    conf_fields = ""
    if confidence:
        conf_fields = (_CONFIDENCE_WHY_FIELD if confidence_why else "") + _CONFIDENCE_FIELD
    fields = f"{_OUTPUT_IS_ERROR_FIELDS}{conf_fields}{_STATE_NOTE_FIELD}"
    ground_field = _GROUND_QUERY_FIELD if grounding else ""
    if reasoning:
        return (
            "First deliberate, then answer. Respond with ONLY a JSON object whose FIRST key is"
            f" your deliberation:\n{_REASONING_FIELD}{fields}{_KB_NOTE_FIELD}{ground_field}}}"
        )
    # No deliberation pass: the base fields + whichever optional fields are on.
    return (
        "Respond with ONLY a JSON object describing the environment's response to this action:\n"
        f"{{{fields}{ground_field}}}"
    )
