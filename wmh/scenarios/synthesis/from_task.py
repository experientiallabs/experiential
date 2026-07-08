"""Scenario synthesis from a plain task description (no source trace).

`wmh scenarios build` distills scenarios out of mined traces; this module is the complementary
entry point: a user types a task ("build a python airbnb clone") and gets a scenario that is
token-realistic for a specific world model. Realism comes from three commitments:

  - the user message is the task exactly as typed — never paraphrased by an LLM;
  - the scenario carries the harness the corpus' agent actually ran under (system prompt + tool
    definitions, captured from traces), so `render_messages()` reproduces what a fresh agent
    would receive;
  - the structured seed state is the modal state observed in real corpus steps, so the session
    starts where recorded episodes started (e.g. `{"cwd": "/workspace", "harness": "pi"}`).

Only the free-text parts an LLM is good at — plausible initial environment facts and a judgeable
checklist — are synthesized, grounded in the harness and sampled corpus steps.
"""

from __future__ import annotations

import hashlib

from pydantic import BaseModel, Field, ValidationError

from wmh.core.parsing import extract_json_object
from wmh.core.render import render_demo, render_harness
from wmh.core.types import EnvState, HarnessContext, JsonObject, Step
from wmh.providers.base import Message, Provider
from wmh.scenarios.synthesis.scenario_set import EvalScenario

FROM_TASK_SYSTEM = """You prepare an evaluation scenario for an AI agent from a task description.
You see the agent's harness (its system prompt and tools) and example steps recorded from real
episodes in the same environment.

Respond with ONLY a JSON object, no prose around it:
{"initial_state": "<0-4 sentences of environment facts the episode starts from (existing files,
records, services), consistent with the example steps; empty string when a fresh environment
needs none>",
 "checklist": ["<3-6 concrete, independently checkable success criteria for an attempt>"]}

Rules:
- initial_state states facts about the world, not about the agent's behavior or the task.
- Checklist items grade the OUTCOME (what ended up true / produced), not the exact tool
  sequence — a different valid strategy must be able to pass.
- Stay inside what this environment supports as evidenced by the harness and examples."""


class _RawFromTask(BaseModel):
    initial_state: str = ""
    checklist: list[str] = Field(default_factory=list)


def scenario_from_task(
    task: str,
    harness: HarnessContext,
    provider: Provider,
    *,
    examples: list[Step],
) -> EvalScenario:
    """Synthesize one token-realistic `EvalScenario` for `task` under `harness`.

    `examples` are real steps from the target world model's corpus (e.g. `wm.sample_steps(8)`);
    they ground the LLM's seed facts/checklist and supply the modal structured seed state. On an
    unparseable LLM reply the scenario is still returned with an empty scratchpad and checklist —
    usable for rollouts, and verification will flag that there is nothing to grade against.
    """
    completion = provider.complete(
        FROM_TASK_SYSTEM,
        [Message(role="user", content=_synthesis_prompt(task, harness, examples))],
        temperature=0.0,
        max_tokens=1024,
    )
    raw = extract_json_object(completion.text)
    parsed: _RawFromTask | None = None
    if raw is not None:
        try:
            parsed = _RawFromTask.model_validate_json(raw)
        except ValidationError:
            parsed = None
    scratchpad = parsed.initial_state.strip() if parsed else ""
    checklist = [item.strip() for item in parsed.checklist if item.strip()] if parsed else []
    return EvalScenario(
        scenario_id=_scenario_id(task),
        task=task,
        seed_state=EnvState(structured=_modal_structured(examples), scratchpad=scratchpad),
        checklist=checklist,
        weight=1.0,  # a from-task scenario is its whole (one-scenario) set
        harness=harness,
    )


def _scenario_id(task: str) -> str:
    """Deterministic id from the task text, so re-creating the same task overwrites cleanly."""
    digest = hashlib.blake2b(task.encode("utf-8"), digest_size=6).hexdigest()
    return f"scenario-{digest}"


def _synthesis_prompt(task: str, harness: HarnessContext, examples: list[Step]) -> str:
    example_block = (
        "\n\n".join(render_demo(step) for step in examples) if examples else "(none available)"
    )
    return (
        f"AGENT HARNESS:\n{render_harness(harness) or '(not recorded)'}\n\n"
        f"EXAMPLE STEPS FROM REAL EPISODES:\n{example_block}\n\n"
        f"TASK THE USER WILL GIVE THE AGENT:\n{task}"
    )


def _modal_structured(examples: list[Step]) -> JsonObject:
    """The most common `state_before.structured` across `examples` (empty when none carry state).

    Real corpora stamp a constant structured state (cwd, harness name) on every step; the mode
    reproduces it without letting the LLM invent machine-readable config.
    """
    counts: dict[str, tuple[int, JsonObject]] = {}
    for step in examples:
        structured = step.state_before.structured
        if not structured:
            continue
        key = EnvState(structured=structured).model_dump_json()
        seen, value = counts.get(key, (0, structured))
        counts[key] = (seen + 1, value)
    if not counts:
        return {}
    winner = max(counts.values(), key=lambda pair: pair[0])[1]
    return dict(winner)
