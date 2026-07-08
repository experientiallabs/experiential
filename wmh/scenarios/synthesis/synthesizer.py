"""Scenario synthesis: turn a selected trace into a self-contained, judgeable eval scenario.

The WildBench pattern: an LLM reads the source trace and writes (1) a self-contained task
statement (the user's goal plus constraints revealed mid-episode), (2) the minimal initial
environment state the episode needs (seeds the world model's scratchpad), and (3) a short
checklist of success criteria a judge can grade a new trajectory against. Every scenario keeps
provenance to its source trace so it stays auditable.
"""

from __future__ import annotations

from pydantic import BaseModel, Field, ValidationError

from wmh.core.parsing import extract_json_object
from wmh.core.types import EnvState, Trace
from wmh.providers.base import Message, Provider
from wmh.scenarios.mining.facets import TraceFacet, trace_digest
from wmh.scenarios.synthesis.scenario_set import EvalScenario

SYNTHESIS_SYSTEM = """You convert one recorded AI-agent episode into a reusable evaluation
scenario. You see a digest of the episode (task, tool calls, observations).

Respond with ONLY a JSON object, no prose around it:
{"task": "<self-contained task statement for a fresh agent: the user's goal plus any constraints
revealed during the episode; no references to 'the trace' or 'above'>",
 "initial_state": "<2-6 sentences of environment facts the episode started from (accounts,
records, files, balances) that a simulator needs to answer the agent consistently>",
 "checklist": ["<3-6 concrete, independently checkable success criteria for a NEW attempt>"]}

Rules:
- The task must be attemptable without seeing the original episode.
- initial_state states facts about the world, not about the agent's behavior.
- Checklist items grade the OUTCOME (what ended up true / communicated), not the exact tool
  sequence — a different valid strategy must be able to pass."""


class _RawSynthesis(BaseModel):
    task: str
    initial_state: str = ""
    checklist: list[str] = Field(default_factory=list)


class ScenarioSynthesizer:
    """LLM synthesis of one `EvalScenario` per selected trace."""

    def __init__(self, provider: Provider) -> None:
        self._provider = provider

    def synthesize(self, trace: Trace, facet: TraceFacet) -> EvalScenario:
        """Synthesize the scenario for one selected trace.

        On an unparseable reply, falls back to the facet's task summary with an empty checklist —
        the scenario stays usable for rollouts, and verification will flag it (no checklist means
        nothing to grade against).
        """
        completion = self._provider.complete(
            SYNTHESIS_SYSTEM,
            [Message(role="user", content=trace_digest(trace))],
            temperature=0.0,
            max_tokens=1024,
        )
        raw = extract_json_object(completion.text)
        parsed: _RawSynthesis | None = None
        if raw is not None:
            try:
                parsed = _RawSynthesis.model_validate_json(raw)
            except ValidationError:
                parsed = None
        if parsed is not None and parsed.task.strip():
            task = parsed.task.strip()
            seed_state = EnvState(scratchpad=parsed.initial_state.strip())
            checklist = [item.strip() for item in parsed.checklist if item.strip()]
        else:
            task = facet.task_summary
            seed_state = EnvState()
            checklist = []
        return EvalScenario(
            scenario_id=f"scenario-{trace.trace_id}",
            task=task,
            seed_state=seed_state,
            checklist=checklist,
            provenance=[trace.trace_id],
            source_outcome=facet.outcome,
            failure_category=facet.failure_category,
        )
