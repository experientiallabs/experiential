"""Scenario synthesis: turn a selected trace into a self-contained, judgeable eval scenario.

The WildBench pattern: an LLM reads the source trace and writes (1) a self-contained task
statement (the user's goal plus constraints revealed mid-episode), (2) the minimal initial
environment state the episode needs (seeds the world model's scratchpad), and (3) a short
checklist of success criteria a judge can grade a new trajectory against. Every scenario keeps
provenance to its source trace so it stays auditable.
"""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, Field, ValidationError

from wmh.core.parsing import extract_json_object
from wmh.core.types import EnvState, Trace
from wmh.env.scenarios import Scenario
from wmh.providers.base import Message, Provider
from wmh.scenarios.clustering import TraceCluster
from wmh.scenarios.facets import Outcome, TraceFacet, trace_digest


class EvalScenario(BaseModel):
    """One reusable eval scenario distilled from a real trace."""

    scenario_id: str
    task: str  # self-contained task statement handed to the agent
    seed_state: EnvState = Field(default_factory=EnvState)  # initial env state for the world model
    checklist: list[str] = Field(default_factory=list)  # judgeable success criteria
    provenance: list[str] = Field(default_factory=list)  # source trace_ids
    cluster_name: str = ""
    weight: float = 0.0  # fraction of the corpus this scenario represents
    source_outcome: Outcome = Outcome.UNKNOWN
    failure_category: str | None = None

    def to_scenario(self) -> Scenario:
        """The minimal `Scenario` view consumed by existing rollout code."""
        return Scenario(task=self.task, provenance=list(self.provenance))


class ScenarioSet(BaseModel):
    """The constructed scenario set plus the corpus statistics that justify it."""

    scenarios: list[EvalScenario]
    clusters: list[TraceCluster] = Field(default_factory=list)
    corpus_traces: int = 0
    corpus_coverage: float = 0.0  # fraction of corpus facets within tau of a selected facet
    coverage_tau: float = 0.0

    def retain(self, scenario_ids: set[str]) -> None:
        """Keep only `scenario_ids`, renormalizing weights and invalidating coverage.

        Dropping scenarios (e.g. `wmh scenarios verify --drop`) breaks two invariants the artifact
        promises: weights sum to 1 over the set, and `corpus_coverage` describes the current
        scenarios. Weights are renormalized over the survivors; coverage needs the facet
        embeddings (gone by verify time), so it is zeroed rather than left stale.
        """
        self.scenarios = [s for s in self.scenarios if s.scenario_id in scenario_ids]
        total_weight = sum(s.weight for s in self.scenarios)
        if total_weight > 0:
            for scenario in self.scenarios:
                scenario.weight /= total_weight
        self.corpus_coverage = 0.0
        self.coverage_tau = 0.0

    def save(self, path: str | Path) -> None:
        Path(path).write_text(self.model_dump_json(indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> ScenarioSet:
        return cls.model_validate_json(Path(path).read_text(encoding="utf-8"))


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
