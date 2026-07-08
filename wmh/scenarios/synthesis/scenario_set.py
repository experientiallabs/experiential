"""The scenario data model: `EvalScenario` records and the versioned `ScenarioSet` artifact."""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, Field

from wmh.core.render import render_agent_messages
from wmh.core.types import EnvState, HarnessContext
from wmh.env.scenarios import Scenario
from wmh.scenarios.mining.clustering import TraceCluster
from wmh.scenarios.mining.facets import Outcome


class EvalScenario(BaseModel):
    """One reusable eval scenario: distilled from a real trace, or created from a task."""

    scenario_id: str
    task: str  # self-contained task statement handed to the agent
    seed_state: EnvState = Field(default_factory=EnvState)  # initial env state for the world model
    checklist: list[str] = Field(default_factory=list)  # judgeable success criteria
    provenance: list[str] = Field(default_factory=list)  # source trace_ids
    cluster_name: str = ""
    weight: float = 0.0  # fraction of the corpus this scenario represents
    source_outcome: Outcome = Outcome.UNKNOWN
    failure_category: str | None = None
    harness: HarnessContext | None = None  # agent-side system prompt + tools, when captured

    def to_scenario(self) -> Scenario:
        """The minimal `Scenario` view consumed by existing rollout code."""
        return Scenario(task=self.task, provenance=list(self.provenance))

    def render_messages(self) -> tuple[str, str]:
        """The token-realistic (system, user) messages a fresh agent would receive.

        Requires the scenario to carry a harness context; scenarios distilled from corpora whose
        traces recorded no `gen_ai.system_instructions` have none.
        """
        if self.harness is None:
            raise ValueError(
                f"scenario {self.scenario_id} has no harness context; build it from a corpus "
                "whose traces record gen_ai.system_instructions / gen_ai.tool.definitions"
            )
        return render_agent_messages(self.harness, self.task)


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
