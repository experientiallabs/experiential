"""Closed-loop scenario verification: back-agreement and solvability.

A synthesized scenario is only trustworthy once two checks pass (the Trajectory2Task filter):

* **Back-agreement** — grade the scenario's own source trajectory against the generated checklist.
  When the corpus records the episode's true outcome, the judge's verdict must match it; a rubric
  that misgrades the very episode it was distilled from can't be trusted on new trajectories.
* **Solvability** — roll a baseline agent against the world model on the synthesized task and
  grade the result. A scenario nothing can complete is usually under-specified, not hard.

Verification reports; the caller decides whether to drop failures (`wmh scenarios verify --drop`).
"""

from __future__ import annotations

from pydantic import BaseModel

from wmh.core.types import Trace
from wmh.engine.world_model import WorldModel
from wmh.env.base import WorldModelEnv
from wmh.env.episode import Agent, run_episode
from wmh.scenarios.mining.facets import Outcome
from wmh.scenarios.synthesis import EvalScenario, ScenarioSet
from wmh.scenarios.verification.judge import ChecklistJudge, ChecklistResult


class ScenarioVerdict(BaseModel):
    """The verification outcome for one scenario."""

    scenario_id: str
    # Back-agreement: did the checklist judge, grading the SOURCE trajectory, reach the recorded
    # outcome? None when the corpus never recorded one (nothing to agree with).
    back_agreement: bool | None = None
    judge_success_on_source: bool = False
    # Solvability: did a baseline agent's world-model rollout pass the checklist?
    solvable: bool = False
    rollout_pass_rate: float = 0.0
    critique: str = ""

    @property
    def ok(self) -> bool:
        """Verified: solvable and (when checkable) back-agreeing."""
        return self.solvable and self.back_agreement is not False


class VerificationReport(BaseModel):
    """Aggregate verification result for a scenario set."""

    verdicts: list[ScenarioVerdict]

    @property
    def back_agreement_rate(self) -> float:
        checkable = [v for v in self.verdicts if v.back_agreement is not None]
        if not checkable:
            return 0.0
        return sum(v.back_agreement is True for v in checkable) / len(checkable)

    @property
    def solvable_rate(self) -> float:
        if not self.verdicts:
            return 0.0
        return sum(v.solvable for v in self.verdicts) / len(self.verdicts)


def verify_scenarios(
    scenario_set: ScenarioSet,
    traces: list[Trace],
    world_model: WorldModel,
    agent: Agent,
    judge: ChecklistJudge,
    *,
    max_steps: int = 12,
) -> VerificationReport:
    """Run back-agreement + solvability for every scenario in the set.

    `traces` is the source corpus (provenance lookups for back-agreement); scenarios whose source
    trace is absent skip the back-agreement half. Solvability rolls `agent` against `world_model`
    seeded with the scenario's task and initial state. The world model is frozen for the whole
    run: rollout steps are never indexed, so verdicts don't depend on verification order.
    """
    by_id = {trace.trace_id: trace for trace in traces}
    with world_model.frozen():
        verdicts = [
            _verify_one(scenario, by_id, world_model, agent, judge, max_steps=max_steps)
            for scenario in scenario_set.scenarios
        ]
    return VerificationReport(verdicts=verdicts)


def _verify_one(
    scenario: EvalScenario,
    traces_by_id: dict[str, Trace],
    world_model: WorldModel,
    agent: Agent,
    judge: ChecklistJudge,
    *,
    max_steps: int,
) -> ScenarioVerdict:
    # Back-agreement against the source trajectory, when we have it and a recorded outcome.
    back_agreement: bool | None = None
    judge_success_on_source = False
    source = next((traces_by_id[tid] for tid in scenario.provenance if tid in traces_by_id), None)
    if source is not None and scenario.checklist:
        source_result = judge.score(scenario.task, scenario.checklist, source.steps)
        judge_success_on_source = source_result.success
        recorded = _recorded_outcome(source, scenario.source_outcome)
        if recorded is not Outcome.UNKNOWN:
            back_agreement = source_result.success == (recorded is Outcome.SUCCESS)

    # Solvability: one baseline-agent rollout in the world model, graded by the same judge.
    env = WorldModelEnv(world_model)
    episode = run_episode(
        env, agent, scenario.task, seed_state=scenario.seed_state, max_steps=max_steps
    )
    rollout = (
        judge.score(scenario.task, scenario.checklist, episode.steps)
        if scenario.checklist
        else ChecklistResult(critique="no checklist synthesized; nothing to grade")
    )
    return ScenarioVerdict(
        scenario_id=scenario.scenario_id,
        back_agreement=back_agreement,
        judge_success_on_source=judge_success_on_source,
        solvable=rollout.success,
        rollout_pass_rate=rollout.pass_rate,
        critique=rollout.critique,
    )


def _recorded_outcome(trace: Trace, facet_outcome: Outcome) -> Outcome:
    """The trace's ground-truth outcome: recorded reward when present, else the facet's guess."""
    reward = trace.metadata.get("reward")
    if isinstance(reward, int | float):
        return Outcome.SUCCESS if float(reward) >= 1.0 else Outcome.FAILURE
    return facet_outcome
