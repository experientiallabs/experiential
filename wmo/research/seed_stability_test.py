"""Tests for the GEPA seed-stability ablation (drives the framework with fakes, no network)."""

from __future__ import annotations

from wmo.core.types import Action, ActionKind, EnvState, Observation, Step, Trace
from wmo.optimize.judge import JudgeResult
from wmo.providers.base import Completion, Message, ProviderConfig, ProviderKind
from wmo.research.ablation import Ablation, run_ablation
from wmo.research.seed_stability import SeedStabilityAblation


class FakeProvider:
    def __init__(self) -> None:
        self.config = ProviderConfig(kind=ProviderKind.ANTHROPIC, model="m")

    def complete(
        self,
        system: str,
        messages: list[Message],
        *,
        temperature: float = 0.7,
        max_tokens: int = 8192,
    ) -> Completion:
        if "improve the system prompt" in system:
            return Completion(text="IMPROVED")
        return Completion(text="predicted")

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [[float(len(t))] for t in texts]

    def verify(self):  # noqa: ANN201
        raise NotImplementedError


class FakeJudge:
    def score(self, predicted: Observation, actual: Observation, context: Step) -> JudgeResult:
        return JudgeResult(score=0.5, critique="c")


def _trace(tid: str, n: int = 2) -> Trace:
    return Trace(
        trace_id=tid,
        steps=[
            Step(
                action=Action(kind=ActionKind.TOOL_CALL, name="f", arguments={"i": i}),
                observation=Observation(content=f"real-{i}"),
                state_before=EnvState(structured={"loc": "shop"}),
                task="t",
            )
            for i in range(n)
        ],
    )


def _ablation() -> SeedStabilityAblation:
    return SeedStabilityAblation(
        [_trace("tr1"), _trace("tr2")],
        [_trace("te1")],
        "BASE",
        make_backends=lambda: (FakeProvider(), FakeJudge(), None),
        budget=6,
    )


def test_satisfies_protocol() -> None:
    assert isinstance(_ablation(), Ablation)


def test_single_baseline_condition() -> None:
    conds = _ablation().conditions()
    assert len(conds) == 1
    assert conds[0].label == "baseline"


def test_run_one_seed_returns_holdout_fidelity() -> None:
    ablation = _ablation()
    # The fake judge returns 0.5 for every held-out step, so mean fidelity is 0.5.
    assert abs(ablation.run(ablation.conditions()[0], seed=0) - 0.5) < 1e-9


def test_run_ablation_aggregates_across_seeds() -> None:
    report = run_ablation(_ablation(), [0, 1, 2])
    assert report.name == "gepa-seed-stability"
    assert len(report.conditions) == 1
    cell = report.conditions[0]
    assert len(cell.per_seed) == 3
    # Deterministic fakes -> identical fidelity across seeds -> zero std (perfectly stable).
    assert abs(cell.mean - 0.5) < 1e-9
    assert cell.std == 0.0
