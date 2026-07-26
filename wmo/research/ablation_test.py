"""Tests for the ablation framework (condition sweep, seed aggregation, reporting)."""

from __future__ import annotations

from collections.abc import Sequence

from wmo.research.ablation import (
    Ablation,
    Condition,
    SeedScore,
    aggregate,
    run_ablation,
)


class _FakeAblation:
    """A deterministic ablation: score = base (per condition) + seed*0.01, to test aggregation."""

    name = "fake"

    def __init__(self, bases: dict[str, float]) -> None:
        self._bases = bases
        self.calls: list[tuple[str, int]] = []

    def conditions(self) -> list[Condition]:
        return [Condition(label=k, params={"base": v}) for k, v in self._bases.items()]

    def run(self, condition: Condition, seed: int) -> float:
        self.calls.append((condition.label, seed))
        return self._bases[condition.label] + seed * 0.01


def test_fake_ablation_satisfies_protocol() -> None:
    assert isinstance(_FakeAblation({"a": 0.5}), Ablation)


def test_aggregate_computes_mean_and_population_std() -> None:
    report = aggregate(
        Condition(label="c"),
        [SeedScore(seed=0, score=0.2), SeedScore(seed=1, score=0.4)],
    )
    assert abs(report.mean - 0.3) < 1e-9
    # population std of {0.2, 0.4} is 0.1
    assert abs(report.std - 0.1) < 1e-9
    assert len(report.per_seed) == 2


def test_aggregate_single_seed_has_zero_std() -> None:
    report = aggregate(Condition(label="c"), [SeedScore(seed=7, score=0.42)])
    assert report.mean == 0.42
    assert report.std == 0.0


def test_run_ablation_sweeps_every_condition_and_seed() -> None:
    ablation = _FakeAblation({"lo": 0.1, "hi": 0.8})
    seeds = [0, 1, 2]
    progress: list[tuple[str, int, float]] = []

    report = run_ablation(
        ablation, seeds, on_run=lambda c, s, score: progress.append((c.label, s, score))
    )

    # Every condition × seed ran, in declared/given order.
    assert ablation.calls == [("lo", 0), ("lo", 1), ("lo", 2), ("hi", 0), ("hi", 1), ("hi", 2)]
    assert len(progress) == 6
    assert report.name == "fake"
    assert report.seeds == [0, 1, 2]
    assert {c.condition.label for c in report.conditions} == {"lo", "hi"}

    hi = next(c for c in report.conditions if c.condition.label == "hi")
    # hi scores: 0.80, 0.81, 0.82 -> mean 0.81
    assert abs(hi.mean - 0.81) < 1e-9
    best = report.best()
    assert best is not None and best.condition.label == "hi"


def test_best_is_none_for_empty_report() -> None:
    class _Empty:
        name = "empty"

        def conditions(self) -> Sequence[Condition]:
            return []

        def run(self, condition: Condition, seed: int) -> float:  # pragma: no cover - never called
            return 0.0

    report = run_ablation(_Empty(), [0])
    assert report.conditions == []
    assert report.best() is None
