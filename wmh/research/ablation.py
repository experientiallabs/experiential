"""The ablation framework: sweep named conditions across seeds, aggregate mean + std.

An *ablation* answers "does knob X change the outcome?" empirically and reproducibly. The contract:

- A `Condition` is a named bundle of knob values (e.g. `train_temp=0.0, eval_temp=1.0`).
- An `Ablation` (Protocol) knows its `conditions()` and how to `run(condition, seed) -> float`
  (one build+eval at one seed, returning a scalar metric — higher is better).
- `run_ablation` is the generic driver: for every condition × seed it calls `run`, collects the
  scalars, and aggregates per-condition mean + (population) std across seeds.

Aggregation is intentionally simple (mean + std, no CIs/significance tests): small trace corpora
make fancier statistics false precision. Adding a new experiment is "write one `Ablation`"; the
driver, seed sweep, and reporting are reused.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Protocol, runtime_checkable

from pydantic import BaseModel, Field

from wmh.core.types import JsonValue

# Per-run progress hook: (condition, seed, score) -> None, called after each individual run.
RunCallback = Callable[["Condition", int, float], None]


class Condition(BaseModel):
    """A named point in the ablation grid: a label plus the knob values it sets.

    `params` is kept as plain JSON (not typed per-experiment) so the framework stays experiment-
    agnostic and the report serializes cleanly; each `Ablation` reads the keys it owns.
    """

    label: str
    params: dict[str, JsonValue] = Field(default_factory=dict)


def as_int(value: JsonValue) -> int:
    """Narrow a `Condition.params` value the ablation wrote as an int back to `int`.

    The keys an ablation owns are written as ints by its own `conditions()`, so this narrowing
    only ever sees ints - but assert to fail loudly if a malformed condition slips in.
    """
    assert isinstance(value, int)
    return value


class SeedScore(BaseModel):
    """The scalar metric a single (condition, seed) run produced."""

    seed: int
    score: float


class ConditionReport(BaseModel):
    """Per-condition outcome: every seed's score plus the across-seed mean and std."""

    condition: Condition
    per_seed: list[SeedScore] = Field(default_factory=list)
    mean: float = 0.0
    std: float = 0.0

    def summary(self) -> str:
        n = len(self.per_seed)
        return f"{self.condition.label:24} mean={self.mean:.3f} std={self.std:.3f} n={n}"


class AblationReport(BaseModel):
    """The full experiment: the canonical artifact a research run records."""

    name: str
    seeds: list[int] = Field(default_factory=list)
    conditions: list[ConditionReport] = Field(default_factory=list)

    def best(self) -> ConditionReport | None:
        """The condition with the highest across-seed mean (None if the report is empty)."""
        return max(self.conditions, key=lambda c: c.mean, default=None)


@runtime_checkable
class Ablation(Protocol):
    """An experiment: a set of conditions and a way to run one of them at one seed."""

    @property
    def name(self) -> str: ...

    def conditions(self) -> Sequence[Condition]: ...

    def run(self, condition: Condition, seed: int) -> float:
        """Run `condition` at `seed` and return a scalar metric (higher is better)."""
        ...


def _mean_std(values: Sequence[float]) -> tuple[float, float]:
    """Mean and population std of `values` (std=0 for a single sample); ()-safe -> (0, 0)."""
    if not values:
        return 0.0, 0.0
    n = len(values)
    mean = sum(values) / n
    var = sum((v - mean) ** 2 for v in values) / n
    return mean, var**0.5


def aggregate(condition: Condition, per_seed: list[SeedScore]) -> ConditionReport:
    """Build a `ConditionReport` from a condition's per-seed scores (mean + std across seeds)."""
    mean, std = _mean_std([s.score for s in per_seed])
    return ConditionReport(condition=condition, per_seed=per_seed, mean=mean, std=std)


def run_ablation(
    ablation: Ablation,
    seeds: Sequence[int],
    *,
    on_run: RunCallback | None = None,
) -> AblationReport:
    """Sweep every condition × seed of `ablation` and aggregate per-condition mean + std.

    `on_run(condition, seed, score)` is called after each individual run for live progress or
    logging. Conditions run in their declared order, seeds in the given order, so a run is fully
    reproducible given the same `ablation` and `seeds`.
    """
    reports: list[ConditionReport] = []
    for condition in ablation.conditions():
        per_seed: list[SeedScore] = []
        for seed in seeds:
            score = ablation.run(condition, seed)
            per_seed.append(SeedScore(seed=seed, score=score))
            if on_run is not None:
                on_run(condition, seed, score)
        reports.append(aggregate(condition, per_seed))
    return AblationReport(name=ablation.name, seeds=list(seeds), conditions=reports)
