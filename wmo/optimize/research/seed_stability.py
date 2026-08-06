"""GEPA seed-stability — the harness's first concrete experiment.

GEPA's search is stochastic: the reflection LM samples at temperature 1.0, and the engine seed
drives minibatch sampling and candidate selection. So two builds of the *same* corpus with the same
budget can land on different evolved prompts. This experiment quantifies that: it runs GEPA across N
seeds and measures the spread of held-out fidelity. A small std means GEPA is reproducible (any seed
is fine); a large std means the winning prompt is seed-dependent and a real build should sweep seeds
and keep the best.

Unlike a temperature sweep, which is parked because the shipped providers reject sampling params,
this needs no sampling knob: it varies only the GEPA engine `seed`, plumbed through `GEPAOptimizer`.
Fidelity is scored with whatever `Judge` the caller injects — pass `RubricJudge` to score on the
canonical 5 dimensions, exactly as `wmo eval` does.

Dependency-injected: it takes a factory for the (provider, judge, embedder) trio and the
already-split train/held-out traces, so tests can drive it with fakes and live callers can use the
same code path with real providers.
"""

from __future__ import annotations

from collections.abc import Callable

from wmo.common.core.types import Trace
from wmo.common.providers.base import Embedder, Provider
from wmo.optimize.judge import Judge
from wmo.optimize.research.ablation import Condition
from wmo.optimize.research.pipeline import optimize_prompt, score_prompt

# The provider/judge/embedder a single run uses. A factory (not a shared instance) so per-run cost
# tracking or stateful fakes can be isolated; `embedder=None` means zero-shot (no RAG).
Backends = tuple[Provider, Judge, Embedder | None]
BackendFactory = Callable[[], Backends]

# This experiment has a single condition — the whole point is the across-seed spread of one config.
BASELINE = Condition(label="baseline", params={})


class SeedStabilityAblation:
    """Run GEPA across seeds at a fixed config; metric = held-out fidelity. Std = reproducibility.

    One `Condition` (the baseline build config), swept across the seeds passed to `run_ablation`.
    The resulting `ConditionReport.std` is the headline: how much GEPA's held-out fidelity wobbles
    when only the seed changes.
    """

    name = "gepa-seed-stability"

    def __init__(
        self,
        train: list[Trace],
        held_out: list[Trace],
        base_prompt: str,
        *,
        make_backends: BackendFactory,
        budget: int,
        top_k: int = 5,
    ) -> None:
        self._train = train
        self._held_out = held_out
        self._base_prompt = base_prompt
        self._make_backends = make_backends
        self._budget = budget
        self._top_k = top_k

    def conditions(self) -> list[Condition]:
        return [BASELINE]

    def run(self, condition: Condition, seed: int) -> float:
        """Build at `seed`, then replay-score the winning prompt's held-out fidelity (0..1)."""
        provider, judge, embedder = self._make_backends()
        result = optimize_prompt(
            self._train,
            self._held_out,
            self._base_prompt,
            provider=provider,
            judge=judge,
            embedder=embedder,
            budget=self._budget,
            seed=seed,
        )
        return score_prompt(
            result.prompt,
            self._held_out,
            provider=provider,
            judge=judge,
            embedder=embedder,
            train=self._train,
            top_k=self._top_k,
        )
