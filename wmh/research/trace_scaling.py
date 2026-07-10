"""Trace scaling law — how world-model fidelity grows with the number of training traces.

The question: feed the harness *more* recorded traces and does its reconstruction fidelity keep
climbing, or saturate? This experiment sweeps the TRAIN trace count (10, 20, 50, … up to the corpus
size) against a **fixed** held-out test set and reports fidelity at each count, so the curve is a
clean "data in → fidelity out" law rather than a moving target (see `scaling_split`).

Two curves, selectable so the cheap one can scout the range before the expensive one runs:

- `base` — the bundled `BASE_ENV_PROMPT`, scored on the fixed test set with a retrieval buffer built
  from the `n` train traces. Isolates how far retrieval *alone* scales (no GEPA, cheap).
- `gepa` — GEPA optimizes the prompt on the `n` train traces, selecting against the fixed valid set,
  then the winning prompt is scored on the fixed test set. The real "learning from more traces"
  curve (one GEPA run per count × seed — expensive).

It is an `Ablation` like every other experiment: conditions = mode × trace-count, swept across seeds
by `run_ablation`, so the across-seed std gives error bars at each point for free. Each `run` reuses
the canonical `optimize_prompt` / `score_prompt` (the same GEPA + replay the harness ships), so the
metric is comparable to `wmh eval` and any judge upgrade lands here automatically. Backends are
factory-injected, so the unit test drives it with fakes and the `scripts/` runner with live Bedrock.

Extensible across benchmarks: it takes an already-ingested corpus and a base prompt, so tau2 today
and terminal-tasks / swe-bench tomorrow are just a different trace file in — no code change.
"""

from __future__ import annotations

from collections.abc import Sequence

from wmh.core.types import JsonValue, Trace
from wmh.optimize.judge import RubricDimension
from wmh.research.ablation import Condition, as_int
from wmh.research.pipeline import optimize_prompt, score_prompt
from wmh.research.scaling_split import CorpusSplit, partition_corpus, subsample_train
from wmh.research.seed_stability import BackendFactory
from wmh.retrieval import RetrievalKey

# The two prompt sources the sweep compares. `base` = the shipped prompt + RAG only (cheap);
# `gepa` = GEPA-optimized on the train sample (expensive). Strings (not an enum) so they read
# straight from a CLI flag and serialize into the report's `Condition.params` unchanged.
Mode = str
BASE: Mode = "base"
GEPA: Mode = "gepa"
MODES: tuple[Mode, ...] = (BASE, GEPA)


def _as_mode(value: JsonValue) -> Mode:
    assert isinstance(value, str)
    return value


class TraceScalingAblation:
    """Sweep train-trace count × mode against a fixed test set; metric = test fidelity (0..1).

    Conditions are the cartesian product of `modes` and `counts` (e.g. base@10, base@20, gepa@10…).
    `run(condition, seed)` subsamples `n` train traces (nested as `n` grows, seeded), then:
      - `base`: scores `base_prompt` on the fixed test set with a RAG buffer over the `n` traces;
      - `gepa`: optimizes on the `n` traces (selecting on fixed valid) and scores the winner.
    Across-seed std (from `run_ablation`) is the error bar at each point.
    """

    name = "trace-scaling-law"

    def __init__(
        self,
        corpus: list[Trace],
        base_prompt: str,
        *,
        make_backends: BackendFactory,
        counts: Sequence[int],
        modes: Sequence[Mode] = MODES,
        budget: int,
        top_k: int = 5,
        test_frac: float = 0.2,
        valid_frac: float = 0.15,
        sample_turns: str = "all",
        test_cap: int | None = None,
        concurrency: int = 1,
        max_retrieved_observation_chars: int | None = None,
        retrieval_key: RetrievalKey = "state_action",
        score_dimension: RubricDimension | None = None,
    ) -> None:
        self._base_prompt = base_prompt
        self._make_backends = make_backends
        self._budget = budget
        self._top_k = top_k
        self._sample_turns = sample_turns
        self._concurrency = concurrency
        self._max_retrieved_observation_chars = max_retrieved_observation_chars
        self._retrieval_key = retrieval_key
        self._score_dimension = score_dimension
        self._modes = list(modes)
        # GEPA optimizes under DEFAULT retrieval (optimize_prompt does not yet thread these knobs),
        # so scoring the evolved prompt under a non-default key/cap would measure it on a retrieval
        # distribution it never optimized for — a confounded gepa-vs-base comparison. Fail fast
        # rather than silently bias the curve; the base arm honours these knobs.
        if GEPA in self._modes and (
            retrieval_key != "state_action" or max_retrieved_observation_chars is not None
        ):
            raise ValueError(
                "gepa mode does not honour retrieval_key / max_retrieved_observation_chars yet; "
                "combining them would score the evolved prompt under retrieval it never optimized "
                "for. Use gepa with default retrieval, or restrict these knobs to base mode."
            )
        self._split: CorpusSplit = partition_corpus(
            corpus, test_frac=test_frac, valid_frac=valid_frac
        )
        # Optionally score against a fixed-size subsample of the (large) test set: replay scores the
        # test serially, so a 200-trace test is ~1000 judge calls per point. A seeded cap keeps the
        # y-axis stable and reproducible across every point at a fraction of the cost. The subsample
        # is fixed (seed 0) so every count/mode/seed is scored on the SAME test traces.
        test = self._split.test
        self._test = subsample_train(test, test_cap, seed=0) if test_cap else test
        # Cap counts at the train pool and drop duplicates created by the cap, preserving order, so
        # a 1000-point sweep on a small corpus collapses cleanly to the few counts it can serve.
        pool = len(self._split.train_pool)
        self._counts = list(dict.fromkeys(min(c, pool) for c in counts if c > 0))

    @property
    def split(self) -> CorpusSplit:
        """The fixed test/valid sets + train pool this sweep runs against (for the header)."""
        return self._split

    @property
    def counts(self) -> list[int]:
        """The effective train counts (capped at the pool, deduped)."""
        return self._counts

    @property
    def scored_test(self) -> list[Trace]:
        """The test traces scored each point (the `test_cap` subsample, or the full test)."""
        return self._test

    def conditions(self) -> list[Condition]:
        return [
            Condition(label=f"{mode}@{n}", params={"mode": mode, "n_train": n})
            for mode in self._modes
            for n in self._counts
        ]

    def run(self, condition: Condition, seed: int) -> float:
        """Score one (mode, n_train) point at `seed` on the fixed test set; fidelity 0..1."""
        mode = _as_mode(condition.params["mode"])
        n_train = as_int(condition.params["n_train"])
        train = subsample_train(self._split.train_pool, n_train, seed=seed)
        provider, judge, embedder = self._make_backends()

        if mode == GEPA:
            result = optimize_prompt(
                train,
                self._split.valid,
                self._base_prompt,
                provider=provider,
                judge=judge,
                embedder=embedder,
                budget=self._budget,
                seed=seed,
            )
            prompt = result.prompt
        else:
            prompt = self._base_prompt

        return score_prompt(
            prompt,
            self._test,
            provider=provider,
            judge=judge,
            embedder=embedder,
            train=train,
            top_k=self._top_k,
            sample_turns=self._sample_turns,
            seed=seed,
            concurrency=self._concurrency,
            max_retrieved_observation_chars=self._max_retrieved_observation_chars,
            retrieval_key=self._retrieval_key,
            score_dimension=self._score_dimension,
        )
