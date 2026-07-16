"""GEPA scaling law - how world-model fidelity scales along GEPA's two knobs.

The trace scaling law (`trace_scaling`) swept how much *data* the world model retrieves over; this
experiment sweeps how much *optimization* it gets: GEPA iterations (`budget`) and the number of
training traces GEPA learns from, against the same fixed held-out test set. Each grid point is a
full GEPA run (`optimize_prompt`) followed by the canonical replay scoring (`score_prompt`), so the
metric is identical to `wmh eval` and directly comparable to the trace scaling law's numbers.

Design choices specific to the GEPA axis:

- **`budget=0` is the anchor**: no optimization, the shipped base prompt + RAG over the same train
  sample - at a given trace count this reproduces the trace scaling law's RAG-only point, a built-in
  consistency check between the two experiments.
- **The GEPA selection valset is capped by steps** (`gepa_val_steps`). GEPA evaluates a promising
  candidate on the full valset, so its size multiplies the per-iteration cost; a ~30-step fixed
  subset (seed-0 shuffle of the valid band, same for every condition and seed) keeps iterations
  affordable while selection pressure stays constant across the grid.
- **Optional hard-step concentration** (`hard_threshold`): most steps on near-saturated benchmarks
  score perfectly, so a random reflection minibatch often contains no failure to learn from ("all
  subsample scores perfect - skipping", a wasted iteration). When a threshold is set, the run first
  replays a probe of the train sample plus the GEPA valset under the *base* prompt, marks the steps
  scoring below the threshold, and hands GEPA a `hard_step_filter` (with `select_on_hard`) so both
  reflection and candidate selection concentrate on the steps with headroom.

It is an `Ablation` like every other experiment: conditions = the (n_train, budget) grid, swept
across seeds by `run_ablation`. Backends are factory-injected, so the unit test drives it with
fakes and any live caller supplies real providers (the published sweeps used a workspace runner;
`.agents/` contents are disposable - the reproduce commands in `docs/research/gepa_scaling_law.md`
are the record).
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

from wmh.core.types import Step, Trace
from wmh.engine.replay import replay
from wmh.optimize.judge import Judge
from wmh.providers.base import Embedder, Provider
from wmh.research.ablation import Condition, as_int
from wmh.research.pipeline import optimize_prompt, score_prompt
from wmh.research.scaling_split import CorpusSplit, partition_corpus, subsample_train
from wmh.research.seed_stability import BackendFactory
from wmh.retrieval import EmbeddingRetriever

# A point on the sweep grid: (number of train traces, GEPA iteration budget).
GridPoint = tuple[int, int]


class GepaScalingAblation:
    """Sweep (train-trace count × GEPA budget) against a fixed test set; metric = test fidelity.

    Conditions are the given `grid` of `(n_train, budget)` points, labelled `t{n}_b{b}`.
    `run(condition, seed)` subsamples `n_train` traces (nested as n grows, seeded), GEPA-optimizes
    the base prompt for `budget` iterations selecting on a fixed step-capped valid subset (budget=0
    skips optimization - the RAG-only anchor), then replay-scores the winning prompt on the fixed
    test set. Across-seed std (from `run_ablation`) is the error bar at each point.
    """

    name = "gepa-scaling-law"

    def __init__(
        self,
        corpus: list[Trace],
        base_prompt: str,
        *,
        make_backends: BackendFactory,
        grid: Sequence[GridPoint],
        top_k: int = 5,
        test_frac: float = 0.2,
        valid_frac: float = 0.15,
        gepa_val_steps: int = 30,
        val_fill: str = "greedy",
        recheck_steps: int = 0,
        minibatch_size: int = 3,
        hard_threshold: float | None = None,
        hard_probe_steps: int = 40,
        sample_turns: str = "all",
        test_cap: int | None = None,
        concurrency: int = 1,
    ) -> None:
        self._base_prompt = base_prompt
        self._make_backends = make_backends
        self._top_k = top_k
        self._hard_threshold = hard_threshold
        self._hard_probe_steps = hard_probe_steps
        self._sample_turns = sample_turns
        self._concurrency = concurrency
        self._split: CorpusSplit = partition_corpus(
            corpus, test_frac=test_frac, valid_frac=valid_frac
        )
        # Fixed seeded test subsample, exactly as the trace scaling law: every grid point and seed
        # is scored on the SAME test traces so the y-axis never moves.
        test = self._split.test
        self._test = subsample_train(test, test_cap, seed=0) if test_cap else test
        # GEPA's selection valset: a fixed step-capped subset of the valid band (seed-0 shuffle,
        # identical across conditions/seeds). GEPA fully evaluates promising candidates on this set
        # every iteration, so its STEP count - not trace count - is the per-iteration cost driver.
        self._minibatch_size = minibatch_size
        # Selection valset construction: "greedy" (historical) fits under the step cap by skipping
        # over-cap traces - cheap but short-trace-biased; "inclusive" takes shuffled traces until
        # the cap is REACHED, keeping long traces so the selection distribution matches the corpus.
        # Validated explicitly: this experiment A/Bs the two fills, so a typo silently selecting
        # the wrong one would corrupt a whole sweep with nothing in the outputs to show for it.
        if val_fill not in ("greedy", "inclusive"):
            raise ValueError(f"val_fill must be 'greedy' or 'inclusive', got {val_fill!r}")
        fill = _cap_by_steps if val_fill == "greedy" else _fill_by_steps
        # One seed-0 shuffle of the valid band feeds BOTH the selection valset and the recheck
        # pool, so their disjointness/ordering relationship is structural, not coincidental.
        shuffled_valid = subsample_train(self._split.valid, len(self._split.valid), seed=0)
        self._gepa_valid = fill(shuffled_valid, gepa_val_steps)
        # Optional guard-v2 re-check set: a step-capped slice of the valid band DISJOINT from the
        # selection valset, so the stagnant-or-improve acceptance re-check runs on steps GEPA never
        # selected against (catches winners whose valset win is real but biased).
        self._gepa_recheck: list[Trace] | None = None
        if recheck_steps > 0:
            val_ids = {t.trace_id for t in self._gepa_valid}
            remaining = [t for t in shuffled_valid if t.trace_id not in val_ids]
            self._gepa_recheck = _cap_by_steps(remaining, recheck_steps) if remaining else None
        # Cap trace counts at the train pool and drop duplicate points created by the cap,
        # preserving order, so an over-tall ladder collapses cleanly on a small corpus. The
        # zero-count filter runs on the CAPPED value: an empty pool must yield an empty grid, not
        # a bogus t0 point silently trained on nothing.
        pool = len(self._split.train_pool)
        capped = [(min(n, pool), b) for n, b in grid]
        self._grid = list(dict.fromkeys((n, b) for n, b in capped if n > 0 and b >= 0))

    @property
    def split(self) -> CorpusSplit:
        """The fixed test/valid sets + train pool this sweep runs against (for the header)."""
        return self._split

    @property
    def grid(self) -> list[GridPoint]:
        """The effective (n_train, budget) points (counts capped at the pool, deduped)."""
        return self._grid

    @property
    def scored_test(self) -> list[Trace]:
        """The test traces scored each point (the `test_cap` subsample, or the full test)."""
        return self._test

    @property
    def gepa_valid(self) -> list[Trace]:
        """The fixed step-capped valid subset GEPA selects candidates on."""
        return self._gepa_valid

    def conditions(self) -> list[Condition]:
        return [
            Condition(label=f"t{n}_b{b}", params={"n_train": n, "budget": b}) for n, b in self._grid
        ]

    def run(self, condition: Condition, seed: int) -> float:
        """Score one (n_train, budget) point at `seed` on the fixed test set; fidelity 0..1."""
        n_train = as_int(condition.params["n_train"])
        budget = as_int(condition.params["budget"])
        train = subsample_train(self._split.train_pool, n_train, seed=seed)
        provider, judge, embedder = self._make_backends()

        prompt = self._base_prompt
        if budget > 0:
            hard_filter = None
            if self._hard_threshold is not None:
                hard_filter = self._hard_step_filter(train, provider, judge, embedder)
            result = optimize_prompt(
                train,
                self._gepa_valid,
                self._base_prompt,
                provider=provider,
                judge=judge,
                embedder=embedder,
                budget=budget,
                seed=seed,
                hard_step_filter=hard_filter,
                select_on_hard=hard_filter is not None,
                recheck=self._gepa_recheck,
                minibatch_size=self._minibatch_size,
            )
            prompt = result.prompt

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
        )

    def _hard_step_filter(
        self,
        train: list[Trace],
        provider: Provider,
        judge: Judge,
        embedder: Embedder | None,
    ) -> Callable[[Step], bool] | None:
        """Pre-score a train probe + the GEPA valset under the base prompt; keep the hard steps.

        Replays every step of a step-capped probe of the train sample and of the fixed GEPA valset
        with the base prompt (leak-free demos from the same train corpus), and returns an
        identity-keyed filter accepting the steps that scored below `hard_threshold`. Identity works
        because GEPA sees the very same `Step` objects from the same `train`/valset trace lists.
        Returns None when nothing scores below the threshold (GEPA then runs unfiltered rather than
        starved). One replay covers both trace sets (they come from disjoint hash bands): a single
        thread pool spans all probe steps instead of two serialized pools. NB: the probe retrieves
        `top_k` demos while GEPA's own evaluations use `DemoRetriever`'s default of 5 - keep
        `top_k=5` (the default) when using `hard_threshold` so hardness is measured under the same
        retrieval GEPA optimizes with.
        """
        assert self._hard_threshold is not None
        retriever = EmbeddingRetriever(embedder) if embedder is not None else None
        probed = _cap_by_steps(train, self._hard_probe_steps) + self._gepa_valid
        report = replay(
            self._base_prompt,
            probed,
            provider,
            judge,
            retriever=retriever,
            train=train,
            top_k=self._top_k,
            concurrency=self._concurrency,
        )
        steps = [step for trace in probed for step in trace.steps]
        hard_ids = {
            id(step)
            for step, result in zip(steps, report.results, strict=True)
            if result.score < self._hard_threshold
        }
        if not hard_ids:
            return None
        return lambda step: id(step) in hard_ids


def _cap_by_steps(traces: list[Trace], max_steps: int) -> list[Trace]:
    """Greedily pick traces (in the given order) whose steps fit within `max_steps`; never empty.

    Greedy fill, not a prefix cut: a trace whose steps no longer fit is skipped and the fill
    continues with later, smaller traces - a prefix cut lets one over-cap trace starve the set
    (on swe-bench a 30-step trace early in the shuffle left GEPA a 2-step valset). If no trace
    fits the cap at all, the first trace is kept so the result is never empty.
    """
    out: list[Trace] = []
    total = 0
    for trace in traces:
        if total + len(trace.steps) <= max_steps:
            out.append(trace)
            total += len(trace.steps)
    if not out and traces:
        out = [traces[0]]
    return out


def _fill_by_steps(traces: list[Trace], min_steps: int) -> list[Trace]:
    """Take traces in the given order until at least `min_steps` steps are covered; never empty.

    Unlike `_cap_by_steps` this never skips a long trace, so the selection valset keeps the
    corpus's trace-length distribution (the greedy cap's short-trace skew is the documented
    valset-representativeness failure). May overshoot the budget by up to one trace. Shares
    `_cap_by_steps`' never-empty guarantee: a non-positive budget still yields one trace (an
    empty valset would make GEPA silently validate on the train traces instead).
    """
    out: list[Trace] = []
    total = 0
    for trace in traces:
        if total >= min_steps:
            break
        out.append(trace)
        total += len(trace.steps)
    if not out and traces:
        out = [traces[0]]
    return out
