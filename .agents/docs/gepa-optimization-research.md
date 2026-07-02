---
source: https://app.notion.com/38e0f8b3f59181d5a524f6279113c826
area: Research
status: Current
migrated: 2026-07-02
---

# GEPA optimization research

This is the **research surface** for the harness's optimization trajectory: prompt optimization
(GEPA) today, heavier training methods tomorrow. It exists to try optimization directions
*empirically* — change a knob, measure reconstruction fidelity, record the result — rather than
guessing. It is the experimental sibling of *Iterating on BASE_ENV_PROMPT with replay fidelity* (which
hand-tunes the base prompt) and *Design note: RAG-aware GEPA* (which explains the RAG-aware,
leak-free evaluation every experiment here inherits). Directions not yet run live in *Research
directions (not yet run)*; the first completed sweep is written up in *Trace scaling law*.

## The harness (`wmh/research`)

Three small pieces, designed so **a new experiment is one new file**:

- **`ablation.py` — the framework.** A `Condition` is a named bundle of knob values. An `Ablation`
  (Protocol) knows its `conditions()` and how to `run(condition, seed) -> float` (one build+eval at
  one seed, returning a scalar metric, higher = better). `run_ablation(ablation, seeds)` is the
  generic driver: it sweeps every condition × seed and aggregates each condition's mean +
  (population) std across seeds into an `AblationReport`. Aggregation is deliberately simple —
  small trace corpora make CIs/significance tests false precision.
- **`pipeline.py` — the reusable primitives.** `optimize_prompt(...)` runs `GEPAOptimizer` at a
  chosen GEPA `seed`; `score_prompt(...)` replay-scores a prompt's held-out fidelity by delegating
  to the canonical `wmh.engine.replay.replay` — the *same* scorer `wmh eval` uses. So an
  experiment's fidelity is directly comparable to the rest of the harness, and a judge/rubric
  upgrade (e.g. the `RubricJudge` 5-dimension scorer) lands in experiments for free.
- **`seed_stability.py` — the first experiment** (below). `trace_scaling.py` + `scaling_split.py` — the trace scaling law sweep (see its own doc) — came next.

Backends (provider, judge, embedder) are dependency-injected via a factory, so the unit tests
drive everything with fakes (no network) and the live runner drives it with Bedrock through the
exact same code path. Pass `RubricJudge` as the judge to score on the canonical 5 dimensions.

### Adding a new experiment

Write a class satisfying `Ablation`: a `name`, a `conditions()` list, and a `run(condition, seed)`
that builds/evaluates under those knobs and returns a scalar. Reuse `optimize_prompt` /
`score_prompt` for the build+eval. The driver, seed sweep, aggregation, and reporting come for free.
Candidate experiments are catalogued in [`research_directions.md`](./research_directions.md).

## The knob this added to the core (coordinate with the eval-scorer chat)

`wmh/optimize/gepa.py` hardcoded the GEPA engine `seed=0`. The research harness needs to vary it, so
the change is **surgical and backward-compatible**: `GEPAOptimizer(..., *, seed=0)` threads it
through to `gepa.optimize(seed=...)`. Production callers (`wmh build`, `wmh serve`, `wmh eval`) pass
nothing and get the old behavior; only the research harness sets it. The change is one additive
keyword-only parameter, so conflicts with the eval-scorer chat's judge work should be mechanical.

> A `temperature` knob was prototyped here too, but every shipped provider (Opus 4.8 / GPT 5.5)
> rejects sampling parameters — passing `temperature` would 400, so a temperature sweep is inert
> until a sampling-capable provider exists. That direction is parked in
> [`research_directions.md`](./research_directions.md); the knob was removed from the core to avoid
> implying a capability that isn't wired.

## Experiment 1 — GEPA seed stability

**Question.** GEPA's search is stochastic: the reflection LM samples at temperature 1.0, and the
engine seed drives minibatch sampling and Pareto candidate selection. So two builds of the *same*
corpus at the *same* budget can evolve *different* winning prompts. How much does the held-out
fidelity of the winner wobble when only the seed changes?

This matters operationally: if the spread is small, any seed is fine and a single build is
trustworthy; if it's large, the winning prompt is seed-dependent and a real build should sweep seeds
and keep the best. It also calibrates every *other* experiment — a knob effect smaller than the
seed-noise band isn't a real effect.

One `Condition` (the baseline build config), swept across N seeds. The metric is held-out
reconstruction fidelity (mean judge score 0..1, scored with `RubricJudge`). The headline number is
the **across-seed std**.

### Run it

```bash
AWS_REGION=us-east-1 uv run python scripts/run_seed_stability.py \
  examples/tau2-bench.otel.jsonl \
  --seeds 0,1,2 --budget 12 --judge rubric --out report.json
```

The runner ingests + splits the corpus exactly as `wmh build` does, runs GEPA per seed on live
Bedrock (offline `HashingEmbedder` for phi), prints per-seed fidelity and the mean ± std, and writes
the full `AblationReport` JSON. Use a corpus with a real held-out split (the bundled
`examples/tau2-bench.otel.jsonl` is 12 traces / 67 steps) for a meaningful number; pass `--judge
match` to use the functional `LLMJudge` instead of the 5-dimension rubric.

### Reading the result

- **std < ~0.05** → GEPA is reproducible at this budget; a single build is fine, and other
  experiments can treat anything below this band as noise.
- **std large** → the winner is seed-dependent; real builds should sweep a few seeds and keep the
  best held-out, and ablation effects must clear this band to count.

This experiment is **live-only** (every run is a real multi-rollout GEPA build), so no canned result
is committed; run the command above to produce the report. The unit tests exercise the apparatus
with deterministic fakes (which trivially give std=0).

## The canonical model

`world-models/tau-bench/` is the repo's committed, GEPA-optimized example world model (built from the
clean `examples/tau2-bench.otel.jsonl` on Bedrock Opus 4.8). It is discovered automatically by the
bundled search path (see `WorldModelStore` and [`ARCHITECTURE.md`](./ARCHITECTURE.md)), so `wmh
list`, `wmh play --name tau-bench`, and `wmh serve` find it with no `--root`. It is the model the
benchmark/reporting chat scores and the README points at.
