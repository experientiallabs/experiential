# GEPA scaling law

**Does giving the world model more prompt optimization improve how faithfully it reconstructs the
environment?** The trace scaling law swept the *data* axis (how many traces retrieval sees); this
experiment sweeps the *optimization* axis: GEPA iterations (`budget`, panel A) and the number of
training traces GEPA learns from (panel B), against the same fixed held-out test set and the same
metric (open-loop reconstruction fidelity, `RubricJudge`/Opus 4.8). Panel C ablates the judge
model itself on byte-identical predictions; panel D densely sweeps the trace count with the
improved optimizer to locate each benchmark's optimal n.

> **Judge version - every number in this doc is `rubric-v1`** (the pre-#83 judge: unweighted mean
> of 5 dimensions, no validity gating), the same instrument as the trace scaling law it compares
> against. These numbers are **NOT comparable** to `rubric-v2` / `JUDGE_VERSION`-stamped fidelity
> on current `main` - the judge overhaul (#83) re-based the scale (roughly 0.58 under v2 where v1
> read ~0.70 on identical predictions). Re-running the Reproduce commands below on current `main`
> will silently score with rubric-v2 and produce numbers matching no table in this document; they
> are only for reproducing the *procedure*, not the printed values.

`wmh/research/gepa_scaling.py` implements the ablation; `budget=0` (GEPA off) reproduces the trace
scaling law's RAG-only point - a built-in consistency anchor between the two experiments.

![GEPA scaling law](gepa_scaling_law.png)

## Finding 1: GEPA does not lift open-loop fidelity above the RAG plateau

Fidelity vs GEPA iterations at a fixed 64 training traces (mean ± std across seeds 0-1):

| b | tau-bench | terminal-tasks | swe-bench |
|---|---|---|---|
| **0 (GEPA off)** | 0.892 ±0.004 | 0.875 ±0.010 | 0.730 ±0.003 |
| 1 | 0.887 ±0.000 | 0.869 ±0.001 | 0.732 ±0.004 |
| 2 | 0.893 ±0.015 | 0.857 ±0.009 | 0.728 ±0.002 |
| 4 | 0.895 ±0.005 | 0.863 ±0.006 | 0.718 ±0.017 |
| 8 | 0.882 ±0.010 | 0.857 ±0.006 | 0.702 ±0.003 |
| 16 | 0.885 ±0.002 | 0.819 ±0.028 | 0.656 ±0.015 |
| **GEPA lift (0 → best b)** | **+0.003** | **−0.006** | **+0.002** |

Every curve is flat to declining: no benchmark's fidelity climbs with iterations, and no
saturation point exists because there is nothing to saturate. Worse, terminal-tasks and swe-bench
actively *degrade* as budget grows - monotonically for swe (0.730 → 0.656 at b=16, both seeds) and
−0.056 for terminal at b=16: with the valset scores near-saturated or noisy, selection is
dominated by pipeline noise - the base prompt's own valset score varied 0.68-0.76 across identical
runs - so a high budget just buys more chances to promote a candidate whose marginal valset win
doesn't transfer. (These sweeps ran the *original* GEPA; the acceptance re-check below was built
from this diagnosis.) At a fixed budget of 8, the trace
axis tells the same story - GEPA does not unlock trace-count scaling either (seed 0; the trace
scaling law's RAG-only value in parens, which carries a ~0.01-0.02 serving-model offset - see
caveats):

| n | tau-bench | terminal-tasks | swe-bench |
|---|---|---|---|
| 1 | 0.860 (0.844) | 0.846 (0.860) | 0.672 (0.726) |
| 4 | 0.877 (0.887) | 0.861 (0.858) | 0.699 (0.729) |
| 16 | 0.876 (0.902) | 0.857 (0.858) | 0.688 (0.723) |
| pool | 0.906 @648 (0.932) | 0.849 @164 (0.873) | 0.718 @173 (0.743) |

tau-bench still climbs with n at b=8 - but that is retrieval doing the work (the same climb the
RAG-only curve shows), not optimization amplifying it.

## Finding 2: GEPA learns real knowledge; the residual errors are unknowable values

The flat curves are **not** "GEPA learned nothing". Spot-reading the evolved prompts (archived
with the raw results): tau-bench's b=8 prompt doubles in length with *correct*, genuinely
environment-specific rules - mutation tools return bare `"Transfer successful"` strings; unknown
ids yield structured not-found errors while known ids always populate fully; per-tool projections
(a usage tool never echoes `phone_number`/`plan_id`); numeric quantities returned as JSON strings.

The knowledge is right; it just doesn't move the metric, because the residual test errors are
almost entirely **unknowable-value errors** - the exact `data_used_gb` on a fresh lookup, a live
API's result count, the actual files in a repo. Format, shape, and outcome (the things prompt
knowledge *can* encode) were already mostly right at b=0 thanks to the base prompt + retrieved
demos. This is the optimization-side mirror of the trace scaling law's conclusion: past a low
floor, the binding constraint is information the trace never captured, and neither more retrieval
data nor more distilled prompt knowledge can manufacture it.

Where the losses actually come from (per-step diff of base vs the b=8 evolved prompt on
terminal's test set - the empirical failure reading):

- **Overgeneralized outcome priors.** The evolved prompt learned *true-on-average, wrong-per-step*
  rules from the training traces: "GitHub calls often rate-limit", "computed counts are often
  unknowable → null/empty". On test steps where the real call succeeded, it now predicts
  rate-limit errors, `null` counts, and empty arrays (0.80→0.42-class regressions); on steps that
  really did rate-limit it now matches exactly (0.44→1.00-class improvements). Net ≈ 0: GEPA
  *redistributes* fidelity across the success/error boundary because *which* call fails is itself
  unknowable.
- **Unknowable-value advice bleeding into derivable outputs.** The worst regression was a
  word-frequency count fully computable from the command's own heredoc (1.00 → 0.62): the
  "values you can't know - pick plausible ones" note discouraged exact computation.
- **Wasted reflection is real but not the explanation.** Random reflection minibatches frequently
  contain no failure ("all subsample scores perfect - skipping"): 7 wasted iterations on tau, 18 on
  terminal. But *concentrating* on the failures doesn't help either: the hard-step arm
  (`--hard-threshold 0.9`, reflection AND selection restricted to below-threshold steps, b=8/n=64/
  seed 0) scored **0.845 / 0.804 / 0.643** (tau / terminal / swe) vs the plain b=8 arm's
  0.872 / 0.852 / 0.705 - uniformly worse. Pushing selection onto unknowable-value failures trades
  away easy-case fidelity rather than converting the hard cases; reflection starvation is not what
  caps GEPA here.

### Why a "won" search can end below its own anchor - and the fix

GEPA never promotes a candidate that loses to base *within* a run. The violation of
"stagnant-or-improve" comes from measurement noise: re-scoring the *same base prompt* on the
*same fixed 30-step valset* three times gives 0.744 / 0.774 / 0.796 (std ≈ 0.02, range 0.05) -
run-to-run serve+judge nondeterminism at temperature 0. Promotion is argmax over single-sample
candidate evaluations inside that noise band, which systematically selects noise-inflated
candidates (the winner's curse); terminal's b=16/seed-1 cell (0.790, −0.056 vs anchor) is that
mechanism caught in the wild - its winner "beat" a base valset measurement of 0.68 that other
runs measured at 0.70-0.76.

Two principled changes landed from this diagnosis (both in `wmh/optimize/gepa.py`):

1. **Stagnant-or-improve acceptance re-check.** A non-base winner must replicate its win on a
   fresh, independent base-vs-winner evaluation or the base prompt is restored
   (`OptimizeMetrics.reverted_to_base`). Two extra eval passes per build - noise-promotion is
   structurally blocked. Live validation on the failing cell surfaced a SECOND mechanism: with the
   re-check on the *same* valset (guard v1), the winner's win replicated (test still 0.769) - the
   valset win was real but **unrepresentative**: the step-capped greedy valset over-represents
   short-trace steps (long traces are skipped to fit the cap), so a candidate can honestly win
   there and still lose on the test distribution. Guard v2 re-checks on a valset-**disjoint**
   slice of the valid band (`recheck` on `GEPAOptimizer.optimize` / `--recheck-steps`); on the
   failing cell it also kept the winner (test 0.788). Three independent replicates of that cell -
   no guard 0.790, v1 0.769, v2 0.788 - agree: the candidate genuinely beats base on *any*
   step-capped valid-band sample and consistently loses ~0.09 on test. The guard eliminates
   noise-promotion (its job); what remains is **selection-data bias** - the disjoint slice shares
   the short-trace skew, so representativeness, not more re-checking, is the residual fix
   (uniform step-sampled valsets; follow-up). Pragmatically, the flat budget curves already give
   the operational answer: nothing past b≈4 buys fidelity, and small budgets rarely produce these
   pathological winners.
2. **Anti-outcome-flip reflection guidance.** The reflection template now forbids rules that flip
   outcomes from frequency priors ("usually fails/empty here") and requires classifying output
   values as derivable-from-the-action (compute exactly), session-established (reuse verbatim), or
   external-unknowable (plausible + consistent) - directly targeting the two regression classes
   above. A/B on terminal b=8 (same split/seed): the new evolved prompt introduces **one** outcome
   flip across the 99-step test set where the old prompt's biggest regressions were dominated by
   success→error flips; its notes explicitly encode "do NOT assume failure by default - predict
   success when state/history shows it"; test fidelity base 0.862 → evolved 0.867 (the derivable-
   value case improves 0.62 → 0.70 but is not fully fixed). GEPA still doesn't *beat* the anchor -
   the unknowable-value ceiling stands - but it no longer damages what already worked.

## Finding 3: the conclusion is judge-robust; absolute fidelity is not

The same predictions (b=0 and b=8 prompts at n=64), re-scored by four judge models through the
same rubric - differences are pure measurement effects:

| judge | tau b0→b8 | terminal b0→b8 | swe b0→b8 |
|---|---|---|---|
| haiku-4.5 | 0.824 → 0.831 | 0.757 → 0.770 | 0.586 → 0.598 |
| gpt-5.4-mini | 0.865 → 0.863 | 0.809 → 0.828 | 0.693 → 0.702 |
| opus-4.8 (headline) | 0.888 → 0.888 | 0.847 → 0.853 | 0.737 → 0.745 |
| gpt-5.5 | 0.888 → 0.885 | 0.767 → 0.806 | 0.700 → 0.709 |

- **The ~zero-lift conclusion is judge-robust.** On tau no judge sees a lift Opus 4.8 missed
  (|Δ| ≤ 0.007 across all four); on swe all four agree on a +0.009-0.012 delta - while the budget
  sweep's own b=8 cell measured −0.03 for the *same configuration*. The sign of GEPA's "lift"
  flips between independent runs; no judge resolves it away from zero.
- **Terminal is the exception that sharpens the story**: every judge sees a positive delta and the
  harsher the judge, the bigger (gpt-5.5 +0.039 vs opus-4.8 +0.006) - Opus 4.8's generosity on
  plausible-but-wrong values partially saturates away a small real improvement in
  shape/outcome fidelity.
- **Absolute levels move by up to ~0.15 across judges** at identical predictions (swe: haiku 0.586
  vs opus 0.737). Fidelity numbers are only comparable with the judge model pinned; the ranking of
  benchmarks is stable under every judge.

## Finding 4: there is no trace count at which GEPA beats its base - more traces only reduce harm

A dense trace-count sweep (panel D: n ∈ {1,2,4,8,16,32,64,128,(256),pool} at b=8, seed 0, one run
per point) with the **improved** GEPA (acceptance re-check + anti-flip template) asks whether some
amount of training data makes GEPA worth running:

| n | tau-bench | terminal-tasks | swe-bench |
|---|---|---|---|
| 1 | 0.776 | **0.836**★ | 0.625 |
| 2 | 0.794 | 0.780 | 0.618 |
| 4 | 0.809 | 0.785 | 0.632 |
| 8 | 0.851 | 0.814 | 0.628 |
| 16 | 0.846 | 0.799 | 0.625 |
| 32 | 0.858 | 0.807 | 0.652 |
| 64 | 0.865 | 0.815 | **0.654**★ |
| 128 | 0.864 | 0.823 | 0.641 |
| 256 | 0.889 | - | - |
| pool | **0.912**★ @648 | 0.782 @164 | 0.642 @173 |

- **tau-bench: optimal n = the full pool**, rising monotonically 0.776 → 0.912 - but that climb
  tracks the RAG-only curve (0.932 at 648); retrieval, not optimization, is doing the work, and
  GEPA never closes the gap to it.
- **terminal-tasks: no learning optimum exists.** The argmax is n=1 - the point where GEPA has
  almost nothing to learn from and mostly leaves the base prompt alone. Among points where GEPA
  actually learns, n=128 is least bad (0.823); every point sits below the ~0.85 base.
- **swe-bench: shallow interior optimum at n≈64** (0.654), everything far below the 0.730 base.
- **Low n is actively dangerous**: at n=2-4 the evolved prompts are built from 2-4 traces' worth
  of patterns, maximizing the overgeneralized-prior failure (terminal drops to 0.780). A same-day
  healthy-pipeline probe (base+RAG@2 = 0.858, base+RAG@64 = 0.851) confirms these depressions are
  GEPA's doing, not measurement drift.
- **Run-to-run variance dominates point-to-point differences**: the identical terminal
  n=64/b=8/seed-0 condition measured 0.867 in one run and 0.815 in another (the reflection LM
  samples at temperature 1.0, and selection cannot discriminate candidates within its noise/bias
  band - see the acceptance-re-check section). Single-run points (this whole panel) carry that
  error bar implicitly.

**Operational answer:** if you run GEPA at all, give it the full trace pool (its harm shrinks with
data and tau even approaches its RAG ceiling); never run it on a handful of traces. Under the
*original* reflection template and data configuration, the fidelity-optimal setup was **base
prompt + RAG, GEPA off** - Finding 5 revises that.

## Finding 5: with the right reflection prompt and data configuration, GEPA finally lifts

Rereading the GEPA paper (arXiv 2507.19457, Alg. 1) against our integration exposed two departures:
the paper draws ~8-example reflection minibatches from `D_feedback` (ours hardcoded 3 - so with
~80% of steps scoring perfectly, ~half of our reflection iterations saw zero failures and were
skipped), and it selects on a sizeable `D_pareto` (ours: 30 short-trace-biased steps - the
Finding-4 anti-transfer mechanism). We exposed both as knobs (`minibatch_size`, `--val-fill
inclusive` + `--gepa-val-steps`) and rewrote the reflection template around the empirical failure
taxonomy (template v2: a mandatory diagnose-then-classify pass; **evidence precedence** - retrieved
demos/session history override distilled notes; no frequency-based outcome flips; derivable vs
unknowable values; compact targeted notes). The 2×2 configuration grid at the standard b=8/n=64
point, all with template v2 (anchor in the header; single run per cell):

| config | tau (0.892) | terminal (0.875) | swe (0.730) |
|---|---|---|---|
| mb=3, val=30 greedy | 0.900 | 0.844 | 0.749 |
| mb=8, val=30 greedy | 0.900 | 0.864 | 0.750 |
| mb=3, val=90 inclusive | **0.914** | 0.858 | **0.751** |
| mb=8, val=90 inclusive | **0.914** | 0.856 | 0.737 |

- **Template v2 is the biggest single factor.** At the identical mb3/val30 configuration the old
  template measured 0.882/0.857/0.702 (sweep) - v2 moves every benchmark up (+0.02 to +0.05) and
  turns swe from GEPA's worst case (−0.03 vs anchor) into its clearest win (+0.02, three of four
  cells). The evidence-precedence rule targets exactly the diagnosed retrieval-override
  regressions.
- **Selection-set representativeness buys real lift where selection signal exists**: tau gains
  +0.014 from val30-greedy → val90-inclusive, reaching **0.914 - +0.022 over its anchor, the first
  configuration in this study where GEPA beats base+RAG**. Terminal/swe are ~neutral to it.
- **Reflection minibatch size is endpoint-neutral** (all pairs within run noise) once the template
  stops writing harmful notes - b=8 just wastes fewer skipped iterations getting to the same
  place.
- **Terminal never lifts** (0.844-0.864 vs 0.875) under any configuration - consistent with its
  failure mass being unknowable-value content where the prompt ceiling is already reached; the
  improved configuration only removes the harm (no more 0.78-0.82 tails).

**Revised operational answer:** template v2 + a ≥90-step *inclusive* selection valset + the full
trace pool + b≈8, minibatch 8. Under that configuration GEPA is worth running on benchmarks whose
failures include fixable convention/evidence errors (tau +0.022, swe +0.021) and is safely neutral
elsewhere. Cells are single runs (pipeline noise ±0.02): tau's lift repeats across two
configurations and swe's across three, so the pattern is consistent, but error bars are implicit.

## Caveats

- Serving/optimization ran on Opus 4.7 (4.8 throttles under GEPA's call volume); the trace scaling
  law served on 4.8. Our own b=0 anchors (0.892/0.875/0.730) sit ~0.01-0.02 below its base@64
  values (0.908/0.859/0.744 - terminal actually higher here), so RAG-parens comparisons above are
  directional; within-experiment comparisons are exact.
- Trace axis, b=16 tails, and the hard-filter arm ran 1 seed (cost control); budget axis b≤8 has
  2 seeds everywhere. Across-seed std at those points is ≤0.017.
- The headline metric is the pre-#83 `rubric-v1` judge (unweighted mean of 5 dims; `format`
  over-generous; no empty-prediction penalty; no validity gating) - see the judge-version notice at
  the top. Finding 3 exists precisely to bound the instrument's influence; the #83 judge overhaul
  later re-based the scale, so these numbers live and die with rubric-v1.
- A Bedrock brownout crashed the three budget sweeps on their final b=16 points (a mistyped
  failover-ladder rung turned the cascade fatal); completed points were verified untainted (every
  exception postdates the last completed point; zero `Rollout failed` critiques) and the tails were
  re-run. During brownout windows individual calls may have been served by fallback ladder models
  (failover is unlogged - a known limitation).

## Reproduce

The experiment is fully specified by the public `wmh` API - the commands below name a workspace
runner for convenience, but nothing here depends on it surviving (`.agents/` contents are
disposable). The procedure each command drives, in API terms: ingest the suite's OTel traces
(`wmh.ingest`, adapter `otel-genai`), split with `wmh.research.partition_corpus(test_frac=0.2,
valid_frac=0.15)`, and run `wmh.research.GepaScalingAblation` over the given `(n_train, budget)`
grid via `run_ablation` with the flags mapping 1:1 onto its constructor: `--gepa-val-steps` →
`gepa_val_steps` (GEPA's selection valset step cap), `--val-fill` → `val_fill`
(greedy|inclusive fill), `--minibatch` → `minibatch_size`, `--recheck-steps` → `recheck_steps`
(guard-v2 disjoint re-check), `--hard-threshold` → `hard_threshold`, `--test-cap` → `test_cap`
(fixed seeded test subsample), `--sample-turns sampled` → Qwen-AgentWorld 5-turn scoring.
Backends: serving/optimize on Bedrock Opus 4.7 behind a capacity-only failover ladder, judge =
`RubricJudge` on Opus 4.8 (**rubric-v1 at publication** - on current `main` the same commands
score with rubric-v2 and will not match these tables), embedder = offline
`HashingEmbedder(dim=512)`. Raw `AblationReport`s, judge-ablation JSONs (including the evolved
prompts), and the RAG baselines are archived under
`.agents/docs/research/gepa_scaling_results/`.

```bash
# budget axis (per benchmark): b in {0,1,2,4,8,16} at n=64, seeds 0,1
AWS_PROFILE=default AWS_REGION=us-east-1 uv run python .agents/scripts/run_gepa_scaling.py \
  tau-bench --counts 64 --budgets 0,1,2,4,8,16 --seeds 0,1 --sample-turns sampled \
  --test-cap 40 --gepa-val-steps 30 --concurrency 8 --out tau_budget.json

# trace axis (per benchmark): n in {1,4,16,pool} at b=8, seed 0
AWS_PROFILE=default AWS_REGION=us-east-1 uv run python .agents/scripts/run_gepa_scaling.py \
  tau-bench --counts 1,4,16,648 --budgets 8 --seeds 0 --sample-turns sampled \
  --test-cap 40 --gepa-val-steps 30 --concurrency 8 --out tau_traces.json

# dense optimal-n sweep (improved GEPA: acceptance re-check + anti-flip template are default;
# --recheck-steps 30 additionally re-checks on a valset-disjoint slice - guard v2)
AWS_PROFILE=default AWS_REGION=us-east-1 uv run python .agents/scripts/run_gepa_scaling.py \
  tau-bench --counts 1,2,4,8,16,32,64,128,256,648 --budgets 8 --seeds 0 --sample-turns sampled \
  --test-cap 40 --gepa-val-steps 30 --concurrency 8 --out tau_dense.json

# data-configuration grid (Finding 5; template v2 ships as the default reflection template) -
# the winning configuration:
AWS_PROFILE=default AWS_REGION=us-east-1 uv run python .agents/scripts/run_gepa_scaling.py \
  tau-bench --counts 64 --budgets 8 --seeds 0 --minibatch 8 --gepa-val-steps 90 \
  --val-fill inclusive --sample-turns sampled --test-cap 40 --concurrency 8 --out tau_mb8v90.json

# hard-step arm: same t64_b8 point with reflection/selection concentrated on failures
AWS_PROFILE=default AWS_REGION=us-east-1 uv run python .agents/scripts/run_gepa_scaling.py \
  tau-bench --counts 64 --budgets 8 --seeds 0 --hard-threshold 0.9 ... --out tau_hard.json

# judge-sensitivity ablation (per benchmark; needs OPENAI_API_KEY for the two GPT judges)
AWS_PROFILE=default AWS_REGION=us-east-1 OPENAI_API_KEY=... uv run python \
  .agents/scripts/run_judge_ablation.py tau-bench --out judge_ablation/tau-bench.json

# the figure (matplotlib is ephemeral, not a project dep)
uv run --with matplotlib python .agents/scripts/plot_gepa_scaling.py \
  --budget-report tau-bench=tau_budget.json ... --trace-report tau-bench=tau_traces.json ... \
  --rag-report tau-bench=rag_baseline/tau-bench.json ... \
  --judge-report tau-bench=judge_ablation/tau-bench.json ... \
  --dense-report tau-bench=tau_dense.json ... \
  --ymin 0.55 --out docs/research/gepa_scaling_law --title "GEPA scaling law"
```

Serving/optimize model `us.anthropic.claude-opus-4-7`, judge `us.anthropic.claude-opus-4-8`, both
behind a capacity-error-only failover ladder (endflow-account 4.6-gen models → default-account
Opus → GPT-5.5); the corpora are the live-captured benchmarks under `examples/`.
