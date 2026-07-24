# Routing optimizer v1: benchmark + implementation proposal (DISCUSSION DRAFT)

Status: awaiting Silen's back-and-forth (his explicit direction 2026-07-24: no fitting code
until the benchmark and approach are agreed). Everything AROUND the optimizer already exists
on feat/optimizer-switch: pool, closed-loop OutcomeMatrix, policy artifact + serve-time
selection, OpenAI-compatible streaming endpoint, improvement report. The fitter drops into
`RoutingPolicy` (wmh/optimize/policy.py) and is judged by the benchmark below.

## 1. The benchmark: what "working" means

One number cannot honestly answer "does routing work"; the benchmark is a fixed table computed
from one precomputed OutcomeMatrix per benchmark corpus (RouterBench methodology: run the pool
over the scenarios once, compare every policy variant offline on identical data).

**Data**: per corpus (start: tau-bench, terminal-tasks, bird-sql), scenarios split by SCENARIO
id into fit/validation/test with task-level leakage control (D26 precedent). The test split is
never seen by any fitting run. Episodes per scenario: 2 (judge-noise floor per D30).

**Baselines (the quartet, all from the same matrix)**:
1. Best single model in hindsight (chosen on fit split, evaluated on test): the "why route at
   all" bar. THE PASS/FAIL GATE IS AGAINST THIS, not against a weak strawman.
2. Random assignment (expectation over the pool): sanity floor.
3. Oracle (per-scenario best on test): the ceiling; how much headroom routing has at all.
4. All-frontier (the D-REPORT baseline model everywhere): the cost-savings anchor.

**Metrics**:
- Cost-quality Pareto frontier plot (policy knob swept) + AIQ (area under cost-quality, absolute
  dollars, RouterBench convention).
- Headline pair at the balanced default knob: accuracy delta vs best-single-model (must be
  >= -1 noise-floor point) AND cost delta (must be materially cheaper; propose >= 30% cheaper,
  else routing is complexity without payoff).
- Latency p50/p95 carried but reported, not gated, in v1.
- Judge noise bar: every headline delta is reported with the seed/episode spread; a delta
  inside the spread is "no effect" by definition (env-luck lesson, stdev 0.34 history).

**PASS for v1** = on >= 2 of 3 corpora, at the default knob: accuracy within noise of
best-single-model AND cost <= 0.7x best-single-model's, on the untouched test split, judge
pinned. FAIL = anything else; we then ship static policies and say so.

**Cache honesty**: matrix episodes are effectively cold-cache single conversations, so v1
benchmark cost is single-shot and the report's cost_assumptions says exactly that. The
multi-turn cache-adjusted benchmark (traffic replay with per-model cache state, switching cost
= residual prefill per GORGO 2602.11688) is phase 2, tied to capturing cached_tokens in
providers. We do NOT quote multi-turn savings until then (2604.12385 is the closest prior art
and is simulation-only; being wired to real cache state is our claim, so it must be real).

## 2. v1 algorithm proposal (to agree on, then build)

Avengers-style cluster-then-assign (2505.19797), the literature's small-data winner:
1. Embed fit-split scenario tasks (HashingEmbedder first; measure, then decide if a provider
   embedder is worth credentials, per the 2505.12601 locality diagnostic).
2. k-means (numpy, k swept over ~4-16, chosen on validation).
3. Per cluster, assign argmax over pool of `mean_reward - lambda * cost_per_run`; lambda is the
   single knob (Hybrid-LLM style: fit once, slide at eval time to trace the frontier; default
   knob = balanced point). Thin clusters (< N scored episodes) fall back to default_model,
   counted and reported, never silently.
4. Emit RoutingPolicy(kind="cluster") + ImprovementReport; `fitted_from` pins the matrix.
Explicitly NOT in v1: trained per-query predictors (MTRouter-style, v2), cascades, bandits,
switching-penalty learning (serving is default-sticky until phase 2).

## 3. Invocation shape (CLI/factory), to reconcile with `wmh optimize <agent>`

`wmh optimize` today means harness optimization (PR #242). Proposal: keep positional semantics
and add the switch as a subcommand group later only if a third CLI-visible optimizer appears;
for the routing optimizer the natural surface is endpoint-centric, not optimizer-centric:

    wmh route fit <world-model> --pool .wmh/pool.toml --episodes 2 [--knob 0.5]
    wmh route report <world-model>   # rebuild the report from the pinned matrix

(name "route" is internal-dev vocabulary; customer copy never says router.) Alternative
steelman: `wmh optimize --optimizer routing <world-model>` keeps one verb as the brief sketched,
but overloads a command whose positional args mean something else today. Recommendation: the
endpoint-centric `wmh route` pair; revisit if Silen prefers verb unification.

## 4. Open questions for Silen

1. Pass/fail thresholds in §1 (within-noise accuracy AND <= 0.7x cost on 2/3 corpora): right
   bar? Steelman for stricter: the demo story wants ~10x; steelman for this bar: 0.7x vs the
   hindsight-best single model is already a strong claim (most of the 10x comes vs all-frontier).
2. Scenario tool surface (2026-07-24 finding): fit only on corpora whose scenarios carry tools,
   or accept the affordance-guessing bias for v1 corpora that lack them? Recommendation: block
   fitting on the wm-create tool-surface contract for agentic corpora; chat-style corpora are
   unaffected.
3. Judge pin for the matrix: Opus 4.8 (house default) vs gpt-5.4-mini (cheap, pinned in
   wm-benchmarks)? Recommendation: Opus 4.8, third-family vs most candidates, and never compare
   across judges.
4. Router training longer-term (affects D-METERING router_cost_usd and the artifact): if v2 is
   a trained predictor, do we host it (own deployment, real per-call cost) or embed it in
   serving (amortized)? No decision needed now; shapes already carry the cost either way.
