# Routing optimizer: findings + method tier list

2026-07-24, optimization chat. Canonical copy (a mirror lives in the Notion experiments area
under Research per Silen's ask). Companion docs: `routing-lit-review.md` (full literature map)
and `../proposals/routing-optimizer-v1.md` (benchmark + v1 design, approved 2026-07-24).

## Findings to date

1. **The literature supports cluster-then-assign at our data scale.** The Avengers
   (arXiv 2505.19797) is the published version of our v1 almost verbatim and shows the one real
   hyperparameter (cluster count) is robust; "Simple kNN Beats Complex Learned Routers"
   (2505.12601) shows neighborhood methods match or beat learned routers whenever win-rates are
   locally smooth in embedding space; UniRoute (2502.08773) proves cluster routing approximates
   the Bayes-optimal rule with an excess-risk bound; CARROT (2502.03261) adds a minimax argument
   for simple cost+accuracy predictors. Added router capacity at 100s-of-scenarios scale buys
   variance, not accuracy.
2. **Label noise is the dominant failure mode.** DARS (2606.06924) shows single-shot outcome
   labels are unstable; our own env-reward-lottery history (reward stdev 0.34) says the same.
   Mitigations adopted: >= 2 episodes/scenario, pinned judge (Opus 4.8), every delta reported
   with its spread, RouterBench exact-match stage as a judge-noise-free control.
3. **Cache-aware cross-model routing is a real, citable gap.** Only 2604.12385 (simulation-only,
   2026) prices cache invalidation in routing; serving-side cache work (SGLang 2312.07104,
   Preble 2407.00023, GORGO 2602.11688) is single-model. Nobody ties real prefix-cache state to
   a cross-model decision. Our serving already does default-sticky affinity; the learned
   switching rule is phase 2, gated on capturing cached_tokens.
4. **Scenario tool surface gates benchmark validity** (live finding, 2026-07-24): on bare-task
   scenarios, models that honestly decline to hallucinate tools score 0 while models that invent
   tool calls the WM plays along with score high. A matrix fitted on that teaches the router to
   prefer confident hallucinators. Fitting on agentic corpora is blocked on the wm-create tool
   surface contract; tau-bench is safe (tools.json exists).
5. **Azure retail-meter API is the pricing source of truth** (prices.azure.com): aggregator
   quotes for Kimi-K2.6 were wrong in both directions. Pool prices now pinned from published
   meters, including cache-read rates.

## The method tier list (Silen's ladder, annotated, with papers)

Ordered by when we reach for each; every step keeps the same `RoutingPolicy` artifact +
`select_model` seam so upgrades swap in without touching serving.

- **Tier 1 - NOW: clustering (Avengers replication).** Embed -> k-means -> per-cluster model
  scoring -> route to nearest cluster's model. Papers: Avengers 2505.19797 (recipe + ablations),
  Avengers-Pro 2508.12631 (the alpha cost/quality knob on top), UniRoute 2502.08773 (theory),
  kNN-beats 2505.12601 (the locality diagnostic we run to know when tier 2 is worth it).
- **Tier 2: clustering + learned predictor.** Per-model correctness classifiers
  (Shnitzer 2309.15789), matrix factorization (RouteLLM 2406.18665), contrastive query<->model
  embeddings (RouterDC 2409.19886), per-prompt Bradley-Terry (P2L 2502.14855), model embeddings
  from correctness matrices (EmbedLLM 2410.02223). Justified only by a measured oracle gap that
  tier 1 leaves unclaimed AND a failed locality diagnostic. Supervision follows DARS
  (2606.06924): multi-sample labels, not single shots.
- **Tier 3: LLM-as-router / compute-shaped routing (quality max).** Router-R1 2506.09033 and
  xRouter 2510.08439 (RL-trained router LLM, can aggregate multiple models); BEST-Route
  2506.22716 (jointly pick model AND best-of-n budget: n samples from a cheap model + selection
  can beat one frontier call on both cost and quality, which is Silen's 4-cheap-calls intuition,
  published). Costs router-side latency + tokens; only for quality-max tiers of the knob.
- **Tier 4: bandits for live updates.** BARP 2510.07429 (multi-objective, learns from
  bandit feedback = exactly deployed-endpoint logs), MixLLM 2502.18482 (continual), NeuralUCB
  2603.30035, dueling feedback 2510.00841. Warm-start from the tier-1 cluster table to skip
  cold-start regret.
- **Tier 5: cascades for escalation.** FrugalGPT 2305.05176, AutoMix 2310.12963 (POMDP
  meta-verifier over self-verification), BEST-Route again. The escalation decision sees the
  cheap model's actual answer (stronger signal than the query), at the price of latency and a
  cache-breaking model switch; use only for low-confidence clusters.
- **Tier 6: R2R + cache-aware serving.** R2R "Roads to Rome" 2505.21600 (token-level
  small/large routing; requires co-hosted models, so it activates only if the distillation leg
  gives us our own hosted model). Cache-aware switching: 2604.12385 (closest prior art),
  GORGO 2602.11688 (residual-prefill cost decomposition we adopt). Annotation: cache-aware is
  ORTHOGONAL to tiers 2-5 and starts as soon as serving captures cached_tokens; it is listed
  last only because its benchmark (multi-turn traffic replay) is the most work.

## Benchmark plan (approved)

Stage A: fit on RouterBench's public precomputed matrix (405k outcomes, 11 models; free,
offline, published baselines) until our implementation lands in the published band. Stage B:
run OUR 9-model pool over subsampled RouterBench prompts with exact-match grading (their matrix
contains their models, not ours; this stage answers "optimize directly against RouterBench with
our models" and is judge-noise-free). Stage C: wm-scenario matrices (tau-bench first;
terminal-tasks; bird-sql blocked on the tool-surface contract). Metrics: cost-quality Pareto +
AIQ (RouterBench 2403.12031), oracle / best-single-in-hindsight / random / Zero-Router
baselines, scenario-split held-out + one held-out-cluster OOD row. Budget approved ~$1-1.5k.

## The quality-vs-cost question (Silen asked back; recorded answer)

The tradeoff is the product knob, not a fixed constant. Default position = cost-saver at
quality parity (matches the "run 10x more for the same budget" claim; strongest honest
headline). The fitted policy exposes lambda so eval can sweep the full frontier
(fit-once-slide-knob, Hybrid LLM 2404.14618), and the future weighted-equation objective over
quality/cost/latency (Silen's direction) arrives as knob positions on the same artifact, with
latency as an SLA constraint rather than a scalar term (MixLLM's pattern).
