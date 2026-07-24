# LLM routing: literature review (2023-2026)

Compiled 2026-07-23 by the optimization chat (endpoint pivot). Consumers: the routing
optimizer v1 built in this repo, and the deferred routing-algorithm + routing-benchmark
chat. Three research passes: eval conventions, methods design space, named-paper
resolution + open-source repo sweep.

## 1. Named-paper identity resolutions

Silen named several papers from memory (voice-transcribed). Resolutions, with confidence:

| Named | Resolution | Confidence | Notes |
| --- | --- | --- | --- |
| "ML Router" | No literal match on arXiv. Candidates: TensorOpera Router "TO-Router" (2408.12320, EMNLP 2024 Industry, BERT-based multi-LLM router), Meta-Router (2509.25535), MetaLLM (2407.10834). Also "LLM Router: Rethinking Routing with Prefill Activations" (2603.20895) | LOW | Needs Silen confirmation. Prefill-activations paper requires white-box access to routed models, so its mechanism does not transfer to an API-model pool |
| "T-Router" | TO-Router (2408.12320) or TagRouter (ACL Findings 2025, tag generator/scorer/decider, no code) | MEDIUM | "TO-Router" phonetic match; may collapse into the same paper as "ML Router" |
| "DARs" | DARS = Distribution-Aware Routing Supervision, in "From Sampled Outcomes to Capability Distributions" (2606.06924) | HIGH | Single-shot outcome labels are noisy; build supervision from multi-sample, multi-paraphrase distributions |
| "Route to roam" | R2R "Roads to Rome" (2505.21600, NeurIPS 2025): token-level SLM/LLM routing, only path-divergent tokens go to the big model. Code: thu-nics/R2R + HF weights | HIGH | Alternative: Route-to-Reason (2505.19435), lower phonetic fit |
| "mixLM" | MixLLM (2502.18482, NAACL 2025): contextual-bandit router, tag-enhanced embeddings, separate quality/cost heads + latency-constrained meta-selector | HIGH | |
| "Router R1" | Router-R1 (2506.09033, NeurIPS 2025, UIUC): the router is itself an RL-trained LLM, multi-round think/route/aggregate, cost-aware reward. Code: ulab-uiuc/Router-R1 | HIGH | Heavier paradigm; relevant only if we want multi-model aggregation, not single-shot assignment |

## 2. Design space map

### Predictive routers (per-query quality/cost predictors; our v2 family)
- Shnitzer et al. 2309.15789: the foundational framing, router = per-model binary
  correctness classifiers. Simplest thing that works; exactly the v2 outcome predictor.
- RouteLLM 2406.18665 (ICLR 2025): strong-vs-weak router on Arena preferences; matrix
  factorization variant best; 2x+ cost cut at fixed quality. Repo: lm-sys/RouteLLM.
- Hybrid LLM 2404.14618 (ICLR 2024): predicts quality GAP; one test-time threshold
  slides the whole cost/quality tradeoff, no retraining. Soft labels from repeated
  sampling beat hard 0/1 labels under judge noise. Two ideas we adopt outright.
- RouterDC 2409.19886 (NeurIPS 2024): dual contrastive losses; fixes tie degeneracy
  (several models all good). Repo: shuhao02/RouterDC.
- EmbedLLM 2410.02223: per-model embedding learned from its correctness matrix; cheap
  model vectors + cross-benchmark forecasting.
- GraphRouter 2410.03834 (ICLR 2025): inductive GNN; add a model = add a node with a
  few probe results. For roster churn; overkill at our data scale. Repo: ulab-uiuc.
- P2L Prompt-to-Leaderboard 2502.14855: per-prompt Bradley-Terry coefficients (1-param
  IRT); was #1 on Chatbot Arena Jan 2025. Repo: lmarena/p2l.
- UniRoute 2502.08773 (Google): unseen models at test time via model feature vectors on
  representative prompts; PROVES cluster-based routing approximates the Bayes-optimal
  rule with an excess-risk bound. Our v1 theory anchor.
- MixLLM 2502.18482: see table above. Three-head design matches our objective shape.
- Arch-Router 2506.16655 (Katanemo): routes to human-authored natural-language
  domain/action policies; 1.5B open weights on HF.
- Smoothie 2412.04692 (NeurIPS 2024, Hazy Research): LABEL-FREE quality estimation via
  weak supervision over output embeddings. De-risks thin/noisy-label clusters. Repo:
  HazyResearch/smoothie.
- IRT-Router 2506.01048 (ACL 2025): Item Response Theory, interpretable, strong
  cold-start. Repo: Mercidaiha/IRT-Router.
- CARROT 2502.03261: minimax theory result: a simple router predicting both cost and
  accuracy per query is rate-optimal. Supports lightweight-over-heavy. Repo:
  somerstep/CARROT.
- DARS 2606.06924: distribution-aware supervision (see table). Caveat on any training
  over single-shot outcome matrices, including reused RouterBench/RouterEval data.
- RadialRouter 2506.03880, xRouter 2510.08439 (Salesforce, RL tool-calling router,
  code + HF), Meta-Router 2509.25535, TO-Router 2408.12320: noted, lower priority.

### Clustering / kNN routers (our v1 family)
- The Avengers 2505.19797 (AAAI 2026): the v1 blueprint, near-verbatim: embed queries,
  k-means, score each model per cluster on a train split, route to nearest cluster's
  best model. Only real hyperparameter (#clusters) shown robust. 10 open ~7B models
  beat GPT-4.1 on 10/15 datasets.
- Avengers-Pro / "Beyond GPT-5" 2508.12631: adds one scalarization knob alpha over the
  same clustering router, traces the cost/quality frontier, Pareto-dominates GPT-5
  routing. This is v1 + our objective, already published.
- "Simple kNN Beats Complex Learned Routers" 2505.12601: tuned kNN over query
  embeddings matches/beats RouteLLM/GraphRouter-class routers. Works iff win-rates are
  locally smooth in embedding space; their locality diagnostic is worth running on our
  scenario embeddings before any v2 investment.
- Verdict: at 100s-1000s of scenarios, cluster-then-assign is near-optimal and does not
  overfit; learned per-query routers win mainly with big data, churning rosters, or
  non-smooth win-rate geometry.

### Cascades (fallback pattern, not primary)
- FrugalGPT 2305.05176: canonical cascade with learned stop-scorer.
- AutoMix 2310.12963 (NeurIPS 2024): small model answers, self-verifies few-shot, a
  POMDP meta-verifier decides escalation. Works over black-box APIs.
- BEST-Route 2506.22716 (ICML 2025): jointly picks model AND best-of-n budget; small
  model + n samples + selection substitutes for escalation. 60% cost cut, <1% drop.
- Tradeoff: cascades see the cheap model's actual answer (stronger signal than the
  query alone) but pay latency and forfeit cache affinity. Keep as low-confidence
  fallback only.

### Bandits / online (v2+, after the endpoint generates logs)
- BARP 2510.07429: multi-objective contextual bandit from bandit feedback (only the
  chosen model's outcome observed = exactly our deployed logs); one policy spans the
  tradeoff family.
- MixLLM 2502.18482 (continual), NeuralUCB 2603.30035, dueling feedback 2510.00841
  (fits pairwise judge outputs), drifting contexts 2506.17670.
- Cold-start remedy: warm-start the bandit from the v1 cluster table.

### Multi-turn and cache-aware (our claimed novelty)
- MTRouter 2604.23530 (ACL 2026; round-2 pass could NOT re-verify this ID, treat as
  unverified): history+model joint embeddings, MLP predicts terminal episode success;
  ACKNOWLEDGES prefix re-processing cost on switch but does NOT model it. The brief's
  v2 predictor style.
- "From Myopic Selection to Long-Horizon Awareness" 2604.12385 (2026): the ONLY paper
  that prices cross-model cache invalidation in routing (sequence-dependent invocation
  cost). Simulation-only, not wired to a real cache or serving layer. Cite as closest
  prior art; do not claim a total gap.
- Serving-layer cache literature (single model, across replicas): SGLang/RadixAttention
  2312.07104, Preble 2407.00023, CachedAttention 2403.19708, GORGO 2602.11688.
  GORGO's cost decomposition (residual prefill after prefix reuse) is the template for
  our switching-cost term.
- Novelty statement that survives review: tying real prefix-cache state to a
  CROSS-MODEL routing decision under a quality/cost/latency objective with an explicit
  switching penalty; benchmarked on multi-turn traffic.

## 3. Objective formalization
- Adopt Hybrid LLM's train-once, slide-threshold-at-test-time knob: one policy, every
  operating point, no retraining. v1 ships one balanced default position; tiers later
  are knob positions, not new machinery.
- Latency enters as a hard constraint (SLA semantics), not folded into the scalar
  (MixLLM's meta-selector pattern).
- Report the full cost/quality Pareto frontier, not a single point (RouterBench,
  Avengers-Pro convention).

## 4. Evaluation conventions (for the algorithm/benchmark chat)
- Baselines quartet: best-single-model, random, oracle (per-scenario best), all-frontier;
  plus RouterBench's Zero Router interpolation floor.
- Metrics: cost-quality Pareto frontier; AIQ (RouterBench 2403.12031, area under the
  cost-quality curve in dollars; fits the value-per-dollar framing); APGR and CPT(x%)
  (RouteLLM) when routing between two tiers. Latency: p50/p95 per scenario, no single
  canonical metric in the literature.
- Held-out: split by scenario (never by turn), ideally by CLUSTER for the OOD row;
  report in-distribution and held-out-category numbers separately.
- Precompute the model x scenario outcome matrix once (RouterBench pattern) so router
  variants compare offline on identical data. Our closed-loop eval stage produces
  exactly this matrix.
- Multi-turn honesty: single-shot eval cost overstates routing gains under prompt
  caching; our benchmark accounts for cache effects and the report states the
  assumption. No published benchmark does this.
- Reusable public assets: RouterBench 405k outcomes (withmartian/routerbench),
  RouterEval 200M+ records (2503.10657), GraphRouter's Router Dataset, RouterArena
  (2510.00202, RouteWorks/RouterArena, live leaderboard), LLMRouterBench (2601.07206,
  ynulihao/LLMRouterBench, re-implements RouterDC/Avengers/RouteLLM baselines).
  DARS caveat applies: these matrices are single-shot labels.

## 5. Failure modes (literature-flagged, mapped to us)
- Judge/label noise, the dominant one. Remedies: repeated-sampling soft labels (Hybrid
  LLM), distribution-aware supervision (DARS 2606.06924), label-free estimation
  (Smoothie) for thin clusters, pairwise/dueling feedback. Matches this repo's own
  env-reward-lottery finding; never compare across unpinned judges.
- Router overfits the eval mix; OOD drop is real (RouterDC numbers). Split by cluster.
- Roster churn breaks static tables (motivates UniRoute model-feature vectors); our
  pool file WILL change, so policy artifacts must record the pool they were fit on.
- Tie degeneracy when several models are all good (RouterDC's motivation).

## 6. Recommendations adopted for this repo
1. v1 = Avengers-style cluster-then-assign over wm scenarios, HashingEmbedder or
   provider embeddings, k-means in numpy; soft labels via repeated sampling where
   budget allows.
2. Objective = balanced quality/cost/latency default behind a single threshold
   parameter; latency as constraint; Pareto frontier in the report artifact.
3. v2 = per-query outcome predictor (MTRouter-style, or per-model correctness
   classifiers per Shnitzer/CARROT), swappable behind the same policy artifact; run the
   2505.12601 locality diagnostic first to justify it.
4. Cache-aware switching penalty in the policy + conversation affinity in serving;
   cite 2604.12385 as closest prior art, borrow GORGO's residual-prefill cost term.
5. Cascade escalation only as a low-confidence fallback, never the primary path.
6. Bandit/online updating deferred until the endpoint generates logs; warm-start from
   the v1 cluster table.
