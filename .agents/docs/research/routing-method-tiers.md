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

## Results (2026-07-24, first benchmark round)

RouterBench 0-shot matrix (36,497 prompts x 11 models), stratified split seed 0
(25,548 fit / 10,949 held-out), fitter = faithful Avengers replication
(k=64, top_k=2, beta=6.0, seed 42). All numbers on the untouched held-out split:

| policy | accuracy | cost/call | note |
| --- | --- | --- | --- |
| best-single (gpt-4-1106, fit-chosen) | 0.7856 | $0.00327 | the bar |
| oracle (per-scenario best) | 0.9138 | $0.00024 | ceiling: +13pts AND 13x cheaper |
| random | 0.5223 | $0.00083 | floor |
| rank, hashing-512, lam=0 | 0.7882 | $0.00264 | beats best-single on BOTH axes |
| rank, hashing-1024, lam=0 | 0.7886 | $0.00253 | |
| rank, hashing-1024, lam=0.02 | 0.7723 | $0.00094 | -1.3pts for 71% cheaper |
| rank, hashing-1024, lam=0.05 | 0.7552 | $0.00069 | |
| rank, hashing-1024, lam=0.2 | 0.6938 | $0.00024 | oracle-cost territory |

Readings: (1) the pure replication clears best-single with cost playing no part in fitting -
the saving falls out of the ~25% of clusters where cheaper models genuinely rank first;
(2) the cost knob (rerank_policy, fit-once-slide) traces a real frontier and answers the
"how much cost for quality" question with data; (3) the gap to oracle (12.5pts) is the
embedder + per-query-predictor headroom - hashing trigram is lexical, and the
text-embedding-3-large comparison run is in flight. Artifacts: .wmh/evals/routerbench/.

## Implementation comparison vs the reference (ZhangYiqun018/Avengers)

Read line-by-line during implementation (core/routing/rank_router.py,
core/generate_rank_router.py); differences, all deliberate and documented in
wmh/optimize/routing.py's docstring:

| aspect | reference | ours |
| --- | --- | --- |
| embed + normalize | external gte-qwen2-7b service + sklearn Normalizer | EmbedderSpec (hashing or Azure deployment) + same Normalizer |
| clustering | sklearn KMeans k-means++/elkan/max_iter=1000/seeded | identical call |
| per-cluster score | correct/total (binary) | mean reward (graded; identical on binary) |
| ranking | accuracy desc, dict-order ties | reward desc, pool-order ties (deterministic) |
| selection | top-k softmax(-beta*dist), sum p/(rank+0.1), missing=1/999 | identical math (shared serve/eval code path) |
| artifacts | centres.npy + rankings.json + joblib normalizer | one versioned policy.json (self-describing) |
| cost | absent | absent at lam=0; stored evidence + rerank knob on top |
| routing output | top-N experts (generator may ensemble) | top-1 (single-model serving); top-N is the tier-3 ensembling step |

No accidental deltas found; the two additive extras (cluster labels for the request log,
cost evidence for the knob) do not alter lam=0 behavior.

## AIQ vs the non-predictive floor (2026-07-24)

RouterBench's own headline metric, computed with their normalization (hull area / shared max
cost) on the held-out split: **rank-router AIQ 0.7447 vs Zero-Router 0.7001** (Zero-Router =
hull of the 11 single models + their random mixes, the best label-free strategy). The fitted
hull dominates the floor across the interior cost range, e.g. at ~$0.00094/call the router
holds 0.772 accuracy where price-mixing toward gpt-4 interpolates ~0.70. Prediction is
genuinely adding value beyond price interpolation. Caveat carried from their code: AIQ's
normalization couples it to the comparison set's max cost, so we always publish hull points
alongside the scalar.

## Closing cross-check facts (2026-07-24)

- Avengers: every evaluate/* harness passes the RAW prompt into `router.route(question)`
  (grep across evaluate/*.py); no preprocessing before embedding, so the replication's
  query path matches. Paper-era artifact names pin k=64 (ranking_centers_split_k64_m22),
  config template pins top_n=2/top_k=2/beta=6.0.
- RouterBench licenses: code MIT (LICENSE carries a stray LangChain copyright line, likely a
  template slip); the HF dataset carries NO license tag. We consume it for internal
  validation and cite it; do not redistribute the data in any public artifact.

## Embedder comparison (2026-07-24): hashing ties text-embedding-3-large on RouterBench

Identical 12k-prompt subsample (seed 7), identical split, knob=0: best-single gpt-4 0.7937 @
$0.00318; hashing-1024 0.7935 @ $0.00240; azure text-embedding-3-large (3072d) 0.7930 @
$0.00262. The semantic embedder bought nothing, and its fit leg took 33 minutes against the
rate-limited deployment vs ~12s for hashing.

WHY (cluster audit of the fitted policy): hashing's 64 clusters align cleanly with dataset
identity (24 hellaswag clusters, per-subject mmlu clusters, ...) because RouterBench prompts
carry strong lexical format signatures, and dataset identity IS most of this benchmark's
routable signal (gpt-4 ranks first in 54/64 clusters; the routing win lives in the 10 clusters
led by Yi-34B/claude-v2/mixtral). LIMIT: this does not transfer to wm-scenario corpora, where
all scenarios share one format and the routable structure (if any) is semantic. Decision:
hashing stays the default for stage A/B; the comparison RERUNS per corpus in stage C before
any product-default call. Follow-up noted: compare full knob-swept frontiers, not just knob=0.

## Stage B (2026-07-24): OUR 9-model pool on certified RouterBench MCQ, exact-match

Setup: gold recovered from the matrix by consensus (23,354/26,821 certified, 87.1%, zero
parse conflicts); 1,199 stratified prompts x 9 models = 10,791 calls, $17.77, judge-free
grading; ~0.5% error rows (Azure content filters), recorded unscored. 843 fit / 356 test.

Single models on test (cost/call, accuracy): gpt-5.4-mini $0.00015/0.827; haiku-4-5
$0.00023/0.888; deepseek-v4-pro $0.00033/0.836; sonnet-5 $0.00084/0.941; opus-4-8
$0.00131/0.949; glm-5.2 $0.00242/0.812; kimi-k2.6 $0.00248/0.805; fable-5 $0.0035/0.904;
gpt-5.5 $0.00351/0.955. Oracle 0.9916 @ $0.00019.

HONEST HEADLINE: on these saturated 2023-era MCQ benchmarks, routing does NOT beat
price-mixing with our pool. AIQ ours 0.8726-0.8802 (k swept 4-64) vs Zero-Router 0.9014;
sonnet-5 alone (0.941 @ $0.00084) anchors a singles hull the fitted frontier stays under.
Best routed point (k=32, knob=0): 0.9607 @ $0.00236 - above best-single gpt-5.5 (+0.6pt,
within the ~1.7pt noise floor of 356 scenarios) at 33% lower cost and faster (routed p50
0.8-1.9s vs gpt-5.5 p50 1.87s / mean 2.33s), but single points do not beat the hull.

WHY, and why this does not kill routing: 2026 models are near-ceiling and almost TOTALLY
ORDERED on 2023 MCQ (every model 0.81-0.955); per-cluster specialization barely exists, and
843 scenarios / 64 clusters = ~13 per cluster fits rankings on noise (the no-support-threshold
characteristic + DARS 2606.06924's warning, live). Contrast stage A: the 2023 pool had REAL
specialization (Yi-34B led whole clusters over gpt-4) and routing beat both best-single and
the floor. Routing pays where models specialize; it cannot pay where quality is totally
ordered and the data is saturated.

Consequences: (1) the fitter is validated (stage A) and the honest-benchmark machinery does
its job - on saturated corpora the improvement report should recommend a STATIC policy
(sonnet-5 at 4x under gpt-5.5's cost, or haiku at 15x under with -6.7pts) and say so;
(2) the decisive test for the product is stage C, wm scenarios from customer traces, where
specialization demonstrably exists (the bird-sql smoke: haiku 1.0 vs gpt-5.4-mini 0.0 on
identical scenarios); (3) fitter iteration queue: per-cluster support thresholds, k selection
on validation, DARS-style multi-sample labels.

## LLMRouterBench flagship track (2026-07-24): the modern-benchmark validation

Per Silen's pointer, github.com/ynulihao/LLMRouterBench (2601.07206; NOT what Avengers used -
it postdates Avengers and re-implements it as a baseline). Its performance-cost track = 13
flagship 2025 models with measured costs on hard datasets (AIME, GPQA, HLE, LiveCodeBench,
arenahard, ...). Our fitter on the shared-coverage matrix (1,560 scenarios, 70/30 split seed 0,
hashing-1024, k=64):

| point | accuracy | cost/call | PerfGain | CostSave |
| --- | --- | --- | --- | --- |
| best-single (gemini-2.5-pro) | 0.7938 | $0.04770 | - | - |
| rank, lam=0 | 0.7895 | $0.00951 | -0.5% | +80.1% |
| rank, lam=0.02 | 0.7799 | $0.00495 | -1.7% | +89.6% |
| rank, lam=0.1 | 0.7682 | $0.00232 | -3.2% | +95.1% |
| oracle | 0.9861 | $0.00242 | +24% | +95% |

AIQ ours 0.7652 vs Zero-Router 0.7506: the router BEATS the price-mixing floor here, unlike
saturated stage B - this matrix is unsaturated and specialized (qwen3-235b $0.001/0.740 vs
claude-sonnet-4 $0.020/0.544), so routing has real signal to harvest. Consistent with the
paper's own findings (top routers ~= each other; gains from coarse domain structure; some
routers fail to beat Best Single on accuracy; Avengers-Pro wins the Pareto). Oracle at 0.986
shows the tier-2 predictor headroom. Full comparison against their published Avengers rows =
next (their baseline configs/seeds), but the replication is squarely in the leading-router
band on their data.

## Avengers successors (2026-07-24 lit review round 3, citation graph + deep dive)

The Avengers lineage continued; per-paper deltas verified, code links checked:

1. **JiSi / "Beyond Gemini-3-Pro" (2601.01330) - the Avengers authors' own successor**
   (Yiqun Zhang + Shanghai AI Lab core team). Fixes three Avengers limits: query-only routing
   (adds query-response MIXED routing: semantics + problem difficulty in the embedding),
   static aggregation (support-set aggregator selection), separate route/aggregate (per-query
   route-vs-aggregate switch). Router-only head-to-head: 69.68 vs Avengers 68.74; full system
   72.15 avg beats Gemini-3-Pro 71.00 at 53% lower cost. Code: github.com/magent4aci/openJiSi.
2. **IrtNet (2510.00844) - strongest direct challenger.** IRT ability x difficulty latent
   model replaces per-cluster reciprocal-rank tables; beats Avengers-Pro head-to-head 67.4 vs
   62.1 routing accuracy (35k queries, 112 models) and needs <4% of the training data (the
   sample-efficiency answer to our thin-cluster noise). Code: github.com/JianhaoChen-nju/IrtNet.
   (Related: IRT-Router 2506.01048 adds the monotonicity constraint + cold-start warm-up.
   NOTE: "MonoRouter" from an earlier agent pass is FABRICATED - does not exist.)
3. **ProxRouter (2510.09852, CMU)** - exponential-tilt reweighting bolted onto existing
   k-means/kNN scores (w ~ p*exp(-phi/tau)); +2.8 to +8.1pp OUTLIER AUC with inliers
   preserved. Best effort-to-payoff for customer traffic drift. No code found.
4. **Federate the Router (2601.22318, same CMU group)** - federated routing from sparse
   decentralized evals, k-means variant composes with our clustering: pool per-cluster stats
   ACROSS customers to beat isolated thin-data tables. Our multi-tenant setting exactly.
5. Also: MetaRouter (2606.06178; learned per-user preference replaces the static alpha, beats
   Avengers-Pro on hypervolume/IGD), Mixture of Thoughts (2509.21164; latent top-K
   aggregation, +2.92% OOD over Avengers, code github.com/jacobfa/mot), EvoRoute (2601.02695;
   online experience base + Thompson sampling, per-step agentic), MoMA (2509.07571; trained
   MoE router + TOPSIS knob), RouteJudge/ORBIT (2606.18774; eval harness with Avengers rows,
   code github.com/LAMDA-Model-Reuse/ORBIT).

Synthesis for our roadmap: tier-1.5 = ProxRouter tilt (drop-in robustness); tier-2's concrete
form = IrtNet-style IRT ability model (sample-efficient, answers thin-cluster noise, open
code) with IRT-Router's monotonicity for cold start; tier-3's concrete form = JiSi's
route-vs-aggregate switch + MoT latent aggregation; multi-tenant lever = federated per-cluster
stats. Cache-aware multi-turn cluster routing remains whitespace nobody claimed - still ours.

## CORRECTION (2026-07-24): LLMRouterBench text-duplication leak, caught by the audit loop

The first IRT run on LLMRouterBench flagship posted +12pt over the rank router - too good, so
it went through the leak audit: zero id overlap, shuffled-label control collapsed (0.698,
no reward leak), BUT 327/468 test scenarios had task text appearing VERBATIM in the fit split
(the arenahard subsets categorize the same prompts across dataset dirs; overall 1,560 -> 809
unique scenarios, nearly half duplicated). Trigram-hashing embeddings make text dupes
near-exact retrieval keys, so the learned head profited most. Adapter now dedupes by task
text (first occurrence wins, logged; regression test), tainted runs purged.

CLEAN flagship numbers (809 scenarios, 566 fit / 243 test, zero residual overlap): best-single
gemini-2.5-pro 0.8045 @ $0.05284; rank lam=0 0.7202 @ $0.01592; IRT lam=0 0.7449 @ $0.01643
(+2.5pt over rank ~= the +-2.5pt noise floor at n=243 - directional, matching IrtNet's
published direction, not conclusive here); at lam=0.1 both ~0.68 @ $0.0024 (95% cheaper).
Honest reading: on this small deduped set, routing is a cost-saver (-6-8pt for -70-95% cost),
not an accuracy win. Dup audit on the other matrices: RouterBench classic 16/36,497 (0.04%,
tables stand), ours9 0/1,199 (clean). Earlier pre-dedupe LLMRouterBench rows in this doc are
SUPERSEDED by this section.

## Credibility correction + hill-climb structure (2026-07-24, Silen review)

Citation screen (which the earlier lit passes skipped): Avengers (Shanghai AI Lab, AAAI'26)
and ProxRouter (CMU) are the established methods; JiSi (2601.01330) and IrtNet (2510.00844)
are <5-citation 2026 preprints from unestablished groups. Corrected stance: jisi/irt/tilt are
RESEARCH DIRECTIONS whose empirical results here are the evidence, not their papers.

Structure going forward: three specialist chats (transfer prompts routing-common/r1/r2/r3 in
~/Downloads/wmh-plan-transfer-prompts/) hill-climb retrieval (r1), cluster (r2, keeper of the
credible Avengers/ProxRouter line), and learned-ability (r3) families against the shared
interfaces: matrices + runs + findings under ~/Desktop/Projects/wmh-routing-data/,
evaluate_choices/RunRecord as the single scorer, 5-seed spreads, margin guards, leak audits,
and look-at-the-outputs discipline (all binding via routing-common.md). The master chat
(optimization) owns interfaces, benchmark sanity, cross-chat comparison, and promotion into
wmh/optimize; next master tasks = drawing-board survey beyond these three families and an
empirical audit of the benchmarks themselves.
