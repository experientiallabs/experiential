# The canonical tau-bench result: routing, compaction, and distillation, jointly

> DRAFT - numbers marked [PENDING] slot in from the grid, the corner analyses, and the
> real-episode validation. Publication is gated (DECISIONS 2026-07-27) on the three corner
> analyses and the cost corner's verdict. Nothing in this document may be quoted externally
> until that gate lifts and this banner is removed.

## The claim this document substantiates

world-model-optimizer takes an agent's traces and returns an OpenAI-compatible endpoint that
serves the same workload cheaper at comparable quality. Behind the endpoint, three optimizers
compose: a learned inference policy (routing) chooses a model per conversation, compaction
compresses request context before the call, and distillation trains a small personal model
that earns traffic as it improves. This document is the measured, end-to-end demonstration of
that composition on one public benchmark, with every number's provenance labeled and every
negative result kept.

## Why this is credible (the method, before the numbers)

1. Simulated and real are never blurred. The world model built from tau-bench traces supplies
   the closed-loop evidence for FITTING policies (its rank agreement with reality was measured
   before it was trusted: the sim picks the real winner, and errs pessimistic). Every headline
   quality/cost claim is then checked on REAL tau2 episodes through the same serving path a
   customer uses. Numbers below carry [wm] or [real] tags.
2. Cost means cache-adjusted effective cost per completed task. Not per-token price: a model
   that is cheap per token and fails tasks, or one that forfeits the provider prompt cache, is
   expensive in the only sense that matters. The metric folds the compressor's own inference
   bill and reports unscored spend rather than hiding it. (The dumb-compression controls in
   the companion accuracy study RAISED effective cost while lowering per-token cost - the
   metric exists because that inversion is real.)
3. The controls are matched and the gates are pre-registered. Compression arms include a
   truncation control matched on ACHIEVED token ratio; the distillation gate and its holdout
   were fixed before any training spend; the router's guard makes its worst case the fallback
   model by construction.
4. Failure is reported as failure. The first distillation cycle was REJECTED by its own gate -
   the teacher had no headroom over the student on this benchmark - and that verdict, its
   $34.94 cost, and its diagnosis are part of this record, not an embarrassment excised
   from it.

## The pipeline (one command per stage; the cookbook walks it)

traces.otel.jsonl -> `wmo build` (the world model) -> `wmo providers set` (the candidate pool)
-> `wmo optimize model <name> [--compressor <id>] [--distill <config>]` (measure, fit, tune,
report) -> `wmo serve` (the endpoint). See docs/cookbook/tau-bench.md for the artifact-by-
artifact walk. Every stage in this study ran through those public commands or the library
functions they wrap.

## The evidence

### The ablation ladder [PENDING: figure + table from the corner analyses]

Rungs, each vs the fable-5 anchor on the same scenarios, quality / effective cost / latency:
distill-only; +routing; +routing+compaction; with leave-one-out ablations. WM-simulated
[wm], 20 held-out scenarios x 11 candidate models x 2 episodes per arm; judge rubric-v2
(opus-4-8); compaction rungs labeled "measured tradeoff, not recommendation" pending the
compression track's accuracy verdict.

### Quality across training stages [PENDING: the shared corners chart]

Student quality by training stage with the three lever ablations. Cycle 1 (teacher
Qwen3.6-27B) appears as measured: no promotion - teacher 73.3% / student-before 71.7% /
student-after 65.0% at k=3 [real], paired sign test p=0.45, verdict "no measurable effect at
this sample size; nothing to distill from a peer teacher". Later stages [PENDING: K3 cycle,
gated on its headroom probe and Silen's go].

### The three corners [PENDING: one subsection per corner analysis]

Quality-max / cost-max (savings) / latency-max as named, mountable policy configurations,
each reporting all three objectives. The latency corner is an offline mount choice - no
online latency-aware routing rule exists yet, recorded as a limitation.

### Real-episode validation [PENDING]

The named corners and the anchor, run on real tau2 through the served endpoint (the user
path: tau2 -> OpenAI-compatible endpoint -> compress -> route -> provider). Preceded by the
WM-vs-real difference probe that decides how much of the corner analysis the world model can
carry alone [PENDING: probe result].

### The compound loop [PENDING: traffic-share and effective-cost vs training step]

The plot this product sells: as the personal model improves, the router measurably shifts
traffic to it and effective cost falls at held quality. Status: infrastructure demonstrated
end to end (train -> gate -> pool entry -> refit -> serve); the first measured shift awaits a
cycle that gates. If no cycle gates on this benchmark, this section reports that plainly:
the gate refusing to promote a non-improvement IS the mechanism working.

## Honest limitations (standing, whatever the numbers say)

- tau-bench's corpus yields 20 distinct held-out scenarios; the routing bank is small, so the
  routing rung's claim is cost-at-parity under a guarded policy, not accuracy lift. The
  evidence-volume law from the routing research (routability emerges near ~1000 scenarios)
  says larger corpora are where routing accuracy gains live.
- The WM-simulated ladder is ~85% telecom (the corpus's mix); the real-episode leg is the
  balanced check.
- 7 of the 20 holdout tasks include tau2's own NL-assertion judge in their reward; rows
  record the fully-deterministic subset separately.
- The dial's five measured anchors were calibrated on routerbench-ours9 and are quoted as
  such until re-measured jointly on this grid (D-DIAL v2).
- Sim-to-real agreement is quoted as per-scenario paired sign agreement; model-mean rank
  correlations at n<=9 models sit inside their null noise band and are descriptive only.

## Reproduce [PENDING: exact pins - corpus sha, model dir config, pool, tips, seeds, judge]

## Provenance of this document

Written by the joint-tau integration effort, 2026-07-27. Companion documents:
docs/cookbook/tau-bench.md (the how-to), docs/usage.md (the CLI map),
docs/research/world_model_findings.md (the layer-by-layer research record).
