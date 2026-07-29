# tb2-cost findings: the product pipeline, end to end, on the terminal-tasks corpus

One page, honest-deltas style. Everything below is `wm_simulated`, judged by the
terminal-tasks build's own rubric (opus-4-8), scorecard accounting (cache-adjusted
effective cost per COMPLETED task, compressor bill folded in, unscored spend excluded and
itemized), paired per-scenario CIs from `common/stats`. Workload relabel binding: this is
the org's HF terminal-tasks bundle (280 one-shot bash-agent tasks), NOT Terminal-Bench 2
(see README.md). Noise floor +-2pt. Numbers: `numbers.json`. Figures render through
`common/build_corners.py --lens tb2-cost` only.

## What was run (the pipeline itself, public commands only)

`wmo build --fidelity medium` (280 traces -> 685 steps; held-out fidelity 0.808, GEPA
+0.024 over base 0.784; $6.80) -> `wmo optimize model` identity arm (13-candidate pinned
pool x 20 scenarios x 2 episodes; $49.90 all-in, measured env/candidate ratio 4.6x) ->
compressed arms llmlingua2-endpoint @ dial 0.5 ($64.14) and truncate @ 0.22 matched on
llmlingua2's MEASURED achieved ratio (partial; see caveat 2) -> fit/tune/report. Total
program spend $151.06 including a $3.59 written-off re-buy attempt into a Bedrock outage.

## Headline: on a one-shot terminal workload, model selection is the whole win

- The fable-5 anchor is deeply dominated: **opus-5 is -63.0% cost at +6.7pt quality with
  the paired CI excluding zero** (+0.4..+13.6pt, n=20) - resolved better AND cheaper.
  sonnet-5 (-88.2%, +7.2pt, CI -0.1..+17.7) and kimi-k3 (-77.0%, +7.3pt) sit at the same
  quality point estimate for 2-8x less; their CIs graze zero at n=20.
- The 40-50% SLA band is cleared at parity by most of the pool: haiku-4-5 -96.1% (+2.6pt,
  within noise), gpt-5.4-mini -98.2% (-2.2pt, within noise), qwen3.6-27b -98.4% (+0.6pt,
  14/40 cells only).
- Cheapest is not free: qwen3.5-9b's -99.0% comes with quality resolved WORSE (-12.1pt,
  CI -23.3..-3.5) - the honesty row against "just use the smallest model".
- **The guarded router agrees**: the fitted knn policy picked sonnet-5 as fallback and
  routed 0.0% of requests away from it. On a workload where one mid-price model is at the
  ceiling, the correct learned policy is a constant - the product declining to add
  complexity is the result, not a failure. (Contrast tau, where the routed rung is the
  open question; here it is answered flatly.)

## Compaction: the cost-inversion reproduces on terminal one-shots

Same-model, arm vs identity (both vs the raw anchor, so the deltas compose): gpt-5.5 goes
-68.7% (identity) -> -29.3% (truncate) and -25.3% (llmlingua2): compression made it ~2-2.4x
DEARER per completed task. sonnet-5: -88.2% -> -41.9% under llmlingua2 (~3.9x dearer).
haiku-4-5: -96.1% -> -91.1% (~2.3x). Only the already-cheapest models stay flat (deepseek
-97.1% -> -96.9%). Mechanism matches tau/financebench: compressed one-shot prompts lengthen
episodes and depress success, and on THIS corpus the entire prompt is the task statement -
exactly the segment the compression lane's seam rule says never to compress. Fresh
calibration datum: llmlingua2 @ dial 0.5 achieves ratio 0.777 mean here vs 0.566 on tau -
per-corpus calibration is mandatory (tau's dial is not portable). Labels stay "measured
tradeoff, not a recommendation".

## tau-vs-terminal contrast (the cross-corpus frame)

tau (cost corner, same runner): best-single opus-5 -59% at +10.9pt; routing still in play;
compression inverts. terminal: bigger spread (-63% to -98% at parity-or-better), routing
degenerates to best-single, compression inverts harder. WM fidelity is corpus-dependent
(0.808 here vs 0.249 tau rebuild). The distill lane's TB2 story (+27pt teacher gap) does
NOT attach to this corpus; the teacher gate run descriptively on this matrix says
existence gap kimi-k3 +19.4pt over qwen3.5-9b (CI +6.6..+32.1), cheapest sufficient
teacher sonnet-5 keeping 99% - recorded verbatim from `wmo.optimize.teacher`, and NOT
acted on: distillation is skipped by Silen's ruling (a distilled model from the earlier
TB2 work already exists on HF).

## The REAL Terminal-Bench 2 leg (Silen directive 2026-07-28: real harness + sim-to-real)

All 13 pool candidates ran the REAL TB2 17-task holdout through harbor terminus-2 on E2B
(1 attempt, litellm/tinker thin config, the distill lane's pins; timeouts and content
refusals count as failures per the benchmark's own convention; 6 of 221 cells are
persistent-infra holes, all on the two weakest models). Data:
`real_tb2_leaderboard.json` (+ p50 episode seconds and $/task), raw trial dirs preserved
at main checkout `.wmo/tb2-real/`, corpus + matrices at `.wmo/jt/tb2real/`.

- REAL leaderboard headline: fable-5 15/17 (88%) at $6.50/task; gpt-5.5 and opus-4-8
  76% at $0.93/$1.82; sonnet-5 71% at $0.72; tail: haiku-4-5 41%, gpt-5.4-mini 36%,
  qwen3.5-9b 20%. REAL quality headroom exists here (52pt spread) and quality costs real
  money - the inverse of the terminal-tasks one-shot corpus.
- HARNESS VALIDATION: the qwen rows reproduce the distill lane's baselines within noise
  (qwen3.6-27b 53% vs 49.0%; qwen3.5-9b 20% vs 21.6%; n=17 vs their 51 trials) - same
  pins, same renderer, independent run.
- THE SIM-TO-REAL RESULT, negative and kept: a WM built from these 201 real episodes
  (`wmo build` medium, fidelity 0.241, $24.07) and swept with the same 13 candidates
  (104 cells, ~$142 all-in, env/candidate ~1.0x) produces model rankings that DO NOT
  track the real benchmark: Spearman rho = -0.162 (n=12 models;
  `real_tb2_rank_agreement.json`), where tau's sim-to-real was +0.639. The sim winner is
  opus-4-8 - the WM's own serve AND judge model (sim 0.769 vs real 76%), while sonnet-5
  (real 71%) sims at 0.287 and haiku-4-5 (real 41%) sims at 0.669. Named hypothesis, not
  proven mechanism: self-model affinity in the serve/judge loop plus a scenario band too
  thin to average over (8 scenarios x 1 episode).
- WHY THIS MATTERS for the corner: on workloads where the WM's fidelity is low (0.24
  here), the routing matrix is not yet evidence about real deployment - the honest
  pipeline must gate routing claims on a sim-to-real check, exactly the machinery this
  leg demonstrates. tau passes that gate; TB2 with a 201-trace corpus does not (corpus
  size, judge calibration, and self-model bias are the three candidate fixes, in
  measurement order).

## Caveats, named

1. **Judge**: wm rewards from the build's own rubric; the fleet-wide wm-scorer autopsy
   (2026-07-28) flagged one-shot-answer corpora specifically. Here failed episodes were
   EXCLUDED as unscored (never hard-zeroed), but reward calibration on this corpus has not
   had a meta-eval pass; quality deltas are wm_simulated, not real-benchmark solve rates.
2. **Truncate arm is partial**: 149/520 scored (a Bedrock ServiceUnavailable window killed
   the premium tier mid-sweep; fable-5 n=3, sonnet/opus/haiku/glm/kimi-k3 absent; a $3.59
   re-buy attempt hit the same outage - 421 ServiceUnavailable - and was stopped). Its
   rows are labeled with their n; the truncate-vs-identity inversion claim rests on the
   four well-covered models (gpt-5.5, gpt-5.4-mini, deepseek, kimi-k2.6 at n=15-20).
3. **qwen rows**: OpenRouter account has zero purchased credits (free tier); qwen3.6-27b
   is 14/40 on identity and both qwens are absent from compressed arms (402s). ~$5-10 of
   credits closes every gap.
4. **Coverage repair**: the product has no re-buy path for errored cells in a completed
   sweep (`--repair` is a filed gap); `--force-from sweep` re-buys everything and was the
   only lever. Uneven-coverage fits used the product's own `--allow-uneven-coverage` with
   the bias printed; the identity fit's bias was immaterial (nothing routed).
