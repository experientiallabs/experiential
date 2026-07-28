# COST-MAX corner: where the savings actually come from

> Status 2026-07-28 (post-repair): identity and truncate matrices are FINAL (440/440
> scored); llmlingua2-endpoint is mid-straggler-pass (last merge 409/440, matrix currently
> re-merging; its numbers are labeled provisional). The identity-arm routed rung is
> measured (fit + dial replay on the held-out band). Coverage travels on every number.
> Publication of anything on this page is gated (DECISIONS 2026-07-27, corner chats entry).

## The accounting rule every number obeys

All cost aggregation on this page comes from `wmo.optimize.scorecard` and nothing else. It
implements the binding D-COMPRESS rule: **cache-adjusted effective cost per COMPLETED
task**, compressor/router inference folded in as overhead, unscored episodes excluded from
numerator and denominator and their spend reported, money conserved on the artifact.
Per-token price is not cost: an arm that fails half its tasks is cheap per run and ruinous
per completed task, and the compaction lane already measured the inversion this metric
exists to catch (below).

Every number is labeled `measured` or `estimate`, `wm_simulated` or `real_episode`, with
the judge named. The grid's WM cells are judged by rubric-v2 pinned on opus-4-8; cycle 1's
training numbers are real tau2 episodes under tau2's own reward.

## The savings decomposition (the honest shares)

Savings behind an optimized endpoint can come from four places. Their current shares:

1. **Model-selection share: today this is the WHOLE measured savings story** [measured,
   wm_simulated, judge rubric-v2 (opus-4-8), 20 scenarios x 2 episodes]. Against the named
   fable-5 anchor ($0.958 per completed task here), the single-model swap already clears
   the SLA band: opus-5 is **-59.0% effective cost at +10.9 pt quality** (paired CI +0.3 to
   +21.9 pt, excludes zero: better AND cheaper), and gpt-5.5 is **-59.2% at +2.8 pt**
   (CI -7.6 to +12.6, parity). Cheaper models buy more savings only by paying quality:
   glm-5.2 -80.5% at -5.8 pt, haiku-4-5 -89.8% at -14.3 pt, gpt-5.4-mini -93.0% at
   -34.1 pt. The routed rung on top of this is measured (see the routed bullet in the 40%
   section): the router discovers opus-5 automatically and holds best-single parity; the
   dial's savings leg does not pay on tau's ~14-scenario bank, per the small-bank law. On
   routerbench-ours9 (different corpus, quoted as measured there, never blended), the
   dial runs -13.9% (quality max, +1.14 pt) to -46.2% (max savings, -0.54 pt) vs ITS best
   single model.
2. **Compression share: the cost inversion on strong models is CONFIRMED for the dumb
   control and MILD for the learned compressor** [measured, same provenance; compressor
   bill folded as RowOverhead. CORRECTION 2026-07-28: the first quote of this section rode
   pre-repair matrices with outage-shaped missingness; the master's re-render flag was
   right, and the repaired numbers moved a lot]. On the FINAL truncate arm (440/440),
   truncation raises fable-5's effective cost **+47.1%** at -8.9 pt quality (n=20; the
   pre-repair read was +23.6% at n=15). On the repaired llmlingua2 arm (mid-straggler-pass,
   provisional), fable-5's inversion collapsed from the outage-inflated +146.2% to
   **+6.7%** at -8.1 pt (n=20): the learned compressor is dramatically better than the
   matched-keep dumb control on cost, while still not a net win on strong models (quality
   pays). The financebench mechanism (deleting load-bearing observations fails tasks and
   lengthens episodes) still shows in truncate's latency column (p50 +170-480% on several
   models). Per-token accounting would have called every arm cheaper; effective cost per
   completed task catches the inversion. Compaction rungs stay "measured tradeoff, not
   recommendation" pending that lane's accuracy verdict.
3. **Student share (distillation): zero, and zero BY PROGRAM DECISION.** Silen's ruling
   (2026-07-28, this chat): distillation is not being pursued. Cycle 1's gate had already
   REJECTED the warmup adapter (teacher Qwen3.6-27B 73.3% vs student base 71.7% at k=3: no
   headroom; before-vs-after p=0.45 at n=60), and no unpromoted student ever enters the
   pool, so no served token is cheaper because of training and none is planned to be. The
   distill-only rung reads "no measurable effect at this sample size", never a lift and
   never a regression. The teacher-gate verdict on this grid (DISTILL: cheapest sufficient
   teacher gpt-5.5 at $0.3907/completed task, keeping 82% of opus-5's +45 pt gain over
   gpt-5.4-mini) is recorded in `numbers.json` as the repo function's descriptive output,
   NOT as a plan. Teacher-selection economics are OWNED
   by `wmo.optimize.teacher.select_teacher` (#329, merged): this page cites its verdict and
   never hand-computes teacher economics. PRICE-ORDERING REVIEW against this corner's
   scorecard conventions (2026-07-27, cost chat): ALIGNED. The primary ladder IS
   `scorecard.effective_cost_per_completed_task` over each model's own rows, so it inherits
   cache-adjusted pricing, the per-completed-task denominator (a chatty teacher that fails
   tasks ranks expensive), unscored-row exclusion, and measured-$0-is-missing-not-free; the
   fallback is a LIST-price ordering key (input+output per Mtok), used only when any model
   lacks a measured figure, applied to the WHOLE ladder (one basis, never mixed) and stamped
   on the verdict as `price_basis`. One caveat worth carrying: the list-price fallback is
   not cache-adjusted, so on cache-dominated workloads two adjacent models could in
   principle order differently than their real serving cost; any verdict quoted here with
   `price_basis="list"` says so. The z=1.96-at-n=8 nit is the master's logged note; it
   matches the program-wide convention. Current live verdict (grid-c2 partial, cited
   verbatim in numbers.json): INSUFFICIENT EVIDENCE, leading candidate opus-5 at +32.0
   points over gpt-5.4-mini on only 5 shared scored scenarios, below the 8 the gate
   requires: the gate refusing on thin evidence is the mechanism working.
4. **Cheap-model-was-already-fine share (the anchor's weakness)**: the share of "savings"
   that any cheap model would have delivered because the anchor is overkill for part of the
   workload. This is real money but not an optimizer achievement, which is why every delta
   is reported against BOTH anchors (next section).

## The two-anchor discipline (inherited from the routing lane)

RULING (Silen, 2026-07-28, this chat): **fable-5 is the anchor for ALL baselines, routed
configs included**; the dominated-anchor situation is understood and accepted. The
best-single column stays on every artifact as the check, per the rest of this section.

### The "is anything cheaper than opus-5" check (Silen ask, measured)

Seven models are cheaper than opus-5 per completed task on this grid; every one pays
quality for it. Five are RESOLVED worse (paired CI excludes zero): gpt-5.4-mini (-82.8%,
-45.0 pt), haiku-4-5 (-75.0%, -25.2 pt), deepseek-v4-pro (-67.5%, -29.0 pt), glm-5.2
(-52.4%, -16.6 pt), opus-4-8 (-8.6%, -15.0 pt). Two are UNRESOLVED at this sample size
(CI spans zero): **kimi-k2.6** (-10.0%, mean -11.8 pt, CI -28.5..+4.5) and **gpt-5.5**
(-0.5%, mean -8.1 pt, CI -18.1..+1.5). Neither is a measurable cheaper-at-parity win:
gpt-5.5's saving is negligible, and kimi-k2.6's real 10% saving comes with p50 246s per
task vs opus-5's 38s (6.5x slower) plus an unresolved but negative-leaning quality mean.
Resolving kimi-k2.6 properly needs more episodes than the cohort pin allows; recorded as
an open cell, not bought unprompted.

Savings quoted against a weak anchor overstate. The headline anchor is **fable-5** (named,
$5/$25 per Mtok class): the frontier reference a customer would otherwise be paying for.
Alongside it, every config is also scored against the **best single pool model by mean
reward on this grid** ([PENDING: name]; on ours9 the equivalent baseline was the bar the
dial's -46.2% was measured against). If the best single model is much cheaper than fable-5
at similar quality, the fable-5 column reads large for reasons that are partly
model-market facts, not optimization; the best-single column is the defensible optimizer
contribution. Both columns ship in `numbers.json` and on the charts.

## What the 40+% target reads as against measurement

The SLA promise is "always at least ~40-50% cheaper than the frontier reference at quality
within a small tolerance".

- Against the **fable-5 anchor** on tau [measured]: CLEARED with headroom by single-model
  selection alone. gpt-5.5 delivers -59.2% at quality parity (+2.8 pt, CI spans zero);
  opus-5 delivers -59.0% while being BETTER (+10.9 pt, CI excludes zero). The 40-50% band
  is not the frontier of this workload; it is comfortably inside it.
- Against the **best-single anchor (opus-5)** [measured], the honest optimizer-contribution
  reading: NOTHING measured beats it yet. opus-5 dominates this grid (it is the quality
  max AND cheaper than fable-5 at $0.393/task), so every other single model saves money
  only by losing quality (gpt-5.5: -0.5% at -8.1 pt), and the compressed arms lose on both
  axes for strong models.
- **The routed rung [measured, identity-arm fit, 6 held-out scenarios, embedding replay
  ~$0.001 logged per the 2026-07-28 spend ruling]**: at the quality and balanced dials the
  router routes ALL six eval scenarios to opus-5, i.e. it automatically discovers the
  dominant model and delivers best-single parity (-67.3% vs fable-5 at +12.5 pt on ITS
  eval band; CI spans zero at n=6; NOT comparable to the 20-scenario single-model numbers,
  different scenario set). Past the balanced point the cost leg routes one scenario to
  kimi-k2.6 and gets WORSE on both axes vs opus-5 (+13.2% cost, -6.25 pt): on a
  ~14-scenario bank the dial's savings leg does not pay, exactly what the small-bank
  caveat predicted. The routing rung's honest tau claim is **cost-at-parity with the best
  single model via automatic discovery**, not savings beyond it (ours9's 1199-scenario
  bank is where the savings leg earns -24.7% to -46.2%).
- HONESTY NOTE the headline must carry: fable-5 is a WEAK anchor on this workload; it is
  dominated by opus-5 on both axes. A "-59% cheaper than frontier" claim that names
  fable-5 is true and clears the SLA, but the defensible optimizer contribution is the
  best-single column, where the measured answer today is "pick opus-5, which the router
  finds automatically". RULING: fable-5 leads all baselines (Silen 2026-07-28); this
  column stays adjacent as the check.
- The sales frame stays "run 10x more for the same budget" (an **estimate** derived from
  measured per-task cost, and labeled as such wherever quoted; vs fable-5 the measured
  multiplier at parity is ~2.5x, not 10x, on this workload).

## Standing caveats on every number here

- WM-simulated cells are ~85% telecom (the corpus mix); the real-episode leg (pinned
  balanced 20 through the served endpoint) is the check, and the master's WM-vs-real probe
  decides how much this corner's analysis the WM can carry alone.
- Latency figures are per-task MODEL seconds (call_seconds excludes env/tool time), so they
  understate wall clock and flatter prompt-shortening optimizers; they are read jointly
  with cost per completed task, never alone.
- 7 of the 20 holdout tasks include tau2's NL-assertion judge in their reward basis.
- Episodes=2 per cell: per-cell variance is halved vs e1 but the noise floor on per-model
  quality deltas remains material at n=20 scenarios; paired stats over shared cells, not
  headline means, carry the load-bearing claims.

## Artifacts

- `lens.py`: this corner's declarative figure spec, rendered by the ONE shared runner
  (`common/build_corners.py`, charter Amendment): scorecard-only cost aggregation,
  paired-delta evidence on every quality claim, the distillation verdict cited verbatim
  from `wmo.optimize.teacher.select_teacher`. Rerun
  `uv run python .agents/docs/research/corners/common/build_corners.py --lens cost` as
  grid-c2 cells land (live #330 sidecars load before any matrix merges); routed rungs
  attach via `rows_for_policy` when the master's per-arm fits are delivered.
- `figures/dial_cost_curve.png`: ours9 dial anchors as measured (tau panel pending fits).
- `figures/training_stage_cost_lens.png`: the shared training-stage chart, cost lens
  (cycle 1 as measured, real_episode; cost deltas attach when the grid's student cells
  land).
- `figures/savings_vs_fable5.png`, `figures/effective_cost_per_task.png`: [PENDING grid].
- `numbers.json`: every computed figure with provenance and the scorecard's own
  cost-assumptions sentence per entry.
