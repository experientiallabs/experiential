# COST-MAX corner: where the savings actually come from

> Status 2026-07-27: the canonical tau grid's three arms launched today (~15h wall); every
> [PENDING] below fills from `numbers.json` when the matrices land. Everything already
> written here is either measured elsewhere (provenance named) or a structural fact of the
> program. Publication of anything on this page is gated (DECISIONS 2026-07-27, corner
> chats entry).

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

1. **Model-selection share (routing)**: [PENDING: identity-arm routed rung vs fable-5].
   The structural caveat is known now: tau's corpus yields 20 distinct test scenarios and a
   fit band of ~14, deep in the small-bank regime (the evidence-volume law wants n>=1000),
   so the guarded router's honest posture here is abstention-heavy and the routing rung's
   claim is **cost-at-parity, not accuracy lift**. On routerbench-ours9, where the bank is
   big enough to route, the dial's measured curve runs -13.9% (quality max, +1.14 pt) to
   -46.2% (max savings, -0.54 pt) against ITS OWN best single model. Those anchors were
   measured on ours9 and are quoted as such; they are not tau numbers and never blend into
   the tau charts (D-DIAL v2 re-anchors them jointly later).
2. **Compression share (compaction)**: [PENDING: truncate and llmlingua2-endpoint arms vs
   identity, per model]. The compaction lane's interim financebench lesson stands as the
   thing to check, not assume: BOTH dumb controls RAISED effective cost 21-36%, because
   deleting load-bearing observations lengthens episodes; per-token accounting misses this,
   effective-cost-per-completed-task catches it. Whether tau shows the same inversion on
   the truncate arm is an explicit [PENDING] check in this analysis. The compressor's own
   bill (~$0.0008-0.0010 per 10k tokens on the endpoint) is folded into every compressed
   figure as RowOverhead. Compaction rungs carry "measured tradeoff, not recommendation"
   until that lane's accuracy verdict lands.
3. **Student share (distillation): zero, and it must be said plainly.** Cycle 1's gate
   REJECTED the warmup adapter (teacher Qwen3.6-27B 73.3% vs student base 71.7% at k=3: no
   headroom to distill; before-vs-after p=0.45 at n=60). An unpromoted student never enters
   the pool, so no served token is cheaper today because of training. The distill-only rung
   reads "no measurable effect at this sample size", never a lift and never a regression.
   The student's share stays zero until a cycle gates (K3 teacher escalation is the live
   candidate, probe authorized, leg gated on Silen).
4. **Cheap-model-was-already-fine share (the anchor's weakness)**: the share of "savings"
   that any cheap model would have delivered because the anchor is overkill for part of the
   workload. This is real money but not an optimizer achievement, which is why every delta
   is reported against BOTH anchors (next section).

## The two-anchor discipline (inherited from the routing lane)

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

- Against the **fable-5 anchor** on tau: [PENDING: the measured per-config and routed-rung
  savings; expectation from the probe economics is that several pool models clear -40% on
  price sheet alone, so the interesting number is savings at the quality tolerance, not
  savings alone].
- Against the **best-single anchor**, the honest optimizer-contribution reading: the ours9
  evidence says the dial buys -24.7% at +0.99 pt (balanced) and -40.8% at +0.87 pt (cost
  saver), so 40+% vs best-single is reachable there only past the balanced point, and tau's
  small-bank regime will be weaker. [PENDING: tau routed rungs.]
- The sales frame stays "run 10x more for the same budget" (an **estimate** derived from
  measured per-task cost, and labeled as such wherever quoted).

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

- `build_cost_corner.py`: the pipeline (scorecard-only cost aggregation, paired-delta
  evidence on every quality claim; rerun it as matrices land; routed rungs attach via
  `rows_for_policy` when the master's per-arm fits are delivered).
- `figures/dial_cost_curve.png`: ours9 dial anchors as measured (tau panel pending fits).
- `figures/training_stage_cost_lens.png`: the shared training-stage chart, cost lens
  (cycle 1 as measured, real_episode; cost deltas attach when the grid's student cells
  land).
- `figures/savings_vs_fable5.png`, `figures/effective_cost_per_task.png`: [PENDING grid].
- `numbers.json`: every computed figure with provenance and the scorecard's own
  cost-assumptions sentence per entry.
