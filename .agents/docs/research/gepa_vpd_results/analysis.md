# GEPA value-per-dollar: targeted cheap-executor cells (analysis)

Plan: `../../proposals/gepa-value-per-dollar.md`. Run 2026-07-19/20, driver
`.agents/scripts/run_gepa_vpd.sh`, per-arm JSONs + metered logs in this directory.
Serve/optimize = the cheap executor itself, judge pinned Opus 4.8 **rubric-v2**, seeds 0,1,
test-cap 40, winning GEPA config (b=8, mb=8, val=90 inclusive, recheck 30), same
deterministic split as #97. Total spend **$200.87**.

## TL;DR

| cell | RAG anchor | GEPA self | GEPA strong-reflect | build $ (self) | VPD (self) |
|---|---|---|---|---|---|
| T1 Haiku x terminal | 0.716 | 0.726 (+0.010) | 0.708 (-0.009) | $35.6 | 0.27 |
| T2 Haiku x tau | 0.864 | **0.893 (+0.029)** | 0.877 (+0.013) | $33.9 | 0.87 |
| T3 Mini x terminal | 0.709 | **0.760 (+0.051)** | not run | $27.1 | **1.87** |

VPD = milli-fidelity-points per build dollar, over the RAG anchor. Build $ = arm total minus
the cell's RAG-arm total. #163 lever comparison line (reason/kb/rag-deep): ~0.7-8.7, median ~2.

## Findings

1. **The cheap-executor lift is real but smaller than the grid promised.** T3 replicates
   directionally (+0.051; grid prior +0.157 at rubric-v1/seed-0/single-run) with both GEPA
   seeds (0.760, 0.759) above both RAG seeds (0.728, 0.690). T2 confirms at +0.029 with the
   same seed-separation (min GEPA seed 0.883 > max RAG seed 0.874). The grid's headline was
   ~3x inflated by judge version + single-seed noise; the direction survives.
2. **The tau parity result is the economically meaningful one.** Haiku+GEPA+RAG hits 0.893,
   AT the Opus 4.7 RAG anchor on the same split/judge (#97: 0.892). GEPA+RAG buys a cheap
   executor frontier-level tau fidelity for a one-time $34; serve cost stays ~5x lower. On
   terminal the gap barely moves (T1 closes ~6%, T3 ~31% of the frontier gap).
3. **Strong-reflector FAILS (H2 rejected).** Opus-reflects-for-Haiku is worse than
   self-reflection in both cells (-0.019 T1, -0.016 T2) at the same cost. Do NOT default it.
   Consistent mechanism guess: the reflector diagnoses errors ITS priors would make, not the
   executor's error distribution; the executor also may not follow rules it didn't write.
   (Evolved-prompt diff not yet examined; worth a look before generalizing.)
4. **T1 did not unlock.** The winning config finds ~nothing for Haiku on terminal (+0.010,
   overlapping seeds), where the old optimizer emitted a base-identical prompt. Terminal's
   cheap-model headroom is apparently Mini-specific, not universal.
5. **Best-cell GEPA VPD is lever-competitive; typical-cell is not.** T3's 1.87 milli-pts/$
   sits at the #163 lever median (~2); T2's 0.87 and T1's 0.27 sit below the cheapest lever
   wins. Frontier GEPA (#97: +0.022 at similar spend) stays ~0.6.

## Verdict vs pre-committed pass criteria

- "**Cheap-executor uplift** iff >= +0.03 on >= 2 of 3 cells AND >= half the frontier gap
  closed in >= 1 cell": **strictly FAILS the first clause** (T3 +0.051 yes; T2 +0.029 misses
  by 0.001; T1 no) and passes the second (T2 closes the full gap). Honest reading: one clear
  win, one borderline win with full parity, one miss. GEPA earns a slot as cheap-executor
  uplift on the cells where the corpus supports it, NOT as a blanket default.
- "**Strong reflector default** iff >= +0.02 over self on >= 2 cells": FAILS decisively
  (negative both cells).
- "**Demote everywhere** iff no cell clears +0.03": does not trigger (T3 cleared).

Predictions: H1 (T3 >= +0.08) failed. H2 (strong reflector) failed. H3 (T2 +0.03-0.05)
essentially confirmed (+0.029). H4 (b=16) not run.

## Limitations

- 2 seeds, no per-step CI: the runner persists per-seed means only, so "significance" here
  is seed-separation, not a bootstrap over test steps. T2/T3's non-overlapping seed ranges
  are the strongest form available from this data.
- Frontier reference for tau/terminal is #97's Opus 4.7 RAG anchor (same runner, split,
  judge version). The grid's rubric-v1 numbers were NOT used for any cross-judge comparison.
- T3 strong-reflect arm wasn't in the driver (OpenAI executor + Bedrock reflector is
  supported by the code; it just wasn't bought). Finding 3 rests on the two Haiku cells.

## Round 2: E1-E3 complete (2026-07-20)

All arms same harness (rubric-v2, seeds 0,1, test-cap 40, #97 split). E1 executor is Opus
4.8 default-account primary + waterfall ladder (4.8@endflow verified impossible,
AccessDeniedException; endflow is 4.6-gen only). Round-2 spend ~$195 (+~$60 lost restarting
E3 after a mid-run kill).

| exp | question | result |
|---|---|---|
| E1 | in-harness frontier lift + parity stress | Opus 4.8 RAG 0.875 (.870/.881); +GEPA **0.892** (.872/.912), +0.017 for $87 build (VPD ~0.2) |
| E2 | config vs executor | tier config on Mini x terminal = **0.708 = the RAG anchor** (0.709); winning config = +0.051. The lift is 100% config |
| E3 | spend (H4) | Haiku x tau b=16 = **0.894** (.895/.893) vs b=8's 0.893: saturated at b=8 |

**R2.1: the parity claim survives the strongest attack.** Give Opus 4.8 the identical
winning-config GEPA and it reaches 0.892 - statistically the same as Haiku+GEPA's 0.893.
Both converge on tau's unknowable-record ceiling (~0.89); the frontier model cannot buy
past it either. The $34 Haiku build genuinely matches the $87 Opus build AND the
un-GEPA'd frontier. Caveat: Opus's seeds spread .872/.912, the noisiest cell in the study.
**R2.2: cheap-executor GEPA is ~4x frontier VPD, now measured in one harness** (0.87 vs 0.2
milli-pts/$), replacing the #97 cross-reference.
**R2.3: the executor x corpus headroom is only harvestable at the winning config.** Tier
config (b=4, mb=3, val=24 greedy) collects ZERO of Mini x terminal's +0.051. Consequence:
`wmh build`'s tier GEPA spend is wasted at every tier as configured; either plumb
mb=8/val=90-inclusive into build or ship search-only tiers.
**R2.4: don't buy b=16.** +0.001 over b=8 for ~2x the cost. #163's budget-50 tau model is
now best explained by its era/config, not by budget scaling.

## Round 3 / N1: salvaged per-step CIs + the real mechanism (2026-07-23)

Round-1 winning prompts were recovered from the gepa library log lines (all 4 seeds; T3
seed-1 rescored 0.722 vs recorded 0.759, flagged - judge noise or a merge-program
extraction miss; direction intact) and rescored beside base with per-step persistence
(`steps/`, $16.57). Paired per-step bootstrap (5000 resamples, seeds pooled):

- **T2 Haiku x tau: +0.0277, 95% CI [+0.017, +0.039] - EXCLUDES 0.** The parity-cell lift
  is now rigorous, not seed-separation.
- **T3 Mini x terminal: +0.0379, 95% CI [+0.001, +0.075] - excludes 0, barely** (terminal
  is genuinely high-variance at n=198 steps).

**N1.1: the mechanism hypothesis was WRONG.** 96-97% of GEPA's gain comes from FIXING
failing steps (base score < 0.8), only 3-4% from polishing passing steps. The
graded-polish explanation for probe v1's failure is dead. The correct diagnosis of the
probe: its failure-counting frame was right and its CLASSIFIER is what's broken - it
labels failures "unknowable" that GEPA demonstrably fixes. What looks unknowable to a
judge staring at one step in isolation is often inferable from corpus-wide conventions.
Probe v2 therefore needs corpus context in the classification call (retrieve similar
steps, ask "is the answer inferable from these examples?") - and must probe the TEST-like
distribution, not the RAG-saturated valid band (probe v1's tau cell saw 3 failures where
the test band had plenty). GEPA also buys its gains with real regressions (about -0.24 per
step of gain across both cells): net-positive but not free.

## Round 3 / N2-N4: task type, model size, substitution (2026-07-23)

Same harness; all GEPA arms winning-config; per-step reports persisted natively
(`results_dir`). Round-3 spend ~$272 (N4's swe arms ran $60-65 each - swe steps are huge;
over the ~$160 estimate). Paired CIs are per-step bootstraps, seeds pooled.

| cell | rag anchor | +GEPA | paired lift | verdict |
|---|---|---|---|---|
| N2 Haiku x bird-sql | 0.767 | 0.795 | **+0.028 [+0.011,+0.047]** | real; derivability does NOT amplify |
| N3 Sonnet 4.6 x tau | 0.823 | **0.898** | **+0.075 [+0.056,+0.094]** | biggest lift in the program |
| N4 Haiku x swe (gepa+rag vs rag) | 0.499 | 0.513 | +0.014 [-0.017,+0.045] | n.s.; nothing works on swe |

N4 full ladder: base 0.522, rag 0.499 (RAG HURTS, -0.023), gepa-norag 0.526 (+0.004 vs
base, ~0), gepa+rag 0.513. **The grid's swe "GEPA substitutes for RAG" story does NOT
replicate at rubric-v2/winning-config for Haiku**: GEPA-alone is flat; GEPA+RAG only
partially repairs RAG's damage; best arm = base. (Grid's +0.147 swe cell was also a
byte-identical-prompt artifact cell; the +0.075 GPT-5.5 cell remains untested at v2.)

**N3.1: the ceiling-convergence law (tau).** Three executors, three anchors, one
destination: Sonnet 0.823 -> 0.898, Haiku 0.864 -> 0.893, Opus 4.8 0.875 -> 0.892. On tau,
GEPA lift is not a property of model size or task type; it is **ceiling(corpus) -
anchor(executor)**. Model-size curves are non-monotone because ANCHORS are non-monotone
(Sonnet's tau anchor sits below Haiku's).
**N3.2: the ceiling is NOT executor-independent everywhere.** Terminal: Mini reached 0.760
but Haiku stopped at 0.726 from a similar anchor; swe: no executor moves at all. So
corpus-common ceilings hold on structured-API corpora (tau), executor-specific limits
dominate on content-heavy ones.
**Predictor (the practical yield):** anchor + one GEPA run on ANY executor per corpus gives
the ceiling; expected lift for a new executor = max(0, ceiling - anchor) on
tau/bird-sql-like corpora, ~0 on swe-like. The $3-8 anchor run IS the per-executor half of
the probe; the per-corpus ceiling is a one-time ~$40 measurement.

Housekeeping: N4's arms collided in `steps/` (rag/no-rag share a condition label; later
arms overwrote earlier per-step reports - base/gepa-norag pairs lost, aggregates intact).
Fix before reuse: include the retrieval flag in the label or use per-arm results dirs.

## Round 2 / E4: headroom probe v1 FAILS validation (2026-07-20)

All 5 cells probed ($3.64 total, `probe_*.json`). The probe does NOT rank-order the measured
lifts: it returns headroom 0.00 on Haiku x tau (measured +0.029, the parity cell) and an
identical 0.07 on Mini x terminal (+0.051) and Haiku x terminal (+0.010). Failure
classifications are ~all "unknowable" everywhere, including cells GEPA demonstrably lifted.

Why it fails, and what it teaches: GEPA's real gains on the winning cells must come from
graded improvements on steps ABOVE the 0.8 failure threshold (convention polish moving 0.85
to 0.95), which a binary fail-then-classify probe cannot see. Also tau's valid band is
near-saturated under RAG for both executors (probe fidelity 0.966/0.971), so a
failure-counting probe has almost nothing to count there. A v2 would classify all imperfect
steps (< 0.95), weight by score deficit, and persist per-step data - deferred until E1-E3
settle whether the gate is even needed. Until then there is NO validated cheap predictor of
executor x corpus headroom; the honest gate remains "run the $3-6 RAG anchor, then decide".

## Build-ladder consequences (proposed)

- Frontier serve models: search-only build (GEPA off) as #163 recommended; #97's +0.02 at
  ~$35 (VPD ~0.6) is the worst paid lever in the stack.
- Cheap serve models: GEPA+RAG at the winning config where a quick probe shows headroom
  (tau-like corpora; Mini-like executors on terminal) - it can be outright parity-buying.
  gate: run the $3-6 RAG anchor first, spend the ~$30 GEPA only if the corpus class matched.
- Keep reflection = self. The strong-reflector knob stays for experiments.
