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

## Build-ladder consequences (proposed)

- Frontier serve models: search-only build (GEPA off) as #163 recommended; #97's +0.02 at
  ~$35 (VPD ~0.6) is the worst paid lever in the stack.
- Cheap serve models: GEPA+RAG at the winning config where a quick probe shows headroom
  (tau-like corpora; Mini-like executors on terminal) - it can be outright parity-buying.
  gate: run the $3-6 RAG anchor first, spend the ~$30 GEPA only if the corpus class matched.
- Keep reflection = self. The strong-reflector knob stays for experiments.
