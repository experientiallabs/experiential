# GEPA value-per-dollar: targeted cheap-executor confirmation

**STATUS: BOTH ROUNDS RUN 2026-07-19/20, ~$460 total.** Results + verdict:
`../research/gepa_vpd_results/analysis.md`. Round 1: T3 +0.051 / T2 +0.029 (tau parity
with Opus-RAG) / T1 +0.010; strong-reflector negative in both cells. Round 2: parity
survives Opus 4.8 getting the same GEPA (0.892 vs 0.893, both at tau's ceiling); the lift
is 100% optimizer config (tier config harvests ZERO); spend saturates at b=8; headroom
probe v1 fails validation.

Supersedes the 5-arm x 3-benchmark sweep version of this proposal. The sweep is unnecessary:
the #98 grid + #97 scaling JSONs already locate where GEPA pays; this buys ONLY the
confirmation cells for that pattern.

## What the existing data already says (no new spend)

Re-analysis of the #98 grid JSONs (`.agents/docs/research/benchmark-grid/grid_*_api.json`),
excluding the 8 cells whose "GEPA" prompt was byte-identical to base. Marginal GEPA lift ON
TOP of RAG (`gepa_rag - base_rag`), per cell:

1. **Model strength is the axis.** Cheap executors (Haiku 4.5, GPT-5.4 Mini): mean **+0.063**
   over 6 cells (max +0.157). Frontier (Opus 4.8, GPT-5.5): mean **+0.013** over 6 cells.
   Roughly 5x. GEPA distills environment conventions frontier models infer on their own.
2. **Benchmark type is the second axis.** Convention-heavy shell/GUI environments lift
   (terminal +0.105 mean, kimi-gui +0.032, 5/6 cells > +0.02); record/content environments
   don't (tau +0.017, swe sign-mixed). On swe, GEPA-alone lifts (+0.075 GPT-5.5) but
   GEPA+RAG doesn't: prompt and retrieval SUBSTITUTE there, they stack on terminal/kimi.
3. **The headline cell is an economics story.** GPT-5.4 Mini x terminal: RAG 0.771 ->
   GEPA+RAG **0.928**, within noise of GPT-5.5 base (0.931) at ~1/7 the serve cost.
4. **Spend axis (from #97 + #163):** lift is not monotone in iterations under the old
   optimizer (declines past b=8); the winning config operates at b=8; #163's unbeaten
   budget-50 tau model hints high spend can pay at the right config. Open, one cheap arm.

**Why confirmation is still required:** the grid is rubric-v1, seed 0, single run, 38-40
steps/cell, and its prompts came from the OLD optimizer (pre template v2). The pattern could
shrink under rubric-v2 + seeds; it could also grow under the winning config. Notably Haiku x
terminal produced a byte-identical prompt under the old optimizer, so the winning config
gets to attempt a cell the old one outright failed on.

## Design: 3 target cells x 3 arms, frontier reference borrowed from #97

Thesis: **GEPA's dollar value concentrates in cheap-executor cells on convention-heavy
benchmarks, where one-time build spend buys serve-time frontier parity.**

Target cells (cheap executor, convention-heavy suite):

| cell | grid prior | why |
|---|---|---|
| T1 Haiku 4.5 x terminal | none (old optimizer emitted base) | can the winning config unlock what the old one couldn't? |
| T2 Haiku 4.5 x tau | +0.049 | confirm the one cheap tau lift at rubric-v2 + seeds |
| T3 GPT-5.4 Mini x terminal | +0.157 | replicate the headline cell (needs OPENAI_API_KEY; drop if absent) |

Arms per cell (runner: `.agents/scripts/run_gepa_scaling.py`, one metered invocation each,
`--opt-model` = the cheap executor, judge pinned Opus 4.8 rubric-v2, seeds 0,1, test-cap 40,
same deterministic split as #97):

- **rag**: budget 0 (anchor; also the scoring-overhead baseline for cost differencing)
- **gepa-self**: winning config (b=8, mb=8, val=90 inclusive, recheck 30), cheap model
  reflects for itself (what the grid measured)
- **gepa-strong-reflect**: same, reflection LM = Opus 4.7, executor stays cheap. Requires a
  ~5-line `reflection_provider` knob in `wmh/optimize/gepa.py` (reflection is currently
  hardwired to the serve provider, line ~708) + a `--reflect-model` runner flag.
- optional **gepa-self b=16** on T2 only: does spend keep paying at the winning config?

No frontier arms are bought: #97 already measured Opus at the winning config on the same
split/judge (tau +0.022, swe +0.021, terminal ~0). That IS the frontier reference line.

**Metrics.** Per cell: (a) marginal lift over the rag anchor, paired bootstrap over test
steps, seeds pooled; (b) build $ = arm total minus rag-arm total (differences out shared
scoring overhead); (c) **cost-to-parity**: gap to the frontier RAG fidelity for that suite
(grid: Opus terminal-rag 0.899, Opus tau-rag 0.894) plus break-even served steps =
build $ / (frontier serve $/step - cheap serve $/step).

## Predictions

- H1: T3 replicates at +0.08 or better (half the grid's +0.157 survives rubric-v2 + seeds).
- H2: strong-reflector >= self-reflect everywhere, and unlocks T1 (>= +0.04) where
  self-reflection failed; Qwen's total self-reflection failure in the grid is the weak-model
  limit of the same mechanism.
- H3: T2 confirms small (+0.03-0.05): tau's residual is unknowable records, cheap-model
  headroom is mostly convention errors RAG doesn't cover.
- H4: the b=16 arm adds <= +0.01 over b=8 (spend saturates at the winning config too).

## Pass criteria (pre-committed)

- **GEPA's documented value prop becomes "cheap-executor uplift"** iff winning-config
  GEPA+RAG lifts >= +0.03 (CI excluding 0) on >= 2 of 3 cells AND closes >= half the gap to
  frontier-RAG fidelity in >= 1 cell. Build ladder consequence: GEPA stays default only for
  cheap serve models; search-only default for frontier (per #163), pending Part B stacking.
- **Strong reflector becomes the build default** iff it beats self-reflect by >= +0.02 on
  >= 2 cells (it costs almost nothing: reflection is ~10 calls/build).
- **GEPA demoted to opt-in everywhere** iff no cell clears +0.03: the grid lifts were
  judge/seed artifacts and #163's search-only recommendation stands unconditionally.
- Kill switch: any arm past $60 metered stops.

Spend: rollouts on cheap executors are cheap and the Opus judge dominates: ~$10-25 per
(cell x GEPA-arm x seed) -> **~$150-250 total** (vs $250-500 for the superseded sweep), all
aimed where the data says the value is.

## Context

Analysis inputs: `.agents/docs/research/benchmark-grid/` (#98), `gepa_scaling_results/`
(#97), #163's refresh table. Results land in `.agents/docs/research/gepa_vpd_results/`;
survives -> layer 3 addendum in `docs/research/world_model_findings.md`.

---

# Round 2: rigor on the round-1 claims (2026-07-20)

Round 1 left four load-bearing claims resting on cross-experiment references or untested
arms. Each gets exactly one experiment; nothing else. Driver:
`.agents/scripts/run_gepa_vpd_round2.sh`.

| exp | claim under test | arms | est. $ |
|---|---|---|---|
| E1 | "cheap gains 2-3x frontier" used #97's +0.022 cross-ref; and does Haiku's tau parity survive when Opus gets the SAME GEPA? | Opus 4.7 x tau: rag anchor + gepa-self winning, in THIS harness (same split/judge/seeds/test-cap as round 1) | ~$165 (Opus rollouts; kill $200) |
| E2 | which matters more for cheap executors, the winning config or the executor? (grid lifts came from the OLD optimizer) | Mini x terminal at the TIER config (b=4, mb=3, val=24 greedy, no recheck); compare to round 1's +0.051 winning-config lift | ~$15 |
| E3 | spend axis (H4, never run): does 2x budget push T2 past parity? (#163's unbeaten budget-50 tau model is the hint) | Haiku x tau at b=16, winning config otherwise | ~$75 (kill $90) |
| E4 | headroom predictor: can a $2-5 probe rank-order which executor x corpus pairs GEPA will pay on, WITHOUT running GEPA? | `probe_gepa_headroom.py` on 5 cells with known outcomes | ~$15 |

**E1 readouts.** (a) In-harness frontier lift replaces the #97 cross-ref in claim 1.
(b) Parity stress: if Opus+GEPA+RAG lands ~0.90+, Haiku's "parity" was parity with an
under-optimized frontier and the claim weakens to "parity with frontier-RAG"; if Opus+GEPA
stays ~0.89 (tau's unknowable-record ceiling), the parity claim survives in full.

**E4 probe design.** Executor-specific learnable headroom, measured on the VALID band
(never test): score ~25 valid-band steps with base+RAG (retrieval from the train pool,
leak-free), collect failures (step score < 0.8), and have the judge model classify each
failure's root cause with template v2's taxonomy: derivable-convention /
session-establishable / external-unknowable. Headroom = fraction of ALL probed steps that
are fixable (derivable or session) failures. Validation: rank-correlate the probe against
the five measured GEPA lifts (T1 +0.010, T2 +0.029, T3 +0.051, E1's Opus x tau, #97's
Opus x terminal ~0). Pass = probe rank-orders the measured lifts with no inversion between
its top and bottom pick; then it becomes the $3 gate in front of every ~$30 cheap-executor
GEPA build. Predicted ordering if the executor x corpus theory is right: Mini-terminal >
Haiku-tau > Haiku-terminal ~ Opus-terminal, with Opus-tau low.

Round-2 total ~$270. Not run: T3 strong-reflect (finding 3 generalization), kimi/swe cells,
retrieval-ablated GEPA. Deliberately deferred until E1-E4 settle the frame.
