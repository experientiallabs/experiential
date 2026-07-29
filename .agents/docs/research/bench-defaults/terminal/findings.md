# bench-defaults/terminal: Terminal-Bench 2.0 measured for real

**Lane:** bench-defaults/terminal · **Cohort:** `bench-defaults-terminal-v1` · **Endpoint:** `ccf6cf45-f92a-4bc7-a92d-fb1f642f290a` (admin org)

Authorship note, stated up front: the harness build, task pin, grid, and findings 1-9 are the
terminal lane's work (its entries in DECISIONS are the primary record). The lane's agent
session died mid-grid on an Anthropic API usage limit ("regain access 2026-08-01"), so the
endgame - the capped-key re-buy, fit, pin, endpoint install, backfill, and this document -
was executed by the master coordinator from the lane's own pre-staged scripts and
DECISIONS record. Every number is reproducible from
`.agents/docs/research/corners/bench-defaults-terminal/numbers.json` and the cohort dir.

## Headline

**The terminal default is a pinned `sonnet-5`, measured 52.9% cheaper per run than the
`fable-5` anchor with a quality point estimate ABOVE the anchor (+2.5 pt) that the sample
size does not resolve.**

| | value |
| --- | --- |
| accuracy | 0.800 (anchor 0.775) |
| cost per run | $0.946 (anchor $2.010) |
| savings | **52.9%** per run; -54.4% per completed task (CI -68.0..-30.2) |
| paired N | 20 tasks, both episodes, zero unscored anywhere in the matrix |
| quality delta | +2.5 pt, CI -15.0..+20.0, sign test p = 1.00 (UNRESOLVED) |
| latency p50 per run | 6.14s vs 7.58s (-19.0%) |
| provenance | `real_episode` |
| judge | TB2 task verifier (pytest via ctrf.json) - NO LLM judge on this path |

This is the only benchmark of the three where the pin's quality point estimate is above the
anchor. The claim ships as "+2.5 pt, unresolved at n=20", never "beats the anchor": the
paired CI spans -15 to +20.

## What was measured

640 real Terminal-Bench 2.0 episodes: 20 pinned registry tasks x 2 episodes x 16 candidates,
harbor 0.20.0 terminus-2 scaffold on E2B task environments, each task graded by its own
pytest suite. Grid spend $527.76, all 640 cells scored. The pin was validated for
discriminative power on prior-leg data before buying (solve rates 0.15-0.92 across 16 of 17
overlap ids); the one all-fail task (gcode-to-text) was KEPT deliberately - cost per
completed task charges every dollar burned failing it, and dropping hard tasks post hoc is
quiet selection.

## The router null result, third benchmark

Given the full matrix, the fitter itself picks sonnet-5 as its fallback and routes away
from it 0.0% of the time. Under leave-one-out refit (every task held out once):

    dial 0.00-0.75   cost -32.9%   quality -7.5 pt    p=0.688   mix sonnet-5 17, opus-5 2, fable-5 1
    dial 1.00        cost -45.9%   quality -2.5 pt    p=1.000   mix sonnet-5 18, opus-5 2

Every rung is dominated by simply serving sonnet-5 (-54.4% at +2.5 pt). Three benchmarks,
one conclusion: at n=20 on a single workload, model selection carries all the value and
routing adds only variance.

## The full table (paired vs fable-5, cost per completed task)

    candidate         solve   $/completed   cost delta          quality delta
    sonnet-5          0.800      $1.18      -54.4% [-68.0,-30.2]   +2.5 pt (p=1.00)   <- installed
    opus-5            0.800      $1.56      -40.0% [-60.7,-11.9]   +2.5 pt (p=1.00)
    fable-5 (anchor)  0.775      $2.59      -                      -
    glm-5.2           0.725      $1.21      -53.4%                 -5.0 pt
    opus-4-8          0.725      $1.37      -47.1%                 -5.0 pt
    kimi-k3           0.700      $0.55      -78.9%                 -7.5 pt
    gpt-5.6-sol       0.700      $0.46      -82.2%                 -7.5 pt
    gpt-5.6-terra     0.675      $0.62      -75.9%                 -10.0 pt
    gpt-5.5           0.650      $1.31      -49.5%                 -12.5 pt
    kimi-k2.6         0.625      $1.25      -51.9%                 -15.0 pt
    deepseek-v4-pro   0.550      $5.74      +121.5% [+40,+282]     -22.5 pt (p=0.07)
    qwen3.6-27b       0.550      $0.16      -93.7%                 -22.5 pt
    gpt-5.6-luna      0.525      $0.36      -86.2%                 -25.0 pt (p=0.07)
    gpt-5.4-mini      0.275      $1.65      -36.5%                 -50.0 pt (p=0.006)
    haiku-4-5         0.275      $1.12      -56.9%                 -50.0 pt (p<0.001)
    qwen3.5-9b        0.100      $1.65      -36.4%                 -67.5 pt (p<0.001)

Value rungs a customer can actually choose: gpt-5.6-sol (0.700 @ -82.2%) and, as a deep
trade, qwen3.6-27b (0.550 @ -93.7%). deepseek-v4-pro is the anti-value exhibit: open-weight
pricing, 2.2x the ANCHOR's cost per completed task, driven by Azure content-filter churn on
terminal content (the lane predicted this before the grid ran). qwen3.5-9b completes almost
nothing on its third consecutive benchmark.

## Caveats that travel with every number

1. QUALITY IS UNRESOLVED both ways: +2.5 pt has p=1.00 and a CI spanning -15..+20. The cost
   CI excludes zero by a wide margin; the savings claim is the resolved one.
2. LATENCY IS DEFINITION-DEPENDENT and the simulated leg's claim does not reproduce as
   stated: -50.6% (sim) reads -19.0% p50 per run on real episodes, and the
   per-completed-task cut INVERTS to +33.5% (sonnet-5 keeps working on tasks the anchor
   abandons). Quote latency per run with the definition named.
3. THE CAPPED-KEY HOLES WERE RE-BOUGHT, NOT IMPUTED. 45 cells landed unscored during the
   grid: 28 because the grid's Anthropic key hit its account usage limit mid-run
   (the same limit that killed the lane's agent session), 17 transient transport faults.
   All 45 were quarantined (originals preserved under quarantine-*/) and re-bought with a
   working key, 13 of them twice after concurrent Anthropic load caused 529 overloads -
   the final sweep ran arms sequentially. End state: 640/640 scored, zero holes.
4. gpt-5.6 family ran with reasoning OFF (the only tool-calling configuration the serving
   path reaches); disclosed in scenario_label.
5. Sim-to-real transfers at the AGGREGATE level only: the earlier simulated terminal leg
   picked the same winner, but per-scenario correlation measured rho = -0.162, so no
   per-task claim survives the crossing. Cookbook (docs/cookbook/terminal-tasks.md) carries
   the joint interpretation.
6. E2B/harbor operational notes from the lane's record: harbor's shared task cache races
   concurrent first-use resolution (serial prewarm + retry covers it); litellm silently
   reports a 1M-token context for routes it cannot resolve (6 of 16 in this pool - fixed by
   keying off `litellm.get_model_info` raising, never provider kind); `LedgerLine` is
   extra="forbid" and both the runner and backfill silently SKIP nonconforming lines
   (consolidation regenerates the ledger from chunk files for exactly this reason).

## Product path (all verified on the live stack)

    endpoint  ccf6cf45-f92a-4bc7-a92d-fb1f642f290a "terminal-bench-2", admin org,
              world_model_id NULL (real-benchmark evidence), is_catalog_default TRUE
    headline  computed by the platform from the installed report: accuracy 0.8,
              savings_fraction 0.529
    serving   POST /v1/chat/completions model="terminal-bench-2" -> 200 (sonnet-5 answered)
    model dir .wmo/models/terminal-bench-2/ holds policy.json (static/sonnet-5) +
              report.json + pareto.json - the artifact home serving consumers read
    runs      `wmo runs backfill` (dry-run verified cell.batch + heartbeat first):
              640 cells, $527.76, visible in /admin/runs beside the tau and swe grids

## Reproduction

    .agents/scripts/run_tb2_grid.py --pool <cohort>/pool.toml --task-ids <cohort>/task-ids.json \
        --out-dir <cohort> --episodes 2            # the grid (resumable, chunked)
    wmo optimize route fit <matrix> --kind knn --embedder auto --rag-num 7 --min-pairs 2 --z 0.5 --floor-q 0.05
    wmo optimize route pin terminal-bench-2 --model sonnet-5 --pool <cohort>/pool.toml \
        --root <checkout>/.wmo                      # the pin, INTO the model dir
    wmo optimize route report <matrix> <model-dir>/policy.json --baseline fable-5 \
        --endpoint terminal-bench-2 --provenance real_episode \
        --judge "TB2 task verifier (pytest via ctrf.json; no LLM judge)" --out <model-dir>/report.json
    build_corners.py --lens bench-defaults-terminal --anchor fable-5 --loo   # stats + figures
