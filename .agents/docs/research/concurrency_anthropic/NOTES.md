# Concurrency scaling law — provenance data

Two things live here, both cited by `docs/research/concurrency_scaling_law.md`:

1. **`gpt54_*.json`** — the three **headline** gpt-5.4-mini reports (tau/terminal/swe). The original
   run used the OpenAI Responses API; that account was deactivated and its raw JSON was lost, so these
   are the exact published per-benchmark table values re-materialized as machine-readable data (see
   each file's `_provenance`). Fields the table doesn't determine (tokens/cost/fidelity) are omitted.
2. **`tau_h_*.json` / `term_h_*.json`** — the **Claude Haiku 4.5** re-run (below), the robustness
   check that the directions hold across a different world model and 3 seeds.

# Anthropic (Claude Haiku 4.5) concurrency re-run — cross-seed aggregates

World model = claude-haiku-4-5-20251001. `--select random`, full levels 1,2,4,8,16.
Preserved because the OpenAI account (GPT-5.4-mini, the committed PR) is deactivated.


## tau-bench (differential T_real/T_world, mean±sd across seeds)
- W=1: 0.24 ± 0.01  (seeds 0.23, 0.25, 0.26)
- W=2: 0.26 ± 0.01  (seeds 0.27, 0.27, 0.24)
- W=4: 0.28 ± 0.01  (seeds 0.28, 0.28, 0.29)
- W=8: 0.23 ± 0.03  (seeds 0.20, 0.22, 0.27)
- W=16: 0.17 ± 0.05  (seeds 0.14, 0.13, 0.25)

## terminal-tasks (differential T_real/T_world, mean±sd across seeds)
- W=1: 2.72 ± 0.83  (seeds 3.50, 1.57, 3.10)
- W=2: 3.47 ± 1.50  (seeds 4.08, 1.41, 4.93)
- W=4: 6.07 ± 2.99  (seeds 6.94, 2.05, 9.23)
- W=8: 7.76 ± 4.84  (seeds 8.59, 1.47, 13.22)
- W=16: 5.70 ± 3.13  (seeds 7.17, 1.36, 8.59)

## swe-bench: partial (killed mid-run). W=1 0.84x, W=2 0.79x, W=4 0.60x (real faster).
Haiku ~6.8s/verbose-obs -> reconstruction rivals the from-source build. NOT throttling
(16 concurrent long Haiku calls: 1.07x latency inflation, 0 errors).

## Ruled out: GIL (0.2% CPU), OpenAI throttle, Anthropic throttle. World ceiling = step-skew + tail.
