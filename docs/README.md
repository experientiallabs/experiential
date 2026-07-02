# docs — finished products only

Production-ready documents: the final deliverable of a PR (AGENTS.md rule 5). Working drafts,
plans, and experiment scratch live in `.agents/docs/` until they earn promotion here. Every file
in this tree must justify its existence — if a doc's claims go stale and nobody refreshes them,
delete it rather than let it mislead.

## Layout

- **`research/`** — completed research reports with their figures, plus the small summary JSONs
  that back each published figure (under `<experiment>_results/`). A report states its method,
  its numbers, and how to reproduce them (commands quoted as of publication).
- **`design-decisions/`** — the *why* behind load-bearing mechanisms: what problem forced the
  design, what was chosen, what was rejected. These exist so the next person changing that code
  knows which properties are deliberate.
- **`reference/`** — how-to references for the harness's user-facing systems: accurate against
  current `main`, verified at promotion time.

## Why each doc exists

| Doc | Justification |
|---|---|
| `research/trace_scaling_law.md` (+ `.png`/`.svg`, `trace_scaling_results/`) | The repo's first completed scaling-law result and a load-bearing product claim: fidelity saturates at ~10 traces, so the leverage is prompt/optimization, not trace count. Cited by launch material and by the benchmark work; the JSONs make the figure auditable and re-plottable. |
| `design-decisions/rag_aware_gepa.md` | Records why GEPA evaluates candidates under the same retrieved demos serving uses (train/serve mismatch fix), why demos are precomputed once, and the two leak-avoidance rules. Anyone touching `wmh/optimize/gepa.py` or retrieval needs these invariants; verified against current code at promotion. |
| `reference/eval_suites.md` | The single reference for the eval-suite system (`examples/<task>/evals/*.toml` + `wmh eval`) — the reproducibility contract every benchmark number in this repo rests on. Commands and schema verified against `main` at promotion. |
