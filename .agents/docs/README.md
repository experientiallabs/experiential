# .agents/docs — working docs (migrated from Notion 2026-07-02)

The former Notion Eng Docs database, migrated per AGENTS.md's Docs policy: the repo is the
single source of truth. Files keep their Notion `area`/`status` in frontmatter. This README is
the promotion analysis — what graduates to `docs/` (production) and what stays here.

## Promotion queue → `docs/` (worthy, but needs a refresh pass first)

| Doc | Blocking staleness |
|---|---|
| `architecture.md` | Predates `wmh/env` (PR #48: Env/WorldModelEnv/run_episode/scenarios) and `wmh/telemetry`; references `docs/gepa_research.md`, `docs/research_directions.md`, `scripts/` — paths that don't exist. Refresh after the current merge wave (BENCH-A grid, RL seam) settles, then promote — this is the flagship dev doc. |
| `runbook-build-tau2-bedrock.md` | Verify every command against current CLI; sample outputs recorded on an older corpus (flagged inline). Promote as `docs/runbook.md` after a live re-run of the commands. |
| `benchmarks-to-traces.md` | Corpus counts stale (swe-bench 87 → 255+, growing). Recipe + contract are the value; refresh numbers, promote. |
| `embeddings.md` | Spot-check provider list (OpenAI Responses landed #46) + embed flags, then promote. |
| `eval-suites.md` | Near-current; verify against post-#38/#48 CLI, then promote. |
| `rag-aware-gepa-design-note.md` | Mechanism-level, low drift — strongest promote-as-is candidate; one read-through against `wmh/optimize/gepa.py` on current main. |

## Stays here (working material, not production docs)

- `research-directions.md` — backlog/plans (definitionally workspace).
- `gepa-optimization-research.md` — half harness-description (fold into architecture on
  promotion), half experiment log.
- `base-env-prompt-iteration.md` — methodology + historical numbers; superseded numerically.
- `benchmark-results-reproducibility.md` — June 2026 snapshot (base 0.74 vs GEPA 0.86) that
  CONFLICTS with the #37 grid finding (GEPA ~0 lift on the current base — different base prompt
  era). Keep for history; do not promote; the 80-cell grid (BENCH-A) supersedes it.
- `closed-loop-eval-spec.md` — "future direction" spec being overtaken by events (PR #58 landed
  session scoring + scenarios; BENCH-B is building closed-loop RL). BENCH-B should mine then
  retire it.
- `sim-real-policy-rank-agreement.md` — unclaimed-in-literature metric proposal
  (Spearman/Kendall policy-rank agreement sim vs real). High-value idea for the full benchmark's
  narrative — flagged to BENCH-A/B via DECISIONS.md.
- `trace-scaling-law-notion.md` — superseded by the promoted `docs/trace_scaling_law.md`
  (already production). Kept only as the Notion-era copy; delete at next prune.

## Scripts analysis (same exercise, `.agents/scripts/`)

- `run_trace_scaling.py` — stays here for now. If trace-scaling reruns become routine (BENCH-A's
  data-efficiency axis), promote into the `wmh research` CLI group (rule 7) once PR #41 lands it.
- `plot_trace_scaling.py` — disposable example of the brand palette; AGENTS.md rule 15 is
  self-contained (the palette is the contract, not this script).
- `examples/*/` capture/convert/run tooling — correctly placed per rule 6; not workspace material.
