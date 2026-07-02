# .agents/docs — working docs

The unclean side of the documentation (AGENTS.md rule 5): drafts, design notes, experiment logs,
raw results, proposals. Committed so it transfers across worktrees and chats; pruned
periodically; nothing outside `.agents/` may depend on it. When something matures, its cleaned
product is promoted to `docs/` (writeups → `docs/research/`, verified how-tos →
`docs/reference/`) and the working copy dies here. The Notion Eng Docs database was migrated
here 2026-07-02; files keep their Notion `area`/`status` in frontmatter.

## Layout

- **`reference/`** — system descriptions that become `docs/reference/` material after a
  freshness pass: `architecture.md`, `embeddings.md`, `benchmarks-to-traces.md`,
  `runbook-build-tau2-bedrock.md`.
- **`design-decisions/`** — the why behind load-bearing mechanisms: `rag-aware-gepa.md`
  (verified accurate 2026-07-02; lives here, not in public docs — design rationale is internal
  material, per user direction).
- **`research/`** — experiment logs, snapshots, and raw results: `gepa-optimization-research.md`,
  `base-env-prompt-iteration.md`, `benchmark-results-reproducibility.md`, plus
  `trace_scaling_law.svg` and `trace_scaling_results/` (the raw JSONs and vector source behind
  the published `docs/research/trace_scaling_law.md` writeup).
- **`proposals/`** — specced-but-not-built directions: `research-directions.md`,
  `closed-loop-eval-spec.md`, `sim-real-policy-rank-agreement.md`.

## Promotion queue → `docs/` (worthy, blocked on a refresh pass)

| Doc | Blocking staleness |
|---|---|
| `reference/architecture.md` | Predates `wmh/env` (PR #48), `wmh/telemetry`, the RL seam (#58+); references Notion-era doc names and old `scripts/` paths. The flagship dev doc — refresh once the merge wave settles, promote to `docs/reference/`. |
| `reference/runbook-build-tau2-bedrock.md` | Re-run every command live (rule 11), refresh sample outputs, promote as `docs/reference/runbook.md`. |
| `reference/benchmarks-to-traces.md` | Corpus counts stale (swe 87 → 255+). The trace contract + add-a-benchmark recipe are the value. |
| `reference/embeddings.md` | Spot-check provider list (post-#46/#67) and embed flags, then promote. |

## Stays here (working material)

- `design-decisions/rag-aware-gepa.md` — internal design rationale (user call: design decisions
  are not public-docs material).
- `research/benchmark-results-reproducibility.md` — June 2026 snapshot (base 0.74 vs GEPA 0.86)
  that CONFLICTS with the #37 grid finding (GEPA ~0 lift, different base-prompt era). History
  only; the 80-cell grid supersedes it.
- `research/base-env-prompt-iteration.md` — methodology + superseded historical numbers.
- `research/gepa-optimization-research.md` — half harness-description, half experiment log; most
  stale doc (references `scripts/run_seed_stability.py` (gone), `examples/tau2-bench.otel.jsonl`
  (now `examples/tau-bench/traces.otel.jsonl`), `world-models/tau-bench/` (now
  `examples/tau-bench/models/`), `ARCHITECTURE.md` (never existed)).
- `proposals/closed-loop-eval-spec.md` — being overtaken by the RL seam (#58+); BENCH-B should
  mine then retire it.
- `proposals/sim-real-policy-rank-agreement.md` — policy-rank-agreement metric proposal
  (unclaimed in the literature) — flagged to the benchmark chats as narrative material.
- `proposals/research-directions.md` — ablation backlog (references old `scripts/` paths).

## Known staleness (checked at migration, 2026-07-02)

Cross-cutting: migrated docs reference `scripts/…` (now `.agents/scripts/…`) and old cross-doc
filenames (`./sim_real_agreement.md`, `./closed_loop.md`) — fix links when refreshing a doc.

## Scripts (`.agents/scripts/`)

- `run_trace_scaling.py` — stays here; promote into the `wmh research` CLI group (rule 7) only
  if trace-scaling reruns become routine.
- `plot_trace_scaling.py` — disposable brand-palette example (the palette in AGENTS.md rule 15
  is the contract, not this script). Writes into `.agents/docs/research/`; the published PNG is
  copied to `docs/research/` at promotion.
