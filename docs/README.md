# docs — finished products only, kept deliberately small

Production-ready documents: the final deliverable of a PR (AGENTS.md rule 5). Everything else —
working drafts, design notes, raw experiment results, plans — lives in `.agents/docs/`. Every
file here must justify its existence in the table below; a doc that can't gets deleted.

## Layout

- **`research/`** — completed research *writeups* and the figure each one renders. Raw result
  JSONs, vector sources, and experiment logs stay in `.agents/docs/research/`.
- **`reference/`** — how-to references for user-facing systems, verified against current `main`
  at promotion time.

## Why each doc exists

| File | Justification |
|---|---|
| `README.md` | The manifest that makes the justification rule enforceable. |
| `research/trace_scaling_law.md` | The repo's first completed scaling-law result and a load-bearing product claim: fidelity saturates at ~10 traces, so the leverage is prompt/optimization, not trace count. Cited by launch material and the benchmark work. |
| `research/trace_scaling_law.png` | The figure the writeup renders; also the brand-system visual reference cited by AGENTS.md rule 15. |
| `reference/eval_suites.md` | The reproducibility contract every benchmark number in this repo rests on (`examples/<task>/evals/*.toml` + `wmh eval`); commands verified against `main` at promotion. |
| `reference/failover.md` | The `.wmh/fallback.toml` failover contract: which calls ride the chain (world-model) and which never do (the judge), plus the cross-account ladder format; verified live against both AWS accounts. |
