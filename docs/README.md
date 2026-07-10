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
| `research/concurrency_scaling_law.md` | The honest cost claim behind the world-model pitch: a world model saves wall-clock **iff real standup cost > reconstruction cost** — so it wins on expensive-build envs (swe ~3–4×, terminal ~5–12×) and *loses* on cheap in-process envs (tau ~0.3×). Documents the fair-comparison methodology and the selection-sensitivity finding (why results use `--select random`, not `simplest`). |
| `research/concurrency_speedup.png` | The cross-benchmark speed-up figure the writeup renders — how many times faster the world model is than the real environment, per benchmark and concurrency level (the "what"). |
| `research/concurrency_cost.png` | The cross-benchmark cost figure the writeup renders — world-model reconstruction vs. real-environment setup cost at W=1, the crossover that explains the speed-up (the "why"). |
| `reference/eval_suites.md` | The reproducibility contract every benchmark number in this repo rests on (`examples/<task>/evals/*.toml` + `wmh eval`); commands verified against `main` at promotion. |
| `reference/failover.md` | The `.wmh/fallback.toml` failover contract: which calls ride the chain (world-model) and which never do (the judge), plus the cross-account ladder format; verified live against both AWS accounts. |
