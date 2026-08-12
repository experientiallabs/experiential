# docs — finished products only, kept deliberately small

Production-ready documents: the final deliverable of a PR (AGENTS.md rule 5). Working drafts,
design notes, raw experiment results, and plans are not docs and never land here. Every file
here must justify its existence in the table below; a doc that can't gets deleted.

## Layout

- **`research/`**: completed research *writeups*, with the figures they render under
  `research/figures/`. Raw result JSONs, vector sources, and experiment logs are not docs.
- **`reference/`** — how-to references for user-facing systems, verified against current `main`
  at promotion time.
- **`cookbook/`**: end-to-end walks through the whole pipeline on one benchmark, each step one
  real CLI command plus the artifact it creates. One file per benchmark.
- **`usage.md`**: the only root page besides this one. The terse map of the CLI surface, one line
  of purpose and one artifact per command.

## Why each doc exists

| File | Justification |
|---|---|
| `README.md` | The manifest that makes the justification rule enforceable. |
| `research/world_model_findings.md` | The single research record: six layered studies (data, retrieval, optimization, test-time compute, self-knowledge, economics; PRs #72, #97, #55, #120, #41, with #83/#98 as instruments) with shared protocol and judge provenance stated once. Every product claim about world-model fidelity and cost traces to a section of this document. |
| `research/figures/trace_scaling_law.png` | The trace-scaling figure (fidelity vs trace count, n=0 anchored) the record's data layer renders; also the brand-system visual reference cited by AGENTS.md rule 15. |
| `research/figures/rag_optimization.png` | The retrieval-optimization figure: optimized vs unoptimized retrieval curves per benchmark. |
| `research/figures/gepa_scaling_law.png` | The GEPA-scaling figure: fidelity vs GEPA budget and trace count, RAG baselines, judge panel. |
| `research/figures/fidelity_tiers.png` | The fidelity-tier figure: the build-tier ladder on three benchmarks. |
| `research/figures/confidence_gated_frontier.png` | The gated-verify cost frontier (fidelity vs $/cell, never/gated/always), the confidence layer's headline Pareto claim. |
| `research/figures/concurrency_speedup.png` | The concurrency speed-up figure: how many times faster the world model is than the real environment, per benchmark and concurrency level (the "what"). |
| `research/figures/concurrency_cost.png` | The concurrency cost figure: world-model reconstruction vs real-environment setup cost at W=1, the crossover that explains the speed-up (the "why"). |
| `reference/failover.md` | The `.wmo/fallback.toml` failover contract: which calls ride the chain (world-model) and which never do (the judge), plus the cross-account ladder format; verified live against both AWS accounts. |
| `reference/closed_loop.md` | The other half of eval: `wmo eval --mode closed-loop` runs a live agent against the world model and scores task success (gold-judged) instead of per-step fidelity; the contract `wmo/simulation/evaluation/closed_loop.py` and `agreement.py` implement. |
| `reference/ingest.md` | The ingestion contract behind `wmo build --source`: one pluggable `TraceAdapter` seam that turns traces from any observability stack (or plain chat logs) into the harness trace format, plus one section per source adapter (Phoenix, Langfuse, LangSmith, Braintrust, PostHog, Mastra) with its export shape and field mapping. |
| `reference/repository_guardrails.md` | The reproducible production LOC report, its 98,489-line W1 baseline boundary, per-PR file and dependency deltas, and the migration inventory rules for file size and public docstrings. |
| `reference/harness_delta.md` | The `HarnessDelta` interface used by harness search: the typed, precondition-guarded update representation that defines the optimizer agent's search space; WMO keeps the shared harness document, runtime, and store contracts under `wmo/runtime/harness/`. |
