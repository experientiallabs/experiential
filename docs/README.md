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
| `usage.md` | The one-page map of the CLI surface: every command's purpose and the artifact it leaves behind, grouped by pipeline / optimizers / traces / platform. The reference-style counterpart to the cookbook's narrative order, and the page that keeps every other doc from re-explaining what a command is. |
| `cookbook/tau-bench.md` | The canonical end-to-end example: one pass through the whole pipeline (setup, build, pool, optimize, optional distill, serve) told on tau-bench, each step one real CLI command plus the artifact it creates. Commands verified against `main`; the provenance rules (world-model simulated vs real-episode, cache-adjusted effective cost per completed task, savings as estimates) are stated once here and inherited by every number in the walk. Ends with the measured 2026-07-28 results (routed vs anchor at both dial points, traffic mix, caveats), the one-command protocol-exact reproduction (`reproduce run tau-bench --yes` in the research repo), and the real-episode leg: the shipped tau default re-measured on Sierra's actual benchmark, with its own free bit-exact reproduction (`reproduce run tau-bench-real` in the research repo) and a provenance rule that the two legs' numbers never blend. |
| `cookbook/routerbench.md` | The shortest routing walk: fit + tune + report on a precomputed RouterBench-style outcome matrix, no world model in the loop. Documents the matrix shape, the guarded kNN fit, the held-out report, the measured `ours9` results (both fits, both anchors, concrete routed requests with the guard's verbatim reasoning), and the one-command bit-exact reproduction (`reproduce run routerbench` in the research repo) against the published dataset. |
| `cookbook/terminal-tasks.md` | Terminal work in two legs, interpreted together and never blended. The public end-to-end walk on the `wmh-terminal-tasks-traces` Hugging Face bundle (measured constant-policy result, why the dial saturates, the compression cost-inversion), then the real Terminal-Bench 2.0 grid behind the product's terminal default: 640 real episodes, the pinned sonnet-5 table, the sim-to-real caveats (aggregate agreement, per-scenario rho -0.162), and the free bit-exact reproduction (`reproduce run terminal-bench-2` in the research repo). |
| `cookbook/swe-bench.md` | The evidence page behind the product's `swe-bench` default: 640 real SWE-bench Verified episodes, the pinned opus-5 table with its resolved-cost/unresolved-quality reading, the router null result and its mechanism, the cautionary rows (cheapest tokens = dearest completions; per-arm cap concentration), and the free bit-exact reproduction (`reproduce run swe-bench` in the research repo) from the published matrix. Deliberately a replay page, not a pipeline walk: the grid harness that bought the episodes is not a product CLI step, and the page says so. |
| `cookbook/deepswe.md` | The credential-free routing walk: DeepSWE v1.1's published long-horizon SWE trials converted into an outcome matrix (`wmo optimize route convert-deepswe`, gated on reproducing every published pass@1), the in-process local embedder (Qwen3-Embedding-0.6B via MLX/torch, no embedding API), the repo-grouped holdout protocol with its measured dial frontier beside the pre-split lab's threshold-rule reference, and the free bit-exact reproduction (`reproduce run deepswe-coding113` in the research repo). |
| `cookbook/fully_local.md` | The zero-cloud walk: two Ollama models as routing candidates, the simulation and judge on a local model via OPENAI_BASE_URL, sweep + knn fit + `wmo serve`, all at $0; includes where traces come from (your stack or `wmo download <benchmark>`). |
| `research/world_model_findings.md` | The single research record: six layered studies (data, retrieval, optimization, test-time compute, self-knowledge, economics; PRs #72, #97, #55, #120, #41, with #83/#98 as instruments) with shared protocol and judge provenance stated once. Every product claim about world-model fidelity and cost traces to a section of this document. |
| `research/figures/trace_scaling_law.png` | The trace-scaling figure (fidelity vs trace count, n=0 anchored) the record's data layer renders; also the brand-system visual reference cited by AGENTS.md rule 15. |
| `research/figures/rag_optimization.png` | The retrieval-optimization figure: optimized vs unoptimized retrieval curves per benchmark. |
| `research/figures/gepa_scaling_law.png` | The GEPA-scaling figure: fidelity vs GEPA budget and trace count, RAG baselines, judge panel. |
| `research/figures/fidelity_tiers.png` | The fidelity-tier figure: the build-tier ladder on three benchmarks. |
| `research/figures/confidence_gated_frontier.png` | The gated-verify cost frontier (fidelity vs $/cell, never/gated/always), the confidence layer's headline Pareto claim. |
| `research/figures/concurrency_speedup.png` | The concurrency speed-up figure: how many times faster the world model is than the real environment, per benchmark and concurrency level (the "what"). |
| `research/figures/concurrency_cost.png` | The concurrency cost figure: world-model reconstruction vs real-environment setup cost at W=1, the crossover that explains the speed-up (the "why"). |
| `reference/eval_suites.md` | The reproducibility contract every benchmark number in this repo rests on (`environment-capture-data/<task>/evals/*.toml` + `wmo eval`); commands verified against `main` at promotion. |
| `reference/failover.md` | The `.wmo/fallback.toml` failover contract: which calls ride the chain (world-model) and which never do (the judge), plus the cross-account ladder format; verified live against both AWS accounts. |
| `reference/eval_grid.md` | `wmo eval grid` - the model × condition fidelity grid (base/+RAG/+GEPA/+GEPA+RAG across models, one pinned judge, target-side cost); commands + judge version self-contained; fresh results land in `.wmo/evals/grid/`. |
| `reference/closed_loop.md` | The other half of eval: `wmo eval --mode closed-loop` runs a live agent against the world model and scores task success (gold-judged) instead of per-step fidelity; the contract `wmo/simulation/evaluation/closed_loop.py` and `agreement.py` implement. |
| `reference/ingest.md` | The ingestion contract behind `wmo build --source`: one pluggable `TraceAdapter` seam that turns traces from any observability stack (or plain chat logs) into the harness trace format, plus one section per source adapter (Phoenix, Langfuse, LangSmith, Braintrust, PostHog, Mastra) with its export shape and field mapping. |
| `reference/repository_guardrails.md` | The reproducible production LOC report, its 98,489-line W1 baseline boundary, per-PR file and dependency deltas, and the migration inventory rules for file size and public docstrings. |
| `reference/local_models.md` | Local OpenAI-compatible servers (Ollama, vLLM, llama.cpp) as routing candidates: registration (interactive + scripted), the explicit $0 pricing rule, the per-entry `enabled` toggle, zero-cost routing semantics, and the container loopback-translation note. |
| `reference/harness_delta.md` | The `HarnessDelta` interface used by harness search: the typed, precondition-guarded update representation that defines the optimizer agent's search space; WMO keeps the shared harness document, runtime, and store contracts under `wmo/runtime/harness/`. |
| `reference/cost_quality_dial.md` | The one operator control on a routing endpoint: `cost_quality` in [0, 1], what each leg of the mapping changes, the five measured anchors with their limits (interpolation is of the knobs, not the outcome; the savings leg trades guard strictness), and the three ways to set it (Python, `wmo optimize route tune`, `endpoint.toml`) plus the live config and savings routes. |
| `reference/distill.md` | The `wmo optimize distill` how-to: distillation of a Tinker LoRA student from real benchmark rollouts (config-selected: harbor's terminus-2 or tau2-bench episodes) (run config TOML, cost/budget model, run-dir artifacts, `report`, resume, troubleshooting); the contract `wmo/optimize/model/` implements, verified against completed end-to-end runs. |
| `reference/connect-library.md` | The programmatic contract behind `wmo.simulation.context`: `get_connector(name).pull(auth, query)` for host-side consumers (the platform's connector tools), the `ConnectorAuth`/`PullQuery`/`ContextItem` shapes, per-service targeting, and the caller-supplies-tokens rule; verified against `wmo/simulation/context/`. |
