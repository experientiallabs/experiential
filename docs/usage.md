# The `wmo` CLI, one page

Every command, what it is for, and what it leaves behind. For the walk that strings them together
in pipeline order, see [the tau-bench cookbook](cookbook/tau-bench.md).

Run `wmo <command> --help` for the full option list; each one documents its own contract.

## The pipeline

| Command | Purpose | Artifact |
|---|---|---|
| `wmo build` | Ingest traces and build a named world model: normalize, split, index, optimize prompts, write. | `.wmo/models/<name>/` (`config.toml`, `card.json`, `index/`, `prompts/`, `metrics.json`) |
| `wmo providers set` | Choose the local worker model, and register the models the router may choose between. | `.wmo/settings.toml` (worker role) and `.wmo/pool.toml` (candidate roster) |
| `wmo optimize model` | The staged one-command routing workflow: preflight, sweep, fit, tune, report, with one plan table and one confirmation. `--dry-run` previews the plan and spends nothing; a non-interactive spending run needs `--yes`; `--max-usd` caps. `--compressor`/`--aggressiveness` measure and fit a compressed arm end to end; `--embedder` picks what the policy routes on. | `policy.json` + `policy.json.bank.npz` in the model dir; `matrix.json`, `report.json`, `optimize-run.json` under `<model>/optimize/` |
| `wmo serve` | Run the local backend: the OpenAI-compatible endpoint plus the world-model step API. | a live server (`/v1/chat/completions`, `/v1/endpoints/<name>/config`, `/v1/endpoints/<name>/savings`) |

## The optimizers

Three optimizers, named for the artifact each produces.

| Command | Purpose | Artifact |
|---|---|---|
| `wmo optimize route sweep` | Measure every pool candidate closed-loop against the world model. The only paid step of routing, and the only thing that produces a matrix. A non-interactive run needs `--yes`, or it prints the projected spend and exits 2. | `matrix.json` (an `OutcomeMatrix`) |
| `wmo optimize route fit` | Fit a routing policy on a matrix: `--kind knn` guarded neighbor evidence (the default), or `--kind rank` cluster ranks. | `policy.json` + its evidence bank |
| `wmo optimize route tune` | Set a fitted policy's cost/quality dial in place, no refit. | the policy, rewritten; `policy.base.json` snapshot |
| `wmo optimize route report` | Build the three-objective improvement report for a policy over a matrix. | `report.json` (an `ImprovementReport`) |
| `wmo optimize route pin` | Serve one pool model as an endpoint, with no matrix and no fit. | a `kind="static"` `policy.json` |
| `wmo optimize route student` | Add a distilled student to the candidate pool as a priced entry. | a `[[model]]` entry in `pool.toml` |
| `wmo optimize route convert-deepswe` | Convert DeepSWE v1.1's published trials into a fit-ready matrix bundle (the research-adapter producer; refuses unless every published pass@1 reproduces). | `matrix.json` + `task_embeddings.npy` + `scenario_groups.json` |
| `wmo optimize distill probe` | Ask a measured outcome matrix whether this workload has a teacher gap worth distilling at all, and which model is the cheapest sufficient teacher. Free. Exits 0 (distill), 3 (no gap), 4 (too thin to say). | nothing (prints) |
| `wmo optimize distill run` | Train the agent model itself: on-policy distillation of a Tinker LoRA student from harbor rollouts, gated on held-out solve rates. A non-interactive run needs `--yes`. | a run dir (config snapshot, metrics, checkpoints, evals, `gate.json`) and, on an accepted gate, an adapter version |
| `wmo optimize distill report` | Read a finished or aborted run back: gate verdict and held-out before/after table. Free. | nothing (prints) |

`wmo optimize model` is the staged path over the four `route` commands and calls the same library
functions they do, so you can drop to any stage and the next run resumes around it.

## Inspecting and driving a world model

| Command | Purpose | Artifact |
|---|---|---|
| `wmo play` | Step into the environment yourself: type actions, get observations back. | nothing (a session) |
| `wmo demo` | Replay a randomly sampled recorded scenario against the world model, open loop. Needs the corpus (`--traces`) unless the model ships one, since a build keeps no copy of what it read. | nothing (prints) |
| `wmo eval` | Score reconstruction fidelity (open-loop, teacher-forced) or run a live agent against the model (`--mode closed-loop`), or run a named example-local suite. | results under `.wmo/evals/` |
| `wmo knowledge` | Print the model's knowledge base directory: editable markdown that is the env's canonical facts. Says so when the model was built without `--knowledge`, which makes those files inert. | nothing (the directory it names is the editing interface) |
| `wmo list` | List every world model built under the project dir. | nothing (prints) |

## Traces and data

| Command | Purpose | Artifact |
|---|---|---|
| `wmo ingest` | Normalize traces from a file, a vendor API, or a Postgres table into OTel JSONL. No model is built. | an OTel GenAI JSONL corpus, ready for `wmo build --file` |
| `wmo download` | Fetch published benchmark data bundles from the Hub: trace corpus, task data, the prebuilt world model(s) built from that corpus, and its named eval suites. | `environment-capture-data/<benchmark>/` (`traces.otel.jsonl`, `models/<name>/`, `evals/*.toml`) |
| `wmo scenarios build` | Distill a trace corpus into a weighted, representative scenario set (facets, cluster, select). | a `ScenarioSet` |
| `wmo scenarios verify` | Closed-loop verification of a scenario set: back-agreement on source traces plus solvability rollouts. | a verification report |

`wmo scenarios build` produces a `ScenarioSet`
JSONL. The two formats are not interchangeable.

## Providers, harnesses, config

| Command | Purpose | Artifact |
|---|---|---|
| `wmo providers verify` | Ping every configured provider on the completion and embedding paths (deduped by kind and model): the `[models.<role>]` roles in `.wmo/settings.toml` **and** the providers each built world model recorded. Run it before `wmo build` — with nothing built yet it still checks the roles, and just skips the embed half. | nothing (prints a row per provider) |
| `wmo config telemetry` | View or change project-local usage telemetry settings. | `.wmo/settings.toml` |

## Running agents, and the platform

| Command | Purpose | Artifact |
|---|---|---|
| `wmo login` / `logout` / `status` | Connect this machine to a platform account, disconnect, or show the current account and organizations. | a saved credential |
| `wmo push` / `pull` | Publish a local world model to the platform registry, or fetch a model or endpoint state from it. | a registry entry, or a local artifact dir |
| `wmo reproduce list` / `run <benchmark>` | Reproduce a published benchmark result from its shipped manifest: download the pinned data, replay the pinned protocol, and compare every published number field by field. `matrix` manifests run offline and bit-exact; `commands` manifests replay live CLI steps, state their estimated spend, and refuse without `--yes`. Exit 0 is REPRODUCED, 4 is DIVERGED. | `verdict.json` plus the run's own artifacts |
| `wmo runs list` / `show` / `tail` | See the runs this organization is feeding: progress, spend, stages, per-candidate cells, and the live event log. `--json` on the first two; `--org` (id or slug) to read another organization the credential can see, which otherwise comes from the login or `WMO_PLATFORM_ORG`. | nothing (prints) |
| `wmo runs stop` / `retry` | Ask the process feeding a run to stop at its next safe boundary, or to re-measure its unscored cells. Pull-based: it takes effect when that process next reports in, and a runner that owns its own retry policy may refuse with a reason. | a queued command |
| `wmo runs backfill <path>` | Replay a finished or interrupted run from its own artifacts (a grid directory, or a world model's `optimize/`), so a run nobody watched still has its history. The run's name comes from where the artifacts live; `--name` supplies it for artifacts that have moved. `--dry-run` writes the events as JSONL instead. | run history on the platform |

A long run reports itself while it works, and only when a platform credential with an organization
resolves; `wmo optimize model --no-emit` turns it off. Reporting never changes what a run measures
and can never fail it: every call is buffered and guarded, and none of them can raise. It is not
free, though. Pushes ride the run's own callbacks rather than a background thread, so an unreachable
platform costs at most a few bounded seconds at stage boundaries while its requests time out.
Re-running a backfill is free: events are keyed by the emitter's sequence number, and the platform
discards ones it already holds.

## Research

`wmo research` holds the experiment drivers behind the published studies (`concurrency`,
`plot-concurrency`, `deepswe-holdout`, and the rest). They write report JSONs and figures (or
print comparison tables) rather than serving artifacts; `wmo research --help` lists what is
currently wired. `deepswe-holdout` runs the repo-grouped router holdout on a converted DeepSWE
bundle and prints the per-split table, the medians, and the pre-split lab's reference numbers.
