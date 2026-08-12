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

## Inspecting and evaluating a world model

| Command | Purpose | Artifact |
|---|---|---|
| `wmo eval` | Score reconstruction fidelity (open-loop, teacher-forced), run a live agent against the model (`--mode closed-loop`), or compare closed-loop reports with `agreement`. | optional JSON report passed to `--out` |
| `wmo knowledge` | Print the model's knowledge base directory: editable markdown that is the env's canonical facts. Says so when the model was built without `--knowledge`, which makes those files inert. | nothing (the directory it names is the editing interface) |
| `wmo list` | List every world model built under the project dir. | nothing (prints) |

## Traces and data

| Command | Purpose | Artifact |
|---|---|---|
| `wmo ingest` | Normalize traces from a file, a vendor API, or a Postgres table into OTel JSONL. No model is built. | an OTel GenAI JSONL corpus, ready for `wmo build --file` |
| `wmo download` | Fetch published benchmark data bundles from the Hub: trace corpus, task data, and prebuilt world model(s) built from that corpus. | `environment-capture-data/<benchmark>/` (`traces.otel.jsonl`, `models/<name>/`) |
| `wmo scenarios build` | Distill a trace corpus into a weighted, representative scenario set (facets, cluster, select). | a `ScenarioSet` |
| `wmo scenarios verify` | Closed-loop verification of a scenario set: back-agreement on source traces plus solvability rollouts. | a verification report |

`wmo scenarios build` produces a `ScenarioSet`
JSONL. The two formats are not interchangeable.

## Providers, harnesses, config

| Command | Purpose | Artifact |
|---|---|---|
| `wmo providers verify` | Ping every configured provider on the completion and embedding paths (deduped by kind and model): the `[models.<role>]` roles in `.wmo/settings.toml` **and** the providers each built world model recorded. Run it before `wmo build` — with nothing built yet it still checks the roles, and just skips the embed half. | nothing (prints a row per provider) |
| `wmo config telemetry` | View or change project-local usage telemetry settings. | `.wmo/settings.toml` |

## Running agents

| Command | Purpose | Artifact |
|---|---|---|
| `wmo run` | Run the built-in pi agent locally, with an explicit local-execution consent prompt and a file-tool jail selected by `--dir`. | local changes inside the selected directory |
| `reproduce list` / `run <benchmark>` (moved to the [research repo](https://github.com/experientiallabs/research)) | Reproduce a published benchmark result from its shipped manifest: download the pinned data, replay the pinned protocol, and compare every published number field by field. `matrix` manifests run offline and bit-exact; `commands` manifests replay live CLI steps, state their estimated spend, and refuse without `--yes`. Exit 0 is REPRODUCED, 4 is DIVERGED. | `verdict.json` plus the run's own artifacts |
