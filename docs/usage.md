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
| `wmo optimize route sweep` | Measure every pool candidate closed-loop against the world model. The only paid step of routing, and the only thing that produces a matrix. | `matrix.json` (an `OutcomeMatrix`) |
| `wmo optimize route fit` | Fit a routing policy on a matrix: `--kind knn` guarded neighbor evidence, or `--kind rank` cluster ranks. | `policy.json` + its evidence bank |
| `wmo optimize route tune` | Set a fitted policy's cost/quality dial in place, no refit. | the policy, rewritten; `policy.base.json` snapshot |
| `wmo optimize route report` | Build the three-objective improvement report for a policy over a matrix. | `report.json` (an `ImprovementReport`) |
| `wmo optimize route pin` | Serve one pool model as an endpoint, with no matrix and no fit. | a `kind="static"` `policy.json` |
| `wmo optimize route student` | Add a distilled student to the candidate pool as a priced entry. | a `[[model]]` entry in `pool.toml` |
| `wmo optimize harness` | Search the agent scaffold (prompts, skills, tool policy, loop params) against a world model or on harbor tasks. | an immutable `vN` `HarnessDoc` in the store, `champion` alias moved, plus a delta archive |
| `wmo optimize distill run` | Train the agent model itself: on-policy distillation of a Tinker LoRA student from harbor rollouts, gated on held-out solve rates. | a run dir (config snapshot, metrics, checkpoints, evals, `gate.json`) and, on an accepted gate, an adapter version |
| `wmo optimize distill report` | Read a finished or aborted run back: gate verdict and held-out before/after table. Free. | nothing (prints) |

`wmo optimize model` is the staged path over the four `route` commands and calls the same library
functions they do, so you can drop to any stage and the next run resumes around it.

## Inspecting and driving a world model

| Command | Purpose | Artifact |
|---|---|---|
| `wmo play` | Step into the environment yourself: type actions, get observations back. | nothing (a session) |
| `wmo demo` | Replay a randomly sampled recorded scenario against the world model, open loop. | nothing (prints) |
| `wmo eval` | Score reconstruction fidelity (open-loop, teacher-forced) or run a live agent against the model (`--mode closed-loop`), or run a named example-local suite. | results under `.wmo/evals/` |
| `wmo knowledge` | Print the model's knowledge base directory: editable markdown that is the env's canonical facts. | nothing (the directory it names is the editing interface) |
| `wmo list` | List every world model built under the project dir. | nothing (prints) |

## Traces and data

| Command | Purpose | Artifact |
|---|---|---|
| `wmo ingest` | Normalize traces from a file, a vendor API, or a Postgres table into OTel JSONL. No model is built. | an OTel GenAI JSONL corpus, ready for `wmo build --file` |
| `wmo download` | Fetch published benchmark data bundles (trace corpus plus task data) from the Hub. | `packages/environment-capture/<benchmark>/` |
| `wmo scenarios build` | Distill a trace corpus into a weighted, representative scenario set (facets, cluster, select). | a `ScenarioSet` |
| `wmo scenarios verify` | Closed-loop verification of a scenario set: back-agreement on source traces plus solvability rollouts. | a verification report |
| `wmo examples list` / `run` | List the self-contained task examples, or launch one's local helper (extra args after `--`). | whatever the example's launcher writes |

`wmo scenarios build` produces a `ScenarioSet`; `wmo optimize harness --tasks` takes `TaskSpec`
JSONL. The two formats are not interchangeable.

## Providers, harnesses, config

| Command | Purpose | Artifact |
|---|---|---|
| `wmo providers verify` | Ping the providers that **built** world models recorded, on the completion and embedding paths (deduped by kind and model). A project with nothing built has nothing to verify. | nothing (prints a row per provider) |
| `wmo harness list` / `show` / `init` | Inspect stored harness versions and aliases, or write the baseline as `v1` with `champion` pointed at it. | a `HarnessDoc` version in the store |
| `wmo config telemetry` | View or change project-local usage telemetry settings. | `.wmo/settings.toml` |
| `wmo e2b` | Inspect and reclaim E2B sandbox capacity. | nothing (prints, or reclaims) |

## Running agents, and the platform

| Command | Purpose | Artifact |
|---|---|---|
| `wmo run` | Run a platform world model or agent by id, or the built-in local pi harness with no target. | a run record under `.wmo/runs/`; uploaded workspaces sync back |
| `wmo login` / `logout` / `status` | Connect this machine to a platform account, disconnect, or show the current account and organizations. | a saved credential |
| `wmo push` / `pull` | Publish a local world model or harness to the platform registry, or fetch one from it. | a registry entry, or a local artifact dir |

`wmo run` is the primary execution surface. Bare runs execute harness code and bash on your
machine behind an explicit consent boundary and a `--dir` file-tool jail; hosted ids run their
champion harness in platform-managed sandboxes and need no local model or sandbox credentials.

## Research

`wmo research` holds the experiment drivers behind the published studies (`concurrency`,
`plot-concurrency`, and the rest). They write report JSONs and figures rather than serving
artifacts; `wmo research --help` lists what is currently wired.
