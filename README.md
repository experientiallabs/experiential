# World Model Harness

`wmh` makes it easy to go from agent traces to faithful replication of your production environment where your agents run.

Basically, an LLM pretends to be a virtual machine executing instructions — but it's 5x faster than a real sandbox.

Just:

```bash
git clone https://github.com/experientiallabs/world-model-harness
cd world-model-harness
uv sync
uv run wmh build
```

The `build` command opens a wizard that walks you through creating your own world model from your traces.

Below is a comparison running 8 SWE-bench tasks: real sandboxes on the left, a world model acting as the sandbox on the right.

![world-model-harness demo](./assets/demo.gif)

## How it works

A frontier LLM acts as the *environment* your agent steps against, reconstructed from your own OpenTelemetry traces. Inspired by **Qwen-AgentWorld** (LLM-as-environment), **GEPA** (reflective prompt evolution), and **DreamGym** (retrieval over a trace replay buffer) — but with **zero training**: we get there with prompt optimization on a frontier model.

1. **Build** from your OTel traces: ingest → normalize → split train/held-out → index a replay buffer → evolve the env prompt with GEPA against the held-out split.
2. **Serve**: agents call `WorldModel.step(action)` (in-process or via the local HTTP backend). Each step retrieves the most similar past `(state, action) → observation` examples and predicts the next observation.

Already have traces in **Braintrust, Arize Phoenix, Langfuse, LangSmith, PostHog, or Mastra** — or just chat/tool-call logs? Pick the source right in `wmh build` (`--source <name>` with `--file` or `--pull`, or choose it in the wizard); it's normalized into the harness's trace format via one pluggable interface, no separate step. See [`docs/ingest.md`](./docs/ingest.md).

## Try it

```bash
uv run wmh examples list          # swe-bench, tau-bench, terminal-tasks
uv run wmh eval list              # eval suites shipped with the examples
uv run wmh eval run tau-bench     # replay + score reconstruction fidelity
uv run wmh scenarios build --file traces.otel.jsonl   # traces -> judgeable eval scenarios
uv run wmh play                   # step into the environment yourself
uv run wmh serve                  # local HTTP backend on :8000
```

Example-local prebuilt models live under `examples/<task>/models/`; pass `--root examples/<task>` to `wmh list`, `wmh demo`, `wmh play`, or `wmh serve` to use one without rebuilding.

## Use it as an API

```python
from wmh import Action, ActionKind
from wmh.config.store import WorldModelStore
from wmh.engine.loader import load_world_model

model_dir = WorldModelStore(".wmh").resolve("airline")
wm, _provider = load_world_model(model_dir)

session = wm.new_session(task="check out the cart")
obs = wm.step(session.id, Action(kind=ActionKind.TOOL_CALL, name="add_to_cart",
                                 arguments={"sku": "A1"}))
print(obs.content)
```

Or over HTTP (same code path), namespaced by model name: `GET /world_models`, then `POST /world_models/{name}/sessions` and `POST /world_models/{name}/sessions/{id}/step`.

## Run after platform login

`wmh run` is the single interactive execution command. After `wmh login`, an opaque platform id
is resolved automatically: a world-model id opens a hosted model session, while an agent id runs
that agent's champion pi harness in the platform's E2B sandbox. No local files are uploaded by
default. Add `-u PATH` (or `--upload-dir PATH`) to upload that directory as the E2B workspace,
live-sync changes, and automatically sync final regular-file changes back. Concurrent local edits
are preserved and the full E2B result is saved under `.wmh-conflicts/` for manual recovery.
Provider and E2B credentials remain platform-side, so no API keys are needed locally.

```bash
wmh login
wmh run <world-model-or-agent-id>
wmh run <agent-id> -u . --task "fix the failing tests"
wmh run --task "fix the failing tests"   # built-in pi harness, also platform-backed when logged in
```

For a deployment-protected preview whose public discovery route is not available to a
non-browser client, pair its browser and backend URLs explicitly:

```bash
wmh login --url https://preview.example --api-url https://preview-api.example
```

Workspace transport skips symlinks, VCS internals, virtual environments, dependency trees, and
common caches. Uploads are capped at 50 MiB compressed and 512 MiB unpacked.

The bare built-in pi path runs locally and requires Node.js 22.19 or newer plus npm on `PATH`. WMH
installs the pinned pi npm dependencies into its user cache on the first run. Harness code and
shell commands run with your normal user permissions: file tools are restricted to `--dir`, but
bash is not OS-sandboxed. The CLI states this boundary before the local pi process starts. A
logged-out bare `wmh run` remains available with local provider environment credentials.

## Real agents in E2B sandboxes

Harness evals normally drive a plain in-process agent loop. With `--harness-backend e2b`, a
`pi-node` harness runs the **real vendored [pi](https://github.com/earendil-works/pi) agent** —
actual context management, actual harness code — as a process inside an
[E2B](https://e2b.dev) sandbox, one sandbox per (scenario × pass), **all rollouts in parallel**.
The environment stays the world-model simulation on every backend: the sandbox only hosts the
agent process, its tool calls come back over a stdin/stdout frame channel and are answered
host-side by the world model, and the worker LLM is completed host-side too — **no provider
credentials ever enter a sandbox**.

```bash
uv sync --extra e2b                # the e2b SDK is an optional extra
export E2B_API_KEY=...             # sandboxes; the only credential involved
uv run wmh harness create my-agent --tasks tasks.jsonl --harness-backend e2b
uv run wmh eval tasks.jsonl --mode closed-loop --harness pi-agent --harness-backend e2b
```

Sandboxes are pooled and reused across the whole search (bootstrap paid once, lifetimes
auto-extended). Set `WMH_E2B_TEMPLATE` to a prebaked template with node ≥ 22.6 and pi's npm deps
at `/home/user/pi-run` to skip per-sandbox installs (~13 s cold episodes); `--eval-concurrency`
caps the fan-out (default: every cell at once). Worker-LLM tokens and sandbox-seconds are metered
on the results (`worker_usage`, `sandbox_usage`).

## Agentic mode: knowledge base, reasoning, web grounding

Beyond retrieval, a world model can act like an *agent* about its own environment (all opt-in):

- **Knowledge base** — `wmh build --knowledge` extracts the environment's canonical facts
  (business rules, state-dependent gates, entities, tool schemas) from the train traces into
  `models/<name>/knowledge/*.md`. It's plain markdown: edit it in any editor (`wmh knowledge`
  prints the path), read/write it over HTTP (`GET/PUT /world_models/{name}/knowledge`), and the
  env keeps it in every prompt and appends its own cross-session notes to `learned.md`.
- **Reasoning** — `--reasoning` switches the output contract to deliberate-then-answer: the env
  checks the knowledge base's gates (auth, availability, preconditions) and the session history
  before deciding success vs. error.
- **Web grounding** — `--grounder brave` (env var `BRAVE_SEARCH_API_KEY`, free tier) lets the env
  issue a bounded web search when an action references a real-world entity outside its traces
  and knowledge — instead of hallucinating it; `--grounder fetch` (keyless) additionally
  live-fetches the action's own read-only `curl` GET URLs. Results are cached into the knowledge
  base; the default is `none`, so tests and evals never touch the network.
- **Verify pass** — `--verify` adds a second self-check completion per step: the env re-examines
  its draft against the gates, history, and exact computations before answering (~2× serve cost;
  measured to pay off exactly where content prediction is hardest).

## Fidelity: one knob at build, one switch at run

Build effort is a **tier**, not an iteration count:

```bash
wmh build --fidelity low     # RAG only — index the traces, ship the base prompt (near-free)
wmh build --fidelity medium  # + a light prompt-optimization (GEPA) pass        (default)
wmh build --fidelity high    # + full GEPA + a cheap auto-config search
wmh build --fidelity max     # deep GEPA + the full config ladder, to be certain
```

`high`/`max` additionally search the agentic configs (reason / +knowledge / +verify / +fetch)
on the build's held-out split — candidates pruned by a zero-token corpus signature, ties going
to the cheaper config — and record the winner in the artifact's `auto_fidelity.json`.

At run time you either just run it (pure RAG, always), or ask for everything:

```bash
wmh serve --max-fidelity     # the build-measured winning config (or all extras if unmeasured)
wmh play  --max-fidelity
```

Measure any configuration explicitly with `wmh eval run <suite> --knowledge --reasoning` (the
eval seeds its knowledge from the train split only — never from held-out traces).

## Providers

One interface, four backends, verified on startup. Credentials are read from the environment:

| Provider | Model | Env vars |
|---|---|---|
| Anthropic | Claude Opus | `ANTHROPIC_API_KEY` |
| AWS Bedrock | Claude Opus | `AWS_REGION`, `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY` |
| Azure OpenAI | GPT | `AZURE_OPENAI_API_KEY`, `AZURE_OPENAI_ENDPOINT` |
| OpenAI | GPT | `OPENAI_API_KEY` |

## The monorepo

This repository is a [uv workspace](https://docs.astral.sh/uv/concepts/projects/workspaces/):
`wmh` is the flagship package at the root (the quickstart above), and sibling packages live under
`packages/`, each installable on its own:

| Package | What it does | Get it |
|---|---|---|
| **wmh** (root) | Agent traces → a faithful world model of your environment | the quickstart above |
| [`packages/llm-waterfall/`](./packages/llm-waterfall) | Pool LLM quota across models, providers, and AWS accounts: stateless failover that spills only on capacity errors, returning cost + the full attempt trail | `pip install "llm-waterfall @ git+https://github.com/experientiallabs/world-model-harness#subdirectory=packages/llm-waterfall"` *(PyPI release pending)* |
| [`packages/environment-capture/`](./packages/environment-capture) | Point it at any agent benchmark: integrate via a small adapter, capture every real agent-environment transition as OTel GenAI JSONL; 27k+ transitions already published on the [Hub](https://huggingface.co/experiential-labs) | `pip install environment-capture` |

One clone, one `uv sync`, one gate (`just gate`); each package is built and released independently.

## Development

Managed with [uv](https://docs.astral.sh/uv/); linting/formatting with [ruff](https://docs.astral.sh/ruff/); type checking with [ty](https://github.com/astral-sh/ty). Conventions live in [AGENTS.md](./AGENTS.md).

```bash
uv sync --extra dev      # env + dev tools
uv run ruff check .      # lint
uv run ruff format .     # format
uv run ty check          # type check
uv run pytest -q         # tests
```

## Usage telemetry

`wmh` uses anonymous usage telemetry to track the volume of usage.
Telemetry is strictly metadata. It never includes prompts, traces, actions, observations, file paths,
model names, provider credentials, or raw user content.

Telemetry is enabled by default. To opt out for a project:

```bash
uv run wmh config telemetry disable
```

This writes `.wmh/settings.toml`. You can re-enable it with `uv run wmh config telemetry enable`,
check the current setting with `uv run wmh config telemetry status`, or disable it for a process
with `DO_NOT_TRACK=1` or `WMH_TELEMETRY=0`.
