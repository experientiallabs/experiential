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
