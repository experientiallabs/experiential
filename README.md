# World Model Optimizer

WMO optimizes agent workflows from traces through a three-step process:

1. Build a simulation using text world models grounded on your traces.
2. Fit a router that determines which model every request should be sent to.
3. Train custom open source models just for your agent.

![Simulation, routing, and optimization](https://raw.githubusercontent.com/experientiallabs/world-model-optimizer/main/assets/wmo-workflow.svg)

<p align="center">
  <a href="https://platform.experientiallabs.ai">Platform</a> ·
  <a href="https://github.com/experientiallabs/world-model-optimizer/tree/main/docs">Docs</a> ·
  <a href="https://discord.gg/B6sM8xTVwU">Discord</a>
</p>

## Getting Started

To get started, install the package and build a project using collected OpenTelemetry traces from
your current agent:

```bash
pip install world-model-optimizer

# Build simulation from your agent traces
wmo build support-agent traces.otel.jsonl

# Optimize a router against the simulation to use the best model for every task
wmo optimize router support-agent

# Run your router as an OpenAI compatible endpoint
wmo run support-agent

# Send a request to your endpoint
curl http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"support-agent","messages":[{"role":"user","content":"Help me"}]}'
```

After collecting traces from your router, fine-tune an open source model you own using
[Tinker](https://tinker.thinkingmachines.ai/).

```bash
wmo optimize model support-agent
```

## Using the API

Call an optimized router programmatically:

```python
from pathlib import Path

from wmo import load_project_router
from wmo.common.models import ModelMessage, ModelRequest

router = load_project_router("support-agent", Path(".wmo"))

result = router.complete(
    ModelRequest(
        messages=(
            ModelMessage(role="user", content="Help me reset my password"),
        ),
    )
)
```

## Telemetry

Anonymous aggregate PostHog product telemetry is enabled by default. It never includes prompts,
traces, actions, observations, paths, model names, credentials, or raw customer content.

```bash
wmo config telemetry status
wmo config telemetry disable
wmo config telemetry enable
```

The preference is stored locally in `.wmo/settings.toml`.

## Development

```bash
uv sync --extra dev
uv run ruff format --check .
uv run ruff check .
uv run ty check
uv run pytest -q
```

Repository and documentation conventions live in [AGENTS.md](./AGENTS.md).
