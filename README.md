# World Model Optimizer

WMO optimizes agent workflows from traces through a three-step process:

1. Build a simulation using text world models grounded on your traces.
2. Fit a router that determines which model every request should be sent to.
3. Train custom open source models just for your agent.

![Your traces flow through simulation into routing and training optimization](https://raw.githubusercontent.com/experientiallabs/world-model-optimizer/main/assets/wmo-workflow.png)

<p align="center">
  🌐 <a href="https://platform.experientiallabs.ai">Platform</a> |
  📚 <a href="https://github.com/experientiallabs/world-model-optimizer/tree/main/docs">Docs</a> |
  <a href="https://discord.gg/B6sM8xTVwU"><img src="https://cdn.simpleicons.org/discord/5865F2" alt="" width="16" height="16"> Discord</a>
</p>

## Getting Started

To get started, install the package and build a project using collected OpenTelemetry traces from
your current agent:

```bash
pip install world-model-optimizer

# Collect secret-free provider connections, including azure and bedrock
wmo config providers

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

To exercise the build path with a public trace export, download the
[terminal-tasks OTLP dataset](https://huggingface.co/datasets/experiential-labs/wmo-terminal-tasks-traces)
and pass it to the same command without a source adapter or conversion step:

```bash
curl -L -o traces.otel.jsonl \
  https://huggingface.co/datasets/experiential-labs/wmo-terminal-tasks-traces/resolve/540883e451dc13d34fb50fdd36b143cb0f1fb0db/traces.otel.jsonl
wmo build terminal-tasks traces.otel.jsonl
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

## Command budgets

Paid commands share one cost preflight and a user-owned per-command ceiling. The default is
`$10.00`; change or inspect it without prompting:

```bash
wmo config budget
wmo config budget 50
```

Every preflight names the command, conservative estimate, configured budget, and major bounded
assumptions. Estimates at or below 50% of the budget run automatically. Higher in-budget estimates
require a clear terminal confirmation or `--yes`. Estimates above the budget fail before
credentials or provider clients, and `--yes` cannot override that ceiling.

## Telemetry

Anonymous aggregate PostHog product telemetry is enabled by default. It never includes prompts,
traces, actions, observations, paths, model names, credentials, or raw customer content.

```bash
wmo config telemetry status
wmo config telemetry disable
wmo config telemetry enable
```

The command budget and telemetry preference are stored locally in `.wmo/settings.toml`.

## Development

```bash
uv sync --extra dev
uv run ruff format --check .
uv run ruff check .
uv run ty check
uv run pytest -q
```

Repository and documentation conventions live in [AGENTS.md](./AGENTS.md).
