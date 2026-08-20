# Experiential

Experiential optimizes agent workflows from traces through a three-step process:

1. Build a simulation using text world models grounded on your traces.
2. Fit a router that determines which model every request should be sent to.
3. Train custom open source models just for your agent.

![Your traces flow through simulation into routing and training optimization](https://raw.githubusercontent.com/experientiallabs/experiential/main/assets/experiential-workflow.png)

<p align="center">
  🌐 <a href="https://platform.experientiallabs.ai">Platform</a> |
  📚 <a href="https://github.com/experientiallabs/experiential/tree/main/docs">Docs</a> |
  <a href="https://discord.gg/B6sM8xTVwU"><img src="https://cdn.simpleicons.org/discord/5865F2" alt="" width="16" height="16"> Discord</a>
</p>

## Getting Started

Start a local OpenAI-compatible gateway. On first run, the setup wizard asks for a provider,
model, and public alias, then prints a one-time virtual key:

```bash
pip install experiential
exp run
```

Choose a public alias such as `support-agent`, capture the issued key, and send a request:

```bash
export EXP_GATEWAY_KEY=...
curl http://127.0.0.1:8000/v1/chat/completions \
  -H "Authorization: Bearer $EXP_GATEWAY_KEY" \
  -H 'Content-Type: application/json' \
  -d '{"model":"support-agent","messages":[{"role":"user","content":"Help me"}]}'
```

## Optimize from Traffic

First, collect OpenTelemetry traces from your current agent. If you just want to try it out, grab
the public [terminal-tasks OTLP dataset](https://huggingface.co/datasets/experiential-labs/wmo-terminal-tasks-traces):

```bash
curl -L -o traces.otel.jsonl \
  https://huggingface.co/datasets/experiential-labs/wmo-terminal-tasks-traces/resolve/540883e451dc13d34fb50fdd36b143cb0f1fb0db/traces.otel.jsonl
```

Then install the package and build a project. The build command walks you through providers,
models, and budget, and asks for your trace file:

```bash
# Build simulation from your agent traces and optimize a router against it
exp build support-agent

# Run your router as an OpenAI compatible endpoint
exp run support-agent

# Send a request to your endpoint
curl http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"support-agent","messages":[{"role":"user","content":"Help me"}]}'
```

After collecting traces from your router, fine-tune an open source model you own using
[Tinker](https://tinker.thinkingmachines.ai/).

```bash
exp optimize model support-agent
```

## Using the API

Call an optimized router programmatically:

```python
from pathlib import Path

from exp import load_project_router
from exp.common.models import ModelMessage, ModelRequest

router = load_project_router("support-agent", Path(".exp"))

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
exp config telemetry status
exp config telemetry disable
exp config telemetry enable
```

The preference is stored locally in `.exp/settings.toml`.

## Development

```bash
uv sync --extra dev
uv run ruff format --check .
uv run ruff check .
uv run ty check
uv run pytest -q
```

Repository and documentation conventions live in [AGENTS.md](./AGENTS.md).
