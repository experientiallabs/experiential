# Experiential

Experiential is an open source gateway and router for agent workflows:

1. Use models across providers and local deployments through one OpenAI-compatible API.
2. Manage spend and access across identities.
3. Use your traffic to optimize a custom model or router for quality, speed, and cost.

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

## Using the API

Create a local gateway programmatically:

```python
import uvicorn

from exp.runtime.gateway.lifecycle import load_local_gateway

gateway = load_local_gateway()
uvicorn.run(gateway.app, lifespan="on")
```

For hosted workers with their own storage and provider services, use the lower-level
`exp.create_gateway_runtime(...)` composition API.

## Optimize from Traffic

First, collect OpenTelemetry traces from your current agent. If you just want to try it out, grab
the public [terminal-tasks OTLP dataset](https://huggingface.co/datasets/experiential-labs/wmo-terminal-tasks-traces):

```bash
curl -L -o traces.otel.jsonl \
  https://huggingface.co/datasets/experiential-labs/wmo-terminal-tasks-traces/resolve/540883e451dc13d34fb50fdd36b143cb0f1fb0db/traces.otel.jsonl
```

Then build a project. The build command walks you through providers,
models, and budget, and asks for your trace file:

```bash
# Build simulation from your agent traces and optimize a router against it
exp build support-agent
```

After collecting traces from your router, fine-tune an open source model you own using
[Tinker](https://tinker.thinkingmachines.ai/).

```bash
exp optimize model support-agent
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
