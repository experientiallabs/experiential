# Experiential

Experiential is an OpenAI-compatible gateway for agent workflows. It routes requests across
provider models, keeps serving budgets and usage accountable, and can run locally on loopback.

![Requests flow through the Experiential gateway](https://raw.githubusercontent.com/experientiallabs/world-model-optimizer/main/assets/experiential-workflow.png)

<p align="center">
  🌐 <a href="https://platform.experientiallabs.ai">Platform</a> |
  📚 <a href="https://github.com/experientiallabs/world-model-optimizer/tree/main/docs">Docs</a> |
  <a href="https://discord.gg/B6sM8xTVwU"><img src="https://cdn.simpleicons.org/discord/5865F2" alt="" width="16" height="16"> Discord</a>
</p>

## Getting Started

Install the package and start the local gateway. The first run guides you through setup:

```bash
pip install experiential
exp run
```

Use it with the OpenAI SDK:

```python
from openai import OpenAI

client = OpenAI(
    api_key="YOUR_EXPERIENTIAL_KEY",
    base_url="http://127.0.0.1:8000/v1",
)

response = client.responses.create(model="YOUR_ALIAS", input="Help me")
print(response.output_text)
```

The gateway exposes OpenAI Chat Completions, Responses, and Models routes. Usage and serving
limits are available at `http://127.0.0.1:8000/usage`.

## Build a Router

When you want to optimize a workflow from traces, build a project and fit its router:

```bash
exp build support-agent traces.otel.jsonl
exp optimize router support-agent
exp run support-agent
```

The build path turns OpenTelemetry traces into grounded task evidence and a frozen routing policy.
See the [docs](https://github.com/experientiallabs/world-model-optimizer/tree/main/docs) for the
full CLI and configuration surface.

## Development

```bash
uv sync --extra dev
uv run ruff format --check .
uv run ruff check .
uv run ty check
uv run pytest -q
```
