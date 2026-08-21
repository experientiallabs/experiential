# Experiential

Experiential is an open source gateway and router for agent workflows:

1. Use hosted, BYOK, and local models through one OpenAI-compatible API.
2. Control which users and agents can use which models, for which use cases, and how much they can spend.
3. Turn production traffic into a custom router or model optimized for quality, speed, and cost.

![Experiential workspace usage dashboard showing model traffic, identities, and spend](https://raw.githubusercontent.com/experientiallabs/experiential/main/assets/experiential-workflow.png)

<p align="center">
  🌐 <a href="https://platform.experientiallabs.ai">Platform</a> |
  📚 <a href="https://github.com/experientiallabs/experiential/tree/main/docs">Docs</a> |
  <a href="https://discord.gg/B6sM8xTVwU"><img src="https://cdn.simpleicons.org/discord/5865F2" alt="" width="16" height="16"> Discord</a>
</p>

## Getting Started

Start a local OpenAI-compatible gateway. On first run, the setup wizard uses the shared provider,
model, and reasoning-effort selectors, persists every selected provider connection, then shows
defaults for the public alias, identity, and `$50.00` command budget before printing a one-time key:

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

## Setup / get started with the hosted gateway

Prefer a managed gateway to running one locally? The hosted platform at
[platform.experientiallabs.ai](https://platform.experientiallabs.ai) serves the same
OpenAI-compatible (and Anthropic Messages) API at `https://api.experientiallabs.ai/v1`.
Each link below is a copy-paste prompt you hand to your coding agent (Claude Code, Cursor,
Codex, and similar); the agent runs the setup for you.

- [Upload your LLM traces as telemetry](https://github.com/experientiallabs/setup-prompts/blob/main/01-upload-traces-as-telemetry.md): create an account instantly from your email, then pull or upload your existing LLM traces onto the platform as telemetry.
- [Connect your inference provider keys (BYOK)](https://github.com/experientiallabs/setup-prompts/blob/main/02-connect-provider-keys-byok.md): create an account, then connect your own OpenAI, Anthropic, Gemini, Azure, Bedrock, Fireworks, or OpenRouter keys for free pass-through.
- [Start calling models on the gateway](https://github.com/experientiallabs/setup-prompts/blob/main/03-start-calling-models.md): make your first `/v1` call with the OpenAI and Anthropic SDKs using an `xpl_` key, and optionally repoint your existing coding agents.
- [Full onboarding](https://github.com/experientiallabs/setup-prompts/blob/main/04-full-onboarding.md): create an account instantly from your email, connect your keys, import your spend, then repoint every coding agent (Claude Code, Cursor, Codex, Aider, and similar) or Conductor at the gateway.

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
