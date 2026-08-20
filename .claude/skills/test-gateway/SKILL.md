---
name: test-gateway
description: End-to-end happy-path test of the Experiential local gateway, covering interactive first-run setup via the exp run TUI, the served OpenAI-compatible routes, usage accounting, restart persistence, and integration with an external OpenAI-compatible client such as opencode. Use when asked to test, validate, or demo the local gateway.
---

# Test the local gateway end to end

Verified procedure for exercising the gateway happy path from a clean state.
Run everything from the repo root with `uv run` so the project environment is used.

## 1. Interactive first-run setup (TUI)

`exp run` on a real TTY with an uninitialized root opens the first-run wizard
(`exp/cli/gateway/setup.py`). Pick a fresh root, then:

```bash
uv run exp run --root /path/to/fresh/.exp --port 8822
```

Known-good prompt answers (OpenAI provider, real key in `OPENAI_API_KEY`):

| Prompt | Answer |
| --- | --- |
| Provider connection name | `openai` |
| Provider adapter | `openai` |
| Credential environment variable name | `OPENAI_API_KEY` |
| Base URL | empty (provider default) |
| Exact provider model ID | `gpt-5.4-mini` |
| Exact logical model identity | `gpt-5.4-mini` |
| Public model alias | `mini` |
| Default identity | accept `default` |
| Create this gateway configuration? | `y` |

Identifiers must match `^[a-z][a-z0-9]*(?:[._-][a-z0-9]+)*$` (lowercase, dot/underscore/hyphen separators).

On success it prints "Gateway ready at http://127.0.0.1:8822/v1", the usage URL,
and a one-time virtual key (`export OPENAI_API_KEY=exp_vk_...`), then serves uvicorn.
The raw key is shown exactly once: capture it immediately.

Non-TTY contexts (pipes, `--json`, `--non-interactive`) never prompt; they exit 2 with
`gateway_not_initialized` plus the exact non-interactive setup commands.

## 2. Verify the served routes

With `K` set to the issued virtual key:

```bash
curl http://127.0.0.1:8822/v1/models -H "Authorization: Bearer $K"                  # expect alias "mini"
curl http://127.0.0.1:8822/v1/chat/completions -H "Authorization: Bearer $K" \
  -H "Content-Type: application/json" \
  -d '{"model":"mini","messages":[{"role":"user","content":"Reply with exactly: gateway-ok"}]}'
# stream:true returns SSE chunks ending in "data: [DONE]"
curl http://127.0.0.1:8822/v1/responses -H "Authorization: Bearer $K" \
  -H "Content-Type: application/json" -d '{"model":"mini","input":"Say hi"}'
curl -o /dev/null -w '%{http_code}' http://127.0.0.1:8822/v1/models                  # keyless: expect 401
```

Usage accounting: open `http://127.0.0.1:8822/usage` (HTML) or run
`uv run exp config gateway usage --root ROOT --json`. With wizard-default empty prices,
attributed cost stays 0 and all attempts are counted in `unknown_cost_attempts`; that is
by design, not a bug.

## 3. Restart persistence

Ctrl+C the server and rerun the same `exp run` command: it must serve immediately with
no setup prompts, and previously issued keys must keep working.

## 4. Issue additional keys non-interactively

`key issue` on a non-TTY requires `--json` or `--output` (the raw key is revealed once):

```bash
uv run exp config gateway key issue default --key-id KEY_ID --root ROOT \
  --output /path/to/key.file --json
```

## 5. External client integration (opencode)

Install opencode with a user-level npm prefix (global install needs no sudo that way):

```bash
npm config set prefix ~/.npm-global
npm install -g opencode-ai
export PATH=$HOME/.npm-global/bin:$PATH
```

In a scratch project directory, write `opencode.json`:

```json
{
  "$schema": "https://opencode.ai/config.json",
  "provider": {
    "exp": {
      "npm": "@ai-sdk/openai-compatible",
      "options": {
        "baseURL": "http://127.0.0.1:8822/v1",
        "apiKey": "{env:EXP_GATEWAY_KEY}"
      },
      "models": { "mini": {} }
    }
  },
  "model": "exp/mini"
}
```

Then run prompts non-interactively:

```bash
export EXP_GATEWAY_KEY=$(cat /path/to/key.file)
opencode run --model exp/mini "Reply with exactly: opencode-through-exp-gateway-ok"
```

Verify agentic tool calls round-trip (ask it to create or read a file) and confirm the
gateway attributed the requests by re-reading the usage report: each opencode turn is
one `POST /v1/chat/completions`, so a tool-using task produces several requests.
