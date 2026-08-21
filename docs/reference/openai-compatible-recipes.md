# OpenAI-compatible connection recipes: Fireworks and Modal

The gateway's `openai-compatible` provider family serves every endpoint that speaks the
OpenAI Chat Completions SSE protocol, selected purely by `base_url`. Fireworks and
Modal-deployed endpoints are two such services. They are deliberately connection recipes
rather than new provider names: the provider vocabulary is a frozen contract shared by
catalog validation, provider setup, the certification matrix, and the CLI, so an endpoint
that differs from the generic adapter only by its base URL never widens that vocabulary.

Every command and constant below is verified against the current tree by
`exp/cli/tests/openai_compatible_recipes_test.py`.

## Fireworks AI

Fireworks serves its hosted models at one fixed base URL:

```text
https://api.fireworks.ai/inference/v1
```

Authentication uses a bearer key. Keep the key in an environment variable (the catalog
stores only the variable name, never the value), and use the Fireworks model ID form
`accounts/fireworks/models/<model-name>`.

```bash
export FIREWORKS_API_KEY=...   # from https://app.fireworks.ai

exp config gateway init --root ROOT --json
exp config gateway provider add fireworks \
  --provider openai-compatible \
  --base-url https://api.fireworks.ai/inference/v1 \
  --credential-env FIREWORKS_API_KEY \
  --root ROOT --non-interactive --json
```

Then bind one exact model as a direct alias. The capability flags are the template for a
current Fireworks chat model: most support tools and structured output, and none require
developer messages. Confirm the flags and the micro-USD-per-million-token prices against
the Fireworks model page before authoring, because the gateway treats both as frozen
authority for routing and accounting.

The `--deployment` model segment is the exact provider-side spelling with its slashes;
`--exact-model` is your stable logical model identity and must use lowercase components
separated by `.`, `_`, or `-` (for example `llama-v3p1-70b-instruct`).

```bash
exp config gateway alias create coding \
  --deployment fireworks:accounts/fireworks/models/llama-v3p1-70b-instruct \
  --exact-model llama-v3p1-70b-instruct \
  --supports-tools --supports-structured-output \
  --input-price 900000 --output-price 900000 \
  --pricing-source "https://fireworks.ai/pricing" \
  --root ROOT --non-interactive --json

exp config gateway identity create default --root ROOT --non-interactive --json
exp config gateway grant add default coding --root ROOT --non-interactive --json
exp config gateway key issue default --key-id key-one --root ROOT --json
exp --root ROOT --check --non-interactive --json
```

`exp --check` proves readiness without a paid call; drop `--check` to serve, then use
the caller loop (`exp config gateway key check`, `models`, and `call`) against the live
gateway.

## Modal

Modal endpoints are org-deployed web apps (for example a vLLM server deployed with
`modal deploy`), so there is no default base URL: every deployment has its own hostname
of the form `https://<workspace>--<app-label>.modal.run`, and the connection fails closed
at startup until an explicit `base_url` is configured. Point `base_url` at the path that
serves the OpenAI protocol, typically the `/v1` suffix, and reference whatever bearer
token the deployment enforces (Modal proxy-auth tokens or an app-defined API key).

```bash
export MODAL_API_KEY=...       # the token your Modal deployment checks

exp config gateway provider add modal-vllm \
  --provider openai-compatible \
  --base-url https://WORKSPACE--APP_LABEL.modal.run/v1 \
  --credential-env MODAL_API_KEY \
  --root ROOT --non-interactive --json

exp config gateway alias create local-serve \
  --deployment modal-vllm:SERVED_MODEL_ID \
  --exact-model served-model-id \
  --input-price 0 --output-price 0 \
  --pricing-source "self-hosted Modal deployment" \
  --root ROOT --non-interactive --json
```

`SERVED_MODEL_ID` is the model name the deployment reports in its own `GET /v1/models`;
self-hosted deployments usually bill through Modal compute rather than per token, so zero
token prices with an explicit `--pricing-source` keep the accounting honest. If the
served endpoint reports a different model identity than the one requested, pin it during
`exp config providers` setup with `served_model_id` instead of authoring a mismatched
alias.

## Build-time equivalents

The same two connections work for evidence builds and router optimization through
`exp config providers` by choosing the `openai-compatible` provider with the identical
`base_url` and credential environment variable; see `reference/providers.md` for that
flow.

Any trusted OpenAI-compatible host is usable from that command, including an identity-only
`GET /v1/models` response. Setup lists those identities and labels missing capabilities and
prices as unknown. The operator then declares the minimum fields for the selected role in the
interactive screens, or authors the same configured metadata with `--non-interactive`
`--connection-json` / `--model-json` (or a hand-written `.exp/models.toml` record). Setup never
infers tools, structured output, token limits, or prices.

A hosted Experiential gateway is the same provider family and an optional richer listing. Point
`base_url` at its `/v1` origin and present a granted virtual key. `GET /v1/models` keeps the
OpenAI list shape and may publish per-alias completion capabilities, limits, and configured
micro-USD prices from the active catalog. `exp config providers --provider openai-compatible`
reads those optional fields, converts micro-USD prices to USD per million tokens, and leaves
unknown any field the catalog did not declare. It does not treat official OpenAI listing
metadata the same way.
