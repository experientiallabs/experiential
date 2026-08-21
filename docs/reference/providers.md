# Model providers

Experiential resolves models from a secret-free `.exp/models.toml` catalog. `RuntimeModelCatalog` is the
only construction service. Provider names do not imply capabilities or prices. Every completion or
embedding alias must declare the protocol features and token prices it uses.

Configure connections with `exp config providers` or the first `exp build` on a clean checkout.
An interactive terminal opens a provider list: Up and Down move focus, Enter selects or deselects
the focused provider, and the Complete row submits the selection. Agents skip that list with
repeatable `--provider` flags (`openai`, `anthropic`, `gemini`, `openrouter`,
`openai-compatible`, `azure`, `bedrock`). Unsupported or duplicate values fail before any catalog
write. Azure and Bedrock still require manual model IDs. Other selected providers use account
model discovery when credentials are available.

Setup writes only secret-free catalog fields and never prints a credential value.

## Supported providers

| Provider | Catalog `provider` | Credential | Endpoint identity |
|---|---|---|---|
| OpenAI | `openai` | `api_key_env` (suggested `OPENAI_API_KEY`) | Official OpenAI origin |
| OpenRouter | `openrouter` | `api_key_env` (suggested `OPENROUTER_API_KEY`) | Official OpenRouter origin |
| Anthropic | `anthropic` | `api_key_env` (suggested `ANTHROPIC_API_KEY`) | Official Anthropic origin |
| Gemini | `gemini` | `api_key_env` (suggested `GEMINI_API_KEY`) | Official Gemini origin |
| OpenAI-compatible | `openai-compatible` | `api_key_env` plus explicit `base_url` | Catalog `base_url` |
| Azure OpenAI / Foundry | `azure` | `api_key_env` plus explicit resource endpoint and `api_version` | Endpoint and API version |
| Amazon Bedrock | `bedrock` | AWS credential chain. No `api_key_env` | Optional catalog `region` |
| Tinker sampling | `tinker` | `api_key_env` | Official Tinker origin |

Native fixed-origin providers reject a custom `base_url`. Use `openai-compatible` for a trusted
third-party OpenAI-compatible host.

## OpenAI-compatible listing metadata

`provider = "openai-compatible"` is the only OpenAI-shaped listing path that reads optional
extension fields. Official `openai` listing stays identity-only: extra keys on a model object are
discarded so unofficial metadata cannot become verified OpenAI capabilities or prices.

When an OpenAI-compatible host publishes the following optional fields, setup copies only values
that match the declared types. Absent or wrongly typed fields stay unknown. No context window or
cache-write price is inferred from a neighboring value.

| Field | Type | Meaning |
|---|---|---|
| `supports_completions` | boolean | The alias serves chat or responses completions |
| `supports_tools` | boolean | The alias accepts tools |
| `supports_structured_output` | boolean | The alias accepts structured output |
| `maximum_output_tokens` | positive integer | Declared output ceiling |
| `context_window_tokens` | positive integer | Declared context window, only when the host publishes one |
| `pricing.input_micro_usd_per_million_tokens` | integer `>= 0` | Configured input price in micro-USD per million tokens |
| `pricing.output_micro_usd_per_million_tokens` | integer `>= 0` | Configured output price in micro-USD per million tokens |
| `pricing.cached_input_micro_usd_per_million_tokens` | integer `>= 0` | Configured cached-input price in micro-USD per million tokens |

Micro-USD prices convert to catalog USD-per-million-token prices by dividing by `1_000_000`. A
hosted WMO gateway publishes these fields from its active catalog for each granted alias that
resolves to one deployment. That is enough for world-model and judge setup when tools, structured
output, and input/output prices are present. Router-candidate setup still requires a published
context window and both cache prices; WMO does not invent those.

The hosted WMO `/v1/models` response keeps the standard OpenAI list shape (`object`, `data`, and
the four OpenAI model keys) and only adds these extension fields plus the existing `wmo`
authority marker.

## Azure

Use `provider = "azure"`. The connection needs:

- `base_url`: the Azure resource endpoint, for example `https://myresource.openai.azure.com`
- `api_key_env`: the environment-variable name that holds that resource's key
- `api_version`: `v1` for the current Azure OpenAI and Foundry `/openai/v1` routes, or a dated
  Azure OpenAI version such as `2024-10-21` for classic deployment-in-path routing

The model record `model` field is the exact Azure deployment identifier sent on the wire. Experiential never
derives a deployment from an alias or a base-model name. Use a separate alias for an embedding
deployment.

`AZURE_OPENAI_API_KEY` is paired with `AZURE_OPENAI_ENDPOINT` when that endpoint variable is set.
A catalog endpoint that is not the same resource cannot use that key. Comparison lowercases the
scheme and host, treats default HTTPS and HTTP ports as equivalent to an omitted port, keeps path
case, and ignores a trailing slash. Credentials may not appear in the endpoint URL, query string,
or fragment.

```toml
[connections.azure]
provider = "azure"
base_url = "https://myresource.openai.azure.com"
api_key_env = "AZURE_OPENAI_API_KEY"
api_version = "v1"

[models.gpt]
connection = "azure"
model = "gpt-5-deployment"
[models.gpt.capabilities]
supports_completions = true
supports_tools = true
input_cost_per_million_tokens_usd = 0
output_cost_per_million_tokens_usd = 0
cached_input_cost_per_million_tokens_usd = 0
cache_write_cost_per_million_tokens_usd = 0

[models.embed]
connection = "azure"
model = "text-embedding-deployment"
[models.embed.capabilities]
supports_embeddings = true
input_cost_per_million_tokens_usd = 0
```

## Bedrock

Use `provider = "bedrock"`. Do not set `api_key_env`. Credentials come from the standard AWS chain
(environment keys, shared config, profile, role, web identity, container, or instance role). Static
`AWS_ACCESS_KEY_ID` variables are optional.

Region is resolved in this order:

1. Catalog `region`
2. `AWS_REGION`
3. The boto session chain, including `AWS_DEFAULT_REGION`, the active profile region, and the
   instance role

The catalog region is recommended and is part of connection identity when set. Catalog loading and
`snapshot()` do not import boto or inspect instance metadata.

The model record `model` field is the exact foundation-model or inference-profile ID. Use a
separate alias for an embedding model such as Titan. Do not combine completion and embedding IDs
on one record.

```toml
[connections.bedrock]
provider = "bedrock"
region = "us-east-1"

[models.claude]
connection = "bedrock"
model = "us.anthropic.claude-sonnet-4-5"
[models.claude.capabilities]
supports_completions = true
supports_tools = true
input_cost_per_million_tokens_usd = 0
output_cost_per_million_tokens_usd = 0
cached_input_cost_per_million_tokens_usd = 0
cache_write_cost_per_million_tokens_usd = 0

[models.titan]
connection = "bedrock"
model = "amazon.titan-embed-text-v2:0"
[models.titan.capabilities]
supports_embeddings = true
input_cost_per_million_tokens_usd = 0
```

## Errors

Missing credentials name the environment variable, never its value. A missing Bedrock region lists
the resolution order. Azure endpoint and key mismatches name `AZURE_OPENAI_ENDPOINT`, not the key.
Malformed provider responses fail closed and do not write partial catalog or evidence artifacts.
