# Model providers

Experiential resolves models from a secret-free `.exp/models.toml` catalog. `RuntimeModelCatalog` is the
only construction service. Provider names do not imply capabilities or prices. Every completion or
embedding alias must declare the protocol features and token prices it uses.

Configure connections with `exp config providers` or the first `exp build` on a clean checkout.
An interactive terminal opens a provider list: Up and Down move focus, Enter selects or deselects
the focused provider, and the Complete row submits the selection. Agents skip that list with
repeatable `--provider` flags (`experiential-cloud`, `openai`, `anthropic`, `gemini`, `openrouter`,
`openai-compatible`, `azure`, `bedrock`). Unsupported or duplicate values
fail before any catalog write. Azure and Bedrock still require manual model IDs. Other selected
providers use account model discovery when credentials are available.

`exp login` is the first-party authentication command for the hosted Platform gateway. It opens
the Platform `/cli/auth` approval page, receives an organization `xpl_` key through the loopback
callback, and stores that key in the user-data credential file. `experiential-cloud` is the setup
picker for the same gateway. It persists
`provider = "openai-compatible"` with `base_url` `https://api.experientiallabs.ai/v1` (or
`EXP_GATEWAY_URL` when that override is set) and `api_key_env = "EXPLABS_API_KEY"`. The CLI does
not rebuild a local gateway authority for that hosted path. Login also registers the
`experiential-cloud` connection in the selected `.exp/models.toml` and synchronizes every model
identity returned for the authenticated account. Local gateway setup can then select that
provider without a second provider-configuration command.

`exp config providers --provider experiential-cloud` remains available when roles need to be
assigned, or when an explicit refresh or replacement workflow is preferred. When that setup flow
has no environment or stored key, it can also open the same Platform approval flow as a
convenience. Set `EXP_PLATFORM_URL` for a preview or staging Platform web origin. Headless and CI
setup should provide `EXPLABS_API_KEY` instead.

Setup writes only secret-free catalog fields and never prints a credential value. Interactive
`exp config providers` persists browser-approved or pasted API keys outside the repository in the platform
user-data file (`$XDG_DATA_HOME/exp/auth.json` or `~/.local/share/exp/auth.json` on Linux).
When the operator edits an existing provider that already has a stored key, the same wizard
can keep, replace, or remove that record. Known providers keep their canonical environment
override names internally. Custom OpenAI-compatible connections keep a generated or
already-configured override name, but the operator pastes the key rather than typing a
variable name.

Runtime credential resolution for one connection ID:

1. An explicit environment mapping supplied by the caller, when one is present
2. Otherwise the configured process environment variable, when it is non-empty
3. The stored local credential for that exact connection

The caller mapping, when supplied, is the only environment consulted and does not rewrite the
store. Runtime, `--non-interactive`, and CI paths never prompt. A missing credential fails with
the environment name and a recovery that points at `exp config providers`. Amazon Bedrock is
unchanged: it uses the AWS credential chain and has no stored API key.

## Supported providers

| Provider | Catalog `provider` | Credential | Endpoint identity |
|---|---|---|---|
| OpenAI | `openai` | `api_key_env` (suggested `OPENAI_API_KEY`) | Official OpenAI origin |
| OpenRouter | `openrouter` | `api_key_env` (suggested `OPENROUTER_API_KEY`) | Official OpenRouter origin |
| Anthropic | `anthropic` | `api_key_env` (suggested `ANTHROPIC_API_KEY`) | Official Anthropic origin |
| Gemini | `gemini` | `api_key_env` (suggested `GEMINI_API_KEY`) | Official Gemini origin |
| OpenAI-compatible | `openai-compatible` | `api_key_env` plus explicit `base_url` | Catalog `base_url` |
| Experiential Cloud | `openai-compatible` (picker `experiential-cloud`) | `EXPLABS_API_KEY` plus the hosted Platform `/v1` origin | `https://api.experientiallabs.ai/v1` or `EXP_GATEWAY_URL` |
| Azure OpenAI / Foundry | `azure` | `api_key_env` plus explicit resource endpoint and `api_version` | Endpoint and API version |
| Amazon Bedrock | `bedrock` | AWS credential chain. No `api_key_env` | Optional catalog `region` |
| Vertex AI | `vertex` | `api_key_env` holding service-account JSON | Project-and-location `base_url` |
| Tinker sampling | `tinker` | `api_key_env` | Official Tinker origin |
| TypeSafe SystemOne | `typesafe` | `api_key_env` (suggested `TYPESAFE_API_KEY`) | `https://api.typesafe.ai/v1/systemone`, native decisions only |

Native fixed-origin providers reject a custom `base_url`. Use `openai-compatible` for a trusted
third-party OpenAI-compatible host.

## TypeSafe SystemOne decisions

Use `provider = "typesafe"` with an explicitly authored catalog connection and deployment.
The interactive provider picker does not offer TypeSafe. The connection uses the official
`https://api.typesafe.ai/v1` root and sends its provider credential as a Bearer header to
`/systemone`; it exposes no chat, Responses, or embeddings endpoint. The public gateway route is
`POST /v1/systemone`, authenticated with a gateway virtual key, not the provider key.

A deployment must declare `gateway.capabilities.supports_decisions = true`. Its ordinary
`ModelCapabilities` should declare `supports_completions = false` and
`supports_embeddings = false`; decision support is gateway metadata, not an addition to the
frozen model-capability identity. Admission requires a known nonnegative
`gateway.prices.input_nano_usd_per_million_tokens` and an explicit
`gateway.prices.output_nano_usd_per_million_tokens = 0`. Unknown prices are not free prices.
Only direct exact-model pools serve decisions; project-backed aliases are refused.

Send exactly three top-level fields: `model` (the granted public alias), `state`, and `questions`.
State and each question's instructions may be a string, JSON object, or array. This example uses
all three question types; replace `systemone` with the decision alias your gateway grants:

```json
{
  "model": "systemone",
  "state": {"payment_received": true, "message": "Please send my invoice."},
  "questions": {
    "paid": {"type": "noul", "instructions": "Was payment received?"},
    "topic": {
      "type": "choice",
      "instructions": "Classify the customer's request.",
      "criteria": {"billing": "Invoices and payments", "other": "Anything else"}
    },
    "urgency": {
      "type": "score",
      "instructions": "Rate the urgency.",
      "criteria": ["low", "medium", "high"]
    }
  }
}
```

| Type | Question criteria | Answer fields |
|---|---|---|
| `noul` | Optional object with `true` and `false` descriptions | `type`, `noul` (probability from 0 to 1) |
| `choice` | Object mapping 2 through 64 category names to text, object, or array descriptions, or `null` | `type`, `choice`, `confidence`, `probabilities` keyed by the requested names |
| `score` | Ordered array of 2 through 10 text, object, or array descriptions | `type`, `score`, `confidence`, `legend`, `probabilities` keyed by zero-based index strings |

The response contains `id`, the requested public `model` alias, `answers` under the original
question IDs, and `usage.input_tokens` / `usage.output_tokens`. These are structured decision
values, not assistant text. Probabilities and confidence must be finite values from 0 to 1,
distributions must sum to 1 within validation tolerance, the selected choice must have the highest
probability, and a score must match the distribution's weighted zero-based index. Missing answers, mismatched types or criteria, and missing or invalid
usage fail closed. Only validated answer fields are returned; extra provider metadata is omitted.

Limits are 1 through 32 questions, 1 through 256 UTF-8 bytes per question ID or choice category
name, and at most 262,144 bytes for both the raw body and the normalized request. Duplicate JSON
keys, non-finite numbers, unknown fields, chat messages, tools, generation controls, and `stream`
are rejected before acceptance. Nested data must use valid UTF-8, integers in the inclusive range
`-2^63` through `2^64 - 1`, and at most 64 JSON levels. These are gateway serialization bounds,
not TypeSafe token limits. Objects and arrays retain their structure instead of becoming strings. Responses are buffered, never streamed. `Idempotency-Key` is
ignored: resubmitting the same request is a new operation, not a replay. There is no continuation,
prompt-based project selection, or chat input/output guardrail processing on this surface.

The gateway reserves a bounded estimate for each dispatch: serialized state is counted for every
question, question instructions and criteria add to input, and per-question protocol allowances
cover input and output. This is an accounting estimate, not a provider-enforced token limit;
no synthetic `max_tokens` field is sent. Settlement uses only TypeSafe's reported token counts,
including reported output tokens even though their configured price is zero. Unknown outcomes
retain a content-free unknown-cost attempt record and keep the monetary reservation held, with
no invented usage, settled charge, or automatic retry. Only HTTP 400, 401, 403, 404, and 422
establish a known rejection that releases the reservation without inventing zero-token usage.
HTTP 402, 429, and 529 do not prove no work occurred: payment, throttle, and overload responses
are terminal unknown outcomes with the hold retained, not automatic fallback signals. A known
401 authentication rejection may advance to the next certified deployment. The route permits at
most eight deployments and one dispatch per deployment, with no same-deployment or throttle redials.

For a local SQLite gateway, an operator resolves one terminal decision hold explicitly through
`SQLiteAttemptLedger.reconcile_decision_liability(attempt_id=..., assigned_cost_nano_usd=...)`
after checking the provider's outcome. Zero releases the hold; a positive assignment records
that budget amount while token usage and the provider cost estimate stay unknown. Repeating the
same assignment is a no-op; a different assignment is refused. Holds survive request completion
and process restarts until this explicit reconciliation, so unresolved work cannot fund repeats.

## OpenAI-compatible listing metadata

`provider = "openai-compatible"` is the only OpenAI-shaped listing path that reads optional
extension fields. Official `openai` listing stays identity-only: extra keys on a model object are
discarded so unofficial metadata cannot become verified OpenAI capabilities or prices.

Discovery and verification stay separate. `exp config providers --provider openai-compatible`
lists every identity returned by a trusted operator-supplied endpoint. Official `openai` listing
never becomes a capability source. When the compatible host also publishes the following optional
fields, setup copies only values that match the declared types. Absent or wrongly typed fields
stay unknown. No context window or cache-write price is inferred from a neighboring value.

Identity-only rows remain visible and selectable. Setup labels them `unknown capabilities/prices`
and does not assign a build role until the operator declares the minimum fields that role needs.
The interactive flow confirms published values and asks only for missing required fields. The
deterministic equivalent is `exp config providers --non-interactive` with `--connection-json` and
`--model-json`, or a hand-authored `.exp/models.toml` record. Those declarations become configured
catalog metadata. Downstream cost and router-candidate preflights stay fail-closed while a
required price or limit remains unknown.

| Field | Type | Meaning |
|---|---|---|
| `supports_completions` | boolean | The alias serves chat or responses completions |
| `supports_tools` | boolean | The alias accepts tools |
| `supports_structured_output` | boolean | The alias accepts structured output |
| `maximum_output_tokens` | positive integer | Declared output ceiling |
| `context_window_tokens` | positive integer | Declared context window, only when the host publishes one |
| `pricing.input_nano_usd_per_million_tokens` | integer `>= 0` | Configured input price in nano-USD per million tokens |
| `pricing.output_nano_usd_per_million_tokens` | integer `>= 0` | Configured output price in nano-USD per million tokens |
| `pricing.cached_input_nano_usd_per_million_tokens` | integer `>= 0` | Configured cached-input price in nano-USD per million tokens |

Nano-USD prices convert to catalog USD-per-million-token prices by dividing by `1_000_000_000`
(one nano-USD is a billionth of a dollar; `1_250_000_000` is $1.25 per million tokens).
When a trusted third-party compatible host publishes these fields, completion, tool,
structured-output, and input/output price declarations can assign world-model and judge roles
without a questionnaire. Router-candidate setup still requires a published or
operator-declared context window and both cache prices. Setup never invents missing values.

The hosted gateway `/v1/models` response is the strict OpenAI discovery surface. Its list
envelope contains only `object` and `data`; each model contains only `id`, `object`,
`created`, and `owned_by`. Hosted capability and price discovery belongs to the platform
catalog API, not to additive fields on the OpenAI endpoint.

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

## Anthropic inference geography

For a reviewed first-party Claude 4.6+ route, set `inference_geo = "us"`
on its `anthropic` connection. The connection identity includes the constraint.
Both ordinary completions and gateway Messages payloads include it; the gateway
applies it after translating Chat Completions, Responses, or Messages input.
Caller-supplied `inference_geo = "global"` cannot override this connection setting.
Each fallback needs its own constrained connection. No setting leaves existing
caller behavior unchanged. Unsupported models return a provider error; model
eligibility and the provider's 10% regional surcharge belong in the host catalog.

```toml
[connections.anthropic_us]
provider = "anthropic"
api_key_env = "ANTHROPIC_API_KEY"
inference_geo = "us"
```

[Anthropic data residency](https://platform.claude.com/docs/en/manage-claude/data-residency)
documents eligible models and scope. This parameter does not establish a
geography guarantee for a proxy, Bedrock, Vertex, or Foundry endpoint.

## Vertex AI

Use `provider = "vertex"` for Google-published models served from a Google Cloud project.
The connection needs two values: `base_url` naming the project-and-location root, and
`api_key_env` naming an environment variable whose value is the full service-account JSON key
file contents. The runtime mints short-lived OAuth bearer tokens from that credential; the
JSON itself never travels on the wire, and the endpoint host is pinned to HTTPS
`*.aiplatform.googleapis.com` so the token can never be sent to an operator-chosen host.
Requests use the same `generateContent` wire protocol as the Gemini provider on
`publishers/google/models/` routes.

The model id spelling picks the wire. A bare id (`gemini-2.5-pro`) or a Google resource path
(`publishers/google/models/gemini-2.5-pro`) is a Google-published model on the Gemini wire. A
`<publisher>/<model>` id (`deepseek-ai/deepseek-v3.2-maas`, `xai/grok-4.20-reasoning`,
`qwen/qwen3-coder-480b-a35b-instruct-maas`) is a Model Garden model served as a managed API
(MaaS), which Vertex serves only over its OpenAI-compatible route
`{base_url}/endpoints/openapi/chat/completions` (dialect `openai_compatible`, the same
Chat Completions request and stream handling as `openai-compatible`), still under the OAuth
bearer. Most MaaS models are addressed through the `global` location
(`https://aiplatform.googleapis.com/v1/projects/PROJECT/locations/global`); a listing-style
`publishers/<publisher>/models/<model>` spelling is collapsed onto the `<publisher>/<model>`
form the route accepts. Google's own managed endpoints share the Gemini resource path in
the listing (`publishers/google/models/gemma-4-26b-a4b-it-maas`) and are told apart by
Vertex's `-maas` endpoint suffix, so both that spelling and `google/gemma-4-26b-a4b-it-maas`
take the MaaS route.

For models available in Google's US or EU multi-region, set the entire project-and-location
root to `https://aiplatform.us.rep.googleapis.com/v1/projects/PROJECT/locations/us`
or `https://aiplatform.eu.rep.googleapis.com/v1/projects/PROJECT/locations/eu`.
Both the synchronous client and native gateway preserve that endpoint. A global endpoint
does not guarantee a processing location; verify the model's availability and the applicable
[Google data residency commitments](https://docs.cloud.google.com/gemini-enterprise-agent-platform/resources/data-residency)
before selecting a jurisdiction. Endpoint support alone does not enforce a routing policy
on other models or fallback connections.

Vertex is catalog-and-API configuration only: the interactive `exp config providers` picker
does not offer it. Like Azure and Bedrock, provider names do not imply protocol support or
prices, so every Vertex alias declares explicit capabilities. Embeddings are not supported on
the Gemini-wire Vertex aliases; use a `gemini` connection for Gemini embeddings. MaaS aliases
expose the compatible embeddings route (`endpoints/openapi/embeddings`) when their
capabilities declare `supports_embeddings`.

```toml
[connections.vertex]
provider = "vertex"
base_url = "https://us-central1-aiplatform.googleapis.com/v1/projects/PROJECT/locations/us-central1"
api_key_env = "VERTEX_SERVICE_ACCOUNT_JSON"

[models.gemini-pro]
connection = "vertex"
model = "gemini-2.5-pro"
[models.gemini-pro.capabilities]
supports_completions = true
supports_tools = true
input_cost_per_million_tokens_usd = 0
output_cost_per_million_tokens_usd = 0
```

## Errors

Missing credentials name the environment variable and the `exp config providers` recovery, never
a secret value. A missing Bedrock region lists the AWS resolution order. Azure endpoint and key
mismatches name `AZURE_OPENAI_ENDPOINT`, not the key. A malformed user-data credential file
fails closed and tells the operator to move or delete it, then run `exp config providers`.
Malformed provider responses fail closed and do not write partial catalog or evidence artifacts.

## Novita error classification (2026-09-16)

Novita is an OpenAI-compatible reseller whose gpt-5.6 lanes front per-region
Azure OpenAI deployments. Three of its error shapes needed engine rules beyond
the shared envelope reader (`crate::error_envelope`), all pinned in
`upstream_reseller_tests.rs`:

- **A 4xx that says the ACCOUNT cannot pay is `provider_quota`.** A drained
  prepaid balance answers `400 "Insufficient quota available for instant
  inference"`; a status-only read filed it as the caller's `invalid_request`.
  A pre-stream 4xx whose code is a quota token or whose sentence carries
  unambiguous funding wording (`rejected_by_account_quota`: insufficient
  quota/balance/credits/funds, not enough balance, exceeded your current
  quota — no bare "billing") takes the quota class, fails over, and keeps the
  sentence ledger-only.
- **A relay decode failure is unwrapped.** Novita's Responses relay sometimes
  cannot decode the UPSTREAM error it received (`failed to decode error
  response: json: cannot unmarshal number into Go struct field
  ResponseError.error.code of type string, raw: {…}`) and answers its own 400
  with the upstream document embedded after `raw: `. Pre-stream and in-stream
  the engine reads that document (`relayed_decode_failure`; a zero code is
  "no code", a truncated document still yields its message), classifies by
  the UPSTREAM code and sentence (a relayed 429 throttles and fails over; a
  relayed caller error keeps this status's caller class), and relays the
  upstream sentence ("Exceeded maximum number of images (50) allowed in the
  request.") instead of the decoder's noise.
- **`reason` tokens classify** (`INVALID_REQUEST_BODY` generic,
  `MODEL_NOT_FOUND` → lane policy, `NOT_ENOUGH_BALANCE` under 403 →
  `provider_quota`, `RATE_LIMIT_EXCEEDED` / `TOKEN_LIMIT_EXCEEDED` throttle,
  `FAILED_TO_AUTH` / `ACCESS_DENY` authenticate), see the architecture
  reference.
