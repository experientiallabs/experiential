---
source: https://app.notion.com/38e0f8b3f5918150b306e24c3c159d21
area: Architecture
status: Current
migrated: 2026-07-02
---

# Embeddings (phi for retrieval)

Retrieval (DreamGym top-k) ranks past steps by cosine similarity of an embedding `phi(state,
action)`. The embedder that produces `phi` is chosen by `HarnessConfig.embed_provider` (an
`EmbedderKind`) and sized by `HarnessConfig.embed_dim`.

## Choosing an embedder

| `embed_provider` | What runs | Needs | Notes |
| --- | --- | --- | --- |
| `hashing` (default) | `HashingEmbedder` | nothing | Offline hashed char-trigram vector, L2-normalized. Lexical, not semantic. Zero config. |
| `bedrock` | Amazon Titan text embeddings | AWS creds + region | `embed_model` defaults to `amazon.titan-embed-text-v2:0`. |
| `openai` | OpenAI embeddings | `OPENAI_API_KEY` | Set `embed_model` (e.g. `text-embedding-3-small`). |
| `azure_openai` | Azure OpenAI embedding deployment | Azure creds + endpoint | `embed_model` is the **deployment name**, not the base model id. |

Anthropic has no embeddings API, so there is no `anthropic` embedder kind — use one of the above (or
`hashing`) for `phi` while serving the world model on Anthropic/Bedrock for *generation*.

## How `embed_model` is configured per provider

`embed_model` lives on the **`ProviderConfig`** for the backing provider (the one whose `kind`
matches `embed_provider`). It is independent of the completion `model`:

- **Bedrock** — `embed_model` is the Titan model id (`amazon.titan-embed-text-v2:0`). Optional; the
  provider defaults to titan-embed-text-v2 when unset.
- **OpenAI** — `embed_model` is the embeddings model id (`text-embedding-3-small`/`-large`).
  Required for `embed()`.
- **Azure OpenAI** — `embed_model` is the **embedding deployment name** (same role `deployment`
  plays for completions). Required for `embed()`.

`get_embedder(config)` (in `wmh.retrieval.embedders`) resolves all of this: for `hashing` it builds
`HashingEmbedder(dim=embed_dim)`; otherwise it constructs the backing provider via the registry and
stamps `embed_dim` onto its `ProviderConfig` so the backend requests a vector of exactly that size.

## Dimensions must match

The persisted index and the query embedder must agree on dimensionality, or cosine similarity is a
shape error. `embed_dim` is the single knob:

- For `hashing`, `embed_dim` is the literal vector length.
- For Bedrock Titan v2 and OpenAI `text-embedding-3-*`, `embed_dim` is sent as the API's
  `dimensions` parameter, so the model returns exactly that width. (Leave `embed_dim` unset on a
  provider to accept the model's native dimension.)

`EmbeddingRetriever.topk` keeps a runtime guard: if the query embedder's dimension differs from the
indexed matrix, it raises a clear error telling you to load the same `embed_dim` used at build time —
never a raw numpy mismatch.

## Verifying the embed path

`wmh providers verify` pings each provider's completion path and, when `embed_provider` is not
`hashing`, also embeds one tiny string through the configured embed provider and reports `ok` + the
produced `dim` (or the failure detail). It never raises — a missing key or wrong model comes back as
`fail`.

## Wiring at build time

`wmh build --embed-provider <kind> [--embed-model <id>] [--embed-dim N]` constructs the embedder via
`get_embedder` and passes it into the build, so the index is built with the same `phi` that
`WorldModel.load` will reconstruct at serve time.
