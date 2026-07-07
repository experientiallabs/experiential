# llm-waterfall

**Pool every LLM quota you own behind one client.** Rate limits are issued per model, per
provider, per account — but a workload pinned to one backend can only ever use one of them, and
stalls the moment it throttles. llm-waterfall chains your backends into a single stateless
client: each call walks the chain in order, capacity errors (throttling, 5xx, timeouts) spill to
the next backend, and real errors (bad request, auth, validation) raise immediately. A six-rung
chain sustains roughly the sum of six rate limits instead of the minimum of one — capacity you
already pay for, actually reachable.

Every call returns which backend actually served it, token usage, USD cost, and the full attempt
trail — as a return value, not a log line — so failover never blurs attribution.

```python
from llm_waterfall import Backend, Waterfall

wf = Waterfall([
    Backend("bedrock", "us.anthropic.claude-opus-4-6-v1", profile="endflow",   region="us-west-1"),
    Backend("bedrock", "us.anthropic.claude-sonnet-4-6",  profile="endflow",   region="us-west-1"),
    Backend("bedrock", "us.anthropic.claude-opus-4-6-v1", profile="stackwise", region="us-west-1"),
    Backend("bedrock", "us.anthropic.claude-opus-4-7",    profile="stackwise", region="us-west-1"),
    Backend("bedrock", "us.anthropic.claude-sonnet-4-6",  profile="stackwise", region="us-west-1"),
    Backend("bedrock", "us.anthropic.claude-opus-4-8",    profile="stackwise", region="us-west-1"),
])

r = wf.complete(system="You are terse.", messages=[{"role": "user", "content": "hi"}], max_tokens=4096)
print(r.text, r.model_used, r.provider_used, r.cost_usd, r.usage, r.attempts)

emb = wf.embed(["some text"])  # embeddings waterfall too
```

Chains can mix providers and credentials freely — the `profile` field maps to a named AWS profile
(`boto3.Session(profile_name=...)`), so one chain can span multiple AWS accounts.

```python
wf = Waterfall([
    Backend("anthropic", "claude-opus-4-8"),
    Backend("bedrock", "us.anthropic.claude-opus-4-8", region="us-west-2"),
    Backend("openai", "gpt-5.5"),
])
```

## Install

```bash
pip install "llm-waterfall[bedrock]"           # extras: bedrock, openai, anthropic, azure, all
pip install "llm-waterfall[all]"
```

The package imports with zero provider SDKs installed; each adapter imports its SDK on first use
and raises a clear error naming the extra to install.

## What spills, what doesn't

The classifier is the core contract. It prefers structured error codes over message matching:

| Signal | Examples | Verdict |
| --- | --- | --- |
| botocore error code in the capacity set | `ThrottlingException`, `ServiceUnavailableException`, `ModelNotReadyException`, `ModelTimeoutException`, `InternalServerException` | spill |
| botocore error code **not** in the set | `ValidationException`, `AccessDeniedException` — even if the message contains "timeout" or "429" | raise |
| OpenAI/Anthropic SDK exception type | `RateLimitError`, `APITimeoutError`, `APIConnectionError`, `InternalServerError`, `OverloadedError` | spill |
| httpx/httpcore transport failure | `ConnectError`, `ReadError`, `RemoteProtocolError`, `ConnectTimeout`, `PoolTimeout`, ... | spill |
| HTTP status on an SDK error | 408 / 429 / 500 / 502 / 503 / 504 / 529 | spill |
| HTTP status on an SDK error | 400 / 401 / 403 / 404 / 422 | raise |
| Transport-level message (no structure) | "read timeout", "connection reset", "throttl…" | spill |
| Anything else | bad request, auth, validation | raise |

Generic tokens like `429`, `503`, or `capacity` are **never** matched against raw messages — a bad
request whose message happens to contain them propagates instead of silently failing over. (That
was a real bug once.)

## Attribution

`model_used` / `provider_used` / `cost_usd` describe the backend that **actually served** the call,
not the one you hoped would. `attempts` records the whole path:

```python
r = wf.complete(system="...", messages=[...])
for a in r.attempts:
    print(a.provider, a.model, a.outcome, f"{a.latency_s:.2f}s", a.error_type or "")
# bedrock us.anthropic.claude-opus-4-6-v1 capacity_error 0.31s ThrottlingException
# bedrock us.anthropic.claude-sonnet-4-6  ok             8.02s
```

When every backend is capacity-constrained, `complete()` raises `WaterfallExhausted` carrying the
full `attempts` list, chained from the last capacity error. `WaterfallExhausted` itself classifies
as a capacity error, so nested waterfalls and outer retry loops treat it as transient.

## Retries

Default is pure failover: one attempt per backend, no sleeping, restart at the primary next call.
For long unattended runs where the whole chain may throttle at once, allow bounded wrap-around:

```python
from llm_waterfall import RetryPolicy

wf = Waterfall(backends, retry=RetryPolicy(rounds=4))  # capped-exponential sleep between rounds
```

Every adapter disables its SDK's internal retries (`botocore total_max_attempts=1`, `max_retries=0`) so
the waterfall solely owns retry policy — otherwise one throttled request becomes SDK-retries ×
chain-length backend calls. Requests are bounded with connect/read timeouts (default 15 s / 600 s —
generous enough not to cut off a long reasoning generation) so a stalled connection raises and
fails over instead of hanging forever.

## Cost

A built-in USD-per-Mtok table covers current Claude, GPT-5.x, and embedding models, keyed by
normalized model id (Bedrock region prefixes and version suffixes are stripped, so
`us.anthropic.claude-opus-4-8-20260101-v1:0` prices as `claude-opus-4-8`). Unknown models cost
`0.0`; use `price_for(model)` (returns `None`) to detect unpriced models. Override or extend per
instance — no global mutation:

```python
from llm_waterfall import ModelPrice

wf = Waterfall(backends, prices={"my-azure-deployment": ModelPrice(input_per_mtok=2.5, output_per_mtok=15.0)})
```

## Embeddings

`wf.embed(texts)` runs the same waterfall. Each backend embeds with its `embed_model` (or the
provider default: Titan v2 on Bedrock, `text-embedding-3-small` on OpenAI). Backends with no
embeddings API (Anthropic) are recorded as `unsupported` in the attempt trail and skipped.

Failover assumes the chain shares **one embedding space** — vectors from different embedding models
are not comparable, and a chain that mixes them can silently poison a retrieval index. Keep
`embed_model` consistent across rungs (e.g. the same Titan model behind several AWS profiles).

## Backends

| Provider | Status | Config |
| --- | --- | --- |
| `bedrock` | ✅ | `region`, `profile` (named AWS profile), creds via boto3 chain |
| `openai` | ✅ | `OPENAI_API_KEY`, optional `endpoint` (custom base URL) |
| `anthropic` | ✅ | `ANTHROPIC_API_KEY` |
| `azure_openai` | ✅ | `endpoint` + `deployment` + `api_version`, `AZURE_OPENAI_API_KEY` |
| `aws_mantle` | 🚧 stub (fails at construction) | |

## Design notes

- **Stateless.** A `Waterfall` is immutable after construction; results are returned, never logged
  to a side channel. No module globals, no env mutation at call time — safe to share one instance
  across a thread pool.
- **Client errors never spill.** Failing over on a bad request just masks a real bug behind a
  different model's answer.
- **Born from long eval runs** dying at 3am because one endpoint throttled while five perfectly
  good backends sat idle.
