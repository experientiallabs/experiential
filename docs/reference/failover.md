# Provider failover chains (`.wmh/fallback.toml`)

Long GEPA/eval runs die (or silently degrade) when the primary model throttles. The harness
fails over **world-model calls** across a chain of backends instead: create a gitignored
`.wmh/fallback.toml` and every CLI entry point that predicts observations (`wmh build`,
`wmh eval`, `wmh serve` / `demo` / `play`) rides the chain automatically. No file → plain
single-backend behavior.

```toml
# .wmh/fallback.toml — tried top to bottom; only capacity errors (throttling, transient 5xx,
# timeouts) spill to the next rung. Real errors (bad request, auth) surface immediately.

[[backend]]
kind = "bedrock"
model = "us.anthropic.claude-opus-4-6-v1"
profile = "team-a"          # named AWS profile: one chain can span accounts
region = "us-east-1"        # profiles don't inherit the env region; set it per rung

[[backend]]
kind = "bedrock"
model = "us.anthropic.claude-opus-4-8"   # no profile = default AWS credentials

[[backend]]
kind = "openai"
model = "gpt-5.5"           # reads OPENAI_API_KEY (or set api_key = "..." here)

[[backend]]
kind = "anthropic"
model = "claude-opus-4-7"   # Anthropic direct (ANTHROPIC_API_KEY): a different capacity pool
                            # AND network path than Bedrock — rides out regional quota
                            # contention and AWS connectivity flaps (both killed live runs)
```

The requested `(kind, model)` always leads as the primary; the file's rungs back it up.
Per-call cost is attributed to the model that actually served (`Completion.model`), so
`.wmh/runs` records stay honest under failover.

**The judge never rides the chain.** Fidelity scoring (`RubricJudge` in `wmh eval` and as
GEPA's fitness signal in `wmh build`) stays pinned to the single requested backend: a judge
that silently switches models mid-run scores steps on different scales and makes fidelity
numbers incomparable. Judge failures are handled at two layers, never by switching models:
a MALFORMED reply (missing dimension, no JSON, scale confusion) is retried once with feedback,
then excluded from aggregates as `valid=False` (reported as `judge-invalid` counts). A judge
call that RAISES (throttle/5xx after the provider's own retries) is treated as judge-invalid
during `wmh build` (GEPA imputes and continues) but aborts `wmh eval` — a partially judged
eval would silently change what the fidelity mean is over.

Keep the file out of git: profile names identify accounts and it may carry an API key
(`.wmh/` is gitignored wholesale). Known quirk: `verify()` pings each rung with a tiny
completion, which can trip reasoning models' output floor (GPT-5.5 reports a max_tokens
error on verify but serves real calls fine).
