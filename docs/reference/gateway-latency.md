# Gateway latency report

CI measures **gateway-added latency** against a local OpenAI-compatible mock.
It does not call paid providers, does not require secrets, and does not claim
end-to-end model latency.

The static research writeup in `docs/research/rust-gateway-engine.md` remains
the source for high-concurrency native-engine evidence. This page documents the
routine CI report only.

## What is measured

The runner starts a loopback mock, then measures the same
`/v1/chat/completions` payload against:

1. the mock directly
2. the Experiential native gateway

The gateway aliases the public model name `latency` to the same mock, so the
request body is identical. The report records gateway p50, p95, and p99, and
also the client-observed difference versus the mock for those percentiles.

The schedule is:

- warmup requests are discarded before the timed window
- mock-direct and gateway arms run sequentially, not in parallel
- percentiles use nearest-rank selection
- the representative result is the median run by gateway p50

Streaming time-to-first-token is included because the mock emits the first
content token immediately, so TTFT is a simple first-byte measurement rather
than a model-generation race.

## Commands

Local report:

```bash
uv sync --extra dev
uv run python -m exp.runtime.gateway.latency_report \
  --output-json gateway-latency.json \
  --output-badge shields/gateway-latency.json
```

Focused tests:

```bash
uv run pytest -q exp/runtime/gateway/latency_report_test.py \
  exp/runtime/gateway/latency_measure_test.py \
  exp/runtime/gateway/latency_badge_test.py
```

The workflow `.github/workflows/gateway-latency.yml` runs on every push to
`main`, on pull requests, and via `workflow_dispatch`. Pull requests and pushes
use `ubuntu-latest`. An explicit `workflow_dispatch` uses the larger hosted
runner `gateway-benchmark-32core`. Functional request failures fail the job.
There is no hard latency threshold.

## Artifact

The job uploads `gateway-latency.json` with schema
`exp.gateway.latency_report` version 1, plus the derived Shields endpoint
`shields/gateway-latency.json`. It also writes a Markdown table to the GitHub
Actions job summary. The report records the commit SHA, runner OS, CPU count
and model, Python version, resolved Experiential engine, every repeat, and the
median run by gateway non-stream p50.

## Numeric latency badge

The root README shows the latest **representative gateway p50 request
latency** (`representative_run.gateway.p50_ms`) from a successful routine run
on `main`. The badge is a Shields endpoint, not a GitHub Actions pass/fail
status image. The label is `gateway latency` and the message is the measured
value, for example `22.2 ms`.

After each successful `push` to `main`, the workflow publishes
`gateway-latency.json` on the `badges` branch. Pull requests and
`workflow_dispatch` (including the 32-core runner) measure and upload the
artifact but do not publish the badge. That branch is not `main`, so the
update does not require a protected-branch push and does not start gate,
latency, or package workflows.

Stable endpoint:

`https://raw.githubusercontent.com/experientiallabs/experiential/badges/gateway-latency.json`

README image:

```markdown
[![gateway latency](https://img.shields.io/endpoint?url=https%3A%2F%2Fraw.githubusercontent.com%2Fexperientiallabs%2Fexperiential%2Fbadges%2Fgateway-latency.json)](https://github.com/experientiallabs/experiential/actions/workflows/gateway-latency.yml?query=branch%3Amain)
```

The `badges` branch is created by the first successful main run after this
workflow lands. Until that run finishes, the Shields image may render as
invalid. Later successful main runs overwrite the same public file.

## The first-token stall bound

Two fail-fast allowances bound the start of every physical attempt, both absolute from the
dial and both per-deployment overridable through the gateway capabilities:

- **Headers**: `time_to_first_byte_seconds` (15 s) plus
  `time_to_first_byte_seconds_per_million_input_tokens` (240) bounds the wait for the
  provider's response headers (`upstream::open_stream`; overrides
  `time_to_first_byte_base_seconds` / `time_to_first_byte_seconds_per_million_input_tokens`).
- **First token**: `time_to_first_token_seconds` (120 s, clamped to three quarters of the
  request budget so a stall can still fail over: under the engine's own 120 s request
  timeout the effective default is 90 s; the platform's 1500 s budget keeps the full
  allowance) plus the same input slope bounds the wait for generation to begin (override
  `time_to_first_token_base_seconds`). Outward semantic output commits the attempt. A nonempty
  private `reasoning_content` token also starts generation timing, but does not commit: its
  text is withheld until this attempt produces outward output or a successful terminal.
  A refusal delta withheld under refusal failover leaves the first-token bound armed.

Response headers, SSE keepalive comments, Anthropic pings and role-only frames satisfy neither
bound's *token* half: on 2026-09-19 a lane answered all of those at once and then stalled ~2
minutes before its first token, and the old single bound, satisfied by the first body byte,
let the request sit on the 60 s per-chunk timeout while it held a worker permit. The
first-token base is two minutes rather than the header bound's fifteen seconds because a
thinking model on a chat wire streams nothing until its first content token: over the seven
days before the change, 6% of gpt-5.6-sol's completions (p99 94 s) and 1-3% of most other
healthy lanes took longer than the header allowance to produce a token, while the stalled
lane's medians sat above two minutes. A stall past the allowance is
`first_byte_timeout_failure()`: class `timeout`, failover-eligible, never redialed on the
stalled lane, so the ladder advances before anything has reached the caller.

## Generation progress and commitment

After genuine generation begins, the connection's existing `timeout_seconds` bounds the idle
gap between normalized output events, not raw network chunks. Nonempty text, reasoning and
tool arguments count; comments, pings, role-only chunks, empty deltas and usage-only frames do
not. Absolute checks before every fresh network read prevent an always-ready ping flood from
bypassing a timer. Already decoded events, including usage and a terminal, drain before another
read is judged. Time spent handing events to the downstream consumer pauses generation-idle
accounting, but never the first-token or total deadlines.

A progress-idle expiry uses the existing `transport` failure class with the distinct message
`provider stopped making progress; retry the request`. It never redials the same deployment.
Before commitment it may advance to an eligible successor; the authored policy still decides,
so a `failover_only_on: ["timeout"]` successor does not match this transport failure. After
commitment it terminates the selected stream without splicing output or replaying work.

Private route-bound `reasoning_content` is genuine progress, not outward commitment. The
waterfall coalesces it into one carrier under the normalizer's existing 64 MiB retained-output
ceiling (the encoders keep their additional carrier bounds). Failed attempts discard this
private buffer. The winning attempt preserves it byte-for-byte for its usual sealed
continuation; plaintext never becomes public merely to keep a request alive. Exposed reasoning,
visible text and client-tool output still commit. A private-only successful terminal retains
the existing encoding and accounting behavior. Known usage received before a stall settles
with that physical attempt; absence of a report remains unknown, never invented zero usage.

Structural output may commit a surface before generation begins, but does not shorten its
first-progress allowance to the connection timeout. In particular omitted Anthropic thinking
with an empty delta can wait for its signature under the full first-token allowance.

**Provider-executed tools are a separate phase.** A server-tool start or hosted invocation
commits irreversibly. While that provider tool is active, its existing byte-idle timeout and
hard request deadline remain the bounds: a legitimate long web search or Codex hosted tool
can emit only keepalives while it works. Its matching result/completed item resumes generation
progress timing. Remote MCP tool discovery (`mcp_list_tools`) uses the same phase without
being recorded as a billed tool invocation. A server-tool argument-block close is not completion
of the server work. A hung provider tool that continues sending keepalives is bounded only by
the hard deadline; the gateway never retries such irreversible work.

The total `request_timeout_seconds` remains a hard budget shared by the whole ladder. Neither
progress nor keepalives extend it. Defaults for first byte, first token and input scaling are
unchanged; the platform's 1500 s total is unchanged too. Active reasoning and long generation
can run beyond individual first-token and idle windows, but not beyond the explicitly authored
total budget. `time_to_first_byte_ms` still records the first body byte; `first_token_at` records
the first provider output token, including private reasoning, not necessarily the first frame
the caller sees.
