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
