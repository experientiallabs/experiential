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
request body is identical. Reported overhead is the client-observed difference
(gateway minus mock) for p50, p95, and p99.

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
  --output-json gateway-latency.json
```

Focused tests:

```bash
uv run pytest -q exp/runtime/gateway/latency_report_test.py \
  exp/runtime/gateway/latency_measure_test.py
```

The workflow `.github/workflows/gateway-latency.yml` runs on every push to
`main`, on pull requests, and via `workflow_dispatch`. Pull requests and pushes
use `ubuntu-latest`. An explicit `workflow_dispatch` uses the larger hosted
runner `gateway-benchmark-32core`. Functional request failures fail the job.
There is no hard latency threshold.

## Artifact

The job uploads `gateway-latency.json` with schema
`exp.gateway.latency_report` version 1 and writes a Markdown table to the
GitHub Actions job summary. The JSON records the commit SHA, runner OS, CPU
count and model, Python version, resolved Experiential engine, every repeat,
and the median run by gateway non-stream p50.

## Status badge

A workflow-status badge is the zero-secret signal. It does not embed a
latency number. Proposed README insertion (requires explicit README approval):

```markdown
[![Gateway latency](https://github.com/experientiallabs/experiential/actions/workflows/gateway-latency.yml/badge.svg?branch=main)](https://github.com/experientiallabs/experiential/actions/workflows/gateway-latency.yml?query=branch%3Amain)
```
