# Gateway latency report

CI measures **gateway-added latency** against a local OpenAI-compatible mock
and, in the same job, compares Experiential with a pinned LiteLLM proxy.
It does not call paid providers, does not require secrets, and does not claim
end-to-end model latency or LiteLLM's public 1K-RPS topology.

The static research writeup in `docs/research/rust-gateway-engine.md` remains
the source for high-concurrency native-engine evidence. This page documents the
routine CI report only.

## What is measured

The runner starts a loopback mock, then measures the same
`/v1/chat/completions` payload against:

1. the mock directly
2. the Experiential native gateway
3. LiteLLM `1.97.0` started as the official config-file proxy without a
   database (`litellm[proxy]==1.97.0`, `litellm --config`, equivalent image
   `ghcr.io/berriai/litellm:v1.97.0`)

Both proxies alias the public model name `latency` to the same mock, so the
request body is identical. Reported overhead is the client-observed difference
(proxy minus mock) for p50, p95, and p99. Gateway measurement order rotates
across repeats: odd runs measure Experiential first, even runs measure LiteLLM
first.

The schedule follows the official LiteLLM mock-isolated method:

- [benchmark_chat_completions_perf.py](https://github.com/BerriAI/litellm/blob/main/scripts/benchmark_chat_completions_perf.py)
  for a local mock, warmup, sequential proxy-versus-direct arms, nearest-rank
  percentiles, and a median representative run
- [benchmark_proxy_vs_provider.py](https://github.com/BerriAI/litellm/blob/main/scripts/benchmark_proxy_vs_provider.py)
  for sequential (not parallel) proxy-versus-direct comparison, success rate,
  and throughput
- [LiteLLM config-file proxy](https://docs.litellm.ai/docs/proxy/configs) and
  [Running without a database](https://docs.litellm.ai/docs/proxy/docker_quick_start)
  for the pinned LiteLLM startup

Streaming time-to-first-token is included because the mock emits the first
content token immediately, so TTFT is a simple first-byte measurement rather
than a model-generation race.

## Commands

Local Experiential-only report:

```bash
uv sync --extra dev
uv run python -m exp.runtime.gateway.latency_report \
  --output-json gateway-latency.json
```

Same-run comparison (installs the pinned LiteLLM release into the uv run):

```bash
uv run --with 'litellm[proxy]==1.97.0' python -m exp.runtime.gateway.latency_report \
  --compare-litellm \
  --output-json gateway-latency.json
```

Focused tests:

```bash
uv run pytest -q exp/runtime/gateway/latency_report_test.py \
  exp/runtime/gateway/latency_measure_test.py \
  exp/runtime/gateway/latency_litellm_test.py
```

The workflow `.github/workflows/gateway-latency.yml` runs on every push to
`main`, on pull requests, and via `workflow_dispatch`. Functional request
failures fail the job. There is no hard latency threshold.

## Artifact

The job uploads `gateway-latency.json` with schema
`exp.gateway.latency_report` version 2 and writes a Markdown table to the
GitHub Actions job summary. The JSON records the commit SHA, runner OS, CPU
count and model, Python version, resolved Experiential engine, pinned LiteLLM
version and startup line, every repeat with its gateway order, and the median
run by Experiential non-stream p50.

## Status badge

A workflow-status badge is the zero-secret signal. It does not embed a
latency number. Proposed README insertion (requires explicit README approval):

```markdown
[![Gateway latency](https://github.com/experientiallabs/experiential/actions/workflows/gateway-latency.yml/badge.svg?branch=main)](https://github.com/experientiallabs/experiential/actions/workflows/gateway-latency.yml?query=branch%3Amain)
```
