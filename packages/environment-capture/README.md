# environment-capture

Run benchmarks **for real** and record every agent-environment transition as
OTel GenAI JSONL — the wmh trace wire format. This is the capture side of the harness: adapters
stand up a real benchmark environment, an agent acts in it, and the real `(action → observation)`
pairs become the trace corpus a world model is built from.

It is a uv workspace member (`environment_capture` package) designed to be extraction-ready: the
package never imports `wmh` — the dependency arrow is absolute (AGENTS.md § Monorepo). The
wire format is pinned from the flagship side: `wmh/ingest/otel_genai_envcap_roundtrip_test.py`
round-trips emitted spans through the real ingest adapter.

## The contract

```python
from environment_capture import (
    BenchmarkAdapter,   # tasks(split) / open_env(task) / grade(task, submission)
    CommandEnv,         # execute(command) -> ExecResult(output, returncode); close()
    run_capture,        # drive an agent over a split against the REAL env -> [Trajectory]
    trajectory_to_spans, write_spans_jsonl,   # Trajectory -> OTel GenAI JSONL
)
```

- **`CommandEnv.execute` is the world-model seam.** A real adapter executes commands in a real
  workspace; swap in a world-model-backed implementation and the identical agent loop runs
  against the WM. That is what "replace the benchmark with a world model" means mechanically.
- **Graders are deterministic.** `grade(task, submission) -> float` must not call an LLM, so
  WM-vs-real comparisons are judged by the same fixed function.
- **Observations are never synthesized.** A corpus comes from `run_capture` against the real
  environment (or a conversion of someone else's REAL runs, with provenance). No hand-written or
  model-imagined observations, ever.

## Layout

```
packages/environment-capture/
  environment_capture/        # the package: contract + emitter + converters (+ inline *_test.py)
  <benchmark>/                # one dir per benchmark: committed traces.otel.jsonl,
                              # provenance README, task data, thin capture/convert scripts
```

Per-benchmark dirs follow the `examples/` discipline: only traces, small task data, and thin
scripts are committed; cloned upstreams, venvs, and raw run output stay local and gitignored.

## Corpora on the Hugging Face Hub

Every publishable corpus is mirrored as a public dataset under the
[`experiential-labs`](https://huggingface.co/experiential-labs) org (repo per benchmark:
`experiential-labs/wmh-<benchmark>-traces`, license tag matching the upstream terms). Corpora are
**local-first**: capture always writes `traces.otel.jsonl` into the benchmark dir and nothing ever
deletes it; the Hub is the sharing/distribution layer.

```bash
# publish or UPDATE one corpus (or 'all') — every push is a Hub commit, history kept
uv run python -m environment_capture.hub push bird-sql
uv run python -m environment_capture.hub push all            # add --private for private repos

# pull the full set (e.g. after cloning without corpora); never clobbers a local file
uv run python -m environment_capture.hub fetch dabstep
uv run python -m environment_capture.hub fetch all --force   # explicit overwrite

# or push straight from a capture wave
uv run python packages/environment-capture/dabstep/capture.py ... --push-hub
```

Pushing needs a write token (`hf auth login` or `HF_TOKEN`); fetching public corpora needs none.
`appworld` is the deliberate exception: its license only allows encrypted redistribution of
derivatives, so that corpus stays local-only (`hub.py` refuses to push it).

## Adding a benchmark

**Agents: follow [INTEGRATION.md](INTEGRATION.md) — it is the complete, self-contained
playbook** (contract, step order, non-negotiables, acceptance checklist). Summary:

1. Implement a `BenchmarkAdapter` in `environment_capture/benchmarks/<name>.py` — fresh code
   against the benchmark's real upstream dataset (tests inline).
2. Create `packages/environment-capture/<name>/` with the task data (license-checked) and a capture or
   conversion script that writes `traces.otel.jsonl`.
3. Verify the corpus round-trips: `wmh build --file packages/environment-capture/<name>/traces.otel.jsonl`
   must ingest every trace, then eval it under the repo's reporting conventions.
