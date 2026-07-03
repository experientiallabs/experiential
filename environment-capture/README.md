# environment-capture

Run benchmarks **for real** and record every agent-environment transition as
OTel GenAI JSONL — the wmh trace wire format. This is the capture side of the harness: adapters
stand up a real benchmark environment, an agent acts in it, and the real `(action → observation)`
pairs become the trace corpus a world model is built from.

It is a uv workspace member (`environment_capture` package) designed to be extraction-ready: the
package never imports `wmh` (one round-trip test pins the wire format against wmh's ingest and
moves across the boundary if this is ever split out to its own repo).

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
environment-capture/
  environment_capture/        # the package: contract + emitter + converters (+ inline *_test.py)
  <benchmark>/                # one dir per benchmark: committed traces.otel.jsonl,
                              # provenance README, task data, thin capture/convert scripts
```

Per-benchmark dirs follow the `examples/` discipline: only traces, small task data, and thin
scripts are committed; cloned upstreams, venvs, and raw run output stay local and gitignored.

## Adding a benchmark

1. Implement a `BenchmarkAdapter` in `environment_capture/benchmarks/<name>.py` — fresh code
   against the benchmark's real upstream dataset (tests inline).
2. Create `environment-capture/<name>/` with the task data (license-checked) and a capture or
   conversion script that writes `traces.otel.jsonl`.
3. Verify the corpus round-trips: `wmh build --file environment-capture/<name>/traces.otel.jsonl`
   must ingest every trace, then eval it under the repo's reporting conventions.
