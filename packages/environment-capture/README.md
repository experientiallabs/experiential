# environment-capture

Run agent benchmarks **for real** and record every agent-environment transition — each
`(action → observation)` pair, exactly as the environment returned it — as OpenTelemetry GenAI
JSONL. Integrating a benchmark is one small adapter; ten are already in (**5,900+ real
trajectories / 27,000+ real transitions** captured and published as license-tagged datasets on
the [Hugging Face Hub](https://huggingface.co/experiential-labs)).

## Why

Because it's annoying to:

- **set up benchmarks and run models against them** — one three-method interface for any
  benchmark, any agent, any provider
- **capture traces in a standardized format** — every `(action → observation)` transition as
  OTel GenAI JSONL, agnostic to benchmark and provider

Adding a benchmark? Point your coding agent at [INTEGRATION.md](INTEGRATION.md) — it's the
complete, self-contained playbook for integrating one.

## The benchmarks

| benchmark | environment | traces / transitions |
|---|---|---|
| bird-sql | text-to-SQL over real SQLite databases | 1,993 / 4,168 |
| financebench | financial-document QA over SEC-filing evidence | 1,254 / 7,402 |
| tau-bench | customer-service tool agents (airline/retail/telecom) | 1,033 / 5,289 |
| dabstep | pandas data analysis over a payments dataset + rules manual | 687 / 4,859 |
| continual-learning | exploration of a large obfuscated product database | 286 / 2,071 |
| terminal-tasks | computer-use agents in real terminal containers | 280 / 685 |
| swe-bench | software engineering in per-instance Docker images | 255 / 1,700 |
| crmarena | CRM analytics over a realistic Salesforce org | 80 / 553 |
| gaia2 | stateful multi-app simulated world (Meta ARE) | 37 / 563 |
| appworld | stateful multi-app simulated world | local-only (license) |

All published bundles include the trace corpus plus the task data needed to run the benchmark
(task index, gold sidecars, evidence/context files).

## Install

```bash
pip install environment-capture            # the library: contract, capture driver, hygiene, hub fetch
pip install 'environment-capture[fetch]'   # + huggingface_hub, for publishing bundles
```

Pure-package usage — capture YOUR benchmark and pull OUR data, no repo checkout needed:

```python
from pathlib import Path
from environment_capture import run_capture, trajectory_to_spans, write_spans_jsonl
from environment_capture.hub import fetch_corpus

# 1) pull a published bundle (lands in ./environment-capture-data/, or $ENVCAP_DATA_ROOT)
corpus = fetch_corpus("bird-sql")

# 2) capture your own benchmark: implement the 3-method adapter + an agent, then
result = run_capture(my_adapter, my_agent, split="train")
spans = [s for t in result.trajectories for s in trajectory_to_spans(t, benchmark="my-bench")]
write_spans_jsonl(spans, Path("traces.otel.jsonl"))
```

The wheel ships the library and every benchmark adapter; benchmark *data* always comes from the
Hub (or your own capture runs). Note: `hub` publishing is currently pinned to the
`experiential-labs` org manifest — pushing your own benchmark's bundle means adding a
`CorpusSpec` (see INTEGRATION.md).

## The contract

```python
from environment_capture import (
    BenchmarkAdapter,   # tasks(split) / open_env(task) / grade(task, submission)
    CommandEnv,         # execute(command) -> ExecResult(output, returncode); close()
    run_capture,        # drive an agent over a split against the REAL env -> [Trajectory]
    trajectory_to_spans, write_spans_jsonl,   # Trajectory -> OTel GenAI JSONL
)
```

- **`CommandEnv.execute` is the substitution seam.** A real adapter executes commands in a real
  workspace; swap in any other implementation (a simulator, a learned environment model) and
  the identical agent loop runs against it, graded by the same deterministic function.
- **Graders are deterministic.** `grade(task, submission) -> float` must not call an LLM.
- **Observations are never synthesized.** A corpus comes from `run_capture` against the real
  environment (or a conversion of someone else's REAL runs, with provenance).

## Getting the data

```bash
# pull a full bundle (corpus + task data); local-first — never clobbers existing files
python -m environment_capture.hub fetch dabstep
python -m environment_capture.hub fetch all --force   # explicit overwrite

# or straight from the Hub with no dependencies at all
curl -LO https://huggingface.co/datasets/experiential-labs/wmh-dabstep-traces/resolve/main/traces.otel.jsonl
```

Fetching is plain-HTTP stdlib: no extra dependency, no token for public repos, per-chunk
progress callbacks for UIs, atomic `.part` writes, file-level resume. Private repos work with a
token (`HF_TOKEN` or the stored `hf auth login`).

## Publishing / updating corpora

```bash
python -m environment_capture.hub_push bird-sql          # create or update (Hub keeps history)
python -m environment_capture.hub_push all --private     # private repos
python <benchmark>/capture.py ... --push-hub             # push straight from a capture wave
```

Pushing needs the `fetch` extra (`environment-capture[fetch]`) and a write token. Every push is
a Hub commit; re-pushing after new capture waves is the update path. Corpora are **local-first**:
nothing here ever deletes a local file.

## Layout

```
environment-capture/
  environment_capture/        # the package: contract + emitter + hygiene + hub (+ inline *_test.py)
  <benchmark>/                # one dir per benchmark: adapter data, provenance README,
                              # thin capture/convert scripts; traces + task data are
                              # Hub-hosted and gitignored here
```

## Adding a benchmark

**Agents: follow [INTEGRATION.md](INTEGRATION.md) — it is the complete, self-contained
playbook.** Summary:

1. Implement a `BenchmarkAdapter` in `environment_capture/benchmarks/<name>.py` — fresh code
   against the benchmark's real upstream dataset (tests inline).
2. Create `<name>/` with the task data (license-checked) and a capture or conversion script
   that writes `traces.otel.jsonl`.
3. Audit hygiene (`scan_spans_jsonl == {}`), verify unique trace ids, eyeball trajectories,
   then publish the bundle with `hub_push`.
