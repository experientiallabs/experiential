# environment-capture

Run agent benchmarks **for real** and record every agent-environment transition — each
`(action → observation)` pair, exactly as the environment returned it — as OpenTelemetry GenAI
JSONL. Ten benchmarks are integrated behind one small contract, with **5,900+ real trajectories
/ 27,000+ real transitions** already captured and published as license-tagged datasets on the
[Hugging Face Hub](https://huggingface.co/experiential-labs).

## Why this over the alternatives

- **Real transitions, never synthesized.** Every observation in a corpus came from a live
  benchmark environment (a real SQLite query, a real container shell, a real simulated-world
  API) — not from a model imagining what an environment might say. If you are training or
  evaluating environment/world models, imitation policies, or reward models, synthetic
  observations teach the model your generator's quirks; these teach it the environment.
- **One contract across wildly different benchmarks.** `tasks(split) / open_env(task) /
  grade(task, submission)` plus a single env seam (`execute(command) -> output, returncode`)
  covers text-to-SQL, document QA, pandas data analysis, CRM analytics, stateful multi-app
  worlds, customer-service tool agents, terminal computer-use, and software engineering. A new
  benchmark is one adapter, not a new harness.
- **Deterministic, LLM-free grading.** Every `grade` is a fixed function — rewards are
  reproducible, free, and comparable across runs. No judge drift, no judge bill.
- **A standard wire format, not a proprietary one.** Corpora are OTel GenAI semantic-convention
  spans, one JSON object per line — readable by any OTel-aware tooling and trivially parseable
  without this package installed.
- **Privacy hygiene is built in, because it bit us.** Agents that can't find their data wander
  the host; a built-in scanner (`scan_spans_jsonl`) detects host-escape content (home paths,
  credentials, machine identity) at capture time and in committed corpora, and flagged
  trajectories are dropped whole, never redacted.
- **Capture runs survive the real world.** Per-task fault isolation with retries: one throttled
  provider call, grader edge case, or backend crash records a failure and moves on — a
  multi-hour capture never loses its completed trajectories.
- **License discipline end to end.** Each corpus ships with provenance and the upstream's
  license tag on its dataset card; benchmarks whose terms forbid plain-text redistribution
  (AppWorld) are refused by the publisher and stay local-only.
- **Agent-first integration.** [INTEGRATION.md](INTEGRATION.md) is a complete, self-contained
  playbook: hand it to a coding agent and it has everything needed to integrate a new benchmark
  — contract, step order, non-negotiables, and the acceptance checklist.

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
