---
source: https://app.notion.com/38e0f8b3f591817aa9e1e6486a9647c4
area: Data & Traces
status: Current
migrated: 2026-07-02
---

# Benchmarks → traces: the real trace source

The harness reconstructs an environment from **recorded traces of past agent runs**. For those
traces to be worth anything, they must come from the **actual downstream benchmark** — if we captured
from a re-implementation, the world model would learn to imitate our approximation instead of the
real benchmark. So we run the real benchmark (including its own LLM user-simulator) and record what
its real environment actually returned.

Captured benchmarks so far. Each is a **self-contained example** under `examples/<task>/`: the corpus (`traces.otel.jsonl`), the isolated capture tooling, prebuilt `models/`, and `evals/` suites all live together (the old separate `tools/<benchmark>-capture/` layout was consolidated into `examples/` in PR #38):

- **`tau-bench`** (1033 traces) — captured live from Sierra's real tau²-bench on Bedrock (airline + retail + telecom; per-step gold rides in metadata). Multi-model capture sharding (Opus 4.6/4.7/4.8) to beat per-model throttling.
- **`terminal-tasks`** (280 traces) — real computer-use-agent runs on a Unix shell (`bash` tool calls with the real command output recorded per call, including failures: tracebacks, HTTP 301s, retries).
- **`swe-bench`** (87 traces) — the real SWE-bench Verified with the standard mini-swe-agent harness, each instance in its own per-instance Docker image; arbitrary code-execution output (pytest logs, tracebacks) makes it the hardest corpus to reconstruct.

## The trace contract

Each capture produces a `wmh.core.types.Trace`. Per agent **tool call**, one `Step`:

- **`action`** — the real tool call (`name` + `arguments`).
- **`observation`** — *exactly* what the real environment returned (the recorded tool result), with
  the recorded error flag. This is the open-loop ground truth the scorer grades predictions against.
- **`state_before`** — the environment state **before** the action. Optional and benchmark-dependent:
  populated only when a benchmark's state is small and non-leaky. For tau2 it is intentionally empty
  (the env DB is huge and would leak the answer — see below). Open-loop replay feeds
  `(state_before, action)` to the world model.
- **`task`** — the originating user instruction.

And **`Trace.metadata`** carries `benchmark`, `domain`, `task_id`, the task's **`gold`** evaluation
criteria (expected actions + assertions), and the achieved `reward`. Gold rides along for the
deferred **closed-loop** eval; the **open-loop** scorer ignores it (its ground truth is the recorded
observation).

Traces are stored as one-span-per-line OTel-GenAI JSONL that `wmh.ingest.otel_genai` reads. The
per-step state and trace metadata travel as optional `wmh.state.*` / `wmh.trace.metadata` span
attributes — a strict superset of the OTel GenAI semconv, so any trace that omits them still parses.

## How tau²-bench is captured

The pipeline lives in `examples/tau-bench/` (capture scripts + converter next to the corpus) and is deliberately **isolated** from `wmh`:

- It runs Sierra's real [tau²-bench](https://github.com/sierra-research/tau2-bench) (`tau2 run`),
  which drives a fixed agent and an LLM user-simulator against the real domain environment. Both LLMs
  run on Bedrock Opus 4.8.
- `wmh` **never imports `tau2`**. tau²-bench needs Python 3.12–3.13 + a heavy dependency tree; `wmh`
  stays 3.11. The capture tool runs in its own `.venv`; only the produced trace JSONL is carried back
  into the repo. (`tools/` is git-ignored except the conversion script + README, and excluded from
  the `wmh` lint/type gate.)
- `convert_to_wmh.py` turns a tau2 `results.json` into the corpus: per agent tool call, the real
  action + the authoritative recorded observation the agent saw, with gold + reward + domain in
  `Trace.metadata`. tau2's `state_before` is left empty by design — the airline/retail DB is
  megabytes per step and would leak the answer (giving the model a DB that already contains the
  reservation it's asked to look up makes the eval a lookup, not a reconstruction). Open-loop replay
  reconstructs the env from the action + retrieved similar steps + teacher-forced history.

See the tool's README for the exact setup + run + convert commands.

## Adding a new benchmark

The model is one **adapter per benchmark** — a self-contained example directory under `examples/<task>/`:

1. **Run the real benchmark.** Install its real upstream package in an isolated env (its own
   `.venv`, whatever Python it needs). Run it with our fixed agent on Bedrock. Do **not** add it as a
   `wmh` dependency — `wmh` must stay importable on 3.11 without it.
2. **Convert to the trace contract.** Write a `convert_to_wmh.py` that, per recorded step, emits the
   real `action` and the real recorded `observation`, and stamps `Trace.metadata` with the benchmark
   name + gold. Populate `state_before` only if the benchmark's state is small and **non-leaky** — if
   it would contain the answer to the action being scored (as tau2's full DB does), leave it empty and
   let replay reconstruct. Never invent state.
3. **Emit OTel-GenAI JSONL** in the same shape (`gen_ai.*` spans + optional `wmh.state.*` /
   `wmh.trace.metadata`) so `wmh.ingest.otel_genai` reads it with no new adapter.
4. **Commit** `examples/<task>/traces.otel.jsonl` plus the conversion script, launcher, README, and
   an `evals/default.toml` suite. Keep the cloned upstream, venv, and raw run output git-ignored.
5. **Gate.** `uv run ruff check .`, `uv run ty check`, and `uv run pytest -q` must be clean over the
   `wmh` package (the isolated `examples/` env is excluded).
