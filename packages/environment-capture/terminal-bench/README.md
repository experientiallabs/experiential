# terminal-bench trace capture (harbor trials -> OTel)

Converts real harbor Terminal-Bench trial dirs into the world-model-optimizer trace
corpus format. These are real terminus-2 episodes on Terminal-Bench task sandboxes: the
agent issues `bash_command` tool calls and the **real terminal output** is recorded per
call, including real failures. The environment being reconstructed is the task sandbox's
shell: predict a command's real output given the command.

This closes the capture gap filed in DECISIONS 2026-07-27: the product runs benchmark
rollouts through harbor (`wmo optimize distill run`, probes, `harbor run`) but had no way
to ingest its own trial dirs as traces. Now any harbor jobs dir converts:

```bash
python convert_to_wmo.py <jobs_dir>... --out traces.otel.jsonl
wmo build --name terminal-bench --file traces.otel.jsonl --fidelity medium
```

Like the sibling examples, this is isolated from `wmo`:

- `convert_to_wmo.py` is **stdlib-only** (no `wmo` import, no third-party deps). It reads
  trial dirs **in place** and writes only the produced OTel JSONL.
- `examples/`-style exclusion from the `wmo` lint/type gate applies (this package is not
  part of the library surface).

## Source data

harbor `>= 0.20` trial dirs (`<jobs_dir>/<job>/<task>__<id>/`), each holding
`result.json` (task name, verifier reward, agent/model info, exception info) and
`agent/trajectory.json` (ATIF schema: per-step `tool_calls[]` paired index-wise with
`observation.results[]`).

## Conversion rules

- One trace per trial; one action/observation span pair per tool call.
- The first span carries the full initial user message as `gen_ai.prompt` (the
  instruction the agent actually saw) and `wmo.trace.metadata` (benchmark, task, trial,
  job, verifier reward, model).
- Trials with recorded `exception_info` are skipped and counted; zero-reward trials are
  KEPT (real failures with real consequences are exactly what a world model must
  reconstruct).
- Convert BEFORE re-running a job into the same jobs_dir: harbor's scorer prunes invalid
  trial dirs on resume.
