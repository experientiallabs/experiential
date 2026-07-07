# terminal-tasks trace capture (isolated)

Converts terminal-task computer-use-agent trajectories into the world-model-harness trace corpus
(`packages/environment-capture/terminal-tasks/traces.otel.jsonl`). These are real agent runs on a terminal/bash
environment —
an LLM agent issues `bash` tool calls and the **real command output** is recorded per call (including
real failures: tracebacks, HTTP 301s, retries). The environment being reconstructed is a Unix shell:
predict a command's real output given the command.

Like the other task examples, this is isolated from `wmh`:

- `convert_to_wmh.py` is **stdlib-only** (no `wmh` import, no third-party deps). It reads the source
  trajectories **in place** and never copies them into the repo — only the produced OTel JSONL is
  written out.
- `examples/` is excluded from the `wmh` lint/type gate.

## Prebuilt world model

This example includes the old committed terminal-tasks world model under:

```text
packages/environment-capture/terminal-tasks/models/terminal-tasks/
```

Use it as a local model root:

```bash
uv run wmh list --root packages/environment-capture/terminal-tasks
uv run wmh demo --root packages/environment-capture/terminal-tasks --name terminal-tasks
uv run wmh play --root packages/environment-capture/terminal-tasks --name terminal-tasks
```

## Source data

The trajectories ship outside this repo as JSONL, one trajectory per line, with a `tool_calls` array;
each tool call has `name`, `arguments`, `output`, and an `isError` flag:

```json
{"task": "...", "task_category": "...", "returncode": 0,
 "tool_calls": [{"name": "bash", "arguments": {"command": "..."}, "output": "...", "isError": false}]}
```

## Convert

```bash
cd packages/environment-capture/terminal-tasks
python convert_to_wmh.py \
  <path/to/trajectories.jsonl> \
  --out traces.otel.jsonl --benchmark terminal-tasks \
  --exclude-substr <source-specific-path-fragment>
```

`--exclude-substr` (repeatable) drops any trajectory whose raw JSON contains the given
case-insensitive substring — used to omit trajectories whose captured command output happens to
reference source-specific filesystem paths. It drops the whole trajectory rather than redacting a
real observation, so every committed observation stays exactly what the environment returned.

Per trajectory, one Step per tool call:

- `action` — the real tool call (`bash` + `{"command": ...}`).
- `observation` — the real recorded `output`, `is_error` from the call's `isError`.
- `task` — the trajectory's task instruction (on the first step as `gen_ai.prompt`).
- `Trace.metadata` — `benchmark`, `task_category`, `returncode`.

`state_before` is empty (a shell has no compact, non-leaky state snapshot; open-loop replay
reconstructs from action + retrieved steps + teacher-forced history).

The output is OTel-GenAI span JSONL that `wmh.ingest.otel_genai` reads directly.

## Run ONE real scenario (the real-environment side of the comparison)

### One command: `run.sh`

`./run.sh [--trace N]` does it end to end — sets up the venv/deps if missing, builds the
environment from scratch, runs the recorded scenario, and streams all stdout, ending with the
total time. That whole standup is the cost the world-model side skips.
Defaults to the simplest held-out scenario; `--trace N` pins one. Details below.

## Run ONE real scenario (manual)

`run_real_scenario.py` is the real half of the scenario comparison. The world-model side runs the
same held-out scenario through the model; this runs the SAME held-out scenario for real — and to be
honest about the standup the world model skips, it **builds a fresh container from scratch**
first (a `debian:bookworm-slim` base + the real `apt-get install` of `curl`, `python3`, `jq`,
`ca-certificates`), streams that build, counts it in the total time, *then* `docker exec`s the exact
recorded `bash` commands. Compare the two end times by eye.

```bash
python run_real_scenario.py --trace 1            # cold --no-cache build (default)
python run_real_scenario.py --trace 1 --cache    # reuse the cached tools image
```

Stdlib-only (needs Docker); reads the committed `traces.otel.jsonl` and
re-implements the harness's blake2b train/holdout split inline so `--trace N` matches the world-model
side. These commands hit live public APIs, so a real re-run reflects *current* data and the output
may differ from the recorded observation (rates change, releases bump) — that is the honest real
environment. Observed (`--trace 1`, cold): build from scratch 8.7s + 10 commands, 10.8s total.

## Hygiene

Capture ran with the agent inside the task's Docker container, so absolute paths in commands and
output (`/tmp/...`, `/usr/...`) are the container's own filesystem — the host-escape audit's
command-path and generic path/toolchain signals do not apply to this corpus (see
`packages/environment-capture/INTEGRATION.md`). The machine-identity markers (real host username and
home path) were verified absent.

## License — read before redistributing

[terminal-bench](https://github.com/laude-institute/terminal-bench) is published under
**Apache 2.0**. The trace corpus embeds task text and container output from it and is
redistributed under the same terms with attribution (see the Hub dataset card).
