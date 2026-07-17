# SWE-bench trace capture (isolated)

This directory is a **self-contained, local-only capture tool**. It runs the *real*
[SWE-bench Verified](https://www.swebench.com/) benchmark with the standard
[mini-swe-agent](https://github.com/SWE-agent/mini-swe-agent) harness and converts the recorded
agent trajectories into the world-model-harness trace corpus
(`packages/environment-capture/swe-bench/traces.otel.jsonl`).

It is deliberately isolated, exactly like `packages/environment-capture/tau-bench/` and
`packages/environment-capture/terminal-tasks/`:

- **`wmh` never imports SWE-bench or mini-swe-agent.** SWE-bench Verified runs each instance in its
  own per-instance Docker image (the buggy repo at a pinned commit + its full test env); the agent
  harness needs its own heavy dependency tree. This tool runs in its own `.venv` and uses the local
  Docker daemon. Only the produced trace JSONL is carried back into the repo.
- The cloned harness, the `.venv/`, the pulled Docker images, and the raw run output are
  **git-ignored**. The tracked example assets are the converter, launcher scripts, README,
  `traces.otel.jsonl`, and `models/`.
- `examples/` is excluded from the `wmh` lint/type gate (`pyproject.toml`), since these task helpers
  can target different Python versions and import packages `wmh` doesn't depend on.

## Prebuilt world model

This example includes the old committed SWE-bench world model under:

```text
packages/environment-capture/swe-bench/models/swe-bench/
```

Use it as a local model root:

```bash
uv run wmh list --root packages/environment-capture/swe-bench
uv run wmh demo --root packages/environment-capture/swe-bench --name swe-bench
uv run wmh play --root packages/environment-capture/swe-bench --name swe-bench
```

## Why capture from the REAL benchmark

The world model's job is to reconstruct the **actual downstream benchmark**. SWE-bench's environment
is a real shell inside a real repo container: the agent runs commands (`ls`, `cat`, `sed`, `python
-m pytest`, …) and the environment returns the **real** stdout/stderr + exit code — including
tracebacks, build errors, and test logs. We record exactly what the real environment returned, never
a re-implementation. (This also makes SWE-bench the *low-fidelity* end of the spectrum for a world
model: its observations are arbitrary code-execution output, the hardest thing to reconstruct — by
design, so the harness is honest about where it's hard.)

## Setup

```bash
cd packages/environment-capture/swe-bench
git clone --depth 1 https://github.com/SWE-agent/mini-swe-agent.git
uv venv --python 3.12 .venv
uv pip install --python .venv ./mini-swe-agent 'swebench' boto3
#   swebench: the Verified dataset loader + (optionally) the official evaluation harness
#   boto3:    litellm's AWS Bedrock route
# Docker must be running locally; the agent execs commands inside the per-instance images.
```

## Run a capture (live, on Bedrock Opus 4.8 — the only creds available here)

mini-swe-agent's `swebench` runner pulls each instance's Docker image, runs the agent loop inside it,
and writes one `<instance_id>.traj.json` per instance under the output dir. Opus 4.8 on Bedrock
rejects the `temperature` parameter, so the model config must not set one.

```bash
export AWS_REGION=us-east-1 AWS_REGION_NAME=us-east-1
# A SMALL slice: --subset verified, the first few instances. Each instance pulls a multi-GB
# x86_64 image (runs under emulation on arm64) and runs a real agent loop, so keep the slice small.
.venv/bin/python -m minisweagent.run.benchmarks.swebench \
  --subset verified --split test --slice 0:3 \
  --environment-class docker \
  -m bedrock/us.anthropic.claude-opus-4-8 \
  -o runs/verified_capture
# -> runs/verified_capture/<instance_id>/<instance_id>.traj.json  (one per instance)
```

(Flag names follow the installed mini-swe-agent version — see `python -m
minisweagent.run.benchmarks.swebench --help`. The shape that matters: a per-instance `*.traj.json`
whose `messages` are the recorded agent loop. The default model config must not set `temperature`
since Opus 4.8 rejects it; the bundled `swebench.yaml` config is a fine base.)

## Convert to the wmh corpus

```bash
.venv/bin/python convert_to_wmh.py \
  runs/verified_capture \
  --out traces.otel.jsonl --benchmark swe-bench
```

`convert_to_wmh.py` (stdlib-only, no `wmh` import) reads every `*.traj.json` under the run dir and
produces, per trajectory, one Step per agent **shell command**:

- `action` — the real command the agent ran (`bash {"command": "..."}`), parsed from the assistant
  message's fenced command block.
- `observation` — the **real recorded command output** the agent saw (the following environment
  message: stdout/stderr inside `<output>…</output>`), with `is_error` from the recorded
  `<returncode>` being non-zero.
- `task` — the instance's problem statement (the GitHub issue), carried on the first step.
- `Trace.metadata` — `benchmark`, `instance_id`, `repo`, and the gold `model_patch`/`exit_status`
  when present (gold rides along for the deferred closed-loop eval; the open-loop scorer ignores it).

`state_before` is left **empty**: the environment state is an entire repo working tree (huge, and a
`cat` of the buggy file would leak the answer to "what does this file contain?"). Open-loop replay
reconstructs from the action + retrieved similar steps + teacher-forced history — the whole point.

Pure-reasoning turns (assistant messages with no command) are not Steps: open-loop replay scores a
predicted observation for `(state, action)`, and a reasoning turn has no environment observation.

The output is OTel-GenAI span JSONL that `wmh.ingest.otel_genai` reads directly.

## Concurrency scaling law

This corpus is one benchmark in the world-model-vs-real-sandbox concurrency study — does
reconstructing observations with a world model beat standing up the real environment? Method,
per-benchmark numbers, and the figure live in
[`docs/research/world_model_findings.md`](../../docs/research/world_model_findings.md).

## Run ONE real scenario (the real-environment side of the comparison)

### One command: `run.sh`

`./run.sh [--trace N]` does it end to end — sets up the venv/deps if missing, builds the
environment from scratch, runs the recorded scenario, and streams all stdout, ending with the
total time. That whole standup is the cost the world-model side skips.
Defaults to the simplest held-out scenario; `--trace N` pins one. Details below.

## Run ONE real scenario (manual)

`run_real_scenario.py` is the real half of the scenario comparison. The world-model side runs the
same held-out scenario through the model; this runs the SAME held-out scenario for real — and
crucially it **builds the environment from scratch** before running anything: the SWE-bench base
image → the environment image (the real conda/pip **dependency install**) → the instance image
(clone repo + checkout commit + install). Every `docker build` line is streamed and the whole
standup is counted in the total time, *then* the recorded commands are `docker exec`'d. That build
is the slow, multi-minute cost the world model skips entirely.

```bash
# from the swebench .venv; same --trace index the world-model side uses (same held-out split)
.venv/bin/python run_real_scenario.py --trace 0           # cold --no-cache build (default)
.venv/bin/python run_real_scenario.py --trace 0 --cache   # reuse cached layers after the first build
```

Imports `swebench` (for the official base/env/instance Dockerfiles + setup scripts) but never `wmh`;
reads the committed `traces.otel.jsonl` and re-implements the harness's blake2b
train/holdout split inline so `--trace N` selects the SAME scenario as the world-model side.

Observed (astropy__astropy-13453, `--trace 0`, cold `--no-cache`): **build from scratch 339.5s + 19
commands, 362.0s total** — vs. the world model reconstructing the same 19-step scenario in ~96s with
**zero** standup. The dependency install is the gap.

## Hygiene

Capture ran with the agent inside the task's Docker container, so absolute paths in commands and
output (`/tmp/...`, `/usr/...`) are the container's own filesystem — the host-escape audit's
command-path and generic path/toolchain signals do not apply to this corpus (see
`packages/environment-capture/INTEGRATION.md`). The machine-identity markers (real host username and
home path) were verified absent.

## License — read before redistributing

[SWE-bench](https://github.com/SWE-bench/SWE-bench) and
[mini-swe-agent](https://github.com/SWE-agent/mini-swe-agent) are published under **MIT**; the
underlying repos are permissively licensed open source. The trace corpus embeds issue/repo text
from them and is redistributed under the same terms with attribution (see the Hub dataset card).
