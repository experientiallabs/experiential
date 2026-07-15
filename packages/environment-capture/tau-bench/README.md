# tau2-bench trace capture (isolated)

This directory is a **self-contained, local-only capture tool**. It runs the *real*
[tau²-bench](https://github.com/sierra-research/tau2-bench) benchmark and converts its trajectories
into the world-model-harness trace corpus (`packages/environment-capture/tau-bench/traces.otel.jsonl`).

It is deliberately isolated:

- **`wmh` never imports `tau2`.** Real tau²-bench needs Python 3.12–3.13 and a heavy dependency tree
  (`litellm`, `boto3`, …); `wmh` stays on 3.11. This tool runs in its own `.venv`. Only the produced
  trace JSONL is carried back into the repo.
- The cloned `tau2-bench/`, the `.venv/`, and any `data/` simulations are **git-ignored**. Only
  the example assets in this folder are tracked: the converter, launcher scripts, README,
  `traces.otel.jsonl`, and `models/`.
- `examples/` is excluded from the `wmh` lint/type gate (`pyproject.toml`), since these task helpers
  can target different Python versions and import packages `wmh` doesn't depend on.

## Prebuilt world models

This example includes the old committed tau world models under:

```text
packages/environment-capture/tau-bench/models/tau-bench/
packages/environment-capture/tau-bench/models/tau-telecom/
```

Use them as a local model root:

```bash
uv run wmh list --root packages/environment-capture/tau-bench
uv run wmh demo --root packages/environment-capture/tau-bench --name tau-telecom
uv run wmh play --root packages/environment-capture/tau-bench --name tau-telecom
```

## Concurrency scaling law

This corpus is one benchmark in the world-model-vs-real-sandbox concurrency study — does
reconstructing observations with a world model beat standing up the real environment? Method,
per-benchmark numbers, and the figure live in
[`docs/research/concurrency_scaling_law.md`](../../docs/research/concurrency_scaling_law.md).

## Why capture from the REAL benchmark

The world model's job is to reconstruct the **actual downstream benchmark**. If we captured traces
from a re-implementation, the model would learn to imitate our approximation, not tau²-bench. So we
run Sierra's real benchmark — including its **LLM user-simulator** — and record what its real
environment actually returned.

## Setup

```bash
cd packages/environment-capture/tau-bench
git clone --depth 1 https://github.com/sierra-research/tau2-bench.git
uv venv --python 3.13 .venv
uv pip install --python .venv ./tau2-bench audioop-lts boto3
#   audioop-lts: backport of the audioop module removed from Python 3.13 stdlib (tau2 imports it)
#   boto3:       litellm's AWS Bedrock route
export TAU2_DATA_DIR="$PWD/tau2-bench/data"
.venv/bin/tau2 check-data    # should report OK
```

## Run a capture (live, on Bedrock Opus 4.8 — the only creds available here)

tau²-bench runs two LLM streams per task (the agent and the user-simulator). Opus 4.8 on Bedrock
rejects the `temperature` parameter, so pass empty LLM args to drop it.

```bash
export TAU2_DATA_DIR="$PWD/tau2-bench/data" AWS_REGION=us-east-1 AWS_REGION_NAME=us-east-1
.venv/bin/tau2 run \
  --domain airline \
  --agent-llm bedrock/us.anthropic.claude-opus-4-8 --agent-llm-args '{}' \
  --user-llm  bedrock/us.anthropic.claude-opus-4-8 --user-llm-args '{}' \
  --num-trials 1 --num-tasks 12 --max-concurrency 4 \
  --save-to airline_capture
# -> tau2-bench/data/simulations/airline_capture/results.json
```

## Bulk capture: the committed ~1000-trace corpus

`traces.otel.jsonl` here holds **~1000 distinct traces** across all three tau2 domains —
airline (50 tasks) + retail (114) + telecom (the bulk, from its 2285-task `full` split). Two helper
scripts capture and merge it; both are idempotent/resumable:

```bash
# airline + retail + telecom (single-model), then convert + merge -> traces.otel.jsonl
./capture_corpus.sh

# telecom at scale, sharded across THREE Opus models to beat per-model Bedrock throttling
TAU2_DATA_DIR="$PWD/tau2-bench/data" AWS_REGION=us-east-1 \
  .venv/bin/python capture_telecom_multimodel.py --total 990 --concurrency 3
```

Why two scripts: a **single** Opus model on Bedrock throttles hard under telecom's long,
call-heavy trajectories (litellm `ServiceUnavailableError` — a single-model run salvaged only
~180/980 telecom sims). `capture_telecom_multimodel.py` shards the telecom task list across
`opus-4-6-v1` / `opus-4-7` / `opus-4-8` — each its own per-model quota — lifting that to ~850+ with
near-zero throttling. Disjoint round-robin slices per model (`--offset` for top-ups) keep traces
unique. All valid sims are kept (reward rides along in metadata), so ~80% are reward-1.0 and the
rest are real partial trajectories. Bedrock model ids (default AWS profile, us-east-1):
`us.anthropic.claude-opus-4-7`, `us.anthropic.claude-opus-4-6-v1` (the `-v1` suffix is required),
`us.anthropic.claude-opus-4-8`.

## Convert to the wmh corpus

```bash
TAU2_DATA_DIR="$PWD/tau2-bench/data" .venv/bin/python convert_to_wmh.py \
  tau2-bench/data/simulations/airline_capture/results.json \
  --out traces.otel.jsonl --benchmark tau2-bench
```

`convert_to_wmh.py` produces, per simulation, one Step per agent **tool call**:

- `action` — the real tool call (name + arguments).
- `observation` — the **real recorded tool result** the agent saw (`gen_ai.tool.message`), error flag
  from the recorded `error`.
- `Trace.metadata` — `benchmark`, `domain`, `task_id`, the task's **gold** `evaluation_criteria`
  (expected actions + assertions), and the achieved `reward`.

`state_before` is intentionally **empty** for tau2. The airline/retail DB (full flight catalog, all
reservations, all users) is megabytes per step *and* would leak the answer — handing the model a DB
that already contains reservation `NM1VX1` turns predicting `get_reservation_details(NM1VX1)` into a
lookup, not a reconstruction. Open-loop replay reconstructs the env from the action + retrieved
similar past steps + the teacher-forced session history, which is the whole point. (The wmh adapter
still *reads* `wmh.state.*` when present, for future benchmarks whose state is small and non-leaky.)

Pure-conversational turns (no tool call) are not Steps: open-loop replay scores predicted
observations for `(state, action)`, and a chat turn has no environment observation to score.

The output is OTel-GenAI span JSONL that `wmh.ingest.otel_genai` reads directly (the per-step state
and gold travel as optional `wmh.state.*` / `wmh.trace.metadata` attributes).

## Run ONE real scenario (the real-environment side of the comparison)

### One command: `run.sh`

`./run.sh [--trace N]` does it end to end — sets up the venv/deps if missing, builds the
environment from scratch, runs the recorded scenario, and streams all stdout, ending with the
total time. That whole standup is the cost the world-model side skips.
Defaults to the simplest held-out scenario; `--trace N` pins one. Details below.

## Run ONE real scenario (manual)

`run_real_scenario.py` is the real half of the scenario comparison. The world-model side runs the
same held-out scenario through the model; this runs the SAME held-out scenario for real — it stands
up Sierra's real tau2 domain environment **from scratch** (import the heavy `tau2` package →
register components → load the domain JSON DB), times that standup and counts it in the total, then
calls the exact recorded tool calls in order, printing the real tool results. Compare the two end
times by eye.

Because tau2 actions are tool calls (not shell commands), this imports the real `tau2` package and
must run in the `.venv` from the Setup section above (NOT under `wmh`, which never imports tau2):

```bash
TAU2_DATA_DIR="$PWD/tau2-bench/data" .venv/bin/python run_real_scenario.py --trace 0
```

Stdlib + tau2 only; reads the committed `traces.otel.jsonl`, re-implements the harness's
blake2b train/holdout split inline so trace selection matches the world-model side, and reads the
`domain` from each trace's metadata. The per-run standup timed here is import + registry + DB load
(the one-time `pip install tau2-bench` is the venv Setup above). Observed (`--trace 0`, airline):
standup 1.74s + 10 tool calls, 1.74s total. tau2's env is an in-memory DB, so the comparison here is
less about speed than about not needing to stand up Sierra's stack at all.

## License — read before redistributing

[tau²-bench](https://github.com/sierra-research/tau2-bench) is published under **MIT**. The
trace corpus embeds task/domain text from it and is redistributed under the same terms with
attribution (see the Hub dataset card).
