---
source: https://app.notion.com/38e0f8b3f5918129b2dbcc1e67e09230
area: Runbook
status: Current
migrated: 2026-07-02
---

# Runbook: building a world model from tau2 traces (real Bedrock)

This walks the full pipeline end-to-end on real data: ingest agent traces, evolve the env prompt
with GEPA on a live LLM, persist a `.wmh/` artifact, then load it and step against it. It was run
and verified on 2026-06-25 with **Bedrock Opus 4.8** for generation and the offline
`HashingEmbedder` for retrieval (no embedding credentials required).

> **Note (corpus changed):** the worked numbers and `get_user u_kath`-style tool calls below were
> recorded against the earlier SIB-derived bash corpus (3 traces). The committed corpus is now
> captured from the **real tau²-bench** (§0) and has grown to **1033 traces**; its actions are real
> tool calls (e.g. `get_user_details(user_id=...)`), not bash commands. The *pipeline steps* are
> unchanged; the specific commands/outputs shown are illustrative and will differ on the current
> corpus.

## 0. Source data

`examples/tau-bench/traces.otel.jsonl` is a ready-to-use OTel GenAI trace corpus (1033 traces)
captured from the **real** [tau²-bench](https://github.com/sierra-research/tau2-bench) benchmark —
see *Benchmarks → traces: the real trace source* for how it is produced (the capture tooling lives
next to the corpus in `examples/tau-bench/`; `wmh` never imports tau2). Each trace is one solved
task: per agent tool call, the real tool-call action and the real recorded observation, with the
task's gold and reward in `Trace.metadata`. Two sibling corpora exist via the same pattern:
`examples/terminal-tasks/` (280 traces) and `examples/swe-bench/` (87 traces).

## 1. Build the world model (ingest → split → index → GEPA → persist)

```bash
AWS_REGION=us-east-1 uv run wmh build \
  --name tau2-airline \
  --file examples/tau-bench/traces.otel.jsonl \
  --root /tmp/tau2_wmh \
  --provider bedrock --model us.anthropic.claude-opus-4-8 --region us-east-1 \
  --gepa-budget 6
```

The build renders a guided, animated pipeline (ingest → split → index → a live GEPA rollout
progress bar with the running held-out score → a summary panel). When piped to a file it degrades to
one plain line per stage, so captured logs stay legible.

Observed (budget 6): `held_out_accuracy=0.562, frontier=2, rollouts=14`. GEPA improved the held-out
judge score from the base prompt's ~0.40 to **0.562** within the budget. The evolved prompt is
genuinely specialized — it inferred the environment is a Unix shell/tool sandbox and even captured
the exact JSON schemas the tau2 tools emit (e.g. the `get_user` record shape and key ordering).

`wmh list --root /tmp/tau2_wmh` shows every model built under the project dir.

### Artifact layout (`/tmp/tau2_wmh`)

World models are named and stored under `models/<name>/`; each is a self-contained artifact:

```
models/tau2-airline/
  config.toml              # HarnessConfig (serve provider, embed_dim, top_k, ...)
  prompts/base.txt         # the un-evolved BASE_ENV_PROMPT
  prompts/optimized.txt    # GEPA winner (what serve uses)
  prompts/frontier.json    # Pareto frontier of candidate prompts
  metrics.json             # held_out_accuracy, rollouts_used
  index/embeddings.npy     # phi(s,a) matrix for the replay buffer
  index/steps.jsonl        # the parallel Steps
```

## 2. Load the stored model and step against it

```python
from wmh.config.store import WorldModelStore
from wmh.core.types import Action, ActionKind
from wmh.engine.loader import load_world_model

# Resolve the named model under the project root and load it with the provider it was built on.
model_dir = WorldModelStore("/tmp/tau2_wmh").resolve("tau2-airline")
wm, _provider = load_world_model(model_dir)

s = wm.new_session(task="Customer request: I am Katherine Johnson (u_kath). Look up my account.")
print(wm.step(s.id, Action(kind=ActionKind.TOOL_CALL, name="bash",
                           arguments={"command": "get_user u_kath"})))
```

Observed:

- `get_user u_kath` → `{"membership": "silver", "name": "Katherine Johnson", "reservations": []}`,
  `is_error=False` — matches the training trace's user record (retrieval grounded the prediction).
- `get_reservation r_999` → `Error: reservation r_999 not found`, `is_error=True` — the model
  *simulates* environment behavior (errors on a missing id) rather than echoing a demo.

## 4. Play it yourself (interactive REPL)

Step into the reconstructed environment as the agent — type tool calls, the world model answers,
and the session scratchpad evolves so later turns stay consistent:

```bash
AWS_REGION=us-east-1 uv run wmh play --root /tmp/tau2_wmh --name tau2-airline \
  --task "Look up user u_kath."
# agent> get_user {"command": "get_user u_kath"}
#   -> observation panel with the env's JSON reply
# agent> :state     # show the task, turn count, and scratchpad "database"
# agent> :quit
```

Verified on 2026-06-25 against real Bedrock Opus 4.8: typing `bash {"command": "get_user u_kath"}`
returned the silver-membership user record, and a follow-up missing-id lookup returned a simulated
`not found` error — the scratchpad accumulated the state notes across turns.

## 5. Serve it over HTTP (same code path)

```bash
AWS_REGION=us-east-1 uv run wmh serve --root /tmp/tau2_wmh
# GET  /world_models                                 ->  {"world_models": ["tau2-airline"]}
# POST /world_models/tau2-airline/sessions           ->  {"session_id": ...}
# POST /world_models/tau2-airline/sessions/{id}/step  with {"action": {"kind": "tool_call", ...}}
```

`wmh serve` serves every built model by default; pass `--name` (repeatable) to serve a subset.

## Reproducing the verification automatically

`wmh/engine/integration_test.py::test_build_load_step_against_real_bedrock` runs build→load→step
against real Bedrock with a tiny budget. It is **skipped unless `AWS_REGION` is set** (same gate as
the provider live smoke tests), so the default `uv run pytest` stays offline and deterministic.

```bash
AWS_REGION=us-east-1 uv run pytest wmh/engine/integration_test.py -q   # ~37s, real LLM
```

## Notes / limitations

- The cache only had 3 tau2 transcripts (→ 3 traces, 4 steps), so the train/held-out split and the
  replay buffer are small; numbers are a smoke signal, not a benchmark.
- `embed_dim` is persisted in `config.toml` and `WorldModel.load` rebuilds the matching embedder; a
  mismatch raises a clear error instead of a cryptic numpy matmul failure.
- Embeddings stay offline by design (`HashingEmbedder`). Wiring Bedrock Titan / a real embed model
  into `BedrockProvider.embed` is a separate, additive change.
