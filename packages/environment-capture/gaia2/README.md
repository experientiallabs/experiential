# gaia2

A **stateful** multi-app simulated world. GAIA2 / Meta Agents Research Environments (ARE) drops an
agent into a simulated universe of apps (Contacts, Email, Messaging, Calendar, RentAFlat, Shopping,
CabApp, CityApp, a sandbox file system, ...) pre-populated with fictional user data. A `USER`
message states a task — e.g. *"save every apartment in zip codes whose violent-crime rate is 5-10"*
or *"add together the ages of all my contacts in Dublin, then in Galway, and give the absolute
difference"* — and the agent completes it by calling the apps' real Python tools against a **live,
mutable world** (state persists across steps), then answering via
`AgentUserInterface__send_message_to_user`. State carries across steps, exactly the world-model
dynamics this benchmark exercises — see `environment_capture/benchmarks/gaia2.py`.

> ## ⚠ The reward here is NOT the official Gaia2 score
> The official Gaia2 verifier grades an agent by matching its write-actions to the scenario's oracle
> actions using exact-match for structured fields **and an LLM rubric (Llama-3.3-70B model-as-a-judge)
> for free-text fields**. An LLM judge is incompatible with this harness's `grade()` contract, which
> must be **deterministic and LLM-free** (a world model has to be judged by the same fixed function
> as the real environment). So we grade with our **own deterministic structural approximation**
> (`score_actions`): exact/numeric match on structured args and **normalized-string equality**
> (lowercased, whitespace-collapsed) on text args — which is **stricter than the official rubric on
> free text** and is order-insensitive. Treat these reward/fidelity numbers as this harness's
> internal signal; they are **not comparable to the Gaia2 leaderboard**.

## Architecture: the engine runs out-of-process

The ARE engine pulls a large dependency tree and loads big scenario universes, so it lives in a
**benchmark-local venv** (`./.venv`, gitignored) and the gate-checked
`environment_capture.benchmarks.gaia2` module never imports it. Instead:

- `Gaia2Adapter.open_env(task)` launches `backend/world_backend.py serve <scenario.json> <state.json>`
  under that venv. The subprocess imports ONE scenario, populates its apps, and speaks a
  line-delimited JSON protocol on stdio; `Gaia2Env` is the client. Each agent action — the
  `CommandEnv.execute` seam — is a block of Python run in a stateful shell where a `tools` dict of
  the scenario's app tools is preloaded, so world mutations persist across steps.
- `Gaia2Agent` (Bedrock, in the gate module, ARE-free) is the capture agent: its only action is the
  `execute_python` tool.
- The backend logs every executed tool call (app, function, args, write-flag) and, on `close`, dumps
  the log to a per-task state file. `grade` reads that log and matches its **write-actions** against
  the scenario's oracle actions with the deterministic `score_actions` (no LLM). Reward is the
  matched fraction (`matches / max(#oracle, #agent-writes)`), so both misses and extra writes lower
  it; 1.0 requires exactly the oracle set with matching args.

Swapping `Gaia2Env` for a world-model-backed `CommandEnv` runs the identical agent loop against a
world model instead of the real engine — the point of the harness.

## Contents

- `capture.py` — fresh real-run capture against this adapter (Bedrock Python-REPL agent, sharded
  across models). Writes the committed `traces.otel.jsonl`.
- `backend/` — the venv-only side (imports the ARE engine / `datasets`; excluded from the repo type
  gate):
  - `fetch_data.py` — reads the `execution` + `search` validation splits from HuggingFace and
    materializes `data/{train,test}.jsonl` (prompt + oracle actions) + `datafiles/<task_id>.json`
    (full scenario universe, gitignored).
  - `world_backend.py` — the `serve` engine subprocess (stateful Python-REPL over the scenario).
  - `smoke.py` — end-to-end plumbing check against a real world (no Bedrock).
- `evals/default.toml` — the open-loop fidelity replay suite.
- `data/{train,test}.jsonl` — committed task index (prompt + oracle actions; CC-BY-4.0).
- `datafiles/` (gitignored) — the large, re-fetchable scenario universe JSONs.

## Getting the data

The ARE engine needs its own venv; the scenario universes are re-fetched (not committed):

```bash
cd packages/environment-capture/gaia2
uv venv --python 3.11 .venv
VIRTUAL_ENV=.venv uv pip install meta-agents-research-environments
./.venv/bin/python backend/fetch_data.py   # materializes data/ + datafiles/ from HuggingFace
```

Then, from the repo root, smoke-test the plumbing and capture:

```bash
uv run python packages/environment-capture/gaia2/backend/smoke.py            # real world, no model
uv run python packages/environment-capture/gaia2/capture.py --split train --limit 40 \
    --models us.anthropic.claude-opus-4-8,us.anthropic.claude-opus-4-7
```

## Scope & splits

We scope to the **`execution`** and **`search`** capabilities (160 scenarios each). These are
completed from the initial universe state by agent tool calls; `fetch_data.py` additionally keeps
only scenarios with no time-driven `ENV`/`CONDITION` events, so a run is faithful without advancing
the simulation clock. GAIA2's other capabilities — `adaptability`, `time`, `ambiguity` — depend on
time-driven events or clarification dialogue and are out of scope. The in-scope scenarios are
seeded-split (seed 7) ~70/30 into train/test; **only train is captured** (the test split never
enters the world model).

## Results (2026-07-03, corpus as committed)

- **In-scope**: 320 `execution` + `search` validation scenarios (224 train / 96 test, seed-7 split;
  test never captured).
- **Corpus**: 37 fresh real Bedrock runs (16 on `us.anthropic.claude-opus-4-7`, 21 on `-4-8`), 563
  `execute_python` transitions, mean reward **0.253** (our strict structural approximation, NOT the
  official score — 2 full solves, 18 partial; the tasks are hard multi-step reasoning and the grader
  is strict on free-text answers). Hygiene audit `scan_spans_jsonl(...) == {}` (clean). The 37 runs
  cover 19 distinct scenarios resampled across models and passes (run tags r1-r5), so treat the
  effective task diversity accordingly (the D33 cross-run-overlap caveat applies to fidelity).
  - Grown across five `--append` waves (10 → 19 → 32 → 37) and rebalanced to opus-4-8 majority;
    a hair short of the ~40+ target because tail tasks kept getting skipped on Bedrock
    `ServiceUnavailableException` (the endpoint flaps under load). `capture.py` emits each trajectory
    durably (a hang never discards completed work) and re-runs with `--append`/`--run-start` grow it.
- **Open-loop fidelity** (Opus 4.8 target + rubric judge, seed 0; final 37-trace corpus):
  mean fidelity **0.773**, error-flag accuracy **0.939**, n=196 held-out steps (the earlier
  0.652 @10 traces was a noisy small-n measurement). Measured via the cross-provider failover
  runner (`.agents/scripts/eval_with_fallback.py`: bedrock-4.8 → anthropic-direct-4.8 →
  bedrock-4.7 — same Opus 4.8 weights on the first two links, so the judge stays comparable).
  Sits mid-family: above financebench's document excerpts (0.586), below
  appworld's structured API observations (0.793); the residual is opaque per-universe identifiers
  (contact/message ids) the model cannot infer from the request alone.

## ⚠ Please do not train on this evaluation data (maintainers' request)

GAIA2 is an **evaluation** benchmark and ships only a `validation` split (with public ground truth).
Its maintainers explicitly ask: *"Please help us keep this benchmark strong by not training on this
evaluation data."* This harness's purpose is to build a **world model from the trace corpus**, and
the corpus here is captured on GAIA2 validation scenarios — so using it to train a model runs
against that request. We surface this for a final call at PR review: the corpus is committed because
CC-BY-4.0 permits redistribution, but whether to *train* on it is a separate decision. Each trace's
metadata records `source: meta-agents-research-environments/gaia2 (CC-BY-4.0)` and
`reward_kind: structural-approx-not-official-gaia2` for provenance.

## License — read before redistributing

GAIA2 is released under **CC-BY-4.0**; the task index and trace corpus are redistributed here under
CC-BY-4.0 **with attribution to Meta** (`meta-agents-research-environments/gaia2`). Additionally, the
scenario content is synthetic data generated by **Llama 3.3 and Llama 4 Maverick** and is subject to
the respective Llama licenses: if you use this data to create, train, fine-tune, or otherwise improve
a distributed AI model, you must include **"Llama"** at the beginning of that model's name (and
comply with the [Llama 3.3](https://github.com/meta-llama/llama-models/blob/main/models/llama3_3/LICENSE)
/ [Llama 4](https://github.com/meta-llama/llama-models/blob/main/models/llama4/LICENSE) license terms).

## Provenance

- **Dataset**: [meta-agents-research-environments/gaia2](https://huggingface.co/datasets/meta-agents-research-environments/gaia2)
  (Meta, "Gaia2: Benchmarking LLM Agents on Dynamic and Asynchronous Environments"), `execution` +
  `search` validation splits. Scenario universes, USER tasks, and oracle actions are the real
  upstream release; all adapter/agent/backend/grader code here is fresh.
- **Engine**: the ARE framework (`meta-agents-research-environments`, PyPI) boots each scenario and
  exposes its app tools; the agent acts by calling them against the live world.
- **Traces**: captured fresh with `capture.py` — a Bedrock Python-REPL agent
  (`us.anthropic.claude-opus-4-8` / `-4-7`) operating each scenario for real, graded by this
  adapter's deterministic structural matcher (NOT the official Gaia2 LLM-judge score). Each trace's
  task id is run-suffixed (`gaia2-train-3#opus48-r1`) so trace ids never collide across models/runs;
  the base task id and reward ride in the trace metadata. Observations are never synthesized.
- **Workspace containment**: the simulated apps run in-process under the venv and never touch the
  host filesystem; the shared hygiene audit (`environment_capture.scan_spans_jsonl`) reports no host
  content on the committed corpus (see Results).
