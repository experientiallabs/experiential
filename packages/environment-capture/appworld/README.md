# appworld

A **stateful** multi-app world. AppWorld drops an agent into a simulated world of nine apps (Amazon,
Gmail, Venmo, Spotify, phone, file system, Splitwise, Todoist, SimpleNote) plus a `supervisor` app
for the account, behind 450+ real Python APIs. A task is a natural-language request — e.g. *"what is
the title of the most-liked song in my Spotify playlists"* — and the agent completes it by writing
Python that calls `apis.<app>.<endpoint>(...)` against a **live, mutable world**, signalling
completion with `apis.supervisor.complete_task(...)`. This is the first adapter whose world state
carries across steps (variables and world mutations persist), which is exactly the world-model
dynamics this benchmark exists to exercise — see
`environment_capture/benchmarks/appworld.py`.

## Architecture: the engine runs out-of-process

The `appworld` engine is Python-3.11-only, pulls a large dependency tree, and ships its data as
encrypted `.bundle` files — so it lives in a **benchmark-local venv** (`./.venv`, gitignored) and the
gate-checked `environment_capture.benchmarks.appworld` module never imports it. Instead:

- `AppWorldAdapter.open_env(task)` launches `backend/world_backend.py serve <appworld_id> <exp>` under
  that venv. The subprocess boots ONE live `AppWorld` for the task and speaks a line-delimited JSON
  protocol on stdio; `AppWorldEnv` is the client. Each agent action — the `CommandEnv.execute` seam —
  is a block of Python run against that live world, so state persists across steps.
- `AppWorldAgent` (Bedrock, in the gate module, appworld-free) is the capture agent: its only
  environment action is the `execute_python` tool, faithful to AppWorld's real action space.
- `grade` shells out to `backend/world_backend.py grade`, which runs AppWorld's **own deterministic
  evaluation tests** over the final world state; reward is the fraction of those tests that pass. No
  LLM judging.

Swapping `AppWorldEnv` for a world-model-backed `CommandEnv` runs the identical agent loop against a
world model instead of the real engine — the point of the harness.

## Contents

- `capture.py` — fresh real-run capture against this adapter (Bedrock Python-REPL agent, sharded
  across models). Writes `traces.otel.jsonl` (gitignored, see License).
- `backend/` — the venv-only side (imports `appworld`; excluded from the repo type gate):
  - `fetch_data.py` — `appworld install` + `appworld download data`, then materialize
    `data/train.jsonl`.
  - `world_backend.py` — the `serve` / `grade` engine subprocess.
  - `smoke.py` — end-to-end plumbing check against a real world (no Bedrock).
- `evals/default.toml` — the open-loop fidelity replay suite.
- `data/` (gitignored) — AppWorld's downloaded data + the materialized `data/train.jsonl`
  (`{task_id: "aw-train-<i>", prompt, data: {appworld_id}}`).

## Getting the data

Python 3.11 is required by `appworld`. Create the benchmark-local venv, install the engine, and
materialize the task index (all gitignored):

```bash
cd packages/environment-capture/appworld
uv venv --python 3.11 .venv
VIRTUAL_ENV=.venv uv pip install appworld
./.venv/bin/python backend/fetch_data.py   # runs `appworld install` + `download data`, writes data/train.jsonl
```

Then, from the repo root, smoke-test the plumbing and capture:

```bash
uv run python packages/environment-capture/appworld/backend/smoke.py            # real world, no model
uv run python packages/environment-capture/appworld/capture.py --split train \
    --models us.anthropic.claude-opus-4-8,us.anthropic.claude-opus-4-7
```

## Splits

Only the `train` split (90 tasks) is materialized and captured. AppWorld's hidden test splits
(`test_normal`, `test_challenge`) are **never** captured, so the world model can't absorb their
dynamics.

## Results (2026-07-02, corpus as captured)

- **Corpus**: 90 fresh real Bedrock runs over the train split (50 on
  `us.anthropic.claude-opus-4-8`, 40 on `-4-7`), covering all 90 train tasks, 1,058 recorded
  `execute_python` transitions, mean reward **0.983** (93% of tasks fully solved by AppWorld's own
  tests). Real-host-identity audit is clean (0 trajectories).
- **Open-loop fidelity** (suite `appworld/default`, seed 0, Opus 4.8 target + rubric judge, run via
  `uv run wmh eval run appworld/default --examples-root packages/environment-capture`): mean fidelity
  **0.802 ± 0.208**, error-flag accuracy **0.957**, n=280 held-out steps. AppWorld's structured JSON API
  observations reconstruct better than financebench's document excerpts (0.581) but below bird-sql's
  fully structured sqlite rows (0.864) — the residual is opaque per-world identifiers and values
  (song ids, tokens, amounts) the model cannot infer from the request alone.

### Hygiene: benchmark-scoped host-escape checks

AppWorld executes agent code in its OWN sandbox (SafetyGuard blocks `os.listdir` / `subprocess` /
`open`), and its *simulated* file system legitimately uses `~/`- and `/home/`-rooted paths as
environment content (e.g. *"export … into `~/backups/spotify_library.csv`"*). The shared hygiene
detector's generic path markers (`~/`, `/home/`, …) would drop those legitimate file-system
trajectories as false host-escapes. So capture passes `partition_contained(...,
generic_path_markers=False)`: the generic path markers are skipped **for observations only**, while
(a) all command-level checks and (b) the runtime IDENTITY markers (this machine's real username and
home path) stay active unconditionally. That still catches a genuine leak — `os.path.expanduser("~")`
does echo the real home through AppWorld's sandbox — so a trajectory that touches the real account is
dropped whole (this happens occasionally; the corpus above is the retained, identity-clean set). The
emitted corpus therefore contains simulated `~/` paths by design; a strict `scan_spans_jsonl` (which
uses the generic markers) flags exactly those, but the identity audit is empty.

## License — read before redistributing

**AppWorld dual-licenses, and its data has an encrypted-redistribution restriction that this corpus
respects by staying uncommitted.**

- The **public** portion (agent baselines, evaluation utilities) is plain **Apache 2.0**.
- The **protected** portion — API documentation and implementation, task data, solutions, and
  evaluation tests — ships in encrypted `.bundle` files under **Apache 2.0 with an additional
  requirement that any public redistribution of it (or of its derivatives) be done in an encrypted
  format**. AppWorld additionally requests that extracted/derived data not be posted online in plain
  text. There is an explicit exemption: *"Training language models and serving their outputs do not
  constitute redistribution."*

A captured trace embeds task instructions and app-API observations, which are **derivatives** of that
protected data. So, unlike `bird-sql` (CC BY-SA 4.0) or `financebench`, **`traces.otel.jsonl` is NOT
committed** — it is gitignored along with `data/` and `experiments/`, and is reproduced locally with
`capture.py`. Building and serving a world model from a locally-captured corpus falls under the
training exemption; committing the plaintext corpus to this public repo would not, so we don't.

- **Upstream**: AppWorld (Stony Brook NLP / AI2) — <https://github.com/StonyBrookNLP/appworld>;
  paper *"AppWorld: A Controllable World of Apps and People for Benchmarking Interactive Coding
  Agents"* (ACL 2024), <https://arxiv.org/abs/2407.18901>. Package `appworld` (PyPI), Apache 2.0.
- **Adapter/agent/backend code here is fresh**, written against AppWorld's public API
  (`load_task_ids`, `AppWorld`, `world.execute`, `world.evaluate` / `evaluate_task`).
