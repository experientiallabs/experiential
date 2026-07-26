# Distill mode (`wmh optimize harness <agent> harbor --mode distill`)

The other optimizer modes edit the agent's *harness*; distill mode trains the agent's *model*.
It runs on-policy distillation of a Tinker LoRA student: the pi agent, with its harness held
fixed, rolls out on real harbor benchmark tasks while sampling from the student's current
weights; a larger teacher model scores the exact tokens the student sampled; and each training
step nudges the student toward the teacher with a per-token reverse-KL objective (Tinker's
`importance_sampling` loss). A holdout gate at the end compares teacher, student-before, and
student-after solve rates, and only an adapter that closes enough of the gap to the teacher is
promoted. The result is a small model that behaves like the big one inside your agent, plus a
ready-to-paste serving snippet.

## Prerequisites

- **The distill extra**: `uv sync --extra distill` (installs the `tinker` SDK and `wandb`).
- **`TINKER_API_KEY`** in the environment: the student trains and samples on Tinker, and the
  teacher scores there too.
- **`E2B_API_KEY`** when the run config sets `harbor.backend = "e2b"` (rollout trials in E2B
  sandboxes); `backend = "local"` runs them on your machine instead.
- **Free E2B sandbox capacity** for `backend = "e2b"`: a running trial holds two concurrent
  sandboxes (harbor's task environment plus the sandbox hosting the pi harness process), so a
  run needs `2 x train.trial_concurrency` free slots against your account's concurrent-sandbox
  limit (100 by default; set `WMH_E2B_SANDBOX_CAP` when yours differs). See
  [Sandbox capacity](#sandbox-capacity).
- **A harbor job template**: the Harbor `JobConfig` YAML/JSON naming the benchmark dataset the
  trials run against, pointed at by the config's `[harbor] job_template`.
- **Task-id splits**: two JSON files, each a plain array of task-id strings. The train split
  feeds rollouts and interim evals; the holdout split (disjoint, enforced) is reserved for the
  baselines and the promotion gate.

## The run config

One TOML file describes one run. `[student]`, `[teacher]`, and `[harbor]` are required; every
other section has complete defaults. A minimal, realistic config:

```toml
[student]
base_model = "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16"  # the Tinker LoRA student
lora_rank = 32

[teacher]
model = "nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-BF16"    # scores student tokens

[harbor]
job_template = "tb2-job-template.yaml"  # Harbor JobConfig for the benchmark
backend = "local"                       # or "e2b" (needs E2B_API_KEY)

[rollout]
max_turns = 20                 # per-episode turn cap, pinned into the harness
context_budget_tokens = 65536  # episodes that outgrow this are dropped whole

[train]
steps = 40           # optimizer steps
tasks_per_batch = 8  # tasks sampled per step
group_size = 4       # attempts per task (the on-policy group)
# loss = "topk_ce"   # optional: weighted CE over the teacher's top-k candidate
# topk = 8           # tokens per position instead of reverse-KL on realized
#                    # tokens; trains ~topk x the token volume per step

[sampling]
temperature = 1.0    # keep 1.0: issued logprobs stay comparable to the teacher
max_tokens = 8192

[warmup]
steps = 2              # optional SFT bootstrap on the teacher's own passing
rollouts_per_task = 2  # trajectories before OPD; steps = 0 disables it

[eval]
every = 0  # interim evals off (the default); N > 0 evals a train subsample every N steps
# Baseline reuse: point these at a prior run's eval reports to skip re-running
# the holdout baselines. Validated before use: identical holdout task ids,
# attempts >= this run's gate.k, and the same teacher (teacher baseline) or
# student base model (student baseline, via the report's recorded base_model).
# teacher_baseline_from = "runs/prior/evals/baseline-teacher.json"
# student_baseline_from = "runs/prior/evals/baseline-student-before.json"

[gate]
k = 3                        # holdout attempts per task for each baseline
min_teacher_fraction = 0.7   # student-after must reach 70% of teacher solve rate
require_no_regression = true # and must not fall below student-before

[pricing]  # USD per million tokens, from the provider's price list
teacher_prefill = 2.49
teacher_sample = 6.225
student_prefill = 0.30
student_sample = 0.70
student_train = 0.60
# cached prefill rates default to 20% of the full prefill price

[budget]
max_usd = 600.0  # hard cap; the run checkpoints and aborts resumably at it

[wandb]
enabled = true   # optional live tracking (needs WANDB_API_KEY or wandb login)
project = "wmh-distill"
```

Billing follows the provider's per-request model: every agent turn re-bills its whole prompt,
with the verbatim repeated prefix at the cached rate, so the projected volumes are much larger
than the unique context size. The CLI prints the per-meter projection and asks for
confirmation before anything is spent; unpriced meters print as `unknown`, and a run with
unpriced meters and no `budget.max_usd` refuses to start non-interactively.

## Running it

```bash
wmh optimize harness pi harbor --mode distill \
  --distill-config run.toml \
  --task-ids train-task-ids.json \
  --holdout-task-ids holdout-task-ids.json \
  --run-dir runs/distill-01
```

The agent argument works like the other optimizer modes: the literal `pi` is the built-in
default agent, and `name@ref` seeds from a stored harness version (the harness must be a
pi-node harness; it is pinned for the whole run). `--backend local|e2b` overrides the config's
`harbor.backend`. `--yes` skips the cost confirmation when the spend is accountable. Add
`--promote` to be offered a `[models.agent]` settings write after an accepted gate.

Before spending anything the run preflights: renderer resolution, a student/teacher tokenizer
fingerprint check, one-token pings, and a tokens-in-tokens-out (TITO) recompute proof that the
sampling and scoring paths agree on the student's own tokens. On `backend = "e2b"` it also
checks sandbox capacity first (below).

## Sandbox capacity

E2B caps concurrent sandboxes per account, and a harbor trial's task-environment sandbox lives
for its own multi-hour timeout. A run that dies without graceful shutdown (crash, SIGKILL,
budget abort, machine sleep) therefore leaves its sandboxes running, and the next run starves at
the cap: every trial fails at sandbox creation with
`RateLimitException: 429 ... maximum number of concurrent E2B sandboxes`, which surfaces as
trials producing zero token spans and looks exactly like a broken model.

Two mechanisms keep that from happening silently.

- Every sandbox a wmh run creates is recorded, with its owning process id, in a per-process
  JSONL ledger under the WMH user state directory (`$WMH_HOME/e2b-sandboxes`, else
  `~/.wmh/e2b-sandboxes`), and marked released when its kill is proved. A ledger entry whose
  owning process is gone is a provable orphan that can be killed by exact id.
- An e2b-backed distill run preflights capacity: it counts running sandboxes, auto-reclaims
  those provable orphans, and refuses to start when `2 x train.trial_concurrency` slots are
  still not free, naming the numbers instead of starving.

To inspect or reclaim capacity by hand:

```bash
wmh e2b reap                      # dry run: what is running, and what would be killed
wmh e2b reap --yes                # kill orphans of dead local runs (exact recorded ids)
wmh e2b reap --stale-minutes 60   # ALSO match harbor trial sandboxes account-wide by age
```

`--stale-minutes` matches on the account, not just this machine, so it can kill a run on another
machine or in another checkout; sandboxes whose local owner process is still alive are never
selected. Both forms are dry runs until `--yes`.

## What a run produces

Everything durable lands under `--run-dir`:

```text
<run-dir>/
  config.toml         # exact snapshot of the run config
  distill-run.json    # pinned CLI inputs (splits, backend, seed harness hash)
  metrics.jsonl       # one row per warmup/training step: solve rate, reverse
                      # KL/token, datum and drop counts, per-meter tokens, USD
  spend.json          # cumulative priced USD, updated on every charge
  checkpoints.json    # saved tinker:// training-state + sampler paths
  evals/<name>.json   # baseline, interim, and student-after eval reports
  gate.json           # the promotion verdict
  model_card.json     # base model, teacher, artifact paths, gate record
  handoff.toml        # the [models.agent] serving snippet
  harbor/  tokens/    # per-step rollout artifacts and token sinks
  eval-rollouts/  warmup-rollouts/  # isolated rollout roots for eval/warmup batches
```

An accepted gate additionally saves the adapter as an immutable version under the project's
`.wmh/adapters/<agent>/vN/` with a movable `champion` alias, and prints the serving handoff:
a `[models.agent]` TOML snippet pointing at the final `tinker://...` sampler path through
Tinker's OpenAI-compatible endpoint (authenticate by setting `WMH_ENDPOINT_API_KEY` to your
Tinker API key). With `[wandb] enabled = true`, steps, evals, spend, and the gate summary
stream to a Weights & Biases run that resumes with the run dir.

## Resume and budget behavior

Training state checkpoints on a cadence (`train.save_state_every`) plus at every abort. If the
run hits `budget.max_usd`, it saves what it can and exits with the exact resume command; raise
the cap in the config and rerun with:

```bash
wmh optimize harness pi harbor --mode distill --run-dir runs/distill-01 --resume
```

A resume needs only `--run-dir`: the CLI reloads the pinned splits, backend, and seed harness
from `distill-run.json`, the config from the `config.toml` snapshot (an explicit
`--distill-config` wins, which is how you raise the cap), and prior spend from `spend.json`,
so a resumed run can never spend the budget twice. Recorded baselines and a finished warmup
are reused, the step count continues from the latest checkpoint, and conflicting explicit
flags are rejected rather than silently changing the run.

## Troubleshooting

| Symptom | Meaning and fix |
|---|---|
| TITO preflight failure (`TITO recompute disagreement ...`) | The sampling and scoring paths disagree on the student's own tokens, so training data would be corrupt; check that the sampler path matches the student base model and that the pinned `tinker` SDK version is unchanged. |
| Empty-batch abort (`... every trial produced zero token spans`) | Consecutive steps sampled no completions at all. Either the student provider or its sessions are failing upstream (check the runner logs for worker completion warnings), or, on `backend = "e2b"`, trials are dying at sandbox creation because the account is at its concurrent-sandbox cap (`wmh e2b reap`). Fix the cause, then `--resume`. |
| Every trial 429s at sandbox creation (`maximum number of concurrent E2B sandboxes`) | The account is at its concurrent-sandbox cap, usually because a crashed run's sandboxes are still running out their multi-hour timeout. Run `wmh e2b reap` to see what is holding the slots, then `wmh e2b reap --yes` (orphans of dead local runs) or `wmh e2b reap --stale-minutes N --yes` (account-wide by age). See [Sandbox capacity](#sandbox-capacity). |
| Start refused (`not enough free E2B sandbox slots ...`) | The capacity preflight found fewer free slots than `2 x train.trial_concurrency` even after reclaiming provable orphans; free slots as above, lower `train.trial_concurrency`, or raise the cap (`WMH_E2B_SANDBOX_CAP`). |
| Deadline expiries (`TinkerDeadlineError: tinker <call> timed out after ...`) | A wedged Tinker session was cut off instead of hanging; transient ones retry with a fresh session on their own, and a persistent one can be given more headroom via the `WMH_TINKER_DEADLINE_<KIND>` env vars the error names. |
| Resume rejected (`LoadWeights can only be called on uninitialized models`) | Tinker accepts a checkpoint restore only on a model nothing has touched yet, so a resume loads its state as the very first call on a freshly created training client. If a restore is slow enough to blow its deadline, the run retries it on another fresh client; `WMH_TINKER_DEADLINE_LOAD_STATE` (600s by default) is how long one attempt may take, which large students may need raised. |
| Fragmentation warning (`N of M datum(s) are fragments ...`) | The agent edited its prompt history mid-episode, so shared context re-prefills at full price and teacher scoring multiplies; keep `rollout.compaction = false` and check the harness keeps its prompt prefix stable across turns. |
