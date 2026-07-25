# Distill mode (`wmh optimize <agent> harbor --mode distill`)

The other optimizer modes edit the agent's *harness*; distill mode trains the agent's *model*.
It runs on-policy distillation of a Tinker LoRA student: the pi agent, with its harness held
fixed, rolls out on real harbor benchmark tasks while sampling from the student's current
weights; a larger teacher model scores the exact tokens the student sampled (or, when it does not
share the student's vocabulary, scores its own tokenization of the same conversation and is
matched to the student's tokens by content, see Cross-tokenizer teachers below); and each training
step nudges the student toward the teacher with a per-token reverse-KL objective (Tinker's
`importance_sampling` loss). A holdout gate at the end compares teacher, student-before, and
student-after solve rates, and only an adapter that closes enough of the gap to the teacher is
promoted. The result is a small model that behaves like the big one inside your agent, plus a
ready-to-paste serving snippet.

## Prerequisites

- **The distill extra**: `uv sync --extra distill` (installs the `tinker` SDK and `wandb`).
- **`TINKER_API_KEY`** in the environment: the student trains and samples on Tinker, and a
  same-vocabulary teacher scores there too.
- **`E2B_API_KEY`** when the run config sets `harbor.backend = "e2b"` (rollout trials in E2B
  sandboxes); `backend = "local"` runs them on your machine instead.
- **`WMH_ENDPOINT_API_KEY`** only when the teacher is reached over an OpenAI-compatible endpoint
  instead of Tinker (`[teacher] backend = "openai_compat"`, the cross-tokenizer path below): it
  carries that endpoint's bearer token, and is the only key ever sent to a custom host.
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

## Cross-tokenizer teachers (`teacher.alignment = "chunk"`)

Everything above assumes teacher and student share one vocabulary, which is what lets the teacher
score the student's exact sampled ids position for position. A teacher from a different model
family cannot do that at all: a GLM-5.2 teacher's vocabulary holds 154,856 tokens and a
Qwen3.5-9B student's holds 248,077, and the same integer spells different text on each side, so
handing the student's ids to such a teacher is not a weaker signal, it is a different sentence
scored with no error raised. Chunk alignment is the mode for that case. The objective, the
training wire format, and the gate are unchanged; only the way the teacher's numbers are matched
to the student's tokens is different.

### How it works

The teacher scores its OWN tokenization of the same conversation, and the two are compared span by
span:

1. **Teacher render.** The canonical message list, the same messages the student saw, is rendered
   with the TEACHER's chat template and tokenized with the teacher's tokenizer. The render reports
   the byte-identical content islands inside itself: assistant reasoning, visible text, and
   tool-call argument values.
2. **Byte offsets.** Exact per-token byte offsets are reconstructed on both sides from the
   vocabularies alone, which puts two foreign token streams into one shared byte space. Nothing is
   re-encoded: sampling routinely emits a non-canonical BPE segmentation of its own text (`'S'` +
   `'olver'` where canonical BPE gives `'Solver'`), so gating on `encode(decode(ids)) == ids`
   would discard perfectly good tokens.
3. **Chunks.** Inside an island the two tokenizations are two partitions of one byte interval, so
   a chunk boundary is cut wherever a student token boundary coincides with a teacher token
   boundary. On real GLM-5.2 and Qwen3.5 traffic that lands at 1.05 to 1.08 student tokens per
   chunk, so the comparison is nearly per-token even though it is not per-token by construction.
4. **Advantages.** A chunk's advantage is `(teacher_logprob_sum - student_logprob_sum)` divided by
   the STUDENT span length, broadcast to that chunk's student tokens. The loss sums
   `advantage * grad log pi` over positions, so the chunk contributes exactly its reverse-KL gap
   however many student tokens it spans. Dividing by the teacher span length instead would scale
   every chunk by the two tokenizers' verbosity ratio.
5. **Training.** The result is the per-token advantage array the existing `importance_sampling`
   loss already carries, so nothing new goes onto the training wire. A student token no chunk
   covers keeps advantage exactly 0.0, which IS the mask, so advantage centering is over chunk
   TOTALS rather than over tokens: subtracting a constant per token shifts a chunk by that
   constant times its length, which can invert a long chunk's sign against a short one's.

The teacher side of the config is what selects all of this:

```toml
[teacher]
backend = "openai_compat"   # not Tinker: this teacher cannot read student ids
model = "accounts/fireworks/models/glm-5p2"   # the served model id
endpoint = "https://api.fireworks.ai/inference/v1"  # OpenAI-compatible base URL
tokenizer = "zai-org/GLM-5.2"                 # HF repo id, for render + tokenize
alignment = "chunk"                           # byte-identical content, span by span

[train]
loss = "importance_sampling"  # topk_ce is rejected under chunk alignment
```

`backend`, `endpoint`, `tokenizer`, and `alignment` travel together and are validated together,
because a half-configured cross-tokenizer teacher is the dangerous case: an `endpoint` left at the
default `alignment = "same_tokenizer"` would ship the student's ids to a foreign vocabulary and
score noise silently, so that pair is refused. `teacher.checkpoint` is a `tinker://` path and is
refused for this backend too. `train.loss = "topk_ce"` is refused as well: topk_ce trains a
weighted cross-entropy whose targets are the teacher's top-k candidate token IDS, and those ids
index a different embedding table on the student side, so the run would train confidently on text
nobody wrote. Chunk alignment needs only a scalar logprob sum per aligned chunk, which is exactly
what `importance_sampling` consumes.

### What is and is not scored

| Part of the trajectory | Scored | Why |
|---|---|---|
| Assistant reasoning, visible assistant text, tool-call argument VALUES | yes | The bytes are identical under both chat templates, so a chunk can pair them. |
| Template framing, turn markers, tool-call syntax, whitespace the template strips | no | There is no counterpart to align against, so those positions keep advantage 0.0. |

The split is a property of the templates, not a tuning knob. Qwen writes `<|im_start|>assistant`
and a tool call as `<function=bash><parameter=command>`; GLM writes `<|assistant|>` and
`<tool_call>bash<arg_key>command</arg_key><arg_value>`. Measured on real run spans, that framing
is 4.51% of sampled tokens and is unscoreable by construction, while the content islands covered
99.72% of content tokens over 186,706 real tokens. GLM-5.2's template also drops
`reasoning_content` from historical turns by default, which would both hide most think tokens and
condition the teacher on a history the student never saw; the render keeps reasoning on every
assistant turn instead.

### Operational warnings

- **A truncated rollout is never trained on.** A cut trajectory's last tokens are an artifact of
  the cap rather than of the policy, and the teacher would score them as if they were a choice.
  Tinker's context ceiling is 65,536 tokens for every model in the catalog, so raising the budget
  out of the problem is not available. Keep `rollout.context_budget_tokens` (with
  `sampling.max_tokens` fitting inside it) at a budget the student can actually finish in, 32,768
  for the GLM-5.2 to Qwen3.5-9B run, and let episodes that outgrow it be dropped whole from
  training: filtering out the problems the student cannot finish, up to about a third of them, is
  the correct trade against training on cut tokens. Report the filtered fraction next to any
  accuracy number the run produces.
- **Hosted teachers are scarce.** Scoring requires a provider that will score TOKENS THE CALLER
  SUPPLIES, byte for byte. Fireworks is currently the only hosted provider that does; the rest
  either re-tokenize a text prompt server-side (which prepends a BOS and shifts every position by
  one against the local offsets, with no error) or expose no prompt positions at all. Anything
  else means hosting the teacher yourself, where the vLLM-native `prompt_logprobs` form is
  available. Prompts always go on the wire as token ids, never as text, and the echoed ids are
  compared against the ids sent before a single logprob is used.
- **The `logprobs` parameter must be an INTEGER.** Sending `true` selects the newer response
  shape, a content array with no prompt positions in it, and the server returns 200 either way.
- **Position 0 comes back as `0.0`, not `null`.** A logprob of 0.0 reads as probability 1, so a
  chunk that starts at position 0 would be inflated toward "the teacher was certain". Position 0
  has no context and is discarded before anything is summed.
- **26 to 38% of echo responses are silently all-zero.** Fireworks' serverless echo has a measured
  placeholder mode: HTTP 200 with every `token_logprobs` entry exactly 0.0 (one 12k-token response
  came back partly real and partly placeholder). Nothing in the body marks it, and training on it
  is not a neutral step: with a zero teacher sum every chunk's advantage collapses to
  `-student_logprob`, which is not a KL signal. Rows where nearly every scored position is exactly
  0.0 are rejected and retried on a FRESH connection, since the fault is per replica. Do not pin
  session affinity to work around it: that locked onto the bad replica 16 times out of 16.

## Running it

```bash
wmh optimize pi harbor --mode distill \
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
sampling and scoring paths agree on the student's own tokens.

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
wmh optimize pi harbor --mode distill --run-dir runs/distill-01 --resume
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
| Empty-batch abort (`... every trial produced zero token spans`) | Consecutive steps sampled no completions at all: the student provider or its sessions are failing upstream; check the runner logs for worker completion warnings, then `--resume` once the provider is healthy. |
| Deadline expiries (`TinkerDeadlineError: tinker <call> timed out after ...`) | A wedged Tinker session was cut off instead of hanging; transient ones retry with a fresh session on their own, and a persistent one can be given more headroom via the `WMH_TINKER_DEADLINE_<KIND>` env vars the error names. |
| Fragmentation warning (`N of M datum(s) are fragments ...`) | The agent edited its prompt history mid-episode, so shared context re-prefills at full price and teacher scoring multiplies; keep `rollout.compaction = false` and check the harness keeps its prompt prefix stable across turns. |
| Config rejection (`teacher.backend = 'openai_compat' requires teacher.alignment = 'chunk'`) | A foreign-vocabulary teacher was configured without cross-tokenizer alignment, which would score the student's ids against the wrong vocabulary and never error; set `alignment = "chunk"`, or drop `endpoint` and use the Tinker backend if the teacher really does share the student's vocabulary. |
| Placeholder scoring row (`... returned N of M scored positions as exactly 0.0`) | The teacher endpoint served the known all-zero echo response instead of logprobs; it is retried on a fresh connection automatically, and if it persists the run stops rather than train on zeros. Do not pin session affinity, which sticks to the bad replica. |
| Echo id mismatch (`teacher scoring echoed ... token id(s) that do not match the N sent`) | The server re-tokenized the prompt instead of scoring the exact ids, so every chunk offset would be shifted; confirm the endpoint serves the model named by `teacher.model` and that `teacher.tokenizer` is the tokenizer that model was built with. |
