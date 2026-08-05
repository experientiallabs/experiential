# `wmo.optimize.model`

Distill a smaller student from a larger teacher on real agent tasks, training a Tinker LoRA and
gating whether the adapter is promoted. Three modes, in descending order of maturity.

Full reference: [`docs/reference/distill.md`](../../../docs/reference/distill.md).

## 1. On-policy

The student samples episodes, the teacher scores the student's own sampled tokens with
`compute_logprobs`, and the per-token gap drives the update. Shipped and measured.

```bash
wmo optimize distill run --config run.toml --run-dir runs/d1 \
  --task-ids train.json --holdout-task-ids holdout.json --backend e2b --yes
```

Objectives are config-selected: `importance_sampling` (equivalently `ppo`) trains the raw
teacher-minus-student gap as a per-token advantage on realized tokens; `topk_ce` is weighted
cross-entropy over the teacher's top-k candidates, which also supervises tokens the student never
sampled, at k times the training-token bill.

Run so far: Qwen3.5-9B student distilled from a Qwen3.6-27B teacher on 17 held-out
TerminalBench-2 tasks, 51 trials per arm. Solve rate went 21.6% to 27.5% against the teacher's
49.0%, and turns per episode went 53.6 to 29.4 against the teacher's 28.5.

## 2. Off-policy

Train on the teacher's **own** trajectories rather than the student's. Different data source, same
datum pipeline: `build_datums` is provenance-agnostic, so teacher rollouts feed it unchanged and the
objective is plain cross-entropy over the teacher's tokens.

On main this exists as the `[warmup]` phase: a one-shot, full-batch pass over a teacher-trajectory
corpus, collected by `_collect_warmup_trials` or reloaded across runs via
`warmup.trajectories_from`.

A first-class `[offpolicy]` section with epochs, minibatching and a resumable datum cursor is in
progress; it collapses `[warmup]` onto the same executor.

Teacher must share the student's tokenizer. `tokenizer_fingerprint_check` enforces this.

## 3. Cross-tokenizer, from a served teacher (WIP)

Teacher and student on different tokenizers, for example GLM-5.2 scoring a Qwen student over an
OpenAI-compatible endpoint. Alignment is by byte boundary: the student's tokens are grouped into
chunks whose byte spans the teacher's tokenization also respects, and advantages are computed per
chunk rather than per token.

**Not runnable on this build.** `wmo/optimize/model/xtoken/` is present but inert (no importers),
and the runtime rejects `teacher.backend = "openai_compat"` even though the config schema
validates it.

## Configs

A run is defined by one TOML passed as `--config`. There is no generator and no default file: you
copy the closest reference config and edit it. They ship in the package, in
[`configs/`](configs):

| file | what it is |
| --- | --- |
| `distill-smoke-dev.toml` | cheapest same-family pair, `backend = "local"`, 3 steps. Start here. |
| `distill-qwen-anchor.toml` | the run in section 1: Qwen3.5-9B from a Qwen3.6-27B teacher |
| `distill-headline.toml` | a full-size Nemotron run |
| `distill-tau2-smoke.toml` | the `[tau2]` rollout source (real tau2-bench episodes), warmup-only |

Each pairs with a train/holdout split passed separately as `--task-ids` and `--holdout-task-ids`:
plain JSON arrays of benchmark task names. The TerminalBench-2 split those configs were run against
is 72 train and 17 holdout, the 89-task set cut in two. Rollouts and interim evals run on train; the
baselines and the promotion gate are measured on holdout, so the two must stay disjoint. They are
separate files rather than config keys so one config can be run against different splits.

Required with no defaults: `[student]`, `[teacher]`, and exactly ONE rollout source, `[harbor]`
(terminus-2 on harbor tasks) or `[tau2]` (real tau2-bench episodes through the loopback proxy;
see `docs/reference/distill.md` for how token exactness works there). Everything
else defaults: `[rollout]`, `[train]`, `[sampling]`, `[warmup]`, `[eval]`, `[gate]`, `[pricing]`,
`[budget]`, `[tripwire]`, `[wandb]`. The schema is `DistillConfig` in
[`config.py`](config.py), and it is `extra="forbid"`, so a typo in a key is an error at load
rather than a silently ignored setting.

Two that are easy to miss. `[harbor] job_template` points at a YAML defining the task environment,
and `[rollout.renderers]` maps each model name to a renderer: the stock reasoning renderers hand
harbor's parser a list and kill the trial, so reasoning models need the `wmo/*_verbatim` wrappers in
[`renderers.py`](renderers.py). Both checked-in smoke configs show the shape.

`--config` is required to start a run and optional to resume: the run store snapshots the exact
config to `<run-dir>/config.toml`, and a bare `--resume` reloads that. Passing `--config` on a
resume is how you raise `budget.max_usd` on a run that hit its cap.

## Before you spend

Run the sub-cent gsm8k probe (an operator scratch script, kept outside the repo). It
exercises sample, score, `forward_backward` and `optim_step` end to end against Tinker for a
fraction of a cent, and catches the failures that otherwise surface forty minutes into a paid run.

Set `TINKER_API_KEY`, `E2B_API_KEY` and `WANDB_API_KEY`. Runs charge real money and abort on the cap
in `[budget]`. Teacher scoring bills prefill at the cached rate where Tinker reports a cache hit, so
`spend.json` is a bill rather than a ceiling.
