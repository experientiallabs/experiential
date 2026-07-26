# `wmo.distill`

Distill a smaller student from a larger teacher on real agent tasks, training a Tinker LoRA and
gating whether the adapter is promoted. Three modes, in descending order of maturity.

Full reference: [`docs/reference/distill.md`](../../docs/reference/distill.md).

## 1. On-policy

The student samples episodes, the teacher scores the student's own sampled tokens with
`compute_logprobs`, and the per-token gap drives the update. Shipped and measured.

```bash
wmo optimize model run --config run.toml --run-dir runs/d1 \
  --task-ids train.json --holdout-task-ids holdout.json --backend e2b --yes
```

Objectives are config-selected: `importance_sampling` (equivalently `ppo`) trains the raw
teacher-minus-student gap as a per-token advantage on realized tokens; `topk_ce` is weighted
cross-entropy over the teacher's top-k candidates, which also supervises tokens the student never
sampled, at k times the training-token bill.

### Measured: Qwen3.5-9B student, Qwen3.6-27B teacher, TerminalBench-2

17 held-out tasks, 3 attempts each, 51 trials per arm. No trial is missing from any denominator.

| arm | solve rate | turns / episode |
| --- | --- | --- |
| teacher (27B) | 49.0% (25/51) | 28.5 |
| student before | 21.6% (11/51) | 53.6 |
| student after | 27.5% (14/51) | 29.4 |

## 2. Off-policy

Train on the teacher's **own** trajectories rather than the student's. Different data source, same
datum pipeline: `build_datums` is provenance-agnostic, so teacher rollouts feed it unchanged and the
objective is plain cross-entropy over the teacher's tokens.

On main this exists as the `[warmup]` phase: a one-shot, full-batch pass over a teacher-trajectory
corpus, collected by `_collect_warmup_trials` or reloaded across runs via `warmup.trajectories_from`.

PR #268 promotes it to a first-class `[offpolicy]` section with epochs, minibatching and a resumable
datum cursor, and collapses `[warmup]` onto the same executor.

Teacher must share the student's tokenizer. `tokenizer_fingerprint_check` enforces this.

## 3. Cross-tokenizer, from a served teacher (WIP)

Teacher and student on different tokenizers, for example GLM-5.2 scoring a Qwen student over an
OpenAI-compatible endpoint. Alignment is by byte boundary: the student's tokens are grouped into
chunks whose byte spans the teacher's tokenization also respects, and advantages are computed per
chunk rather than per token.

**Not runnable on this build.** `wmo/distill/xtoken/` is present but inert (no importers), and the
runtime rejects `teacher.backend = "openai_compat"` even though the config schema validates it.
PR #258 activates the path.

## What a run produces

Everything durable lands under `--run-dir`: `config.toml`, `metrics.jsonl` (one row per step),
`spend.json`, `checkpoints.json`, `evals/`, `gate.json`, `model_card.json`, and `handoff.toml` with
the serving snippet. Resume with `--run-dir <same> --resume`; splits, backend and prior spend all
reload from the run dir.

Read `scaffold_loss_rate` before any solve rate. It is the share of episodes that never reached an
explicit `submit`, so those episodes measure where the harness cut them off rather than what the
model can do. `graded_solve_rate` sits beside every solve rate and reads the same trials at test
resolution, so partial progress inside a task is visible even though the gate stays binary.

## Before you spend

Run the sub-cent gsm8k probe (`.agents/distill/tinker_probe_gsm8k.py`). It exercises sample, score,
`forward_backward` and `optim_step` end to end against Tinker for a fraction of a cent, and catches
the failures that otherwise surface forty minutes into a paid run.

Set `TINKER_API_KEY`, `E2B_API_KEY` and `WANDB_API_KEY`. Runs charge real money and abort on the cap
in `[budget]`. Teacher scoring bills prefill at the cached rate where Tinker reports a cache hit, so
`spend.json` is a bill rather than a ceiling.
