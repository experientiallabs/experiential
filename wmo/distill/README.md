# `wmh.distill`

On-policy distillation of a Tinker LoRA student from agent rollouts on harbor tasks. The student
samples real episodes, a larger teacher scores the student's own sampled tokens, and the per-token
gap drives the update. A gate decides whether the resulting adapter is promoted.

Full reference: [`docs/reference/distill.md`](../../docs/reference/distill.md). This file is the
short version plus the one measured result.

## Run one

```bash
wmh optimize harness pi harbor --mode distill \
  --distill-config run.toml \
  --task-ids train-task-ids.json \
  --holdout-task-ids holdout-task-ids.json \
  --run-dir runs/distill-01
```

`--run-dir` is required and holds every durable artifact: the config snapshot, `metrics.jsonl`,
`spend.json`, `checkpoints.json`, `evals/`, `gate.json`, `model_card.json` and the `handoff.toml`
serving snippet. Resume with `--run-dir <same> --resume`; nothing else is needed, since the splits,
backend and prior spend are all reloaded from the run dir.

Read `scaffold_loss_rate` before you read any solve rate. It is the share of episodes that never
reached an explicit `submit`. Those episodes measure where the harness cut them off, not what the
model can do.

## Measured result: Qwen3.5-9B student, Qwen3.6-27B teacher, TerminalBench-2

17 held-out tasks, 3 attempts each, 51 trials per arm. No trial is missing from any denominator.

| arm | solve rate | turns / episode |
| --- | --- | --- |
| teacher (27B) | 49.0% (25/51) | 28.5 |
| student before | 21.6% (11/51) | 53.6 |
| student after | 27.5% (14/51) | 29.4 |

**No solve-rate lift is claimed.** The paired per-task delta is **+0.059, 95% CI [+0.000, +0.157]**,
an interval that includes zero. Only 2 of the 17 tasks moved; the other 14 were pinned at floor or
ceiling in both arms. The promotion gate (70% of teacher, i.e. 34.3%) was **not met**.

The finding that does hold is behavioural: the student went from 53.6 turns per episode to 29.4,
against the teacher's 28.5. It learned the teacher's working shape, on a held-out task set, without
that showing up as a solve-rate win at this sample size.

Two properties of this experiment bound how much it can say. With 51 trials per arm and 14 of 17
tasks saturated, the design has little room to detect a small effect. And the gate is binary
because that is TerminalBench-2's own definition of done, so partial progress inside a task is
invisible to it; `graded_solve_rate` records that separately.

### Rerunning it

The run is driven entirely by its config plus the two task-id files, so the recipe is the command
above with the Qwen config. Set `TINKER_API_KEY`, `E2B_API_KEY` and `WANDB_API_KEY` in the
environment first; the run charges real money and aborts on the budget cap in `[budget]`.

Before spending, run the sub-cent gsm8k probe in `.agents/distill/tinker_probe_gsm8k.py`. It
exercises the sample, score, forward_backward and optim_step path end to end against Tinker for a
fraction of a cent, and it is the designated pre-check for exactly the failures that otherwise
surface forty minutes into a paid run.

Costs are metered per call in `spend.json`. Teacher scoring bills prefill at the cached rate where
Tinker reports a cache hit, so the ledger is an actual bill rather than a ceiling.
