# Cycle 1 result note: warmup-only distillation on real tau2 (REJECTED, no effect)

**Capture cohort**: own tau2 harness (wmo `[tau2]` rollout source, PR #297 as merged 7e163298),
post tool-argument-realignment fix, canonical real-tau2 protocol pins (max_turns=100,
episode_timeout_s=1800, max_tokens=8192, user simulator azure/gpt-5.4-mini, tau2
`--max-retries 0` plus one episode-level retry). Never pool these rows with pre-fix or
different-pin cohorts.

**Verdict**: gate REJECTED, adapter NOT promoted, no pool entry created. Total spend $34.94 of
the approved $120 cap.

## The three measured arms (one pre-registered gate read, k=3, 20 pinned holdout tasks)

| arm | model | solve rate | executed | infra failures |
|---|---|---|---|---|
| teacher | Qwen/Qwen3.6-27B | 73.3% | 60/60 | 0 |
| student-before | Qwen/Qwen3.5-9B (base) | 71.7% | 60/60 | 0 |
| student-after | the warmed LoRA (r32) | 65.0% | 60/60 | 0 |

Scoring is tau2's own reward. Note honestly that 7 of the 20 holdout tasks include tau2's
NL-assertion judge in their reward basis, so this is "tau2 reward", not a purely deterministic
check. Per-task per-arm episode rows: `episode-rows.jsonl` in the run dir (180 rows).

## What happened: no measurable effect, not degeneration

The regression is 4 episodes out of 60. A paired sign test over the 7 tasks that moved at all
(5 down, 2 up) gives a two-sided p of 0.45, so before-vs-after is indistinguishable from noise;
13 of 20 tasks did not move at all.

Degeneration is ruled out by the behavioral metrics, which are flat:

| | student-before | student-after |
|---|---|---|
| episodes ending on a clean protocol stop | 60/60 | 60/60 |
| mean messages per episode | 29.5 | 30.4 |
| mean episode duration | 94s | 93s |

No collapse, no length runaway, no scaffold loss, no truncation. The warmed student behaves like
the base student and scores within noise of it.

## Why there was nothing to measure

1. **The teacher had almost no headroom over the student: 73.3% vs 71.7%, a 1.6-point gap.**
   Warmup-only distillation copies the teacher's trajectories; when the teacher is effectively a
   peer, there is nothing to copy that the student does not already do.
2. **The holdout cannot resolve a small effect.** 9 of 20 tasks sit at the student's own ceiling
   (pinned before this cycle, in the power check), so the measurable surface is the 10 interior
   tasks and 60 episodes total. A ~2-point true effect is far below what this pin can detect.
3. **Data yield was NOT the problem.** The teacher's 194 episodes produced 133 kept (68.6% pass
   rate on the train split) and 133 datums with zero overflow or overlong drops, 918,869 loss
   tokens per pass over 1,066,513 context tokens. The supply side worked exactly as designed.

## Follow-ups this cycle earned

- **Warmup has no degeneration-tripwire coverage.** Tripwires are computed per training step, and
  a warmup-only run takes no training steps, so entropy and length guards never ran. Here the
  behavioral metrics happened to be flat, but a warmup-only run currently trains with no
  automated collapse detector. Owner: the distill lane (#268 territory).
- **The learning rate was the OPD default (1e-4) applied to ~1.8M loss tokens across two full
  passes.** No damage is visible in this run's behavior, but the anchor config uses 5e-5 for
  warmup specifically, and a warmup-only cycle should probably not inherit the on-policy default.
- **The escalation premise now has evidence behind it.** The failure mode is "teacher too close to
  the student", which is exactly what a stronger teacher fixes. Before spending on a K3 leg, the
  cheap decisive measurement is K3's own solve rate on TRAIN tasks (never the holdout, which would
  select on gate data): roughly 10 episodes, a few dollars.

## Artifacts

`.wmo/distill-runs/tau2-cycle1/`: config snapshot, `metrics.jsonl`, `spend.json` ($34.94),
`checkpoints.json`, `gate.json`, `model_card.json`, `warmup-trials.json` (the 194-episode teacher
manifest, reusable by any future run via `warmup.trajectories_from`), `evals/` (three arm
reports), `episode-rows.jsonl` (per-task per-arm rows), plus the raw episode dirs with every
transcript. W&B run `tau2-cycle1-warmup` in project `wmo-distill`.

The trained weights and resumable state exist and are recorded in the model card, but the adapter
was NOT versioned into `.wmo/adapters` and NOT priced into the routing pool, because the gate
rejected it. An unpromoted student must never enter the pool.
