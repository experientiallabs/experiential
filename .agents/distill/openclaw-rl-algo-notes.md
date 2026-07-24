# OpenClaw-RL (Gen-Verse) — algorithm notes for the `openclaw-tinker/` path

Source: github.com/Gen-Verse/OpenClaw-RL @ main (read 2026-07-24). All `file:line` refs are
into `openclaw-tinker/` unless prefixed with another top-level dir. Cross-checks against the
Slime-side reference implementations (`openclaw-combine/combine_loss.py`,
`slime/slime/backends/megatron_utils/loss.py`) are marked "[slime]".

Three methods behind one trainer (`run.py --method {rl, opd, combine}`), all trained as
policy-gradient through Tinker: the "distillation" is encoded entirely in the per-token
**advantages** of a `tinker.Datum`, never as a KL/CE loss (except the separate, unused-by-default
top-K variant, §6).

---

## 1. OPD loss (`--method opd`)

**Per-token advantage** (data_formatter.py:106-128, built in `sample_to_datum`):

```
A_i = [ grpo_adv + (teacher_lp_i − student_lp_i) ] * loss_mask_i
```

- `grpo_adv` is the scalar sequence reward broadcast to every token
  (`compute_grpo_advantages`, data_formatter.py:212-217 — literally `[s.reward for s in batch]`).
- In OPD mode every accepted sample carries `reward = 1.0` (api_server.py:822; README.md:118
  "All samples get reward = 1.0"), so the effective OPD per-token advantage is
  **`A_i = 1.0 + (teacher_lp_i − student_lp_i)`** — a constant +1 REINFORCE term rides on top
  of the distillation gap. (The pure Slime estimator has no +1: advantage is exactly
  `teacher_logp − student_logp` [slime] slime/slime/backends/megatron_utils/loss.py:716-730.)
- **No coefficient, no clipping, no centering, no std/length normalization** on
  `(teacher_lp − student_lp)`. Docstring (data_formatter.py:110-113): "advantage =
  teacher_logp − old_logp (raw, no coefficient)". Only hygiene: non-finite values are zeroed
  (`_sanitize`, data_formatter.py:62-71) and teacher lps are truncated/zero-padded to the
  response length (scorers.py:267-271; api_server.py:813-816).
- `student_lp_i` = the **rollout sampling logprobs** of the student's own generation, captured
  from Tinker at sample time (`_raw_response_logprobs`, api_server.py:266-268,295-297) — not
  recomputed. ([slime] recomputes the student side through Megatron: combine_loss.py:8-11,83-85.)

**What the teacher is** (trainer.py:87-95; config.py:28-31,66-68):
- A **fixed base model served on Tinker as a plain `SamplingClient` (no LoRA)**, created once
  at setup. **Not an EMA of the student; there is no EMA anywhere in the repo.** It never
  updates during training.
- Default `teacher_model_name` = **the same base model as the student policy**
  (`resolved_teacher_model()`, config.py:66-68). The distillation signal therefore comes from
  the *privileged hindsight hint in the teacher's context* (§3), not from model capacity.
  A larger teacher is possible via `--teacher-model-name` but is not the default.
- [slime] combine path: teacher defaults to the *PRM model* checkpoint recomputed through
  Megatron (`OPENCLAW_COMBINE_OPD_TEACHER_SOURCE=megatron`, `--prm-teacher-load`,
  combine_loss.py:13-22; openclaw-combine/run_qwen3_4b_openclaw_combine.sh:24,55). Also fixed,
  never EMA-updated.

**Sample gating in OPD mode:** a turn only becomes a training sample if (a) it has a
next-state observation and (b) the hint judge produced an accepted hint; otherwise it is
dropped entirely (api_server.py:753-806, "no valid hint, sample dropped" scorers.py:374).
`loss_mask = [1]*len(response)` on everything that survives (api_server.py:822).

## 2. "Combined" mode (`--method combine`)

**Per-token advantage** (data_formatter.py:149-185, `sample_to_datum_combined`):

```
A_i = w_opd * (teacher_lp_i − student_lp_i) * mask_i  +  w_rl * reward * mask_i
```

- OPD term is **per-token**; RL term is a **per-sequence scalar reward broadcast** to all
  tokens. Defaults `w_opd = w_rl = 1.0` (config.py:43-44; env `OPENCLAW_COMBINE_W_OPD/W_RL`,
  run.py:49-52). [slime] identical: `combined_advantages = w_opd * teacher_advantages +
  w_rl * grpo_advantages`, combine_loss.py:127-130.
- **Gating is a three-way dispatch per turn**, not only-on-failure
  (api_server.py:839-847 docstring, 992-1031 implementation). After every turn, the
  `CombinedScorer` runs *both* the hint judge and the PRM eval concurrently (scorers.py:432-455):
  - hint accepted AND eval ∈ {+1, −1}  → `"opd+rl"` sample: teacher lps + `reward = eval_score`
    (api_server.py:999-1009).
  - hint accepted, eval = 0/unparsable → `"opd"` sample: teacher lps + `reward = 0.0`
    (api_server.py:1010-1020).
  - no hint, eval ∈ {+1, −1}           → `"rl"` sample: `reward = eval_score` and
    **`teacher_logprobs := student's own rollout logprobs`, so the OPD term is exactly 0**
    (api_server.py:1060-1079, comment "teacher = student -> OPD advantage = 0" at :1077).
  - no hint, eval neutral              → dropped ("no signal", api_server.py:1030-1031).
- Rewards from the PRM eval are in {+1, −1, 0} by majority vote over `m` samples; ties or
  all-unparsable → 0.0 (`majority_vote`, scorers.py:143-152). Note hints (and hence the
  distillation term) fire on *successful* turns too — the judge criterion is "does the next
  state reveal useful hindsight info", not "did the turn fail" (scorers.py:102-103).
- `--train-epochs N` (combine typically 2) simply duplicates the collected batch N× before
  the single gradient step (rollout.py:132-142; README.md:17,80).

## 3. Hindsight hints

Pipeline per turn t, once the environment produces the next state at t+1 (the following user
message or tool result — captured as `messages[-1]` of the *next* main-turn request,
api_server.py:711-718 / 914-921):

1. **Judge** (`build_hint_judge_messages`, scorers.py:85-116): teacher model sees
   `(assistant response @ t, next state @ t+1 with its role)` and must output `\boxed{1}` or
   `\boxed{-1}`, plus — iff 1 — a "concise, information-dense hint in 1-3 sentences" wrapped in
   `[HINT_START]…[HINT_END]`. Run `m = prm_m` times (default 3; README quickstart uses
   `--prm-m 1` for opd/combine) at temperature 0.6, max_tokens 4096, top_k 50, top_p 0.95
   (config.py:51-53; scorers.py:200-202).
2. **Hint selection**: among positive votes, take the **longest** hint with len > 10 chars
   (`select_best_hint`, scorers.py:155-161). No hint → sample dropped (OPD) or demoted to
   rl-only/nothing (combine).
3. **Injection** (`append_hint_to_messages`, scorers.py:164-182): the hint is appended to the
   **last user message of the student's original prompt** as
   `"\n\n[user's hint / instruction]\n{hint}"` — i.e. the teacher is conditioned on exactly the
   student's context *plus* the hindsight hint disguised as a user instruction.
4. **Teacher logprobs on the student's own tokens** (`_tinker_teacher_logprobs`,
   scorers.py:212-275): re-render the hint-enhanced messages with the chat template (+tools),
   concatenate the student's decoded `response_text`, encode the whole thing, and call the
   teacher with `temperature=0.0, max_tokens=1, include_prompt_logprobs=True,
   topk_prompt_logprobs=1` (scorers.py:242-247). The per-token teacher lps are the
   `prompt_logprobs` sliced after the enhanced-prompt token count (scorers.py:252-264),
   truncated/zero-padded to the response length (scorers.py:267-271). Any failure → all-zeros
   teacher lps (scorers.py:273-275), which makes the OPD term ≈ −student_lp per token.
   Known caveat: re-encoding decoded text can drift from the original rollout tokenization;
   they log a warning but proceed (scorers.py:254-260).

So "hindsight hints" = the mechanism that makes a *same-size* teacher produce higher logprobs
on good continuations: the teacher's distribution is shifted by privileged future information,
and `teacher_lp − student_lp` measures how much each student token gains credibility under
that privileged context.

## 4. Loss functions through Tinker

- Datums carry `target_tokens`, `logprobs` (= rollout sampling lps; prompt positions 0.0), and
  `advantages` (prompt positions 0.0) (data_formatter.py:13-18,74-99). Loss is whatever Tinker
  applies on top: `forward_backward_async(datums, loss_fn=self.config.loss_fn)` (trainer.py:159).
- **Default `loss_fn = "ppo"`** (config.py:36; env `LOSS_FN`, run.py:43). README documents the
  supported set: **`ppo`, `importance_sampling`, `cispo`** (README.md:70,142-143). No
  `loss_fn_args` are passed, so PPO clipping uses **Tinker's defaults (ε = 0.2 symmetric)**;
  the repo sets no custom epsilon on the Tinker path.
- `kl_loss_coef` exists in config (config.py:37 default 0.0; run.py:44 env default 0.02) but is
  **never referenced by the trainer — dead config on the Tinker path** (no KL term).
- Optimizer: Adam via `optim_step_async(AdamParams(learning_rate=1e-4))`, LoRA rank 32
  (trainer.py:171-173; config.py:26,33). One gradient step per collected batch (batch_size
  default 4; README examples use 16).
- [slime] reference constants: SLIME `compute_policy_loss` PPO clip with
  **`--eps-clip 0.2 --eps-clip-high 0.28`**, `--kl-loss-coef 0.0` (`low_var_kl` type), lr 1e-5
  (openclaw-combine/run_qwen3_4b_openclaw_combine.sh:132-142; combine_loss.py:132-137
  `ppo_kl = old_log_probs − new_log_probs` → ratio clipping, identical objective for OPD and RL
  samples — only the advantages differ).

## 5. Advantage normalization / baselines

- **None.** `compute_grpo_advantages` broadcasts the raw reward with an explicit note:
  "no normalization … Matches OpenClaw-RL's --disable-rewards-normalization"
  (data_formatter.py:212-217; [slime] `--disable-rewards-normalization`,
  run_qwen3_4b_openclaw_combine.sh:129). No batch centering, no std whitening, no group
  baseline, no length normalization anywhere in this repo's code.
- Despite the "GRPO" name, **group size is 1**: `--n-samples-per-prompt 1` [slime]
  (run scripts :101/:71), and on the Tinker path every sample is enqueued as its own
  single-element group (api_server.py:645-649, 828-832). There is no group-mean baseline;
  the reward in {+1, −1, 0} *is* the advantage.
- The only baseline-ish tricks are **masking/dispatch rules**:
  - RL method: turns with no next_state or PRM score 0 get `loss_mask = 0` (advantage zeroed,
    api_server.py:619,625); "at-least-one guarantee" — if a session would end with zero
    effective turns, one zero-scored turn is promoted to `loss_mask = 1`
    (api_server.py:621-623). (README.md:107 claims the promoted turn gets reward = +1, but the
    code keeps `reward = score` — i.e. 0.0 — at api_server.py:614,630, so the promoted sample
    is actually a no-op under advantage-weighted losses. Code ≠ README here.)
  - Non-finite logprobs/advantages sanitized to 0.0 (data_formatter.py:62-71).

## 6. (Adjacent, non-Tinker) top-K distillation variant

`openclaw-opd/topk_distillation_loss.py` is a separate Slime custom loss (not used by
openclaw-tinker): reverse KL D_KL(student‖teacher) over the teacher's top-K (K=50) logits plus
a tail bin (`log(1 − Σ topk p)` via `log(−expm1(logsumexp))`), per-token sum, sample-mean
reduced (topk_distillation_loss.py:80-101). Requires `--disable-compute-advantages-and-returns`.
Cited refs: SDFT (arXiv 2601.19897), SDPO (arXiv 2601.20802).

---

## What our implementation does differently

Ours: `advantage = clip(teacher_lp − student_lp, ±4)`, batch-mean centered,
`importance_sampling` loss, teacher = fixed **larger** model (no hints), rewards unused in the
loss, warmup = SFT on teacher-passing trajectories.

- **Signal source & shaping.** Their teacher is (by default) the *same-size* base model made
  smarter via hindsight-hint prompt injection — the advantage measures information gain from
  privileged future context; the raw gap is used **unclipped and uncentered** (plus a constant
  +1.0 broadcast in pure-OPD mode). We use a genuinely larger fixed teacher with no context
  augmentation, and we post-process the gap (clip ±4 + batch-mean centering) — regularization
  they get instead from PPO ratio clipping (Tinker default ε≈0.2; slime 0.2/0.28), since their
  default `loss_fn` is `ppo`, not `importance_sampling`.
- **Reward usage.** They fold the task/PRM reward into the *same* per-token advantage
  (`w_opd·(t−s) + w_rl·reward`, three-way gated: opd+rl / opd-only(reward=0) /
  rl-only(teacher:=student ⇒ OPD term=0)); nothing is failure-gated. We keep rewards entirely
  out of the loss — closest to their pure-OPD mode minus the +1.0 reward broadcast.
- **Cold start.** They have no SFT warmup at all — every method is on-policy RL-style from
  step 1, with data gating (drop turns lacking next_state/hint; at-least-one guarantee) doing
  the curriculum work. Our SFT-on-teacher-passing-trajectories warmup has no analogue in their
  pipeline.
