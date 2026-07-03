# BENCH-B2 results — SFT / PPO / REINFORCE++ vs the tau world model

*B2 arms of the BENCH-B RL-transfer ladder (D22): Qwen3.5-9B trained closed-loop against
the wmh tau-bench world model (Bedrock Haiku 4.5 backend), evaluated on the pinned
held-out scenarios against a pinned GPT-5.5-backed WM with an Opus 4.8 reward judge
(protocol D30). ICL is the coordinator's row; GRPO/SDPO are chat 3's.*

## Held-out results (20 pinned eval scenarios × 2 episodes, temp 1.0)

| arm | success rate | mean reward | episodes | notes |
|---|---|---|---|---|
| base Qwen3.5-9B | **55.0%** | 0.568 | 40/40 clean | wandb `wm_tau-eval-base-v3` |
| SFT (LoRA, 698 steps) | **60.0%** | 0.584 | 40/40 clean | `wm_tau-eval-sft-ep3-v2` |
| REINFORCE++ ckpt-0096 | **52.5%** | 0.562 | 40/40 clean | `wm_tau-eval-rpp96` |
| PPO ckpt-0096 | **54.1%** | 0.577 | 37/40 clean (3 judge-timeout errors excluded) | `wm_tau-eval-ppo96` |

95% CI on a 40-episode success rate is roughly ±15 points: **no arm separates from base
at this eval size.** The paired per-scenario view is more informative:

- **REINFORCE++ − base: mean Δ −0.005, median 0.000 (2 wins / 5 losses / 13 ties).**
  The trained policy is nearly indistinguishable from base — expected given how little
  the weights moved (86 training steps, KL-regularized LoRA, max |Δw| = 1.6e-4).
  Training-side reward on the WM did drift up within the single pass
  (quartile means 0.782 → 0.812), but it does not transfer to held-out scenarios
  at this magnitude of update.
- **PPO − base: mean Δ +0.009, median 0.000 (6 wins / 5 losses).** Same picture as
  REINFORCE++: statistically flat, tiny weight movement (75 steps, max |Δw| = 1.4e-4).
  Its training-side reward was flat-to-down within the pass (0.835 → 0.776 quartiles).
- **SFT − base: mean Δ +0.016 with heavy redistribution (5 wins / 8 losses).**
  SFT newly solves scenarios base never solves (97f8d3b7 +0.93, bf04d8b8 +0.78,
  56917256 +0.67 — tasks resembling its 97 recorded demonstrations) while breaking
  scenarios base aced (ea1f0245 −0.93, 772c0b9b −0.83). Audits show the failure mode:
  the SFT model acts without checking policy constraints (e.g. cancelling a basic-economy
  booking past the 24h window) — it imitates action patterns and, trained without think
  blocks, no longer deliberates.

## Per-domain breakdown (D35 caveat applies)

| arm | airline (14 eps) | retail (16 eps) | telecom (10 eps) | excl. telecom |
|---|---|---|---|---|
| base | 29% (r 0.31) | 50% (r 0.53) | **100%** (r 1.00) | 40% |
| SFT | **50%** (r 0.46) | 56% (r 0.57) | 80% (r 0.78) | **53%** |
| PPO ckpt-0096 | 25% (r 0.30) | 50% (r 0.56) | **100%** (r 0.98) | 39% |
| R++ ckpt-0096 | 29% (r 0.29) | 44% (r 0.54) | **100%** (r 0.99) | 37% |

**Telecom saturates (10/10) for base/PPO/R++ and is familiarity, not generalization
headroom (D35):** the 5 telecom eval tasks have ~725 near-duplicate captures in the WM's
retrieval buffer (the same traces the D32 leakage rule dropped from the RL scenario
lists), so the env simulates them with near-recorded fidelity. Identical for every arm
(still apples-to-apples), but read the non-telecom columns for signal. There, the story
sharpens: **SFT's lift concentrates in airline (29% → 50%)** — the domain with the most
surviving demonstrations — while both RL arms stay flat everywhere. SFT is also the only
arm that *drops* telecom (100% → 80%): it trained on zero telecom demonstrations and
un-learned some of the base model's saturated behavior.

## Training dynamics (wandb) — why no RL lift, empirically

Nothing in the loop is broken; the runs were **cold and short**:

- **Learning rate**: both arms inherited the algorithm groups' `lr 5e-6`. The proven IH
  GRPO recipe (adopted by the B3 arm, D36) uses **3e-5 — 6× hotter**. At 5e-6 × 75–86
  steps with grad norms ~0.3, total LoRA movement tops out at |Δw| ≈ 1.5e-4 — the flat
  eval rows are the arithmetic consequence.
- **The policy did move, slowly**: R++ `actor/kl_loss` grows monotonically
  4.8e-4 → 1.8e-2 across the run (ref-model path verified working); entropy healthy
  (0.26 → 0.31, no collapse); grad norms stable.
- **PPO's clipping and IS were identity operations**: `ppo_kl = 0`, `pg_clipfrac = 0`
  for all 75 steps, and R++ rollout-IS ratios exactly 1.0 ± 0.0 — with
  `recompute_old_log_probs=false` and low buffer age, both arms trained as plain
  on-policy policy gradient.
- **PPO's critic never converged**: `critic/vf_explained_var` stayed negative through
  all 75 steps (final quartile −0.20), so GAE advantages were mostly noise — PPO was
  effectively REINFORCE with an uncooked baseline for this run length.

**Queued follow-up (next 2-GPU window):** rerun PPO/R++ at lr 3e-5 with 2–3 epochs
(~250+ steps) and critic warmup — a config change, not a code change.

## Lift experiment (box-6): the REINFORCE++ learning-rate window is empty

The queued follow-up ran. Single-variable LR sweep on R++, everything else identical:

| lr | KL anchor | outcome |
|---|---|---|
| 5e-6 (original) | 0.01 | flat — policy barely moves (max Δw ≈ 1.6e-4) |
| 1e-5 | 0.05 | epoch 1 healthy and rising (0.64 → 0.91 within-epoch), then a **slow slide**: paired epoch-2 vs epoch-1 on the *same scenarios* −0.131 (14W/24L, n=56), entropy 0.37 → 0.21, KL accelerating — stopped at ep. ~151, early-stop checkpoint 0072 kept |
| 3e-5 (proven IH-GRPO rate) | 0.01 | **collapse**: rewards 0.81 → 0.00 by episode ~70, entropy 0.31 → 0.08, KL 2.2 by step 45; degenerate fixed point = immediate `done()` every episode |

**Reading:** the pipeline's training signal is strong — it moves the policy hard in every
regime — but n=1 binary-reward REINFORCE++ has no productive LR window on this task:
below ~1e-5 it under-drives, at 1e-5 it drifts toward reward-destroying policies after
~1 epoch, at 3e-5 it collapses outright. This is the variance problem group-relative
baselines exist to solve: the failure *predicts* GRPO (n=8 group advantages — the arm the
WM uniquely enables, running as chat 3's cell) rather than more single-rollout tuning.
The early-stopped ckpt-0072 held-out row is being measured (partial: tracking the same
flat band as the cold run). PPO at the safe profile (ratio clipping as the remaining
single-rollout stabilizer) is the last cell of this sweep.

Ops note from the run: a us-east-1 Bedrock `ServiceUnavailableException` storm stalled
judge calls twice — the serve script's failover chains now end in a us-west-2 link,
which absorbed the third flare live (episodes progressed through 10+ 503s).

## Training runs (97 pinned train scenarios × 1 epoch, n=1 rollout/scenario)

| arm | reward | train steps | episodes clean | WM cost (serve/judge) |
|---|---|---|---|---|
| REINFORCE++ (binary success) | mean 0.793, success 70.1% | 86 | 96/97 | $3.75 / $0.39 |
| PPO (scalar EpisodeScore.reward) | mean 0.803, success 71.1% | 75 | 97/97 | $4.52 / $0.43 |

Raw per-episode records (actions, step rewards, judge critiques, WM costs) are committed
under the agents workspace at `.agents/docs/research/wm_tau_eval_results/` (raw run
outputs stay out of `examples/` per the repo layout rules). The paired table reproduces
from those records with any per-scenario mean comparison, e.g.:

```python
import json
rows = [json.loads(l) for l in open("base_v3.jsonl") if not json.loads(l)["errors"]]
by = {}
for r in rows: by.setdefault(r["scenario_id"][:8], []).append(r["reward"])
means = {k: sum(v)/len(v) for k, v in by.items()}  # repeat per arm, subtract vs base
```

wandb project: `wmh-rl-transfer` (server runs `qwen3_5_9b_wm_tau_reinforce_real`,
`qwen3_5_9b_wm_tau_ppo_real_v2`; eval runs listed above).

## Honest reading (v1)

At this scale — single pass over ~100 scenarios, LoRA rank 32, lr 5e-6, KL-pinned —
closed-loop RL against the WM produces a policy that is statistically flat vs base on
held-out tau scenarios, and imitation (SFT) redistributes competence rather than adding
it. This is consistent with Kion's ~4% SFT experience and with the CLaaS paper's setting
being a *reference point*: detecting small deltas here needs either more training signal
(multiple epochs / more scenarios / larger LoRA lr) or a bigger eval set. The
infrastructure result stands independently: all data paths (rollout → WM reward →
TITO feedback → train → hot-reload) run end-to-end for every arm.

## Dataset facts that shape the rows

- The corpus's test split holds only **20 unique tasks** (repeat captures dominate);
  eval = all 20 × 2 episodes (D26/D30).
- The SFT dataset is **97 episodes / 698 steps**: 5 eval tasks account for 725 of 822
  train traces, and the leakage rule (never train on eval task text) drops them (D32).
- tau traces are 100% tool calls (zero message actions); judge critiques occasionally
  phrase expectations conversationally — identical bias for every arm.

## Failure analysis / infra findings (full ladder in the claas-verl journals)

See `experiments/07_02_2026_wm-tau-{sft-lora,ppo-reinforce}.md` in claas-verl and
DECISIONS D34/D40–D43: Qwen3.5 XML tool-format (vLLM `qwen3_xml` parser + scaffold text
fallback), stale-keep-alive Bedrock hangs (tcp_keepalive + same-model FallbackProvider
chains + WM warm-up probe), PPO critic token-cap and LM-head-logits OOMs (sequence
length is the lever), the wake/sleep LoRA checkpoint namespace trap (peft silently
matches zero keys; checkpoints are evaluated via direct W += α/r·B·A application), and
the non-thinking SFT template mismatch (immediate-EOS without
`chat_template_kwargs={"enable_thinking": false}`).
