"""Full-vocabulary SDPO and importance-corrected veRL scalar policy gradients.

SDPO computes the feedback-conditioned teacher distribution on the student's
original sampled responses. The teacher is detached, never a decoded/re-encoded
replacement target. Generalized JSD and clipped importance weighting follow
https://arxiv.org/abs/2601.20802; no top-k or sampled-token approximation is used.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor
from torch.nn import functional as F
from verl.trainer.ppo.core_algos import agg_loss, compute_policy_loss_reinforce
from verl.workers.config import ActorConfig

from exp.optimize.claas.training_contracts import ClaasTrainingSpec


def feedback_objective(
    *,
    student_logits: Tensor,
    teacher_logits: Tensor | None,
    response_tokens: Tensor,
    rollout_logprobs: Tensor,
    scalar_reward: float | None,
    spec: ClaasTrainingSpec,
    batch_response_tokens: int,
) -> tuple[Tensor, dict[str, float]]:
    """Compute one sequence's contribution to one token-normalized batch update.

    Args:
        student_logits: Raw current logits, exactly one row per response token.
        teacher_logits: Same-vocabulary EMA logits conditioned on text feedback.
        response_tokens: Original sampled IDs, without a decoded text round trip.
        rollout_logprobs: Original behavior probabilities at temperature one.
        scalar_reward: Optional outcome reward in [-1, 1].
        spec: Frozen objective settings.
        batch_response_tokens: Total response tokens across the complete update.

    Returns:
        Differentiable scalar contribution and observed, finite diagnostic metrics.
    """
    if student_logits.ndim != 2 or student_logits.shape[0] != response_tokens.numel():
        raise ValueError("student logits must align with every original response token")
    if rollout_logprobs.shape != response_tokens.shape or batch_response_tokens < 1:
        raise ValueError("rollout probabilities and normalization must cover the original response")
    if spec.objective == "hybrid" and (teacher_logits is None or scalar_reward is None):
        raise ValueError(
            "hybrid requires both feedback-conditioned teacher logits and scalar reward"
        )
    student_lp = F.log_softmax(student_logits.float(), dim=-1)
    chosen_lp = student_lp.gather(-1, response_tokens.unsqueeze(-1)).squeeze(-1).unsqueeze(0)
    old_lp = rollout_logprobs.float().unsqueeze(0)
    mask = torch.ones_like(chosen_lp)
    ratio = (chosen_lp.detach() - old_lp).clamp(-20, 20).exp().clamp(max=spec.importance_ratio_cap)
    total = chosen_lp.sum() * 0
    metrics = {"importance_ratio_mean": float(ratio.mean().detach())}
    if spec.objective in {"sdpo", "hybrid"} and teacher_logits is not None:
        if teacher_logits.shape != student_logits.shape:
            raise ValueError("SDPO teacher and student must have identical token/vocabulary shapes")
        teacher_lp = F.log_softmax(teacher_logits.detach().float(), dim=-1)
        alpha = spec.sdpo_alpha
        if alpha == 0:
            divergence = F.kl_div(student_lp, teacher_lp, reduction="none", log_target=True)
        elif alpha == 1:
            divergence = F.kl_div(teacher_lp, student_lp, reduction="none", log_target=True)
        else:
            mixture = torch.logaddexp(student_lp + math.log1p(-alpha), teacher_lp + math.log(alpha))
            teacher_kl = F.kl_div(mixture, teacher_lp, reduction="none", log_target=True)
            student_kl = F.kl_div(mixture, student_lp, reduction="none", log_target=True)
            divergence = torch.lerp(student_kl, teacher_kl, alpha)
        distillation = agg_loss(
            loss_mat=divergence.sum(-1).unsqueeze(0) * ratio,
            loss_mask=mask,
            loss_agg_mode="token-mean",
            batch_num_tokens=batch_response_tokens,
        )
        total = total + distillation
        metrics["sdpo_loss"] = float(distillation.detach())
    elif spec.objective == "sdpo":
        raise ValueError("SDPO requires feedback-conditioned full-vocabulary teacher logits")
    if spec.objective in {"reinforce", "hybrid"} and scalar_reward is not None:
        config = ActorConfig(
            strategy="fsdp",
            rollout_n=1,
            ppo_micro_batch_size_per_gpu=1,
            global_batch_info={"batch_num_tokens": batch_response_tokens},
        )
        scalar_loss, _ = compute_policy_loss_reinforce(
            rollout_log_prob=old_lp,
            log_prob=chosen_lp,
            advantages=torch.full_like(chosen_lp, scalar_reward),
            response_mask=mask,
            loss_agg_mode="token-mean",
            config=config,
            rollout_is_weights=ratio,
        )
        total = total + spec.scalar_loss_weight * scalar_loss
        metrics["scalar_loss"] = float(scalar_loss.detach())
    elif spec.objective == "reinforce":
        raise ValueError("REINFORCE requires a scalar reward")
    if not bool(torch.isfinite(total)):
        raise ValueError("nonfinite training objective; no optimizer update is permitted")
    return total, metrics
