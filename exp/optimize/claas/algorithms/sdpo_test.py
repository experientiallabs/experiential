"""Numerical SDPO and veRL scalar-gradient tests on CPU tensors."""

import pytest
import torch

from exp.optimize.claas.algorithms.sdpo import feedback_objective
from exp.optimize.claas.training_contracts_test import spec


@pytest.mark.parametrize("alpha", [0.0, 0.25, 0.5, 0.75, 1.0])
def test_full_vocabulary_divergence_and_teacher_stop_gradient(alpha: float) -> None:
    """Match an independent distribution-level oracle and stop teacher gradients."""
    student = torch.tensor([[0.4, 1.0, -0.3], [1.2, 0.0, 0.7]], requires_grad=True)
    teacher = torch.tensor([[1.0, 0.2, 0.8], [-0.2, 1.0, 0.3]], requires_grad=True)
    ids = torch.tensor([1, 2])
    old = student.detach().log_softmax(-1).gather(-1, ids[:, None]).squeeze(-1)
    loss, metrics = feedback_objective(
        student_logits=student,
        teacher_logits=teacher,
        response_tokens=ids,
        rollout_logprobs=old,
        scalar_reward=None,
        spec=spec().model_copy(update={"sdpo_alpha": alpha}),
        batch_response_tokens=2,
    )
    p, q = student.softmax(-1), teacher.detach().softmax(-1)
    if alpha == 0:
        oracle = (q * (q.log() - p.log())).sum(-1).mean()
    elif alpha == 1:
        oracle = (p * (p.log() - q.log())).sum(-1).mean()
    else:
        mix = (1 - alpha) * p + alpha * q
        oracle = (
            ((1 - alpha) * p * (p.log() - mix.log()) + alpha * q * (q.log() - mix.log()))
            .sum(-1)
            .mean()
        )
    torch.testing.assert_close(loss, oracle)
    loss.backward()
    assert student.grad is not None and student.grad.abs().sum() > 0
    assert teacher.grad is None
    assert metrics["sdpo_loss"] >= 0


def test_scalar_policy_gradient_uses_original_behavior_probs_and_clipped_detached_ratio() -> None:
    """veRL REINFORCE receives the real behavior denominator and finite IS cap."""
    logits = torch.tensor([[0.0, 0.0]], requires_grad=True)
    loss, metrics = feedback_objective(
        student_logits=logits,
        teacher_logits=None,
        response_tokens=torch.tensor([0]),
        rollout_logprobs=torch.tensor([-10.0]),
        scalar_reward=1,
        spec=spec().model_copy(update={"objective": "reinforce"}),
        batch_response_tokens=1,
    )
    loss.backward()
    torch.testing.assert_close(logits.grad, torch.tensor([[-1.0, 1.0]]))
    assert metrics["importance_ratio_mean"] == 2.0


def test_full_vocabulary_rejects_mismatched_teacher() -> None:
    """No top-k or different-tokenizer teacher is mistaken for full SDPO."""
    with pytest.raises(ValueError, match="identical"):
        feedback_objective(
            student_logits=torch.zeros(1, 3),
            teacher_logits=torch.zeros(1, 2),
            response_tokens=torch.tensor([0]),
            rollout_logprobs=torch.tensor([-1.0]),
            scalar_reward=None,
            spec=spec(),
            batch_response_tokens=1,
        )


@pytest.mark.parametrize("missing", ["teacher", "scalar"])
def test_hybrid_loss_never_silently_omits_a_component(missing: str) -> None:
    """Direct objective callers have the same complete-recipe requirement as jobs."""
    with pytest.raises(ValueError, match="hybrid requires both"):
        feedback_objective(
            student_logits=torch.zeros(1, 2, requires_grad=True),
            teacher_logits=None if missing == "teacher" else torch.ones(1, 2),
            response_tokens=torch.tensor([0]),
            rollout_logprobs=torch.tensor([-1.0]),
            scalar_reward=None if missing == "scalar" else 1.0,
            spec=spec().model_copy(update={"objective": "hybrid"}),
            batch_response_tokens=1,
        )
