"""Real tensor loss-callback alignment tests without claiming worker training."""

from pathlib import Path

import torch

from exp.optimize.claas.algorithms.sdpo import feedback_objective
from exp.optimize.claas.backends.verl.inputs import build_engine_batch
from exp.optimize.claas.backends.verl.native_test import tokenizer
from exp.optimize.claas.backends.verl.objective import FeedbackLoss
from exp.optimize.claas.training_contracts_test import job


def test_teacher_and_student_callbacks_match_exact_original_response_objective(
    tmp_path: Path,
) -> None:
    """The upstream loss interface aligns feedback-conditioned and original contexts."""
    training = job(tmp_path)
    student_batch = build_engine_batch(training, tokenizer())
    teacher_batch = build_engine_batch(training, tokenizer(), teacher=True)
    teacher_logits = torch.randn(len(teacher_batch["input_ids"].unbind()[0]), 8)
    student_logits = torch.randn(4, 8, requires_grad=True)
    teacher = FeedbackLoss(training, teacher=True)
    teacher(
        model_output={
            "claas_logits": torch.nested.as_nested_tensor([teacher_logits], layout=torch.jagged)
        },
        data=teacher_batch,
    )
    student = FeedbackLoss(training)
    student.teacher_logits = teacher.teacher_logits
    actual, _ = student(
        model_output={
            "claas_logits": torch.nested.as_nested_tensor([student_logits], layout=torch.jagged)
        },
        data=student_batch,
    )
    item = training.batch.examples[0]
    assert item.experience.exact_tokens is not None
    expected, _ = feedback_objective(
        student_logits=student_logits[1:3],
        teacher_logits=teacher.teacher_logits[0],
        response_tokens=torch.tensor([3, 4]),
        rollout_logprobs=torch.tensor(item.experience.exact_tokens.response_logprobs),
        scalar_reward=item.scalar_reward,
        spec=training.spec,
        batch_response_tokens=2,
    )
    torch.testing.assert_close(actual, expected)
    actual.backward()
    assert student_logits.grad is not None
    assert student_logits.grad[0].abs().sum() == student_logits.grad[3].abs().sum() == 0
    assert not teacher.teacher_logits[0].requires_grad
