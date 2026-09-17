"""Exact nested-batch alignment and feedback cost accounting tests."""

from pathlib import Path

import pytest

from exp.optimize.claas.backends.verl.inputs import build_engine_batch, teacher_context
from exp.optimize.claas.backends.verl.native_test import tokenizer
from exp.optimize.claas.training_contracts_test import example, job


def test_feedback_context_preserves_original_ids() -> None:
    """Teacher formatting tokenizes only newly supplied feedback."""
    item = example()
    context = teacher_context(item, tokenizer(), 128)
    assert context is not None and context[:2] == (1, 2)
    assert item.experience.exact_tokens is not None
    assert item.experience.exact_tokens.response_token_ids == (3, 4)
    with pytest.raises(ValueError, match="shorten feedback"):
        teacher_context(item, tokenizer(), 4)


def test_actual_tensordict_preserves_shifted_prediction_positions(tmp_path: Path) -> None:
    """veRL receives all original prompt/response IDs with the correct causal loss mask."""
    training = job(tmp_path)
    student = build_engine_batch(training, tokenizer())
    teacher = build_engine_batch(training, tokenizer(), teacher=True)
    assert student["input_ids"].unbind()[0].tolist() == [1, 2, 3, 4]
    assert student["loss_mask"].unbind()[0].tolist() == [0, 1, 1, 0]
    assert student["claas_response_start"].tolist() == [1]
    assert teacher["input_ids"].unbind()[0][-2:].tolist() == [3, 4]
    assert teacher["input_ids"].unbind()[0][:2].tolist() == [1, 2]
    assert int(teacher["claas_response_start"][0]) > 1
    assert student["loss_mask"].sum() == teacher["loss_mask"].sum() == 2


def test_teacher_tokens_count_toward_hard_input_limit(tmp_path: Path) -> None:
    """Feedback cannot add hidden unbudgeted forward tokens."""
    training = job(tmp_path)
    small = training.model_copy(
        update={"spec": training.spec.model_copy(update={"max_batch_tokens": 5})}
    )
    with pytest.raises(ValueError, match="student and teacher inputs"):
        build_engine_batch(small, tokenizer())
