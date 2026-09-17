"""Token-exact conversion from portable training jobs to veRL nested TensorDicts."""

from __future__ import annotations

import torch
from tensordict import TensorDict
from transformers import PreTrainedTokenizerBase
from verl.utils import tensordict_utils as tu

from exp.optimize.claas.training_contracts import (
    TrainingExample,
    TrainingJob,
    teacher_feedback_text,
)


def teacher_context(
    item: TrainingExample, tokenizer: PreTrainedTokenizerBase, max_sequence_tokens: int
) -> tuple[int, ...] | None:
    """Append newly tokenized feedback while retaining every original sampled ID."""
    tokens = item.experience.exact_tokens
    if tokens is None:
        raise ValueError("training requires original token evidence")
    if item.text_feedback is None:
        return None
    added = tuple(
        tokenizer.encode(teacher_feedback_text(item.text_feedback), add_special_tokens=False)
    )
    context = tokens.prompt_token_ids + added
    if len(context) + len(tokens.response_token_ids) > max_sequence_tokens:
        raise ValueError(
            "feedback-conditioned sequence exceeds max_sequence_tokens; shorten feedback"
        )
    return context


def build_engine_batch(
    job: TrainingJob, tokenizer: PreTrainedTokenizerBase, *, teacher: bool = False
) -> TensorDict:
    """Build jagged model inputs without decoding, truncating or re-tokenizing rollouts."""
    contexts = [
        teacher_context(item, tokenizer, job.spec.max_sequence_tokens)
        if job.spec.objective != "reinforce"
        else None
        for item in job.batch.examples
    ]
    total = 0
    ids, positions, masks, indices, response_starts = [], [], [], [], []
    for index, (item, context) in enumerate(zip(job.batch.examples, contexts, strict=True)):
        tokens = item.experience.exact_tokens
        if tokens is None:
            raise ValueError("training requires original token evidence")
        total += len(tokens.prompt_token_ids) + len(tokens.response_token_ids)
        if context is not None:
            total += len(context) + len(tokens.response_token_ids)
        selected = context if teacher else tokens.prompt_token_ids
        if selected is None:
            raise ValueError("teacher inference requires text feedback")
        sequence = selected + tokens.response_token_ids
        ids.append(torch.tensor(sequence, dtype=torch.long))
        positions.append(torch.arange(len(sequence)))
        # Loss positions predict response tokens from the preceding input position.
        masks.append(
            torch.tensor([0] * (len(selected) - 1) + [1] * len(tokens.response_token_ids) + [0])
        )
        indices.append(index)
        response_starts.append(len(selected) - 1)
    if total > job.spec.max_batch_tokens:
        raise ValueError("student and teacher inputs exceed max_batch_tokens; split the batch")

    def nested(rows: list[torch.Tensor]) -> torch.Tensor:
        """Keep variable sequence lengths explicit at the veRL batch boundary."""
        return torch.nested.as_nested_tensor(rows, layout=torch.jagged)

    return tu.get_tensordict(
        tensor_dict={
            "input_ids": nested(ids),
            "position_ids": nested(positions),
            "loss_mask": nested(masks),
            "claas_index": torch.tensor(indices),
            "claas_response_start": torch.tensor(response_starts),
        },
        non_tensor_dict={
            "temperature": 1.0,
            "global_token_num": [len(row) for row in ids],
            "use_remove_padding": False,
            "use_fused_kernels": False,
            "use_dynamic_bsz": False,
            "micro_batch_size_per_gpu": 1,
            "pad_token_id": tokenizer.pad_token_id or 0,
            "compute_loss": True,
            "update_lr_scheduler": not teacher,
        },
    )
