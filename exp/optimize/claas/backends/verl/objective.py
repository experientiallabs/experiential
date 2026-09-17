"""Custom SDPO loss callbacks executed by the upstream veRL training worker."""

from __future__ import annotations

import torch
from tensordict import TensorDict
from torch import Tensor
from torch.distributed import ProcessGroup

from exp.optimize.claas.algorithms.sdpo import feedback_objective
from exp.optimize.claas.training_contracts import TrainingJob


class FeedbackLoss:
    """Capture detached teacher distributions or evaluate the student objective.

    veRL invokes this callback during its forward step and owns all backward and
    optimizer operations. Teacher distributions are bounded by the validated batch
    and kept on CPU between teacher inference and student microbatches.
    """

    def __init__(self, job: TrainingJob, *, teacher: bool = False) -> None:
        """Bind immutable examples and an initially empty teacher distribution buffer."""
        self.job = job
        self.teacher = teacher
        self.teacher_logits: dict[int, Tensor] = {}
        self.response_tokens = sum(
            len(item.experience.exact_tokens.response_token_ids)
            for item in job.batch.examples
            if item.experience.exact_tokens is not None
        )

    def __call__(
        self,
        *,
        model_output: dict[str, Tensor],
        data: TensorDict,
        dp_group: ProcessGroup | None = None,
    ) -> tuple[Tensor, dict[str, float]]:
        """Return a differentiable, globally token-normalized loss to the engine."""
        del dp_group
        rows = model_output.pop("claas_logits").unbind()
        loss = rows[0].sum() * 0
        metrics: dict[str, float] = {}
        for row, index_value, start_value in zip(
            rows, data["claas_index"], data["claas_response_start"], strict=True
        ):
            index, start = int(index_value), int(start_value)
            item = self.job.batch.examples[index]
            tokens = item.experience.exact_tokens
            if tokens is None:
                raise ValueError("training requires original token evidence")
            logits = row[start : start + len(tokens.response_token_ids)]
            if self.teacher:
                self.teacher_logits[index] = logits.detach().float().cpu().clone()
                continue
            target = self.teacher_logits.get(index)
            contribution, observed = feedback_objective(
                student_logits=logits,
                teacher_logits=target.to(logits.device) if target is not None else None,
                response_tokens=torch.tensor(tokens.response_token_ids, device=logits.device),
                rollout_logprobs=torch.tensor(tokens.response_logprobs, device=logits.device),
                scalar_reward=item.scalar_reward,
                spec=self.job.spec,
                batch_response_tokens=self.response_tokens,
            )
            loss = loss + contribution
            for name, value in observed.items():
                metrics[name] = metrics.get(name, 0.0) + value
        return loss, metrics
