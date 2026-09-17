"""Full-logit output extension of the published veRL FSDP language-model engine.

Only model-output preparation is extended. Model construction, LoRA wrapping,
forward/backward microbatching, optimizer updates and checkpoint state belong to
veRL. One CUDA rank uses ordinary padded attention and full-vocabulary SDPO.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import cast

import torch
from tensordict import TensorDict
from torch import Tensor
from transformers.modeling_outputs import CausalLMOutputWithPast
from verl.trainer.config import CheckpointConfig
from verl.workers.config import FSDPEngineConfig, FSDPOptimizerConfig, HFModelConfig
from verl.workers.engine import EngineRegistry, FSDPEngineWithLMHead

ENGINE_MODEL_TYPE = "claas_feedback_language_model"


@EngineRegistry.register(model_type=ENGINE_MODEL_TYPE, backend="fsdp", device="cuda")
class ClaasFeedbackEngine(FSDPEngineWithLMHead):
    """Expose nested full logits through veRL's existing custom-loss interface."""

    def __init__(
        self,
        model_config: HFModelConfig,
        engine_config: FSDPEngineConfig,
        optimizer_config: FSDPOptimizerConfig,
        checkpoint_config: CheckpointConfig,
    ) -> None:
        """Keep the registry extension distinct from the Hugging Face model kind."""
        model_config.model_type = "language_model"
        super().__init__(model_config, engine_config, optimizer_config, checkpoint_config)

    def prepare_model_outputs(
        self,
        output: CausalLMOutputWithPast,
        output_args: dict[str, Tensor | int],
        micro_batch: TensorDict,
        logits_processor_func: Callable,
    ) -> dict[str, Tensor]:
        """Preserve upstream outputs and add logits aligned with the exact input IDs."""
        prepared = super().prepare_model_outputs(
            output, output_args, micro_batch, logits_processor_func
        )
        logits = output.logits
        if logits is None or logits.ndim != 3:
            raise ValueError("CLaaS requires full-vocabulary causal logits")
        ids = cast(Tensor, micro_batch["input_ids"])
        lengths = [len(row) for row in ids.unbind()]
        rows = [logits[index, :length] for index, length in enumerate(lengths)]
        prepared["claas_logits"] = torch.nested.as_nested_tensor(rows, layout=torch.jagged)
        return prepared
