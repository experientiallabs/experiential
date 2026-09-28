"""CPU numeric contracts for the full-logit veRL output extension, not GPU training."""

from pathlib import Path
from typing import cast

import torch
from transformers.modeling_outputs import CausalLMOutputWithPast
from verl.workers.config import FSDPEngineConfig

from exp.optimize.claas.backends.verl.engine import ClaasFeedbackEngine
from exp.optimize.claas.backends.verl.inputs import build_engine_batch
from exp.optimize.claas.backends.verl.native_test import tokenizer
from exp.optimize.claas.training_contracts_test import job


def test_upstream_output_preparation_keeps_full_logits_and_causal_probabilities(
    tmp_path: Path,
) -> None:
    """Invoke the actual upstream output-preparation method on ordinary CPU tensors."""
    engine = object.__new__(ClaasFeedbackEngine)
    engine.engine_config = FSDPEngineConfig(entropy_checkpointing=False)
    batch = build_engine_batch(job(tmp_path), tokenizer())
    logits = torch.randn(1, 4, 8, requires_grad=True)
    ids = batch["input_ids"]
    output = engine.prepare_model_outputs(
        CausalLMOutputWithPast(logits=cast(torch.FloatTensor, logits)),
        {"temperature": torch.ones(1), "input_ids_rmpad_rolled": ids.values().roll(-1)},
        batch,
        lambda: None,
    )
    torch.testing.assert_close(output["claas_logits"].unbind()[0], logits[0])
    selected = logits[0].log_softmax(-1).gather(1, ids.values().roll(-1).unsqueeze(1)).squeeze(1)
    torch.testing.assert_close(output["log_probs"].unbind()[0], selected)
    output["claas_logits"].values().sum().backward()
    assert logits.grad is not None
