"""Opt-in CUDA proof of the resident public veRL rollout/training weight-sync loop."""

import asyncio
import os
from pathlib import Path

import pytest
import torch
from transformers import LlamaConfig, LlamaForCausalLM

from exp.common.claas.generation import GenerationRequest, GenerationResult
from exp.optimize.claas.backends.verl import resident
from exp.optimize.claas.backends.verl.configuration import ResidentVerlSettings
from exp.optimize.claas.backends.verl.native_test import tokenizer
from exp.optimize.claas.backends.verl.runtime import ResidentVerlFactory
from exp.optimize.claas.training_contracts import TrainingBatch
from exp.optimize.claas.training_contracts_test import example, spec


def rollout_batch(result: GenerationResult, batch_id: str) -> TrainingBatch:
    """Use the sampler's actual IDs and logprobs directly as the training experience."""
    item = example(policy=result.exact_tokens.policy_revision)
    item = item.model_copy(
        update={
            "experience": item.experience.model_copy(
                update={
                    "response_id": result.response_id,
                    "exact_tokens": result.exact_tokens,
                    "experience_id": result.response_id,
                }
            )
        }
    )
    return TrainingBatch(
        batch_id=batch_id,
        expected_policy_revision=result.exact_tokens.policy_revision,
        examples=(item,),
    )


@pytest.mark.skipif(
    os.environ.get("CLAAS_RUN_CUDA_ROLLOUT") != "1" or not torch.cuda.is_available(),
    reason="requires explicit CLAAS_RUN_CUDA_ROLLOUT=1, vLLM 0.22 and an authorized CUDA GPU",
)
def test_cuda_resident_rollout_training_and_adapter_refresh(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Run real generation, native updates and IPC LoRA refresh on one retained Ray server."""
    snapshot = tmp_path / "base"
    config = LlamaConfig.from_dict(
        {
            "vocab_size": 8,
            "hidden_size": 128,
            "intermediate_size": 256,
            "num_hidden_layers": 1,
            "num_attention_heads": 2,
            "num_key_value_heads": 2,
            "max_position_embeddings": 128,
            "bos_token_id": 1,
            "eos_token_id": 7,
        }
    )
    LlamaForCausalLM(config).save_pretrained(snapshot)
    tokenizer().save_pretrained(snapshot)
    monkeypatch.setattr(resident, "snapshot_download", lambda *_args, **_kwargs: str(snapshot))
    monkeypatch.setattr(resident, "_validate_model_reference", lambda _id, _revision: None)
    training_spec = spec().model_copy(update={"target_modules": ("q_proj", "v_proj")})
    settings = ResidentVerlSettings(
        checkpoint_root=tmp_path / "results",
        decoder="text",
        maximum_output_tokens=4,
    )

    async def exercise() -> None:
        """Retain both public engines while generation and training alternate twice."""
        runtime = await ResidentVerlFactory(settings).open(training_spec, mode="run")
        actor, teacher, rollout = runtime.trainer.actor, runtime.trainer.teacher, runtime._rollout
        try:
            first = await runtime.generate(
                GenerationRequest(
                    request_id="r1",
                    model=training_spec.adapter_id,
                    prompt="a b",
                    maximum_output_tokens=4,
                )
            )
            trained = await runtime.train(rollout_batch(first, "batch-1"))
            second = await runtime.generate(
                GenerationRequest(
                    request_id="r2",
                    model=training_spec.adapter_id,
                    prompt="a b",
                    maximum_output_tokens=4,
                )
            )
            assert second.exact_tokens.policy_revision == trained.checkpoint.policy_revision
            await runtime.train(rollout_batch(second, "batch-2"))
            assert (await runtime.checkpoint()).step == 2
            assert runtime.trainer.actor is actor and runtime.trainer.teacher is teacher
            assert runtime._rollout is rollout
            assert await runtime.train(rollout_batch(first, "batch-1")) == trained
        finally:
            await runtime.close()

    asyncio.run(exercise())
