"""Pinned public veRL worker configuration and exclusive single-device process ownership."""

from __future__ import annotations

import importlib.metadata
import os
import re
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import timedelta
from pathlib import Path

import torch
from verl.trainer.config import CheckpointConfig
from verl.workers.config import (
    FSDPEngineConfig,
    FSDPOptimizerConfig,
    HFModelConfig,
    TrainingWorkerConfig,
)

from exp.optimize.claas.backends.verl.engine import ENGINE_MODEL_TYPE
from exp.optimize.claas.training_contracts import ClaasTrainingError, ClaasTrainingSpec


def require_worker_runtime() -> None:
    """Reject incompatible dependencies and ambiguous GPU placement before loading weights."""
    if importlib.metadata.version("verl") != "0.9.0":
        raise ClaasTrainingError(
            "this worker requires verl==0.9.0; install experiential[claas-verl]"
        )
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if not visible or "," in visible or visible == "-1":
        raise ClaasTrainingError("set CUDA_VISIBLE_DEVICES to exactly one authorized GPU")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise ClaasTrainingError("the CLaaS veRL worker requires exactly one visible CUDA GPU")
    if not torch.cuda.is_bf16_supported():
        raise ClaasTrainingError("the selected GPU must support BF16; no implicit dtype fallback")


def _validate_model_reference(identifier: str, revision: str) -> None:
    """Require immutable remote model and tokenizer revisions."""
    if Path(identifier).is_absolute() or Path(identifier).exists():
        raise ValueError("mutable local model/tokenizer directories are not revision-bound")
    if re.fullmatch(r"[0-9a-f]{40}", revision) is None:
        raise ValueError(
            "remote model and tokenizer revisions must be immutable 40-character commits"
        )


@contextmanager
def single_rank_process_group() -> Iterator[None]:
    """Own one finite, private distributed process group without starting a Ray cluster."""
    if torch.distributed.is_initialized():
        raise ClaasTrainingError(
            "run the CLaaS worker in its own process, outside an existing process group"
        )
    with tempfile.TemporaryDirectory(prefix="claas-verl-rendezvous-") as directory:
        previous = {
            name: os.environ.get(name)
            for name in (
                "WORLD_SIZE",
                "RANK",
                "LOCAL_WORLD_SIZE",
                "LOCAL_RANK",
                "MASTER_ADDR",
                "MASTER_PORT",
            )
        }
        os.environ.update(
            WORLD_SIZE="1",
            RANK="0",
            LOCAL_WORLD_SIZE="1",
            LOCAL_RANK="0",
            MASTER_ADDR="127.0.0.1",
            MASTER_PORT="0",
        )
        try:
            torch.cuda.set_device(0)
            torch.distributed.init_process_group(
                backend="cpu:gloo,cuda:nccl",
                rank=0,
                world_size=1,
                init_method=(Path(directory) / "store").as_uri(),
                timeout=timedelta(seconds=120),
            )
            yield
        finally:
            if torch.distributed.is_initialized():
                torch.distributed.destroy_process_group()
            for name, value in previous.items():
                if value is None:
                    os.environ.pop(name, None)
                else:
                    os.environ[name] = value


def worker_config(
    spec: ClaasTrainingSpec,
    model_path: Path,
    tokenizer_path: Path,
    *,
    teacher: bool,
) -> TrainingWorkerConfig:
    """Configure original-model LoRA and native state using veRL's typed public API."""
    model = HFModelConfig(
        path=str(model_path),
        tokenizer_path=str(tokenizer_path),
        trust_remote_code=False,
        use_remove_padding=False,
        use_fused_kernels=False,
        enable_gradient_checkpointing=True,
        lora_rank=spec.lora_rank,
        lora_alpha=spec.lora_alpha,
        target_modules=list(spec.target_modules),
        override_config={"attn_implementation": "sdpa"},
    )
    if model.hf_config.is_encoder_decoder or getattr(model.hf_config, "quantization_config", None):
        raise ValueError("CLaaS veRL requires an unquantized causal language model")
    if (
        str(model.hf_config.model_type).startswith("qwen3_5")
        and model.hf_config.model_type != "qwen3_5"
    ):
        raise ValueError("CLaaS requires the original dense Qwen3.5 wrapper checkpoint")
    contents = ["model"] if teacher else ["model", "optimizer", "extra"]
    return TrainingWorkerConfig(
        model_type=ENGINE_MODEL_TYPE,
        model_config=model,
        engine_config=FSDPEngineConfig(
            strategy="fsdp",
            forward_only=teacher,
            fsdp_size=1,
            use_orig_params=True,
            wrap_policy={"disable": True},
            model_dtype="bf16",
            dtype="bfloat16",
            mixed_precision={"param_dtype": "bf16", "reduce_dtype": "fp32", "buffer_dtype": "fp32"},
            use_torch_compile=False,
            use_dynamic_bsz=False,
            micro_batch_size_per_gpu=1,
            infer_micro_batch_size_per_gpu=1,
            use_remove_padding=False,
            seed=spec.seed,
        ),
        optimizer_config=FSDPOptimizerConfig(
            lr=spec.learning_rate,
            weight_decay=0.0,
            clip_grad=spec.max_gradient_norm,
            lr_scheduler_type="constant",
            total_training_steps=1,
        ),
        checkpoint_config=CheckpointConfig(
            save_contents=contents,
            load_contents=contents,
            save_lora_only=True,
        ),
    )
