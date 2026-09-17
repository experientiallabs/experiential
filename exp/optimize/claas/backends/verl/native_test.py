"""Upstream worker configuration contracts and explicit opt-in CUDA training proof.

CPU tests do not claim to execute veRL's CUDA-only FSDP optimizer. The separate
GPU test runs real worker initialization, updates, native resume and PEFT export.
"""

from pathlib import Path

import pytest
import torch
from safetensors.torch import load_file
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import (
    AutoTokenizer,
    LlamaConfig,
    LlamaForCausalLM,
    PreTrainedTokenizerFast,
    Qwen3_5Config,
    Qwen3_5ForConditionalGeneration,
)
from verl.models.transformers.monkey_patch import apply_monkey_patch
from verl.workers.engine import BaseEngine, FSDPEngineWithLMHead

from exp.optimize.claas.backends.verl.engine import ClaasFeedbackEngine
from exp.optimize.claas.backends.verl.native import (
    _validate_model_reference,
    require_worker_runtime,
    worker_config,
)
from exp.optimize.claas.training_contracts import ClaasTrainingError
from exp.optimize.claas.training_contracts_test import job


def tokenizer() -> PreTrainedTokenizerFast:
    """Create a local eight-token tokenizer that cannot download model assets."""
    backend = Tokenizer(
        WordLevel(
            {"[UNK]": 0, "a": 1, "b": 2, "c": 3, "d": 4, "e": 5, "f": 6, "g": 7},
            unk_token="[UNK]",
        )
    )
    backend.pre_tokenizer = Whitespace()
    return PreTrainedTokenizerFast(
        tokenizer_object=backend, unk_token="[UNK]", pad_token="[UNK]", bos_token="a", eos_token="g"
    )


def tiny_snapshot(path: Path) -> Path:
    """Save deterministic tiny full-model and tokenizer fixtures without a download."""
    torch.manual_seed(1)
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
            "pad_token_id": 0,
            "attention_dropout": 0.0,
        }
    )
    LlamaForCausalLM(config).save_pretrained(path)
    tokenizer().save_pretrained(path)
    return path


def test_worker_config_selects_native_model_optimizer_and_checkpoint_ownership(
    tmp_path: Path,
) -> None:
    """Exercise actual upstream typed configuration and inherited execution methods."""
    snapshot = tiny_snapshot(tmp_path / "base")
    training = job(tmp_path / "results")
    training = training.model_copy(
        update={"spec": training.spec.model_copy(update={"target_modules": ("q_proj", "v_proj")})}
    )
    actor = worker_config(training.spec, snapshot, snapshot, teacher=False)
    teacher = worker_config(training.spec, snapshot, snapshot, teacher=True)
    assert actor.model_config.local_path == str(snapshot)
    assert actor.model_config.lora_rank == training.spec.lora_rank
    assert actor.optimizer_config.lr == training.spec.learning_rate
    assert actor.checkpoint_config.save_contents == ["model", "optimizer", "extra"]
    assert actor.checkpoint_config.load_contents == ["model", "optimizer", "extra"]
    assert actor.checkpoint_config.save_lora_only
    assert teacher.engine_config.forward_only
    assert teacher.checkpoint_config.save_contents == ["model"]
    assert ClaasFeedbackEngine.initialize is FSDPEngineWithLMHead.initialize
    assert ClaasFeedbackEngine.train_batch is BaseEngine.train_batch
    assert ClaasFeedbackEngine.forward_backward_batch is FSDPEngineWithLMHead.forward_backward_batch
    assert ClaasFeedbackEngine.optimizer_step is FSDPEngineWithLMHead.optimizer_step
    assert ClaasFeedbackEngine.save_checkpoint is FSDPEngineWithLMHead.save_checkpoint
    assert ClaasFeedbackEngine.load_checkpoint is FSDPEngineWithLMHead.load_checkpoint


def test_tiny_fixture_supports_upstream_patching_and_persists_token_ids(tmp_path: Path) -> None:
    """Exercise upstream model patching and preserve exact fixture token identities."""
    snapshot = tiny_snapshot(tmp_path / "base")
    model = LlamaForCausalLM.from_pretrained(snapshot, attn_implementation="sdpa")
    apply_monkey_patch(model, use_remove_padding=False, use_fused_kernels=False)
    restored = AutoTokenizer.from_pretrained(snapshot)
    assert isinstance(restored, PreTrainedTokenizerFast)
    assert model.config.hidden_size // model.config.num_attention_heads == 64
    assert model.config.num_key_value_heads == 2
    assert (restored.bos_token_id, restored.eos_token_id, restored.pad_token_id) == (1, 7, 0)
    assert restored.encode("a b", add_special_tokens=False) == [1, 2]


def test_gpu_placement_is_explicit(monkeypatch: pytest.MonkeyPatch) -> None:
    """A CPU host or multiple visible GPUs cannot silently become a training run."""
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    with pytest.raises(ClaasTrainingError, match="exactly one authorized"):
        require_worker_runtime()
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1")
    with pytest.raises(ClaasTrainingError, match="exactly one authorized"):
        require_worker_runtime()
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(ClaasTrainingError, match="visible CUDA"):
        require_worker_runtime()


def test_mutable_local_model_references_are_not_treated_as_pinned(tmp_path: Path) -> None:
    """A revision string cannot freeze model or tokenizer files in a mutable directory."""
    with pytest.raises(ValueError, match="not revision-bound"):
        _validate_model_reference(str(tmp_path), "a" * 40)
    with pytest.raises(ValueError, match="immutable"):
        _validate_model_reference("owner/model", "main")
    _validate_model_reference("owner/model", "a" * 40)


def tiny_qwen() -> Qwen3_5ForConditionalGeneration:
    """Create the actual hybrid text architecture plus tiny unused vision weights locally."""
    config = Qwen3_5Config.from_dict(
        {
            "text_config": {
                "vocab_size": 8,
                "hidden_size": 32,
                "intermediate_size": 48,
                "num_hidden_layers": 2,
                "num_attention_heads": 2,
                "num_key_value_heads": 1,
                "head_dim": 16,
                "max_position_embeddings": 128,
                "tie_word_embeddings": True,
                "linear_conv_kernel_dim": 4,
                "linear_key_head_dim": 8,
                "linear_value_head_dim": 8,
                "linear_num_key_heads": 2,
                "linear_num_value_heads": 2,
                "layer_types": ["linear_attention", "full_attention"],
                "rope_parameters": {
                    "rope_type": "default",
                    "rope_theta": 10000,
                    "partial_rotary_factor": 0.5,
                    "mrope_section": [1, 1, 2],
                },
            },
            "vision_config": {
                "depth": 1,
                "hidden_size": 16,
                "intermediate_size": 32,
                "num_heads": 2,
                "out_hidden_size": 32,
                "num_position_embeddings": 16,
                "patch_size": 2,
                "spatial_merge_size": 1,
                "temporal_patch_size": 1,
            },
            "tie_word_embeddings": True,
        }
    )
    config._attn_implementation = "eager"
    return Qwen3_5ForConditionalGeneration(config)


def test_upstream_lora_construction_preserves_original_qwen_wrapper_names(tmp_path: Path) -> None:
    """Run upstream LoRA construction on CPU without claiming a CUDA optimizer update."""
    model = tiny_qwen()
    snapshot = tmp_path / "qwen"
    model.save_pretrained(snapshot)
    tokenizer().save_pretrained(snapshot)
    training = job(tmp_path / "results")
    training = training.model_copy(
        update={"spec": training.spec.model_copy(update={"target_modules": ("q_proj", "v_proj")})}
    )
    config = worker_config(training.spec, snapshot, snapshot, teacher=False)
    engine = object.__new__(ClaasFeedbackEngine)
    engine.model_config = config.model_config
    adapted = engine._build_lora_module(model)
    exported = tmp_path / "adapter"
    adapted.save_pretrained(exported, safe_serialization=True)
    weights = load_file(str(exported / "adapter_model.safetensors"))
    assert weights
    assert all(name.startswith("base_model.model.model.language_model.") for name in weights)
    assert all("lora_" in name for name in weights)
