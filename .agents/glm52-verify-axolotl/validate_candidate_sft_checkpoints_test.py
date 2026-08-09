"""Tests for candidate SFT checkpoint validation."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

from validate_candidate_sft_checkpoints import RUN_PREFIX, validate_checkpoint


def write_checkpoint(root: Path, *, seed: int = 1, step: int = 25) -> Path:
    checkpoint = root / f"{RUN_PREFIX}{seed}" / f"checkpoint-{step}"
    checkpoint.mkdir(parents=True)
    tensors = {}
    for index in range(200):
        tensors[f"layer.{index}.lora_A.weight"] = torch.ones(1)
        tensors[f"layer.{index}.lora_B.weight"] = torch.ones(1)
    save_file(tensors, checkpoint / "adapter_model.safetensors")
    (checkpoint / "trainer_state.json").write_text(json.dumps({"global_step": step}))
    (checkpoint / "adapter_config.json").write_text(
        json.dumps({"r": 64, "lora_alpha": 128})
    )
    return checkpoint


def test_validate_checkpoint_accepts_expected_adapter(tmp_path: Path) -> None:
    write_checkpoint(tmp_path)
    result = validate_checkpoint(tmp_path, 1, 25)
    assert result["tensor_count"] == 400
    assert result["all_finite"] is True
    assert result["all_nonzero"] is True


def test_validate_checkpoint_rejects_zero_tensor(tmp_path: Path) -> None:
    checkpoint = write_checkpoint(tmp_path)
    tensors = {}
    for index in range(200):
        tensors[f"layer.{index}.lora_A.weight"] = torch.ones(1)
        tensors[f"layer.{index}.lora_B.weight"] = torch.zeros(1) if index == 0 else torch.ones(1)
    save_file(tensors, checkpoint / "adapter_model.safetensors")
    with pytest.raises(ValueError, match="all-zero tensors"):
        validate_checkpoint(tmp_path, 1, 25)


def test_validate_checkpoint_rejects_wrong_step(tmp_path: Path) -> None:
    checkpoint = write_checkpoint(tmp_path)
    (checkpoint / "trainer_state.json").write_text(json.dumps({"global_step": 24}))
    with pytest.raises(ValueError, match="global_step"):
        validate_checkpoint(tmp_path, 1, 25)
