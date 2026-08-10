#!/usr/bin/env python3
"""Validate candidate SFT checkpoint structure and all LoRA tensor values."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch
from safetensors import safe_open

RUN_PREFIX = "qwen35-4b-glm52-candidate-realverified-sft-lr1e5-r64-seed"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_checkpoint(
    root: Path,
    seed: int,
    step: int,
    *,
    run_prefix: str = RUN_PREFIX,
) -> dict[str, object]:
    checkpoint = root / f"{run_prefix}{seed}" / f"checkpoint-{step}"
    model_path = checkpoint / "adapter_model.safetensors"
    state_path = checkpoint / "trainer_state.json"
    config_path = checkpoint / "adapter_config.json"
    for path in (model_path, state_path, config_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    state = json.loads(state_path.read_text())
    if int(state["global_step"]) != step:
        raise ValueError(f"{state_path}: global_step={state['global_step']} != {step}")
    config = json.loads(config_path.read_text())
    if int(config["r"]) != 64 or int(config["lora_alpha"]) != 128:
        raise ValueError(f"{config_path}: unexpected LoRA rank or alpha")

    tensor_count = 0
    lora_a = 0
    lora_b = 0
    zero_tensors: list[str] = []
    nonfinite_tensors: list[str] = []
    with safe_open(model_path, framework="pt", device="cpu") as handle:
        for name in handle.keys():
            tensor = handle.get_tensor(name)
            tensor_count += 1
            lora_a += "lora_A" in name
            lora_b += "lora_B" in name
            if not bool(torch.isfinite(tensor).all()):
                nonfinite_tensors.append(name)
            if int(torch.count_nonzero(tensor)) == 0:
                zero_tensors.append(name)

    if (tensor_count, lora_a, lora_b) != (400, 200, 200):
        raise ValueError(
            f"{model_path}: expected 400/200/200 tensors, found "
            f"{tensor_count}/{lora_a}/{lora_b}"
        )
    if nonfinite_tensors:
        raise ValueError(f"{model_path}: nonfinite tensors: {nonfinite_tensors}")
    if zero_tensors:
        raise ValueError(f"{model_path}: all-zero tensors: {zero_tensors}")

    return {
        "seed": seed,
        "step": step,
        "checkpoint": str(checkpoint),
        "adapter_sha256": sha256(model_path),
        "tensor_count": tensor_count,
        "lora_a_tensors": lora_a,
        "lora_b_tensors": lora_b,
        "all_finite": True,
        "all_nonzero": True,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--seeds", nargs="+", type=int, default=[20260809, 20260810])
    parser.add_argument("--steps", nargs="+", type=int, default=[100, 200])
    parser.add_argument("--run-prefix", default=RUN_PREFIX)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()

    records = [
        validate_checkpoint(args.root, seed, step, run_prefix=args.run_prefix)
        for seed in args.seeds
        for step in args.steps
    ]
    payload = {
        "schema": "candidate-realverified-sft-checkpoint-validation-v1",
        "records": records,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))  # noqa: T201
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
