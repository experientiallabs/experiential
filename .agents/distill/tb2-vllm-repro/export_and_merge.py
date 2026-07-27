"""Take a Tinker LoRA checkpoint to a vLLM-servable HuggingFace model directory.

Downloads the adapter archive, then merges it into the base model. Merging is not an
optimization here, it is the only option: Tinker trains with ``train_unembed=True``, so the
adapter carries ``unembed_tokens`` LoRA that exports as ``lm_head``, and vLLM refuses LoRA on
embedding modules unless the model class declares a non-empty ``embedding_modules``. See the
README for the full reasoning.

The merge is CPU-only and shard-by-shard, so it needs disk (~2x the model) but little RAM.
Point ``--output`` at a large volume; the base model's own snapshot directory is not it.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def download(tinker_path: str, dest_parent: Path, python: str, force: bool) -> Path:
    """Download + extract a Tinker checkpoint; return the extracted directory."""
    dest_parent.mkdir(parents=True, exist_ok=True)
    before = set(dest_parent.iterdir())

    cmd = [
        python,
        "-m",
        "tinker.cli",
        "checkpoint",
        "download",
        tinker_path,
        "--output",
        str(dest_parent),
    ]
    if force:
        cmd.append("--force")
    print("+ " + " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)

    new = [p for p in dest_parent.iterdir() if p.is_dir() and p not in before]
    if len(new) != 1:
        candidates = [
            p
            for p in dest_parent.iterdir()
            if p.is_dir() and (p / "adapter_model.safetensors").exists()
        ]
        if len(candidates) != 1:
            raise RuntimeError(
                f"expected exactly one extracted checkpoint dir under {dest_parent}, "
                f"found new={new} with-adapter={candidates}"
            )
        return candidates[0]
    return new[0]


def merge(base_model: str, adapter_dir: Path, output: Path, trust_remote_code: bool) -> None:
    from tinker_cookbook.weights import build_hf_model

    if output.exists():
        raise SystemExit(f"--output already exists, refusing to overwrite: {output}")

    print(f"merging {adapter_dir} into {base_model} -> {output}", flush=True)
    build_hf_model(
        base_model=base_model,
        adapter_path=str(adapter_dir),
        output_path=str(output),
        merge_strategy="auto",
        trust_remote_code=trust_remote_code,
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--tinker-path", help="tinker://<uuid>:train:0/sampler_weights/<name>")
    src.add_argument("--adapter-dir", type=Path, help="already-downloaded adapter directory")
    ap.add_argument("--base-model", required=True, help="HF id or local snapshot path")
    ap.add_argument("--output", type=Path, required=True, help="merged model dir (must not exist)")
    ap.add_argument("--download-dir", type=Path, default=Path("./tinker-adapters"))
    ap.add_argument("--python", default=sys.executable, help="interpreter holding the tinker SDK")
    ap.add_argument("--force", action="store_true", help="re-download over an existing dir")
    ap.add_argument("--no-trust-remote-code", action="store_true")
    args = ap.parse_args()

    adapter = (
        args.adapter_dir
        if args.adapter_dir
        else download(args.tinker_path, args.download_dir, args.python, args.force)
    )

    weights = adapter / "adapter_model.safetensors"
    if not weights.exists():
        raise SystemExit(f"no adapter_model.safetensors in {adapter}")
    print(f"adapter: {adapter}  ({weights.stat().st_size / 1e6:.0f} MB)")

    merge(args.base_model, adapter, args.output, not args.no_trust_remote_code)

    print(f"\nmerged model written to {args.output}")
    print("next: verify_merge.py --base <base> --merged <output> --adapter <adapter>")
    print("      a merge that silently no-ops looks identical to a good one on disk")


if __name__ == "__main__":
    main()
