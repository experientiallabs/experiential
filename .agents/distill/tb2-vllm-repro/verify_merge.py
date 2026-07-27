"""Assert a Tinker LoRA merge actually landed on every module the adapter targets.

A merge that silently no-ops produces a model directory that looks perfect: right size, right
shard count, loads fine in vLLM. Every downstream number is then a duplicate of the base arm,
and nothing about the run says so. This compares the set of base tensors that changed against
the set the adapter targets, so a partial merge fails loudly instead of quietly.

Two remaps have to be applied before the sets can be compared:

* ``unembed_tokens`` in the adapter is ``lm_head`` in the model.
* ``linear_attn.in_proj_{q,k,v}`` are three separate adapter modules that merge into the single
  fused ``linear_attn.in_proj_qkv`` weight.

Exits non-zero on any adapter module whose target did not move.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

from safetensors import safe_open

# Adapter module suffix -> base tensor suffix it merges into.
SUFFIX_REMAP = {
    "unembed_tokens": "lm_head",
    "linear_attn.in_proj_q": "linear_attn.in_proj_qkv",
    "linear_attn.in_proj_k": "linear_attn.in_proj_qkv",
    "linear_attn.in_proj_v": "linear_attn.in_proj_qkv",
}

_ADAPTER_PREFIX = re.compile(r"^base_model\.model\.")
_LORA_SUFFIX = re.compile(r"\.lora_[AB]\.weight$")
_PARAM_SUFFIX = re.compile(r"\.(weight|bias)$")
# Container segments that differ between adapter naming and model naming. A VL checkpoint
# nests the text stack under `language_model`, which the adapter names never carry.
_CONTAINERS = ("model", "language_model")


def _strip_layer_index(name: str) -> str:
    return ".".join(p for p in name.split(".") if not p.isdigit())


def _canonical(name: str) -> str:
    """Drop leading container segments so adapter and model naming can be compared."""
    parts = _strip_layer_index(name).split(".")
    while parts and parts[0] in _CONTAINERS:
        parts.pop(0)
    return ".".join(parts)


def model_class(tensor_name: str) -> str:
    return _canonical(_PARAM_SUFFIX.sub("", tensor_name))


def adapter_targets(adapter_dir: Path) -> set[str]:
    """Module classes the adapter touches, in canonical (container-stripped) naming."""
    with safe_open(adapter_dir / "adapter_model.safetensors", framework="pt") as f:
        keys = list(f.keys())

    targets: set[str] = set()
    for key in keys:
        name = _LORA_SUFFIX.sub("", _ADAPTER_PREFIX.sub("", key))
        name = _strip_layer_index(name)
        for src, dst in SUFFIX_REMAP.items():
            if name.endswith(src):
                name = name[: -len(src)] + dst
                break
        targets.add(_canonical(name))
    return targets


def _weight_map(root: Path) -> dict[str, str]:
    index = root / "model.safetensors.index.json"
    if index.exists():
        return json.loads(index.read_text())["weight_map"]
    single = "model.safetensors"
    with safe_open(root / single, framework="pt") as f:
        return {k: single for k in f.keys()}


def changed_classes(base: Path, merged: Path, per_class: int) -> tuple[set[str], set[str]]:
    """Return (classes sampled, classes where at least one sampled tensor moved)."""
    bmap, mmap = _weight_map(base), _weight_map(merged)
    shared = sorted(set(bmap) & set(mmap))

    groups: dict[str, list[str]] = defaultdict(list)
    for name in shared:
        groups[model_class(name)].append(name)

    handles: dict[tuple[Path, str], object] = {}

    def tensor(root: Path, wmap: dict[str, str], name: str):
        key = (root, wmap[name])
        if key not in handles:
            handles[key] = safe_open(root / wmap[name], framework="pt")
        return handles[key].get_tensor(name)  # type: ignore[attr-defined]

    sampled: set[str] = set()
    moved: set[str] = set()
    for cls, names in sorted(groups.items()):
        names.sort()
        # First / middle / last layer is enough to catch a layer-range miss.
        picks = list(dict.fromkeys([names[0], names[len(names) // 2], names[-1]]))[:per_class]
        sampled.add(cls)
        for name in picks:
            a = tensor(base, bmap, name).float()
            b = tensor(merged, mmap, name).float()
            if a.shape != b.shape:
                print(f"  SHAPE MISMATCH {name}: {tuple(a.shape)} vs {tuple(b.shape)}")
                continue
            if (a - b).abs().max().item() > 0:
                moved.add(cls)
                break
    return sampled, moved


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base", type=Path, required=True, help="base HF model dir")
    ap.add_argument("--merged", type=Path, required=True, help="merged HF model dir")
    ap.add_argument("--adapter", type=Path, required=True, help="raw Tinker adapter dir")
    ap.add_argument("--per-class", type=int, default=3, help="tensors sampled per module class")
    args = ap.parse_args()

    targets = adapter_targets(args.adapter)
    sampled, moved = changed_classes(args.base, args.merged, args.per_class)

    unknown = sorted(t for t in targets if t not in sampled)
    missed = sorted(t for t in targets if t in sampled and t not in moved)
    spurious = sorted(moved - targets)

    print(f"adapter targets      : {len(targets)}")
    print(f"changed in merge     : {len(moved)}")
    for t in sorted(targets):
        state = "ok" if t in moved else ("NOT IN MODEL" if t in unknown else "UNCHANGED")
        print(f"  [{state:12s}] {t}")

    ok = True
    if missed:
        print(f"\nFAIL: {len(missed)} targeted module(s) did not change: {missed}")
        ok = False
    if unknown:
        print(f"\nFAIL: {len(unknown)} adapter target(s) absent from the model: {unknown}")
        print("      the remap table is probably incomplete for this architecture")
        ok = False
    if spurious:
        # Not fatal, but it means the merge touched something the adapter did not target.
        print(f"\nWARN: {len(spurious)} untargeted module(s) changed: {spurious}")

    print("\nPASS: merge covers every adapter target" if ok else "\nMERGE IS NOT TRUSTWORTHY")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
