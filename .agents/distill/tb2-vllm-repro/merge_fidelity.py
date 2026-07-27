"""Quantify how much of a LoRA update actually survived being baked into bf16 weights.

`verify_merge.py` answers "did every targeted module move at all". That is necessary but weak:
a merge can touch every tensor and still lose most of the update to rounding. This answers the
quantitative version, and it matters whenever the adapter is small relative to bf16's ~0.4%
relative resolution -- which is the normal case for a few-step on-policy-distillation LoRA.

Two statistics, and the second is the one to trust:

* **element change fraction** -- what share of adapted elements differ from base. Intuitive but
  misleading on its own: a genuine small delta legitimately rounds away on most elements, so a
  low fraction is not evidence of failure. A sibling lane's >=40% gate was calibrated on a
  checkpoint trained 66-200 steps and does not transfer to a 4-step one.
* **delta norm ratio** ``||merged - base|| / ||(alpha/r) B A||`` -- what share of the update's
  magnitude landed. ~1.0 means the merge is faithful in aggregate whatever the element count
  says. This is the number that distinguishes "small update, correctly merged" from
  "update evaporated".

Also reported: cosine between intended and realized delta. Well below 1.0 means bf16 rounding
noise is comparable in scale to the update itself, so the merged model is not bit-equivalent to
runtime LoRA application. That is a real caveat to state, not necessarily a defect -- a merged
bf16 model is what actually gets deployed.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import torch
from safetensors import safe_open

# Adapter module -> merged tensor. Multiple sources mean the merge concatenated them.
FUSED = {
    "linear_attn.in_proj_qkv": [
        "linear_attn.in_proj_q",
        "linear_attn.in_proj_k",
        "linear_attn.in_proj_v",
    ]
}
RENAME = {"unembed_tokens": "lm_head"}


def _weight_map(root: Path) -> dict[str, str]:
    return json.loads((root / "model.safetensors.index.json").read_text())["weight_map"]


class Reader:
    def __init__(self) -> None:
        self._h: dict[tuple[Path, str], object] = {}

    def get(self, root: Path, wmap: dict[str, str], name: str) -> torch.Tensor:
        key = (root, wmap[name])
        if key not in self._h:
            self._h[key] = safe_open(root / wmap[name], framework="pt")
        return self._h[key].get_tensor(name)  # type: ignore[attr-defined]


def element_fraction(base: Path, merged: Path, adapted: list[str], r: Reader) -> float:
    bmap, mmap = _weight_map(base), _weight_map(merged)
    per_class: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    changed = total = 0
    unchanged: list[str] = []
    for name in adapted:
        a, b = r.get(base, bmap, name), r.get(merged, mmap, name)
        nch, ne = int((a != b).sum()), a.numel()
        changed += nch
        total += ne
        cls = ".".join(p for p in name.split(".") if not p.isdigit())
        per_class[cls][0] += nch
        per_class[cls][1] += ne
        if nch == 0:
            unchanged.append(name)

    print(f"\n{'class':58s} {'elements changed':>17s}")
    print("-" * 78)
    for cls, (c, e) in sorted(per_class.items()):
        print(f"{cls:58s} {100.0 * c / e:>16.2f}%")
    print("-" * 78)
    print(f"{'OVERALL':58s} {100.0 * changed / total:>16.2f}%")
    if unchanged:
        print(f"\nFAIL: {len(unchanged)} adapted tensor(s) entirely unchanged: {unchanged[:8]}")
    return 100.0 * changed / total


def norm_ratios(
    base: Path, merged: Path, adapter: Path, scaling: float, layers: list[int], r: Reader
) -> list[float]:
    bmap, mmap = _weight_map(base), _weight_map(merged)
    ad = safe_open(adapter / "adapter_model.safetensors", framework="pt")
    keys = set(ad.keys())

    def delta(module: str) -> torch.Tensor:
        a = ad.get_tensor(f"base_model.model.{module}.lora_A.weight").double()
        b = ad.get_tensor(f"base_model.model.{module}.lora_B.weight").double()
        return scaling * (b @ a)

    cases: list[tuple[str, str, list[str]]] = []
    for li in layers:
        ap, mp = f"model.layers.{li}", f"model.language_model.layers.{li}"
        for suffix in (
            "linear_attn.in_proj_qkv",
            "linear_attn.in_proj_z",
            "mlp.gate_proj",
            "mlp.down_proj",
            "self_attn.q_proj",
        ):
            srcs = FUSED.get(suffix, [suffix])
            cases.append((f"L{li} {suffix}", f"{mp}.{suffix}.weight", [f"{ap}.{s}" for s in srcs]))
    cases.append(("lm_head", "lm_head.weight", ["model.unembed_tokens"]))

    print(f"\n{'tensor':40s} {'norm ratio':>12s} {'cosine':>9s} {'||d||/||W||':>13s}")
    print("-" * 78)
    ratios: list[float] = []
    for label, tensor, modules in cases:
        if tensor not in bmap or tensor not in mmap:
            continue
        if not all(f"base_model.model.{m}.lora_A.weight" in keys for m in modules):
            continue
        w = r.get(base, bmap, tensor).double()
        realized = r.get(merged, mmap, tensor).double() - w
        parts = [delta(m) for m in modules]
        intended = torch.cat(parts, dim=0) if len(parts) > 1 else parts[0]
        if intended.shape != realized.shape:
            print(f"{label:40s} shape {tuple(intended.shape)} vs {tuple(realized.shape)}")
            continue
        ni = intended.norm().item()
        ratio = realized.norm().item() / ni if ni else float("nan")
        # float64 throughout: an fp32 dot over ~1e9 elements accumulates enough error to
        # report a cosine above 1.
        cos = float(
            torch.dot(intended.flatten(), realized.flatten()) / (intended.norm() * realized.norm())
        )
        ratios.append(ratio)
        print(f"{label:40s} {ratio:>12.4f} {cos:>9.4f} {ni / w.norm().item():>13.2e}")
    return ratios


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base", type=Path, required=True)
    ap.add_argument("--merged", type=Path, required=True)
    ap.add_argument("--adapter", type=Path, required=True)
    ap.add_argument("--layers", type=int, nargs="*", default=[0, 11, 23])
    ap.add_argument(
        "--min-norm-ratio",
        type=float,
        default=0.80,
        help="fail below this; the update's magnitude did not land",
    )
    args = ap.parse_args()

    cfg = json.loads((args.adapter / "adapter_config.json").read_text())
    scaling = cfg["lora_alpha"] / cfg["r"]
    print(f"adapter r={cfg['r']} alpha={cfg['lora_alpha']} scaling={scaling:g}")

    with safe_open(args.adapter / "adapter_model.safetensors", framework="pt") as f:
        zero_b = sum(
            1 for k in f.keys() if ".lora_B." in k and f.get_tensor(k).abs().max().item() == 0
        )
        n_b = sum(1 for k in f.keys() if ".lora_B." in k)
    print(f"lora_B matrices: {n_b}, exactly zero: {zero_b}")

    bmap, mmap = _weight_map(args.base), _weight_map(args.merged)
    adapted = sorted(
        n
        for n in bmap
        if n in mmap
        and (
            n == "lm_head.weight"
            or (
                "model.language_model.layers." in n
                and n.endswith(".weight")
                and any(
                    s in n
                    for s in (
                        "linear_attn.in_proj_qkv",
                        "linear_attn.in_proj_z",
                        "linear_attn.out_proj",
                        "mlp.down_proj",
                        "mlp.gate_proj",
                        "mlp.up_proj",
                        "self_attn.k_proj",
                        "self_attn.o_proj",
                        "self_attn.q_proj",
                        "self_attn.v_proj",
                    )
                )
            )
        )
    )
    print(f"adapted tensors: {len(adapted)}")

    reader = Reader()
    element_fraction(args.base, args.merged, adapted, reader)
    ratios = norm_ratios(args.base, args.merged, args.adapter, scaling, args.layers, reader)

    if ratios:
        mean = sum(ratios) / len(ratios)
        print("-" * 78)
        print(f"mean norm ratio {mean:.4f}  (min {min(ratios):.4f}, max {max(ratios):.4f})")
        verdict = "PASS" if mean >= args.min_norm_ratio and zero_b == 0 else "FAIL"
        print(f"\nMERGE FIDELITY: {verdict}  (threshold {args.min_norm_ratio})")
        print(
            "Judge on the norm ratio, not the element fraction -- a small update correctly "
            "merged\nlegitimately leaves most individual elements at their base value."
        )


if __name__ == "__main__":
    main()
