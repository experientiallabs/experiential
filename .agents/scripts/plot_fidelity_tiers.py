#!/usr/bin/env python
"""Render the fidelity-tier ladder figure: low -> medium -> high -> max per benchmark.

Reads the per-suite JSONs under `.agents/docs/research/fidelity_tiers/` (base + `.rest` files
merge; `.rest` wins on overlap) and draws the house-style two-panel figure: (left) the ladder —
fidelity per tier, one line per benchmark, winner config annotated at each point; (right) what
each tier spent (serve-side USD, log scale).

    uv run --with matplotlib python .agents/scripts/plot_fidelity_tiers.py \
        --results .agents/docs/research/fidelity_tiers --out docs/research/fidelity_tiers
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

_INK = "#0a0a0a"
_MUTED = "#8a8a8a"
_GRID = "#ececec"
_BENCH_COLORS = {
    "tau-bench": "#0070f3",
    "terminal-tasks": "#7928ca",
    "swe-bench (healthy)": "#f5a623",
}
_TIERS = ["low", "medium", "high", "max"]


def _ladder(results: Path, stem: str) -> dict[str, dict[str, object]]:
    merged: dict[str, dict[str, object]] = {}
    for name in (f"{stem}.json", f"{stem}.rest.json"):
        path = results / name
        if path.exists():
            merged |= json.loads(path.read_text(encoding="utf-8"))
    return merged


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", default=".agents/docs/research/fidelity_tiers")
    parser.add_argument("--out", default="docs/research/fidelity_tiers")
    args = parser.parse_args()

    import matplotlib.pyplot as plt

    results = Path(args.results)
    data = {
        "tau-bench": _ladder(results, "tau-bench"),
        "terminal-tasks": _ladder(results, "terminal-tasks"),
        "swe-bench (healthy)": _ladder(results, "swe-bench"),
    }

    fig, (ax, axc) = plt.subplots(
        1, 2, figsize=(12.5, 4.6), gridspec_kw={"width_ratios": [1.6, 1.0]}
    )
    fig.patch.set_facecolor("white")
    xs = range(len(_TIERS))

    for bench, color in _BENCH_COLORS.items():
        ladder = data[bench]
        ys = [float(str(ladder[t]["fidelity"])) for t in _TIERS]
        costs = [float(str(ladder[t]["serve_cost_usd"])) for t in _TIERS]
        ax.plot(xs, ys, marker="o", markersize=5, linewidth=1.8, color=color, label=bench)
        for x, tier in zip(xs, _TIERS, strict=True):
            winner = str(ladder[tier]["winner"])
            note = "" if winner == "base" else winner.replace("reason+", "r+").replace("reason", "r")
            if note:
                ax.annotate(
                    note,
                    (x, ys[x]),
                    textcoords="offset points",
                    xytext=(0, 8),
                    ha="center",
                    fontsize=7,
                    color=color,
                )
        axc.plot(xs, costs, marker="o", markersize=5, linewidth=1.8, color=color)

    ax.set_title(
        "Fidelity tiers: monotone where headroom exists, saturating where it doesn't",
        loc="left",
        fontsize=12,
        color=_INK,
        pad=14,
    )
    ax.set_ylabel("open-loop reconstruction fidelity", fontsize=10, color=_INK)
    ax.legend(frameon=False, fontsize=9, loc="lower right")
    axc.set_title("serve-side spend per tier (USD, log)", loc="left", fontsize=11, color=_INK, pad=14)
    axc.set_yscale("log")
    axc.set_ylabel("USD", fontsize=10, color=_INK)

    for a in (ax, axc):
        a.set_facecolor("white")
        a.set_xticks(list(xs))
        a.set_xticklabels(_TIERS, fontsize=10, color=_INK)
        for side in ("top", "right"):
            a.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            a.spines[side].set_color(_GRID)
        a.tick_params(colors=_MUTED, labelsize=9)
        a.grid(axis="y", color=_GRID, linewidth=0.8, zorder=0)

    fig.tight_layout()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(f"{out}.png", dpi=200, bbox_inches="tight", facecolor="white")
    print(f"wrote {out}.png")


if __name__ == "__main__":
    main()
