#!/usr/bin/env python
"""Render the optimized-vs-unoptimized RAG trace-scaling comparison: one panel per benchmark.

Each panel overlays two curves from `run_trace_scaling.py` `AblationReport` JSONs — an *unoptimized*
baseline (muted, dashed) and an *optimized* config (benchmark accent, solid) — as reconstruction
fidelity vs. number of training traces (log x). A `base@0` point (no-RAG) is drawn in a "0" slot
left of 10^0 with a dotted connector to the first RAG point. Brand system (rule 15): white bg,
near-black ink, hairline grid, left-aligned titles. Writes PNG + SVG. matplotlib is ephemeral:

    uv run --with matplotlib python .agents/scripts/plot_rag_compare.py \
        --panel tau-bench=base_tau.json,opt_tau.json \
        --panel terminal-tasks=base_term.json,opt_term.json \
        --panel swe-bench=base_swe.json,opt_swe.json \
        --out docs/research/figures/rag_optimization --ymin 0.6

Each --panel is `label=base_path,opt_path`.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np

_INK = "#0a0a0a"
_MUTED = "#8a8a8a"
_GRID = "#ececec"
_BASE_COLOR = "#b0b0b0"  # unoptimized: muted grey
_ACCENTS = ["#0070f3", "#7928ca", "#f5a623"]  # optimized: blue, purple, amber (per panel)
_ZERO_X = 0.4
_POINT_RE = re.compile(r"^(?P<mode>base|gepa)@(?P<n>\d+)$")


def _series(path: str, mode: str = "base") -> tuple[list[int], list[float], list[float]]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    pts: list[tuple[int, float, float]] = []
    for cell in data.get("conditions", []):
        m = _POINT_RE.match(cell.get("condition", {}).get("label", ""))
        if not m or m.group("mode") != mode:
            continue
        pts.append((int(m.group("n")), float(cell.get("mean", 0.0)), float(cell.get("std", 0.0))))
    pts.sort(key=lambda p: p[0])
    return [p[0] for p in pts], [p[1] for p in pts], [p[2] for p in pts]


def _draw(ax, path: str, color: str, label: str, dashed: bool) -> int:  # noqa: ANN001
    """Draw one report's curve on `ax`; returns the max RAG count (for tick layout)."""
    counts, means, stds = _series(path)
    pts = list(zip(counts, means, stds, strict=True))
    zero = next((p for p in pts if p[0] == 0), None)
    rest = [p for p in pts if p[0] >= 1]
    style = "--" if dashed else "-"
    maxc = 1
    if rest:
        rc, rm, rs = zip(*rest, strict=True)
        ax.plot(rc, rm, style, marker="o", color=color, label=label, linewidth=2.2, markersize=5,
                markerfacecolor="white", markeredgecolor=color, markeredgewidth=1.6, zorder=3)
        lo = [m - s for m, s in zip(rm, rs, strict=True)]
        hi = [m + s for m, s in zip(rm, rs, strict=True)]
        ax.fill_between(rc, lo, hi, color=color, alpha=0.10, linewidth=0, zorder=2)
        maxc = max(rc)
    if zero is not None:
        ax.plot([_ZERO_X], [zero[1]], "o", color=color, markersize=5, markerfacecolor="white",
                markeredgecolor=color, markeredgewidth=1.6, zorder=3)
        if rest:
            ax.plot([_ZERO_X, rest[0][0]], [zero[1], rest[0][1]], ":", color=color, linewidth=1.3,
                    alpha=0.6, zorder=2)
    return maxc


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel", action="append", default=[], metavar="LABEL=BASE,OPT",
                        help="A benchmark panel as label=base_path,opt_path (repeatable).")
    parser.add_argument("--out", default="rag_optimization", help="Output stem (.png + .svg).")
    parser.add_argument("--title", default="RAG optimization — trace scaling law")
    parser.add_argument("--ymin", type=float, default=0.6)
    parser.add_argument("--ymax", type=float, default=1.0)
    args = parser.parse_args()

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FixedFormatter, FixedLocator

    panels = []
    for spec in args.panel:
        label, paths = spec.split("=", 1)
        base_path, opt_path = paths.split(",", 1)
        panels.append((label, base_path, opt_path))
    if not panels:
        raise SystemExit("pass at least one --panel label=base,opt")

    fig, axes = plt.subplots(1, len(panels), figsize=(5.2 * len(panels), 4.6), dpi=200, sharey=True)
    if len(panels) == 1:
        axes = [axes]
    fig.patch.set_facecolor("white")

    for i, (label, base_path, opt_path) in enumerate(panels):
        ax = axes[i]
        ax.set_facecolor("white")
        accent = _ACCENTS[i % len(_ACCENTS)]
        maxc = max(
            _draw(ax, base_path, _BASE_COLOR, "unoptimized (hashing, k=5)", dashed=True),
            _draw(ax, opt_path, accent, "optimized (hashing, k=20, obs-cap)", dashed=False),
        )
        ax.set_xscale("log")
        decades = [10**k for k in range(0, int(np.floor(np.log10(maxc))) + 1)]
        ax.xaxis.set_major_locator(FixedLocator([_ZERO_X, *decades]))
        ax.xaxis.set_major_formatter(FixedFormatter(["0", *[str(d) for d in decades]]))
        ax.xaxis.set_minor_locator(FixedLocator([]))
        ax.set_xlim(_ZERO_X * 0.75, maxc * 1.4)
        ax.set_ylim(args.ymin, args.ymax)
        ax.set_title(label, fontsize=13, color=_INK, fontweight="bold", loc="left", pad=10)
        ax.set_xlabel("training traces", fontsize=10, color=_INK)
        if i == 0:
            ax.set_ylabel("reconstruction fidelity", fontsize=11, color=_INK)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(_GRID)
        ax.grid(axis="y", color=_GRID, linewidth=1)
        ax.set_axisbelow(True)
        ax.tick_params(colors=_MUTED, labelsize=9, length=0)
        if i == 0:
            leg = ax.legend(frameon=False, fontsize=9, loc="lower right")
            for t in leg.get_texts():
                t.set_color(_INK)

    fig.suptitle(args.title, fontsize=15, color=_INK, fontweight="bold", x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    png, svg = f"{args.out}.png", f"{args.out}.svg"
    fig.savefig(png, bbox_inches="tight", facecolor="white")
    fig.savefig(svg, bbox_inches="tight", facecolor="white")
    print(f"wrote {png} and {svg} ({len(panels)} panels)")


if __name__ == "__main__":
    main()
