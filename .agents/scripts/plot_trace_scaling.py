#!/usr/bin/env python
"""Render the trace scaling-law figure: RAG fidelity vs. number of training traces, all benchmarks.

Reads one or more `AblationReport` JSONs (produced by `run_trace_scaling.py`) — one per benchmark —
and draws a single clean, Notion/Vercel-style line chart: x = number of training traces (log), y =
open-loop reconstruction fidelity, one line per benchmark with a ±std band across seeds. Writes both
PNG and SVG.

matplotlib is not a project dependency (this is a one-off research figure), so run it ephemerally:

    uv run --with matplotlib python scripts/plot_trace_scaling.py \
        --report tau-bench=tau.json --report terminal-tasks=term.json --report swe-bench=swe.json \
        --out docs/trace_scaling

Each --report is `label=path`. A report's conditions are the sweep points (label `base@N` or
`gepa@N`); by default only `base` (RAG-only) points are plotted — pass --mode gepa for that curve.
A `base@0` condition (the no-RAG baseline, run with `run_trace_scaling.py --no-rag`) is drawn in a
fixed "0" slot left of 10^0 with a dotted connector to the first RAG point (log(0) is undefined).
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np

# Vercel/Notion-ish palette: near-black text, restrained accent lines, generous whitespace.
_INK = "#0a0a0a"
_MUTED = "#8a8a8a"
_GRID = "#ececec"
_COLORS = ["#0070f3", "#7928ca", "#f5a623", "#e00", "#50e3c2"]  # blue, purple, amber, red, teal

# Fixed log-axis slot for the n=0 (no-RAG) baseline, sitting just left of 10^0.
_ZERO_X = 0.4

_POINT_RE = re.compile(r"^(?P<mode>base|gepa)@(?P<n>\d+)$")


def _series(report_path: str, mode: str) -> tuple[list[int], list[float], list[float]]:
    """Extract (counts, means, stds) for `mode` from an AblationReport JSON, sorted by count."""
    data = json.loads(Path(report_path).read_text(encoding="utf-8"))
    points: list[tuple[int, float, float]] = []
    for cell in data.get("conditions", []):
        label = cell.get("condition", {}).get("label", "")
        m = _POINT_RE.match(label)
        if not m or m.group("mode") != mode:
            continue
        n = int(m.group("n"))
        points.append((n, float(cell.get("mean", 0.0)), float(cell.get("std", 0.0))))
    points.sort(key=lambda p: p[0])
    counts = [p[0] for p in points]
    means = [p[1] for p in points]
    stds = [p[2] for p in points]
    return counts, means, stds


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--report",
        action="append",
        default=[],
        metavar="LABEL=PATH",
        help="A benchmark's report as label=path (repeatable).",
    )
    parser.add_argument("--mode", default="base", help="Which curve to plot: base | gepa.")
    parser.add_argument("--out", default="trace_scaling", help="Output path stem (.png + .svg).")
    parser.add_argument("--title", default="Trace scaling law", help="Figure title.")
    parser.add_argument("--ymin", type=float, default=0.6, help="Y-axis floor (fidelity is 0..1).")
    parser.add_argument("--ymax", type=float, default=1.0, help="Y-axis ceiling.")
    args = parser.parse_args()

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 5), dpi=200)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    plotted = 0
    any_zero = False
    max_count = 1
    for i, spec in enumerate(args.report):
        if "=" not in spec:
            raise SystemExit(f"--report must be label=path, got {spec!r}")
        label, path = spec.split("=", 1)
        counts, means, stds = _series(path, args.mode)
        if not counts:
            print(f"warning: no {args.mode} points in {path} ({label})")
            continue
        color = _COLORS[i % len(_COLORS)]
        # n=0 is the no-RAG baseline (log(0) is undefined), so pin it at a fixed slot left of 10^0
        # and connect it to the first RAG point with a dotted line to mark the retrieval "jump".
        pts = list(zip(counts, means, stds, strict=True))
        zero = next((p for p in pts if p[0] == 0), None)
        rest = [p for p in pts if p[0] >= 1]
        if rest:
            rc, rm, rs = zip(*rest, strict=True)
            ax.plot(rc, rm, "-o", color=color, label=label, linewidth=2.2, markersize=5,
                    markerfacecolor="white", markeredgecolor=color, markeredgewidth=1.6, zorder=3)
            lo = [m - s for m, s in zip(rm, rs, strict=True)]
            hi = [m + s for m, s in zip(rm, rs, strict=True)]
            ax.fill_between(rc, lo, hi, color=color, alpha=0.10, linewidth=0, zorder=2)
            max_count = max(max_count, max(rc))
        if zero is not None:
            any_zero = True
            zlabel = label if not rest else None  # keep one legend entry per series
            ax.plot([_ZERO_X], [zero[1]], "o", color=color, label=zlabel, markersize=5,
                    markerfacecolor="white", markeredgecolor=color, markeredgewidth=1.6, zorder=3)
            if rest:
                ax.plot([_ZERO_X, rest[0][0]], [zero[1], rest[0][1]], ":", color=color,
                        linewidth=1.4, alpha=0.7, zorder=2)
        plotted += 1

    if not plotted:
        raise SystemExit("no series plotted — check --report paths and --mode")

    ax.set_xscale("log")
    if any_zero:
        # Explicit ticks: a "0" slot for the no-RAG baseline, then decade ticks for the RAG sweep.
        from matplotlib.ticker import FixedFormatter, FixedLocator

        decades = [10**k for k in range(0, int(np.floor(np.log10(max_count))) + 1)]
        ax.xaxis.set_major_locator(FixedLocator([_ZERO_X, *decades]))
        ax.xaxis.set_major_formatter(FixedFormatter(["0", *[str(d) for d in decades]]))
        ax.xaxis.set_minor_locator(FixedLocator([]))
        ax.set_xlim(_ZERO_X * 0.75, max_count * 1.4)
    ax.set_xlabel("training traces", fontsize=11, color=_INK)
    ylabel = "reconstruction fidelity" + ("" if args.mode == "base" else f" ({args.mode})")
    ax.set_ylabel(ylabel, fontsize=11, color=_INK)
    ax.set_title(args.title, fontsize=15, color=_INK, fontweight="bold", loc="left", pad=14)
    ax.set_ylim(args.ymin, args.ymax)

    # Minimal chrome: no top/right spine, soft horizontal grid only, muted ticks.
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(_GRID)
    ax.grid(axis="y", color=_GRID, linewidth=1)
    ax.set_axisbelow(True)
    ax.tick_params(colors=_MUTED, labelsize=10, length=0)
    leg = ax.legend(frameon=False, fontsize=10, loc="lower right")
    for text in leg.get_texts():
        text.set_color(_INK)

    fig.tight_layout()
    png, svg = f"{args.out}.png", f"{args.out}.svg"
    fig.savefig(png, bbox_inches="tight", facecolor="white")
    fig.savefig(svg, bbox_inches="tight", facecolor="white")
    print(f"wrote {png} and {svg} ({plotted} benchmark curves, mode={args.mode})")


if __name__ == "__main__":
    main()
