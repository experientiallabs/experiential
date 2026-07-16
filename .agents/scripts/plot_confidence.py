#!/usr/bin/env python
"""Render the WS-A6 confidence-calibration figures (brand palette, Vercel/Notion style).

Reads the per-suite calibration summaries produced by `analyze_confidence.py` and draws:

1. reliability diagrams — one panel per suite, one line per mode: stated confidence (x) vs mean
   judge score (y), diagonal = perfect calibration; marker size ~ bin population.
2. risk-coverage ("selective fidelity") curves — one panel per suite: coverage (x) vs fidelity of
   the covered steps (y), sweeping the abstention threshold tau over the 11 one-decimal levels.
3. confidence histograms — stated-confidence mass per mode (shows degenerate/clustered outputs).

matplotlib is not a project dependency; run ephemerally:

    uv run --with matplotlib python .agents/scripts/plot_confidence.py \
        --summary tau-bench=.../tau-bench.calibration.json \
        --summary terminal-tasks=.../terminal-tasks.calibration.json \
        --summary swe-bench=.../swe-bench.calibration.json \
        --modes base+conf,reason+conf --out .agents/docs/research/confidence
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt

_INK = "#0a0a0a"
_MUTED = "#8a8a8a"
_GRID = "#ececec"
_COLORS = ["#0070f3", "#7928ca", "#f5a623", "#ee0000", "#50e3c2"]


def _style(ax: plt.Axes) -> None:
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(_GRID)
    ax.tick_params(colors=_MUTED, labelsize=8)
    ax.grid(True, color=_GRID, linewidth=0.6)
    ax.set_axisbelow(True)


def _load(pairs: list[str]) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for pair in pairs:
        label, path = pair.split("=", 1)
        out[label] = json.loads(Path(path).read_text(encoding="utf-8"))
    return out


def _panels(n: int) -> tuple[plt.Figure, list[plt.Axes]]:
    fig, axes = plt.subplots(1, n, figsize=(4.2 * n, 3.6), dpi=200)
    return fig, list(axes) if n > 1 else [axes]


def plot_reliability(summaries: dict[str, dict], modes: list[str], out: Path) -> None:
    fig, axes = _panels(len(summaries))
    for ax, (suite, summary) in zip(axes, summaries.items()):
        _style(ax)
        ax.plot([0, 1], [0, 1], color=_GRID, linewidth=1.2, zorder=1)
        for i, mode in enumerate(modes):
            st = summary["modes"].get(mode)
            if not st or "reliability" not in st:
                continue
            by_level = sorted((float(k), b) for k, b in st["reliability"].items())
            levels = [lv for lv, _ in by_level]
            ys = [b for _, b in by_level]
            ns = [b["n"] for b in ys]
            total = sum(ns) or 1
            ax.plot(levels, [b["mean_judge_score"] for b in ys], color=_COLORS[i % len(_COLORS)],
                    linewidth=1.6, label=mode, zorder=3)
            ax.scatter(levels, [b["mean_judge_score"] for b in ys],
                       s=[12 + 220 * n / total for n in ns],
                       color=_COLORS[i % len(_COLORS)], alpha=0.55, edgecolors="none", zorder=4)
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.02, 1.02)
        ax.set_title(suite, loc="left", fontsize=10, color=_INK)
        ax.set_xlabel("stated confidence", fontsize=8, color=_MUTED)
    axes[0].set_ylabel("mean judge fidelity", fontsize=8, color=_MUTED)
    axes[0].legend(frameon=False, fontsize=7, loc="upper left")
    fig.suptitle("Reliability: stated confidence vs measured fidelity (marker ~ bin mass)",
                 x=0.01, ha="left", fontsize=11, color=_INK)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out.with_name(out.name + "_reliability.png"), facecolor="white")
    fig.savefig(out.with_name(out.name + "_reliability.svg"), facecolor="white")
    plt.close(fig)


def plot_risk_coverage(summaries: dict[str, dict], modes: list[str], out: Path) -> None:
    fig, axes = _panels(len(summaries))
    for ax, (suite, summary) in zip(axes, summaries.items()):
        _style(ax)
        for i, mode in enumerate(modes):
            st = summary["modes"].get(mode)
            if not st or "risk_coverage" not in st:
                continue
            pts = [p for p in st["risk_coverage"] if p["fidelity_covered"] is not None]
            ax.plot([p["coverage"] for p in pts], [p["fidelity_covered"] for p in pts],
                    marker="o", markersize=3, linewidth=1.6,
                    color=_COLORS[i % len(_COLORS)], label=mode)
        ax.set_xlim(1.02, -0.02)  # coverage shrinks left->right as tau rises
        ax.set_title(suite, loc="left", fontsize=10, color=_INK)
        ax.set_xlabel("coverage (frac steps answered)", fontsize=8, color=_MUTED)
    axes[0].set_ylabel("fidelity of covered steps", fontsize=8, color=_MUTED)
    axes[0].legend(frameon=False, fontsize=7, loc="lower left")
    fig.suptitle("Selective fidelity: abstain below tau, keep the confident steps",
                 x=0.01, ha="left", fontsize=11, color=_INK)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out.with_name(out.name + "_risk_coverage.png"), facecolor="white")
    fig.savefig(out.with_name(out.name + "_risk_coverage.svg"), facecolor="white")
    plt.close(fig)


def plot_histograms(summaries: dict[str, dict], modes: list[str], out: Path) -> None:
    fig, axes = _panels(len(summaries))
    width = 0.08 / max(len(modes), 1)
    for ax, (suite, summary) in zip(axes, summaries.items()):
        _style(ax)
        for i, mode in enumerate(modes):
            st = summary["modes"].get(mode)
            if not st or "reliability" not in st:
                continue
            total = sum(b["n"] for b in st["reliability"].values()) or 1
            xs = [float(k) + (i - (len(modes) - 1) / 2) * width for k in st["reliability"]]
            ax.bar(xs, [b["n"] / total for b in st["reliability"].values()], width=width,
                   color=_COLORS[i % len(_COLORS)], label=mode, alpha=0.9)
        ax.set_xlim(-0.06, 1.06)
        ax.set_title(suite, loc="left", fontsize=10, color=_INK)
        ax.set_xlabel("stated confidence", fontsize=8, color=_MUTED)
    axes[0].set_ylabel("fraction of steps", fontsize=8, color=_MUTED)
    axes[0].legend(frameon=False, fontsize=7, loc="upper left")
    fig.suptitle("Stated-confidence distribution", x=0.01, ha="left", fontsize=11, color=_INK)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out.with_name(out.name + "_histogram.png"), facecolor="white")
    fig.savefig(out.with_name(out.name + "_histogram.svg"), facecolor="white")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", action="append", required=True, metavar="LABEL=PATH")
    parser.add_argument("--modes", required=True, help="Comma-separated mode labels to draw.")
    parser.add_argument("--out", required=True, help="Output path prefix (no extension).")
    args = parser.parse_args()
    summaries = _load(args.summary)
    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "sans-serif", "text.color": _INK})
    plot_reliability(summaries, modes, out)
    plot_risk_coverage(summaries, modes, out)
    plot_histograms(summaries, modes, out)
    print(f"wrote {out}_reliability/_risk_coverage/_histogram .png/.svg")


if __name__ == "__main__":
    main()
