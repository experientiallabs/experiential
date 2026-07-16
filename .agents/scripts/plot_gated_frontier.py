#!/usr/bin/env python
"""Render the WS-A6 confidence-gated-verify cost frontier: fidelity vs serve-$/cell.

Numbers are the measured phase-2 cells (D12 protocol: n_train tau 200 / terminal 160 / swe 24,
test caps 40/40/20, sampled turns, seeds 0+1, serve Opus 4.7, judge pinned 4.8; swe healthy via
--drop-degenerate). Serve cost = MeteredProvider per-cell serve-side $ (judge excluded, D12),
mean across seeds, from the run logs / .usage.json files under
.agents/docs/research/agentic_results/confidence/. verify% = mean StepResult.verified.

    uv run --with matplotlib python .agents/scripts/plot_gated_frontier.py \
        --out .agents/docs/research/agentic_results/confidence/figs/gated_frontier
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt

_INK = "#0a0a0a"
_MUTED = "#8a8a8a"
_GRID = "#ececec"
_BLUE, _PURPLE, _AMBER = "#0070f3", "#7928ca", "#f5a623"

# (label, fidelity mean, fidelity std, serve $/cell mean, verify rate)
FRONTIERS = {
    "tau-bench": [
        ("never", 0.9156, 0.001, 5.47, 0.0),
        ("gate@0.6", 0.9207, 0.005, 7.03, 0.23),
        ("always", 0.9192, 0.000, 11.30, 1.0),
    ],
    "terminal-tasks": [
        ("never", 0.8835, 0.002, 3.16, 0.0),
        ("gate@0.5", 0.8904, 0.007, 4.12, 0.25),
        ("gate@0.7", 0.8809, 0.002, 4.61, 0.35),
        ("always", 0.8849, 0.002, 6.45, 1.0),
    ],
    "swe-bench": [
        ("never", 0.8006, 0.006, 6.36, 0.0),
        ("gate@0.5", 0.8203, 0.002, 8.69, 0.35),
        ("gate@0.7", 0.8262, 0.003, 10.28, 0.64),
        ("always", 0.8118, 0.000, 13.11, 1.0),
    ],
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    plt.rcParams.update({"font.family": "sans-serif", "text.color": _INK})
    fig, axes = plt.subplots(1, 3, figsize=(12.6, 3.6), dpi=200)
    for ax, (suite, pts) in zip(axes, FRONTIERS.items()):
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(_GRID)
        ax.tick_params(colors=_MUTED, labelsize=8)
        ax.grid(True, color=_GRID, linewidth=0.6)
        ax.set_axisbelow(True)
        xs = [p[3] for p in pts]
        ys = [p[1] for p in pts]
        es = [p[2] for p in pts]
        ax.plot(xs, ys, color=_GRID, linewidth=1.2, zorder=1)
        for (label, y, e, x, rate), color in zip(
            pts, [_BLUE, _PURPLE, _PURPLE, _AMBER][: len(pts)]
        ):
            color = {"never": _BLUE, "always": _AMBER}.get(label, _PURPLE)
            ax.errorbar(x, y, yerr=e, fmt="o", color=color, markersize=6, capsize=2, zorder=3)
            note = label if rate in (0.0, 1.0) else f"{label} ({rate:.0%} verified)"
            ax.annotate(
                note, (x, y), textcoords="offset points", xytext=(6, 6), fontsize=7.5, color=_INK
            )
        ax.set_title(suite, loc="left", fontsize=10, color=_INK)
        ax.set_xlabel("serve $ per cell (judge excluded)", fontsize=8, color=_MUTED)
        ax.margins(x=0.18, y=0.25)
    axes[0].set_ylabel("fidelity", fontsize=8, color=_MUTED)
    fig.suptitle(
        "Confidence-gated verify: fidelity vs serve cost (verify only when confidence < tau)",
        x=0.01,
        ha="left",
        fontsize=11,
        color=_INK,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out.with_suffix(".png"), facecolor="white")
    fig.savefig(out.with_suffix(".svg"), facecolor="white")
    print(f"wrote {out}.png/.svg")


if __name__ == "__main__":
    main()
