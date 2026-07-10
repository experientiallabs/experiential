#!/usr/bin/env python
"""Render the agentic-mode lever figure: fidelity by serving configuration, all benchmarks.

Reads the AblationReport JSONs under `.agents/docs/research/agentic_results/` (the v3 sweep +
mode-specific cells) and draws two panels in the house style: (left) grouped fidelity bars per
benchmark x mode with the winner highlighted, (right) terminal-tasks serve-cost vs fidelity —
the panel that shows live fetch dominating the knowledge base at 40% of its cost.

matplotlib is not a project dependency (one-off research figure):

    uv run --with matplotlib python .agents/scripts/plot_agentic_modes.py \
        --results .agents/docs/research/agentic_results --out docs/research/agentic_fidelity
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

_INK = "#0a0a0a"
_MUTED = "#8a8a8a"
_GRID = "#ececec"
# Base = muted gray (the baseline); the five levers take the brand accents; fetch (terminal's
# extra, only where the corpus has URLs) uses ink.
_MODE_COLORS = {
    "base": "#8a8a8a",
    "reason": "#0070f3",
    "reason+kb": "#f5a623",
    "reason+verify": "#e00",
    "reason+source": "#7928ca",
    "reason+profile": "#50e3c2",
    "reason+fetch": "#0a0a0a",
    "reason+source2": "#c026d3",
    "reason+workspace": "#0f9d58",
    "reason+poll": "#f81ce5",
}


def _cells(path: Path) -> dict[str, tuple[float, float]]:
    """AblationReport JSON -> {mode: (mean, std)} keyed by the label's mode part."""
    data = json.loads(path.read_text(encoding="utf-8"))
    out: dict[str, tuple[float, float]] = {}
    for cell in data.get("conditions", []):
        label = cell.get("condition", {}).get("label", "")
        mode = label.split("@")[0]
        out[mode] = (float(cell.get("mean", 0.0)), float(cell.get("std", 0.0)))
    return out


def _mean_cost(usage_path: Path, mode: str) -> float | None:
    if not usage_path.exists():
        return None
    runs = json.loads(usage_path.read_text(encoding="utf-8"))
    costs = [r["total"]["cost_usd"] for r in runs if r["label"].split("@")[0] == mode]
    return sum(costs) / len(costs) if costs else None


def _assemble(results: Path) -> dict[str, dict[str, tuple[float, float]]]:
    """Per benchmark: the final (post-iteration) cell per mode, from the committed reports."""
    tau = (
        _cells(results / "tau-bench.json")
        | _cells(results / "tau-bench.v3.json")
        | _cells(results / "tau-bench.levers.json")
    )
    terminal = (
        _cells(results / "terminal-tasks.json")
        | _cells(results / "terminal-tasks.v3.json")
        | _cells(results / "terminal-tasks.fetch.json")
        | _cells(results / "terminal-tasks.levers.json")
        | _cells(results / "terminal-tasks.poll.json")
    )
    swe = _cells(results / "swe-bench.healthy.json") | _cells(results / "swe-bench.healthy.v3.json")
    swe |= _cells(results / "swe-bench.verify.json") | _cells(results / "swe-bench.levers.json")
    swe |= _cells(results / "swe-bench.source2.json") | _cells(results / "swe-bench.workspace.json")
    swe |= _cells(results / "swe-bench.poll.json")
    keep = list(_MODE_COLORS)
    return {
        "tau-bench": {m: tau[m] for m in keep if m in tau},
        "terminal-tasks": {m: terminal[m] for m in keep if m in terminal},
        "swe-bench (healthy)": {m: swe[m] for m in keep if m in swe},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", default=".agents/docs/research/agentic_results")
    parser.add_argument("--out", default="docs/research/agentic_fidelity")
    args = parser.parse_args()

    import matplotlib.pyplot as plt

    results = Path(args.results)
    data = _assemble(results)

    fig, (ax, axc) = plt.subplots(
        1, 2, figsize=(12.5, 4.6), gridspec_kw={"width_ratios": [1.9, 1.0]}
    )
    fig.patch.set_facecolor("white")

    # Left: grouped fidelity bars per benchmark x mode, winner annotated.
    benches = list(data)
    for bi, bench in enumerate(benches):
        modes = data[bench]
        width = min(0.125, 0.92 / max(len(modes), 1))  # groups can't bleed into each other
        winner = max(modes, key=lambda m: modes[m][0])
        offsets = [(i - (len(modes) - 1) / 2) * width for i in range(len(modes))]
        for (mode, (mean, std)), off in zip(modes.items(), offsets, strict=True):
            x = bi + off
            ax.bar(
                x,
                mean,
                width=width * 0.92,
                color=_MODE_COLORS[mode],
                yerr=std,
                error_kw={"ecolor": _MUTED, "elinewidth": 1, "capsize": 2},
                edgecolor="white",
                linewidth=0.5,
                zorder=3,
            )
            ax.text(
                x,
                mean + std + 0.006,
                f"{mean:.3f}" + (" ★" if mode == winner else ""),
                ha="center",
                va="bottom",
                fontsize=7,
                color=_INK if mode == winner else _MUTED,
                fontweight="bold" if mode == winner else "normal",
            )
    ax.set_xticks(range(len(benches)))
    ax.set_xticklabels(benches, fontsize=10, color=_INK)
    ax.set_ylim(0.72, 0.97)
    ax.set_ylabel("open-loop reconstruction fidelity", fontsize=10, color=_INK)
    ax.set_title(
        "Fidelity by serving configuration — the winning lever is task-shaped",
        loc="left",
        fontsize=12,
        color=_INK,
        pad=14,
    )
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in _MODE_COLORS.values()]
    ax.legend(handles, list(_MODE_COLORS), frameon=False, fontsize=7.5, ncol=4, loc="upper left")

    # Right: terminal-tasks cost vs fidelity — fetch beats the KB at ~40% of its cost.
    term_modes = data["terminal-tasks"]
    for mode, (mean, _std) in term_modes.items():
        stem = {
            "base": "terminal-tasks",
            "reason": "terminal-tasks.v3",
            "reason+kb": "terminal-tasks.v3",
            "reason+fetch": "terminal-tasks.fetch",
        }.get(mode)
        cost = _mean_cost(results / f"{stem}.usage.json", mode) if stem else None
        if cost is None:
            continue
        axc.scatter(cost, mean, s=64, color=_MODE_COLORS[mode], zorder=3)
        axc.annotate(
            mode,
            (cost, mean),
            textcoords="offset points",
            xytext=(7, 4),
            fontsize=8,
            color=_INK,
        )
    axc.set_xlabel("serve cost per run (USD)", fontsize=10, color=_INK)
    axc.set_ylabel("fidelity", fontsize=10, color=_INK)
    axc.set_title("terminal-tasks: cost vs fidelity", loc="left", fontsize=11, color=_INK, pad=14)

    for a in (ax, axc):
        a.set_facecolor("white")
        for side in ("top", "right"):
            a.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            a.spines[side].set_color(_GRID)
        a.tick_params(colors=_MUTED, labelsize=9)
        a.grid(axis="y", color=_GRID, linewidth=0.8, zorder=0)

    fig.tight_layout()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "svg"):
        fig.savefig(f"{out}.{ext}", dpi=200, bbox_inches="tight", facecolor="white")
        print(f"wrote {out}.{ext}")


if __name__ == "__main__":
    main()
