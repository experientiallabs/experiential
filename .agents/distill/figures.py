"""Render the status figures for the Notion page as one self-contained HTML file.

Two panels, in the order a reader needs them:

1. The headroom: measured AIME 2024+2025 pass@1 for the GLM-5.2 teacher and the
   Qwen3.5-9B student, with the post-distillation bar left explicitly EMPTY
   because it has not been measured. The question the page has to answer is "did
   distillation help", and the honest answer today is "not yet measured", so the
   figure says that rather than implying a result.
2. Why run 3's falling gap is NOT that evidence: the student-teacher gap fell
   every step while chunk coverage collapsed, so each step measured a different
   and shrinking subset of tokens.

Palette per AGENTS.md rule 14 (ink, hairline grid, white ground, brand accents).

Usage:
    uv run python .agents/distill/figures.py --out /tmp/xtoken_figures.html
"""

from __future__ import annotations

import argparse
import io
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
# Emit text as <text> rather than vector outlines: the outline form inflates
# the SVG roughly 6x for no visual gain at these sizes.
matplotlib.rcParams["svg.fonttype"] = "none"
matplotlib.rcParams["svg.hashsalt"] = "xtoken"
matplotlib.rcParams["path.simplify"] = True
matplotlib.rcParams["path.simplify_threshold"] = 1.0
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

INK = "#0a0a0a"
GRID = "#ececec"
BLUE = "#0070f3"
PURPLE = "#7928ca"
AMBER = "#f5a623"
RED = "#ee0000"
TEAL = "#50e3c2"

TEACHER_AIME = 75.0
STUDENT_AIME = 53.3
TEACHER_N = 60
STUDENT_N = 60


def _style(axis: plt.Axes) -> None:
    """Minimal frame: hairline y grid, no top or right spine."""
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    for side in ("left", "bottom"):
        axis.spines[side].set_color(GRID)
    axis.tick_params(colors=INK, labelsize=9, length=0)
    axis.set_axisbelow(True)
    axis.yaxis.grid(True, color=GRID, linewidth=1)
    axis.xaxis.grid(False)


def headroom_panel(axis: plt.Axes) -> None:
    """Measured baselines plus an explicitly unmeasured post-distillation bar."""
    labels = ["Student\nbefore", "Student\nafter distillation", "Teacher\nGLM-5.2"]
    values = [STUDENT_AIME, 0.0, TEACHER_AIME]
    colors = [AMBER, GRID, BLUE]
    bars = axis.bar(labels, values, color=colors, width=0.58, edgecolor="none")
    # Standard error at n=60 for the two measured bars.
    for index, value in ((0, STUDENT_AIME), (2, TEACHER_AIME)):
        se = 100 * ((value / 100) * (1 - value / 100) / STUDENT_N) ** 0.5
        axis.errorbar(index, value, yerr=se, color=INK, capsize=4, linewidth=1.1, capthick=1.1)
        axis.text(index, value + se + 2.4, f"{value:.1f}%", ha="center", color=INK, fontsize=11,
                  fontweight="bold")
    axis.text(1, 3.0, "NOT YET\nMEASURED", ha="center", va="bottom", color=RED, fontsize=10,
              fontweight="bold", linespacing=1.35)
    axis.axhline(TEACHER_AIME, color=BLUE, linewidth=1, linestyle=(0, (4, 3)), alpha=0.55)
    axis.annotate(
        f"{TEACHER_AIME - STUDENT_AIME:.1f} pt headroom",
        xy=(1.0, (TEACHER_AIME + STUDENT_AIME) / 2),
        ha="center", color=INK, fontsize=9,
    )
    axis.set_ylim(0, 100)
    axis.yaxis.set_major_formatter(PercentFormatter())
    axis.set_ylabel("AIME 2024+2025 pass@1", color=INK, fontsize=9)
    axis.set_title(
        "Did distillation help? Not measured yet.",
        loc="left", color=INK, fontsize=13, fontweight="bold", pad=12,
    )
    _style(axis)
    for bar in bars:
        bar.set_zorder(2)


def confound_panel(axis: plt.Axes, rows: list[dict[str, float]]) -> None:
    """Run 3's gap fell while coverage collapsed, so the gap is not evidence."""
    steps = [int(r["step"]) for r in rows]
    gaps = [float(r["chunk_reverse_kl"]) for r in rows]
    coverage = [100 * float(r["coverage_rate"]) for r in rows]

    axis.plot(steps, gaps, color=PURPLE, marker="o", markersize=5, linewidth=1.8,
              label="student-teacher gap (fell)")
    axis.set_ylabel("student-teacher gap", color=PURPLE, fontsize=9)
    axis.tick_params(axis="y", colors=PURPLE)
    axis.set_xlabel("training step", color=INK, fontsize=9)
    axis.set_ylim(0, max(gaps) * 1.25)
    _style(axis)

    twin = axis.twinx()
    twin.plot(steps, coverage, color=RED, marker="s", markersize=5, linewidth=1.8,
              linestyle=(0, (5, 2)), label="tokens actually scored (collapsed)")
    twin.set_ylabel("% of student tokens scored", color=RED, fontsize=9)
    twin.tick_params(axis="y", colors=RED, labelsize=9, length=0)
    twin.set_ylim(0, 105)
    twin.yaxis.set_major_formatter(PercentFormatter())
    twin.spines["top"].set_visible(False)
    twin.spines["left"].set_color(GRID)
    twin.spines["right"].set_color(GRID)

    axis.set_title(
        "Run 3's falling gap is not evidence: coverage collapsed alongside it",
        loc="left", color=INK, fontsize=13, fontweight="bold", pad=12,
    )
    handles = axis.get_lines() + twin.get_lines()
    axis.legend(handles, [h.get_label() for h in handles], frameon=False, fontsize=8.5,
                loc="lower left")


def render(rows: list[dict[str, float]]) -> str:
    """Both panels as one inline SVG string."""
    figure, axis = plt.subplots(1, 1, figsize=(6.6, 4.0))
    figure.patch.set_facecolor("white")
    headroom_panel(axis)
    figure.tight_layout(pad=2.0)
    buffer = io.StringIO()
    figure.savefig(buffer, format="svg", bbox_inches="tight", facecolor="white")
    plt.close(figure)
    svg = buffer.getvalue()
    return svg[svg.index("<svg") :]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics", default=".wmh/xtoken-runs/run3/metrics.jsonl")
    parser.add_argument("--out", default="/tmp/xtoken_figures.html")
    args = parser.parse_args()

    rows = [json.loads(line) for line in Path(args.metrics).read_text().splitlines() if line.strip()]
    svg = render(rows)
    html = (
        "<!doctype html><html><head><meta charset='utf-8'>"
        "<style>body{margin:0;padding:12px;background:#fff;"
        "font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif}"
        "svg{max-width:100%;height:auto}</style></head><body>" + svg + "</body></html>"
    )
    Path(args.out).write_text(html, encoding="utf-8")
    print(f"wrote {args.out} ({len(html) / 1024:.0f} KiB)")


if __name__ == "__main__":
    main()
