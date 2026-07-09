#!/usr/bin/env python
"""Render the judge-overhaul figure (.agents/docs/research/judge_overhaul.png), brand system per AGENTS.md.

Three panels. Panel A/C read the committed meta-eval runs in .agents/docs/research/raw/;
panel B (and the "fixed" run) read bulky outputs that are NOT committed — regenerate them first:
    uv run python .agents/scripts/run_judge_quality.py --out .agents/docs/research/raw/judge-quality-fixed.json
    uv run python .agents/scripts/run_judge_regression.py \
        --cache .wmh/judge-regression-preds.json --out .agents/docs/research/raw/judge-regression.json
Panels:
  A. Judge-quality meta-eval per case: expected band, baseline score, overhauled score.
  B. Old vs new judge on 47 identical real predictions, colored by new-judge factuality band.
  C. Judge-model sweep: high-band control mean vs hard-defect mean per candidate (separation).

Run from the repo root:
    uv run python .agents/scripts/plot_judge_overhaul.py
"""

from __future__ import annotations

import json
from pathlib import Path
from statistics import fmean

import matplotlib.pyplot as plt

from wmh.optimize.judge_quality import JUDGE_QUALITY_CASES

INK = "#0a0a0a"
MUTED = "#8a8a8a"
GRID = "#ececec"
BLUE = "#0070f3"
PURPLE = "#7928ca"
AMBER = "#f5a623"
RED = "#ee0000"

RAW = Path(".agents/docs/research/raw")
OUT = Path(".agents/docs/research/judge_overhaul.png")


def _style(ax: plt.Axes) -> None:
    ax.set_facecolor("white")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def _verdicts(path: Path) -> dict[str, float]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return {v["case_id"]: v["score"] for v in data["report"]["verdicts"]}


def panel_meta_eval(ax: plt.Axes) -> None:
    baseline = _verdicts(RAW / "judge-quality-baseline.json")
    fixed = _verdicts(RAW / "judge-quality-fixed.json")
    cases = list(JUDGE_QUALITY_CASES)
    ys = range(len(cases))
    for y, case in zip(ys, cases, strict=True):
        ax.plot(
            [case.expected.lo, case.expected.hi], [y, y],
            color=GRID, linewidth=7, solid_capstyle="butt", zorder=1,
        )
        base = baseline.get(case.id)
        if base is not None:
            in_band = case.expected.lo <= base <= case.expected.hi
            ax.scatter(
                [base], [y], s=42, zorder=3, marker="o",
                facecolors="white", edgecolors=MUTED if in_band else RED, linewidths=1.6,
            )
        ax.scatter([fixed[case.id]], [y], s=42, color=BLUE, zorder=4)
    ax.set_yticks(list(ys))
    ax.set_yticklabels([c.id for c in cases], fontsize=8.5, color=INK)
    ax.invert_yaxis()
    ax.set_xlim(-0.03, 1.03)
    ax.set_xlabel("judge headline score", color=MUTED, fontsize=10)
    ax.set_title("Meta-eval: every case in band after the overhaul",
                 loc="left", fontsize=12, fontweight="bold", color=INK, pad=12)
    # Direct legend (visible labels, not color-alone).
    ax.scatter([], [], s=42, facecolors="white", edgecolors=RED, linewidths=1.6,
               label="baseline (out of band)")
    ax.scatter([], [], s=42, facecolors="white", edgecolors=MUTED, linewidths=1.6,
               label="baseline (in band)")
    ax.scatter([], [], s=42, color=BLUE, label="overhauled")
    ax.plot([], [], color=GRID, linewidth=7, label="labeled band")
    ax.legend(loc="lower right", frameon=False, fontsize=8.5, labelcolor=INK)


def panel_regression(ax: plt.Axes) -> None:
    steps = json.loads((RAW / "judge-regression.json").read_text(encoding="utf-8"))["steps"]
    steps = [s for s in steps if s["new_valid"]]
    bands = [
        ("factuality ≥ 0.9", lambda f: f >= 0.9, BLUE),
        ("0.3 < factuality < 0.9", lambda f: 0.3 < f < 0.9, AMBER),
        ("factuality ≤ 0.3", lambda f: f <= 0.3, RED),
    ]
    ax.plot([0, 1], [0, 1], color=GRID, linewidth=1.4, zorder=1)
    for name, pred, color in bands:
        group = [s for s in steps if pred(s["new_dims"]["factuality"])]
        xs = [s["old_score"] for s in group]
        ys = [s["new_score"] for s in group]
        shift = fmean(y - x for x, y in zip(xs, ys, strict=True))
        ax.scatter(xs, ys, s=44, color=color, alpha=0.85, zorder=3,
                   edgecolors="white", linewidths=0.8,
                   label=f"{name}  n={len(group)}, shift {shift:+.2f}")
    ax.set_xlim(-0.03, 1.03)
    ax.set_ylim(-0.03, 1.03)
    ax.set_xlabel("old judge (unweighted mean)", color=MUTED, fontsize=10)
    ax.set_ylabel("new judge (weighted)", color=MUTED, fontsize=10)
    ax.set_title("Same 47 real predictions, both judges",
                 loc="left", fontsize=12, fontweight="bold", color=INK, pad=12)
    ax.legend(loc="upper left", frameon=False, fontsize=8.5, labelcolor=INK)
    ax.annotate("Spearman 0.963", xy=(0.97, 0.06), xycoords="axes fraction",
                ha="right", fontsize=9.5, color=MUTED)


def panel_sweep(ax: plt.Axes) -> None:
    cases = {c.id: c for c in JUDGE_QUALITY_CASES}
    models = {
        "Opus 4.8": "judge-quality-fixed.json",
        "Opus 4.7": "judge-model-opus-4-7.json",
        "Sonnet 4.6": "judge-model-sonnet-4-6.json",
        "GPT-5.5": "judge-model-gpt-5.5.json",
        "Opus 4.6-v1": "judge-model-opus-4-6-v1.json",
    }
    rows = []
    for name, filename in models.items():
        verdicts = json.loads((RAW / filename).read_text(encoding="utf-8"))["report"]["verdicts"]
        hi = fmean(v["score"] for v in verdicts if cases[v["case_id"]].expected.lo >= 0.6)
        lo = fmean(v["score"] for v in verdicts if cases[v["case_id"]].expected.hi <= 0.4)
        rows.append((name, hi, lo))
    rows.sort(key=lambda r: r[1] - r[2])  # separation ascending -> winner on top after invert
    ys = range(len(rows))
    for y, (name, hi, lo) in zip(ys, rows, strict=True):
        emphasize = name == "Opus 4.8"
        color = BLUE if emphasize else MUTED
        ax.plot([lo, hi], [y, y], color=color, linewidth=2, zorder=2)
        ax.scatter([lo], [y], s=46, color=RED if emphasize else MUTED, zorder=3)
        ax.scatter([hi], [y], s=46, color=BLUE if emphasize else MUTED, zorder=3)
        ax.annotate(f"{hi - lo:.3f}", xy=((hi + lo) / 2, y), xytext=(0, 7),
                    textcoords="offset points", ha="center", fontsize=8.5,
                    color=INK if emphasize else MUTED)
    ax.set_yticks(list(ys))
    ax.set_yticklabels([r[0] for r in rows], fontsize=9.5, color=INK)
    ax.set_xlim(-0.03, 1.03)
    ax.set_xlabel("hard-defect mean → control mean\n(number = separation)",
                  color=MUTED, fontsize=9.5)
    ax.set_title("Judge-model sweep (all pass 12/12)",
                 loc="left", fontsize=12, fontweight="bold", color=INK, pad=12)


def main() -> None:
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.4), facecolor="white",
                             gridspec_kw={"width_ratios": [1.25, 1.0, 1.0]})
    panel_meta_eval(axes[0])
    panel_regression(axes[1])
    panel_sweep(axes[2])
    _style(axes[0])
    _style(axes[1])
    _style(axes[2])
    fig.suptitle("A principled judge: proven fixes, targeted score correction, model-robust rubric",
                 x=0.005, ha="left", fontsize=15, fontweight="bold", color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=200, facecolor="white", bbox_inches="tight")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
