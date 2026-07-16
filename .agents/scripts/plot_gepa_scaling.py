#!/usr/bin/env python
"""Render the GEPA scaling-law figure: fidelity vs. GEPA iterations and vs. training traces.

Reads `AblationReport` JSONs produced by `run_gepa_scaling.py` (conditions labelled `t{n}_b{b}`)
and draws one clean, Notion/Vercel-style chart of up to four panels (each optional after A):

- Panel A — x = GEPA iterations (log), y = open-loop reconstruction fidelity, one line per
  benchmark with a ±std band across seeds. `b=0` (GEPA off — the RAG-only anchor) is drawn in a
  fixed "0" slot left of 10^0 with a dotted connector (log(0) is undefined), exactly like the trace
  scaling law's n=0 slot.
- Panel B — x = training traces (log), y = fidelity at a fixed GEPA budget (--trace-budget), one
  line per benchmark, with the trace scaling law's RAG-only curves (label `base@N`, from PR #72's
  reports) overlaid dashed for comparison.
- Panel C (--judge-report) — judge sensitivity: the same predictions (b=0 open / b=8 filled dots,
  whisker = the GEPA delta) scored by each judge model.
- Panel D (--dense-report) — the dense n-sweep run with the improved GEPA: per-benchmark optimal
  trace count starred, original-GEPA trace-axis points overlaid faint dashed.

matplotlib is not a project dependency (this is a one-off research figure), so run it ephemerally:

    uv run --with matplotlib python .agents/scripts/plot_gepa_scaling.py \
        --budget-report tau-bench=tau_budget.json --trace-report tau-bench=tau_traces.json \
        --rag-report tau-bench=tau_rag.json ... --out docs/research/gepa_scaling_law

Each --*-report is `label=path` (repeatable; repeats with the same label merge, so the shared
t64_b8 point can come from either sweep). Writes both PNG and SVG.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

# Vercel/Notion-ish palette: near-black text, restrained accent lines, generous whitespace.
_INK = "#0a0a0a"
_MUTED = "#8a8a8a"
_GRID = "#ececec"
_COLORS = ["#0070f3", "#7928ca", "#f5a623", "#ee0000", "#50e3c2"]  # blue, purple, amber, red, teal

# Fixed log-axis slot for the zero point (b=0 anchor), sitting just left of 10^0.
_ZERO_X = 0.4

_GEPA_RE = re.compile(r"^t(?P<n>\d+)_b(?P<b>\d+)$")
_RAG_RE = re.compile(r"^base@(?P<n>\d+)$")

Point = tuple[int, float, float]  # (x, mean, std)


def _gepa_points(paths: list[str], key: str, fixed: tuple[str, int] | None = None) -> list[Point]:
    """Extract (x, mean, std) from `t{n}_b{b}` conditions; the non-`key` knob may be held fixed."""
    points: dict[int, tuple[float, float]] = {}
    for path in paths:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        for cell in data.get("conditions", []):
            label = cell.get("condition", {}).get("label", "")
            m = _GEPA_RE.match(label)
            if not m:
                continue
            if fixed is not None and int(m.group(fixed[0])) != fixed[1]:
                continue
            points[int(m.group(key))] = (float(cell.get("mean", 0.0)), float(cell.get("std", 0.0)))
    return sorted((x, mean, std) for x, (mean, std) in points.items())


def _rag_points(paths: list[str]) -> list[Point]:
    """Extract the trace scaling law's RAG-only curve (`base@N` labels)."""
    points: dict[int, tuple[float, float]] = {}
    for path in paths:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        for cell in data.get("conditions", []):
            label = cell.get("condition", {}).get("label", "")
            m = _RAG_RE.match(label)
            if not m:
                continue
            points[int(m.group("n"))] = (float(cell.get("mean", 0.0)), float(cell.get("std", 0.0)))
    return sorted((x, mean, std) for x, (mean, std) in points.items())


def _by_label(specs: list[str]) -> dict[str, list[str]]:
    """Group repeated `label=path` specs into {label: [paths]}, preserving label order."""
    grouped: dict[str, list[str]] = defaultdict(list)
    for spec in specs:
        if "=" not in spec:
            raise SystemExit(f"report specs must be label=path, got {spec!r}")
        label, path = spec.split("=", 1)
        grouped[label].append(path)
    return dict(grouped)


def _draw_series(ax, pts: list[Point], color: str, label: str | None, *, dashed: bool = False):  # noqa: ANN001, ANN202
    """One benchmark's curve on `ax`: markers + ±std band, zero point in the `_ZERO_X` slot."""
    zero = next((p for p in pts if p[0] == 0), None)
    rest = [p for p in pts if p[0] >= 1]
    style = "--" if dashed else "-"
    if rest:
        xs, means, stds = zip(*rest, strict=True)
        ax.plot(
            xs,
            means,
            style,
            marker="o",
            color=color,
            label=label,
            linewidth=1.6 if dashed else 2.2,
            markersize=4 if dashed else 5,
            markerfacecolor="white",
            markeredgecolor=color,
            markeredgewidth=1.2 if dashed else 1.6,
            alpha=0.55 if dashed else 1.0,
            zorder=3,
        )
        if not dashed:
            lo = [m - s for m, s in zip(means, stds, strict=True)]
            hi = [m + s for m, s in zip(means, stds, strict=True)]
            ax.fill_between(xs, lo, hi, color=color, alpha=0.10, linewidth=0, zorder=2)
    if zero is not None:
        zlabel = label if not rest else None  # keep one legend entry per series
        ax.plot(
            [_ZERO_X],
            [zero[1]],
            "o",
            color=color,
            label=zlabel,
            markersize=5,
            markerfacecolor="white",
            markeredgecolor=color,
            markeredgewidth=1.6,
            zorder=3,
        )
        if rest:
            ax.plot(
                [_ZERO_X, rest[0][0]],
                [zero[1], rest[0][1]],
                ":",
                color=color,
                linewidth=1.4,
                alpha=0.7,
                zorder=2,
            )
    return max((p[0] for p in pts), default=1)


def _style_axis(ax, ticks: list[int], *, zero_slot: bool, xlabel: str, ymin: float, ymax: float):  # noqa: ANN001, ANN202
    """Shared minimal chrome: log x with explicit ticks (optional "0" slot), soft y grid."""
    from matplotlib.ticker import FixedFormatter, FixedLocator

    ax.set_xscale("log")
    locs = ([_ZERO_X] if zero_slot else []) + ticks
    labels = (["0"] if zero_slot else []) + [str(t) for t in ticks]
    ax.xaxis.set_major_locator(FixedLocator(locs))
    ax.xaxis.set_major_formatter(FixedFormatter(labels))
    ax.xaxis.set_minor_locator(FixedLocator([]))
    left = _ZERO_X * 0.75 if zero_slot else min(ticks) * 0.75
    ax.set_xlim(left, max(ticks) * 1.4)
    ax.set_ylim(ymin, ymax)
    ax.set_xlabel(xlabel, fontsize=11, color=_INK)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(_GRID)
    ax.grid(axis="y", color=_GRID, linewidth=1)
    ax.set_axisbelow(True)
    ax.tick_params(colors=_MUTED, labelsize=10, length=0)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--budget-report",
        action="append",
        default=[],
        metavar="LABEL=PATH",
        help="A benchmark's budget-axis report (repeatable; same-label repeats merge).",
    )
    parser.add_argument(
        "--trace-report",
        action="append",
        default=[],
        metavar="LABEL=PATH",
        help="A benchmark's trace-axis report (repeatable; same-label repeats merge).",
    )
    parser.add_argument(
        "--rag-report",
        action="append",
        default=[],
        metavar="LABEL=PATH",
        help="A benchmark's trace-scaling-law report (base@N labels) for the dashed RAG overlay.",
    )
    parser.add_argument(
        "--judge-report",
        action="append",
        default=[],
        metavar="LABEL=PATH",
        help="A benchmark's judge-ablation JSON (from run_judge_ablation.py) for panel C.",
    )
    parser.add_argument(
        "--dense-report",
        action="append",
        default=[],
        metavar="LABEL=PATH",
        help="A benchmark's dense n-sweep report (improved GEPA) for panel D; the per-benchmark "
        "optimum is starred and the --trace-report curve is overlaid faint for comparison.",
    )
    parser.add_argument("--trace-budget", type=int, default=8, help="Fixed budget for panel B.")
    parser.add_argument("--out", default="gepa_scaling", help="Output path stem (.png + .svg).")
    parser.add_argument("--title", default="GEPA scaling law", help="Figure title.")
    parser.add_argument("--ymin", type=float, default=0.6, help="Y-axis floor (fidelity is 0..1).")
    parser.add_argument("--ymax", type=float, default=1.0, help="Y-axis ceiling.")
    args = parser.parse_args()

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    budget_reports = _by_label(args.budget_report)
    trace_reports = _by_label(args.trace_report)
    rag_reports = _by_label(args.rag_report)
    judge_reports = _by_label(args.judge_report)
    dense_reports = _by_label(args.dense_report)
    benchmarks = list(
        dict.fromkeys([*budget_reports, *trace_reports, *judge_reports, *dense_reports])
    )
    if not benchmarks:
        raise SystemExit("no reports given — pass --budget-report and/or --trace-report")
    colors = {label: _COLORS[i % len(_COLORS)] for i, label in enumerate(benchmarks)}

    n_panels = 1 + int(bool(trace_reports)) + int(bool(judge_reports)) + int(bool(dense_reports))
    widths = {1: (8, 5), 2: (11, 4.6), 3: (15, 4.4), 4: (19, 4.2)}
    fig, axes = plt.subplots(1, n_panels, figsize=widths[n_panels], dpi=200)
    fig.patch.set_facecolor("white")
    axes = [axes] if n_panels == 1 else list(axes)
    ax_a = axes[0]
    ax_b = axes[1] if trace_reports else None
    ax_c = axes[1 + int(bool(trace_reports))] if judge_reports else None
    ax_d = axes[-1] if dense_reports else None

    # Panel A: fidelity vs GEPA iterations at fixed n (the b=0 anchor sits in the "0" slot).
    budget_ticks: set[int] = set()
    for label, paths in budget_reports.items():
        pts = _gepa_points(paths, "b")
        if not pts:
            print(f"warning: no t*_b* points in {paths} ({label})")
            continue
        _draw_series(ax_a, pts, colors[label], label)
        budget_ticks.update(p[0] for p in pts if p[0] >= 1)
    _style_axis(
        ax_a,
        sorted(budget_ticks) or [1],
        zero_slot=True,
        xlabel="GEPA iterations",
        ymin=args.ymin,
        ymax=args.ymax,
    )
    ax_a.set_ylabel("reconstruction fidelity", fontsize=11, color=_INK)
    leg = ax_a.legend(frameon=False, fontsize=10, loc="lower right")
    for text in leg.get_texts():
        text.set_color(_INK)

    # Panel B: fidelity vs training traces at the fixed budget, RAG-only overlaid dashed.
    if ax_b is not None:
        ax_b.set_facecolor("white")
        count_ticks: set[int] = set()
        for label in benchmarks:
            paths = trace_reports.get(label, []) + budget_reports.get(label, [])
            pts = _gepa_points(paths, "n", fixed=("b", args.trace_budget))
            if pts:
                _draw_series(ax_b, pts, colors[label], None)
                count_ticks.update(p[0] for p in pts if p[0] >= 1)
            rag = _rag_points(rag_reports.get(label, []))
            rag = [p for p in rag if p[0] >= 1]
            if rag:
                _draw_series(ax_b, rag, colors[label], None, dashed=True)
                count_ticks.update(p[0] for p in rag)
        decades = [10**k for k in range(0, 4) if any(10**k <= c for c in count_ticks)]
        ticks = sorted({t for t in decades if t <= max(count_ticks, default=1)} | {1})
        if count_ticks:
            ticks = sorted({*ticks, max(count_ticks)})
        _style_axis(
            ax_b,
            ticks,
            zero_slot=False,
            xlabel="training traces",
            ymin=args.ymin,
            ymax=args.ymax,
        )
        style_legend = ax_b.legend(
            handles=[
                Line2D([], [], color=_INK, linewidth=2.2, label=f"GEPA b={args.trace_budget}"),
                Line2D(
                    [],
                    [],
                    color=_INK,
                    linewidth=1.6,
                    linestyle="--",
                    alpha=0.55,
                    label="RAG only (b=0)",
                ),
            ],
            frameon=False,
            fontsize=10,
            loc="lower right",
        )
        for text in style_legend.get_texts():
            text.set_color(_INK)

    # Panel C: judge sensitivity — the same predictions (b=0 open, b=8 filled) scored by each
    # judge model. Vertical whisker = the GEPA delta each judge measures; per-benchmark x offsets.
    if ax_c is not None:
        ax_c.set_facecolor("white")
        judge_order = ["haiku-4.5", "gpt-5.4-mini", "opus-4.8", "gpt-5.5"]
        offsets = {
            label: (i - (len(benchmarks) - 1) / 2) * 0.18 for i, label in enumerate(benchmarks)
        }
        seen_judges: list[str] = []
        for label in benchmarks:
            for path in judge_reports.get(label, []):
                data = json.loads(Path(path).read_text(encoding="utf-8"))
                budget_key = f"b{data.get('budget', args.trace_budget)}"
                for j, judge in enumerate(judge_order):
                    cells = data.get("judges", {}).get(judge)
                    if not cells:
                        continue
                    if judge not in seen_judges:
                        seen_judges.append(judge)
                    x = j + offsets[label]
                    b0 = cells.get("b0", {}).get("mean")
                    b8 = cells.get(budget_key, {}).get("mean")
                    if b0 is None or b8 is None:
                        continue
                    ax_c.plot(
                        [x, x],
                        [b0, b8],
                        "-",
                        color=colors[label],
                        linewidth=1.2,
                        alpha=0.6,
                        zorder=2,
                    )
                    ax_c.plot(
                        [x],
                        [b0],
                        "o",
                        markersize=5,
                        markerfacecolor="white",
                        markeredgecolor=colors[label],
                        markeredgewidth=1.6,
                        zorder=3,
                    )
                    ax_c.plot([x], [b8], "o", markersize=5, color=colors[label], zorder=3)
        ax_c.set_xticks(range(len(judge_order)))
        ax_c.set_xticklabels(judge_order, fontsize=9)
        ax_c.set_xlim(-0.6, len(judge_order) - 0.4)
        ax_c.set_ylim(args.ymin, args.ymax)
        ax_c.set_xlabel("judge model", fontsize=11, color=_INK)
        for side in ("top", "right"):
            ax_c.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax_c.spines[side].set_color(_GRID)
        ax_c.grid(axis="y", color=_GRID, linewidth=1)
        ax_c.set_axisbelow(True)
        ax_c.tick_params(colors=_MUTED, labelsize=10, length=0)
        judge_legend = ax_c.legend(
            handles=[
                Line2D(
                    [],
                    [],
                    marker="o",
                    linestyle="",
                    markerfacecolor="white",
                    markeredgecolor=_INK,
                    markeredgewidth=1.6,
                    label="base (b=0)",
                ),
                Line2D([], [], marker="o", linestyle="", color=_INK, label="GEPA (b=8)"),
            ],
            frameon=False,
            fontsize=10,
            loc="lower right",
        )
        for text in judge_legend.get_texts():
            text.set_color(_INK)

    # Panel D: dense n-sweep (improved GEPA) — per-benchmark optimum starred, original-GEPA
    # trace-axis points overlaid faint dashed for comparison.
    if ax_d is not None:
        ax_d.set_facecolor("white")
        dense_ticks: set[int] = set()
        for label in benchmarks:
            pts = _gepa_points(dense_reports.get(label, []), "n", fixed=("b", args.trace_budget))
            if not pts:
                continue
            _draw_series(ax_d, pts, colors[label], None)
            dense_ticks.update(p[0] for p in pts if p[0] >= 1)
            old = _gepa_points(
                trace_reports.get(label, []) + budget_reports.get(label, []),
                "n",
                fixed=("b", args.trace_budget),
            )
            if old:
                _draw_series(ax_d, old, colors[label], None, dashed=True)
            best_n, best_mean, _ = max(pts, key=lambda p: p[1])
            ax_d.plot(
                [best_n],
                [best_mean],
                marker="*",
                markersize=14,
                color=colors[label],
                markeredgecolor="white",
                markeredgewidth=0.8,
                zorder=4,
            )
        if dense_ticks:
            decades_d = [10**k for k in range(0, 4) if 10**k <= max(dense_ticks)]
            ticks_d = sorted({*decades_d, 1, max(dense_ticks)})
            _style_axis(
                ax_d,
                ticks_d,
                zero_slot=False,
                xlabel="training traces",
                ymin=args.ymin,
                ymax=args.ymax,
            )
        dense_legend = ax_d.legend(
            handles=[
                Line2D([], [], color=_INK, linewidth=2.2, label="improved GEPA"),
                Line2D(
                    [],
                    [],
                    color=_INK,
                    linewidth=1.6,
                    linestyle="--",
                    alpha=0.55,
                    label="original GEPA",
                ),
                Line2D(
                    [], [], marker="*", linestyle="", color=_INK, markersize=11, label="optimal n"
                ),
            ],
            frameon=False,
            fontsize=10,
            loc="lower right",
        )
        for text in dense_legend.get_texts():
            text.set_color(_INK)

    ax_a.set_title(args.title, fontsize=15, color=_INK, fontweight="bold", loc="left", pad=14)
    fig.tight_layout()
    png, svg = f"{args.out}.png", f"{args.out}.svg"
    fig.savefig(png, bbox_inches="tight", facecolor="white")
    fig.savefig(svg, bbox_inches="tight", facecolor="white")
    print(f"wrote {png} and {svg} ({len(benchmarks)} benchmark curves)")


if __name__ == "__main__":
    main()
