#!/usr/bin/env python
"""Render the four routing-research figures for the PR summary.

Every number except the confidence curve is RECOMPUTED from the run records through the
dashboard's own aggregation path (`build_dashboard.py`), so a figure can never drift from the
dashboard: same loading rules, same matrix-cohort resolution, same knob-vs-fitted-value
detection, same seed aggregation, same verdict tiers. Re-run it after new captures land and
the figures move with the data.

    uv run --with matplotlib python .agents/scripts/render_routing_figures.py

Figures (into .agents/docs/research/figures/):
  routing_pareto_ours9.png    cost vs accuracy on our 9-model pool, seed-aggregated with sd
  routing_verdict_census.png  how every group scored under the honest verdict rules
  routing_confidence_curve.png  the "route only when confident" dial (numbers from PR #259)
  routing_signal_map.png      where signal lives vs how big the test set is

Two deliberate deviations from the brief, both to avoid a misleading chart:

1. Amber and teal are the DARKENED chart steps (#b8770a, #0d9488), not the raw brand accents
   (#f5a623, #50e3c2). The raw pair scores 1.97:1 and 1.56:1 contrast on white, far under the
   3:1 floor, and fails the lightness band; as a thin line on white the raw teal is close to
   invisible. The darkened steps pass every check. `build_dashboard.py` made the same call.
2. The confidence curve is two stacked panels sharing an x-axis, not one dual-axis chart.
   Two y-scales on one frame let the crossing point be placed anywhere by choosing the
   scaling, and here it would matter: accuracy moves 0.56pt across the whole sweep while the
   routed-away fraction moves 38pt. Stacked panels keep both readable and comparable, and
   plotting accuracy against the fable-5-alone reference shows the real shape - flat until
   the dial is turned hard.
"""

from __future__ import annotations

import importlib.util
import statistics
import sys
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from types import ModuleType

    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

BUILDER = Path(__file__).with_name("build_dashboard.py")
OUT_DIR = Path(".agents/docs/research/figures")

# House style: near-black ink, hairline grid, restrained accents, generous whitespace.
INK = "#0a0a0a"
MUTED = "#8a8a8a"
GRID = "#ececec"
BLUE = "#0070f3"
PURPLE = "#7928ca"
AMBER = "#b8770a"  # darkened brand amber; raw #f5a623 is 1.97:1 on white
RED = "#ee0000"
TEAL = "#0d9488"  # darkened brand teal; raw #50e3c2 is 1.56:1 on white
GRAY_MID = "#9a9a9a"
GRAY_LIGHT = "#c9c9c9"

# Fable-5 used alone on routerbench-ours9, and the z sweep for the confidence dial.
# Source: PR #259 validation output (ours9, fallback fable-5). Hardcoded on purpose; these
# come from the confidence-gating run, not from the ablation run records.
FABLE_ALONE_ACC = 0.9045
CONFIDENCE_Z = [0.0, 0.25, 0.5, 1.0, 2.0]
CONFIDENCE_ROUTED_AWAY = [69.1, 68.0, 64.9, 48.0, 30.9]
CONFIDENCE_ACC = [0.9607, 0.9607, 0.9607, 0.9635, 0.9579]

HEADLINE_MATRIX = "routerbench-ours9"


def load_groups() -> tuple[list[dict], int]:
    """Seed-aggregated groups straight from the dashboard builder (its rules, not a copy)."""
    spec = importlib.util.spec_from_file_location("build_dashboard", BUILDER)
    if spec is None or spec.loader is None:
        raise SystemExit(f"cannot import the dashboard builder from {BUILDER}")
    builder = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(builder)

    runs, _ = builder.load_runs()
    runs = [r for r in runs if r["matrix"] not in builder.FOREIGN_POOLS]
    builder.resolve_matrices(runs)
    groups, _ = builder.aggregate(runs, builder.knob_keys(runs))
    groups += builder.synthetic_anchors(runs, groups)
    return groups, len(runs)


def style_axes(ax: Axes, *, xgrid: bool = False) -> None:
    """Minimal chrome: no top/right spine, hairline grid, muted tickless labels."""
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.grid(axis="x" if xgrid else "y", color=GRID, linewidth=1)
    ax.set_axisbelow(True)
    ax.tick_params(colors=MUTED, labelsize=10, length=0)


def titled(ax: Axes, title: str, subtitle: str = "") -> None:
    """Left-aligned title with the subtitle on its own line beneath it, never overlapping."""
    ax.set_title(
        title, fontsize=15, color=INK, fontweight="bold", loc="left", pad=34 if subtitle else 14
    )
    if subtitle:
        ax.annotate(
            subtitle,
            xy=(0, 1),
            xytext=(0, 10),
            xycoords="axes fraction",
            textcoords="offset points",
            fontsize=9.5,
            color=MUTED,
            va="bottom",
            ha="left",
        )


def save(fig: Figure, name: str) -> Path:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / name
    fig.savefig(path, bbox_inches="tight", facecolor="white", dpi=200)
    return path


def curve(ax: Axes, points: list[dict], color: str, label: str) -> None:
    """One variant's cost-knob sweep: a connected line, hollow markers, sd whiskers."""
    pts = sorted(points, key=lambda g: g["cost"])
    xs = [g["cost"] for g in pts]
    ys = [g["acc"] for g in pts]
    es = [g["sd"] for g in pts]
    ax.errorbar(
        xs,
        ys,
        yerr=es,
        fmt="-o",
        color=color,
        label=label,
        linewidth=2.0,
        markersize=5,
        markerfacecolor="white",
        markeredgecolor=color,
        markeredgewidth=1.6,
        elinewidth=1.2,
        capsize=0,
        ecolor=color,
        alpha=0.95,
        zorder=3,
    )


def fig_pareto(groups: list[dict], plt: ModuleType) -> Path:
    """Figure 1: the headline. Cost vs accuracy on our own pool, one point per group."""
    rows = [g for g in groups if g["m"] == HEADLINE_MATRIX]
    by_variant: dict[str, list[dict]] = defaultdict(list)
    for g in rows:
        by_variant[g["v"]].append(g)

    anchor = next(g for g in rows if g["v"] == "best-single")
    champion = max(
        (g for g in rows if g["fam"] == "knn" and g["tier"] == "beats"),
        key=lambda g: g["d"] or 0.0,
    )
    fable_cost = statistics.median([g["pmc"]["fable-5"] for g in rows if "fable-5" in g["pmc"]])

    fig, ax = plt.subplots(figsize=(8.4, 5.2), dpi=200)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    # The three families as cost-knob sweeps: one connected curve each, cheapest to dearest.
    curve(ax, by_variant["r3-knn-frontier"], TEAL, "kNN retrieval (cost knob)")
    curve(ax, by_variant["r3-irt-frontier"], PURPLE, "IRT learned predictor (cost knob)")
    curve(ax, by_variant["rank"], BLUE, "rank cluster scoreboard (cost knob)")

    ax.plot(
        [anchor["cost"]],
        [anchor["acc"]],
        "D",
        color=INK,
        markersize=9,
        markeredgecolor="white",
        markeredgewidth=1.5,
        zorder=6,
        label="best single model",
    )
    ax.annotate(
        f"best single: gpt-5.5\n{anchor['acc']:.1%}",
        xy=(anchor["cost"], anchor["acc"]),
        xytext=(13, 0),
        textcoords="offset points",
        fontsize=9.5,
        color=INK,
        ha="left",
        va="center",
    )

    ax.plot(
        [champion["cost"]],
        [champion["acc"]],
        "o",
        color=TEAL,
        markersize=11,
        markeredgecolor="white",
        markeredgewidth=1.8,
        zorder=7,
    )
    ax.errorbar(
        [champion["cost"]],
        [champion["acc"]],
        yerr=[champion["sd"]],
        elinewidth=1.4,
        capsize=0,
        ecolor=TEAL,
        zorder=6,
        fmt="none",
    )
    ax.annotate(
        f"kNN champion {champion['acc']:.1%}\n{100 * champion['d']:+.1f}pt vs best single, "
        f"{champion['w']}/{champion['s']} seeds, "
        f"{1 - champion['cost'] / anchor['cost']:.0%} cheaper",
        xy=(champion["cost"], champion["acc"]),
        xytext=(-8, 20),
        textcoords="offset points",
        fontsize=9.5,
        color=TEAL,
        ha="right",
        fontweight="bold",
    )

    ax.plot(
        [fable_cost],
        [FABLE_ALONE_ACC],
        "X",
        color=RED,
        markersize=9,
        markeredgecolor="white",
        markeredgewidth=1.2,
        zorder=6,
    )
    ax.annotate(
        f"fable-5 alone {FABLE_ALONE_ACC:.1%}\n(dearer AND worse)",
        xy=(fable_cost, FABLE_ALONE_ACC),
        xytext=(13, 0),
        textcoords="offset points",
        fontsize=9.5,
        color=RED,
        ha="left",
        va="center",
    )

    ax.set_xscale("log")
    from matplotlib.ticker import FixedFormatter, FixedLocator

    ticks = [3e-4, 5e-4, 1e-3, 2e-3, 3e-3, 5e-3]
    ax.xaxis.set_major_locator(FixedLocator(ticks))
    ax.xaxis.set_major_formatter(FixedFormatter([f"${t:.4f}" for t in ticks]))
    ax.xaxis.set_minor_locator(FixedLocator([]))
    ax.set_xlim(2.0e-4, 7.5e-3)  # right margin so the anchor labels sit clear of the curves
    ax.set_xlabel("cost per call, USD (log)", fontsize=11, color=INK)
    ax.set_ylabel("accuracy on held-out scenarios", fontsize=11, color=INK)
    seeds = anchor["s"]
    titled(
        ax,
        "Routing our 9-model pool: what a cheaper call costs you",
        f"routerbench-ours9 · {round(anchor['nt'])} held-out scenarios/seed · {seeds} seeds · "
        "points are seed means, whiskers +-1 sd",
    )
    style_axes(ax)
    ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    leg = ax.legend(frameon=False, fontsize=9.5, loc="lower right")
    for text in leg.get_texts():
        text.set_color(INK)
    fig.tight_layout()
    return save(fig, "routing_pareto_ours9.png")


def fig_census(groups: list[dict], plt: ModuleType, total_runs: int) -> tuple[Path, Counter]:
    """Figure 2: every group's verdict under the two-axis power rules."""
    counts = Counter(g["tier"] for g in groups if g["tier"] != "anchor")
    order = [
        ("beats", "BEATS baseline", BLUE),
        ("promising", "promising (small test set)", AMBER),
        ("ties", "ties baseline (within spread)", GRAY_MID),
        ("identical", "identical to baseline (never routed away)", GRAY_MID),
        ("mixed", "mixed seeds (unclear)", GRAY_LIGHT),
        ("unfavourable", "unfavourable (small test set)", GRAY_LIGHT),
        ("worse", "WORSE than baseline", RED),
        ("underpowered", "underpowered (under 3 seeds)", GRAY_LIGHT),
    ]
    labels = [label for key, label, _ in order if counts.get(key)]
    values = [counts[key] for key, _, _ in order if counts.get(key)]
    colors = [color for key, _, color in order if counts.get(key)]

    fig, ax = plt.subplots(figsize=(8.4, 4.6), dpi=200)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    ypos = range(len(labels))
    ax.barh(list(ypos), values, color=colors, height=0.62, zorder=3)
    for y, value in zip(ypos, values, strict=True):
        ax.annotate(
            f"{value}",
            xy=(value, y),
            xytext=(6, 0),
            textcoords="offset points",
            va="center",
            ha="left",
            fontsize=10.5,
            color=INK,
            fontweight="bold",
        )
    ax.set_yticks(list(ypos))
    ax.set_yticklabels(labels, fontsize=10.5, color=INK)
    ax.invert_yaxis()
    ax.set_xlabel("seed-aggregated groups", fontsize=11, color=INK)
    ax.set_xlim(0, max(values) * 1.12)
    titled(
        ax,
        "The honest scoreboard: how every configuration actually scored",
        f"{sum(values):,} groups from {total_runs:,} runs · a verdict needs 3+ seeds AND 30+ "
        "test scenarios/seed · paired by seed against best-single",
    )
    style_axes(ax, xgrid=True)
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    return save(fig, "routing_verdict_census.png"), counts


def fig_confidence(plt: ModuleType) -> Path:
    """Figure 3: the confidence dial. Two panels, shared x - never a dual axis (see docstring)."""
    fig, (top, bottom) = plt.subplots(
        2, 1, figsize=(8.4, 5.8), dpi=200, sharex=True, gridspec_kw={"height_ratios": [1, 1.15]}
    )
    fig.patch.set_facecolor("white")
    for ax in (top, bottom):
        ax.set_facecolor("white")

    top.plot(
        CONFIDENCE_Z,
        CONFIDENCE_ROUTED_AWAY,
        "-o",
        color=BLUE,
        linewidth=2.2,
        markersize=6,
        markerfacecolor="white",
        markeredgecolor=BLUE,
        markeredgewidth=1.8,
        zorder=3,
    )
    for x, y in zip(CONFIDENCE_Z, CONFIDENCE_ROUTED_AWAY, strict=True):
        top.annotate(
            f"{y:.1f}%",
            xy=(x, y),
            xytext=(0, 9),
            textcoords="offset points",
            ha="center",
            fontsize=9.5,
            color=INK,
        )
    top.set_ylabel("routed away from fable-5", fontsize=11, color=INK)
    top.set_ylim(20, 82)
    top.yaxis.set_major_formatter(lambda v, _: f"{v:.0f}%")
    titled(
        top,
        "Route only when confident: turning the z dial",
        "routerbench-ours9, fallback fable-5 · higher z = a stricter bar before routing away "
        "· numbers from PR #259 validation",
    )
    style_axes(top)

    bottom.plot(
        CONFIDENCE_Z,
        CONFIDENCE_ACC,
        "-o",
        color=TEAL,
        linewidth=2.2,
        markersize=6,
        markerfacecolor="white",
        markeredgecolor=TEAL,
        markeredgewidth=1.8,
        zorder=3,
        label="routed accuracy",
    )
    bottom.axhline(FABLE_ALONE_ACC, color=MUTED, linewidth=1.6, linestyle="--", zorder=2)
    bottom.annotate(
        f"fable-5 alone, no routing ({FABLE_ALONE_ACC:.1%})",
        xy=(CONFIDENCE_Z[-1], FABLE_ALONE_ACC),
        xytext=(0, 7),
        textcoords="offset points",
        fontsize=9.5,
        color=MUTED,
        ha="right",
    )
    bottom.annotate(
        "routed accuracy",
        xy=(CONFIDENCE_Z[0], CONFIDENCE_ACC[0]),
        xytext=(6, 10),
        textcoords="offset points",
        fontsize=9.5,
        color=TEAL,
        ha="left",
        fontweight="bold",
    )
    best_i = max(range(len(CONFIDENCE_ACC)), key=lambda i: CONFIDENCE_ACC[i])
    bottom.annotate(
        f"peak {CONFIDENCE_ACC[best_i]:.2%} at z={CONFIDENCE_Z[best_i]:g},\n"
        f"routing away only {CONFIDENCE_ROUTED_AWAY[best_i]:.0f}% of calls",
        xy=(CONFIDENCE_Z[best_i], CONFIDENCE_ACC[best_i]),
        xytext=(8, -34),
        textcoords="offset points",
        fontsize=9.5,
        color=TEAL,
        ha="left",
        fontweight="bold",
    )
    bottom.set_xlabel("z (confidence margin required before routing away)", fontsize=11, color=INK)
    bottom.set_ylabel("accuracy", fontsize=11, color=INK)
    bottom.set_ylim(0.895, 0.972)
    bottom.set_xticks(CONFIDENCE_Z)
    bottom.set_xticklabels([f"{z:g}" for z in CONFIDENCE_Z])
    bottom.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    style_axes(bottom)
    fig.tight_layout()
    return save(fig, "routing_confidence_curve.png")


def fig_signal_map(groups: list[dict], plt: ModuleType) -> Path:
    """Figure 4: signal lives where the test set is big. One point per matrix section."""
    per: dict[str, dict[str, float]] = defaultdict(lambda: {"groups": 0, "signal": 0, "n": 0.0})
    for g in groups:
        if g["tier"] == "anchor":
            continue
        row = per[g["m"]]
        row["groups"] += 1
        row["n"] = g["ntmed"]
        if g["tier"] in {"beats", "promising"}:
            row["signal"] += 1

    fig, ax = plt.subplots(figsize=(8.4, 5.2), dpi=200)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    gate = 30
    ax.axvspan(0.7, gate, color=GRID, alpha=0.7, zorder=0)
    ax.annotate(
        "under 30 scenarios/seed a directional\ndelta is only ever a candidate",
        xy=(0.9, 0.97),
        xycoords=("data", "axes fraction"),
        fontsize=9.5,
        color=MUTED,
        ha="left",
        va="top",
    )

    for row in per.values():
        big = row["n"] >= gate
        ax.scatter(
            row["n"],
            row["signal"],
            s=24 + 6.0 * row["groups"] ** 0.5,
            facecolor=BLUE if big else "white",
            edgecolor=BLUE if big else GRAY_MID,
            linewidth=1.6,
            alpha=0.9 if big else 0.75,
            zorder=4,
        )
    for matrix, dx, dy, ha in [
        (HEADLINE_MATRIX, -18, -6, "right"),
        ("tau-bench", 14, 8, "left"),
        ("wm-all", 6, 20, "left"),
    ]:
        if matrix not in per:
            continue
        row = per[matrix]
        ax.annotate(
            f"{matrix}\n{int(row['signal'])} of {int(row['groups'])} groups, "
            f"{row['n']:.0f} scenarios/seed",
            xy=(row["n"], row["signal"]),
            xytext=(dx, dy),
            textcoords="offset points",
            fontsize=9.5,
            color=INK,
            ha=ha,
            fontweight="bold" if matrix == HEADLINE_MATRIX else "normal",
        )

    ax.set_xscale("log")
    ax.set_xlim(0.7, 900)
    ax.set_xlabel("median test scenarios per seed (log)", fontsize=11, color=INK)
    ax.set_ylabel(
        "groups that beat baseline, or would with a real test set", fontsize=11, color=INK
    )
    titled(
        ax,
        "Signal lives where the test set is big",
        "one point per matrix section, sized by how many configurations it holds · "
        "this is why the scaled captures exist",
    )
    style_axes(ax)
    fig.tight_layout()
    return save(fig, "routing_signal_map.png")


def main() -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    groups, total_runs = load_groups()
    written = [fig_pareto(groups, plt)]
    census_path, counts = fig_census(groups, plt, total_runs)
    written.append(census_path)
    written.append(fig_confidence(plt))
    written.append(fig_signal_map(groups, plt))

    stamp = datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC")
    sys.stderr.write(
        f"snapshot {stamp}: {total_runs} runs -> "
        f"{sum(v for k, v in counts.items() if k != 'anchor')} groups\n"
    )
    for key in (
        "beats",
        "promising",
        "ties",
        "identical",
        "worse",
        "unfavourable",
        "mixed",
        "underpowered",
    ):
        sys.stderr.write(f"  {key:14s} {counts[key]}\n")
    for path in written:
        sys.stderr.write(f"wrote {path}\n")


if __name__ == "__main__":
    main()
