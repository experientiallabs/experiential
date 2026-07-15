"""Render a concurrency scaling-law report to a brand-styled matplotlib figure.

`render_report` draws one benchmark's report (`wmh research concurrency --out`) as three panels:
batch wall-clock per side (endpoint seconds labelled), the headline T_real/T_world differential
(parity crossing shaded), and how each side parallelizes (speed-up vs ideal, log-log). The
cross-benchmark story is split into two standalone figures: `render_speedup` (the "what" — how many
times faster the world model is, per benchmark) and `render_cost` (the "why" — reconstruction vs.
real-setup cost). Each is a single clean panel with one left-aligned title (no subtitle). Styling
follows the brand system (AGENTS.md rule 15): white background, near-black ink, hairline grid,
brand-palette accents, left-aligned titles — matching plot_trace_scaling.py.

Needs the `viz` extra (matplotlib/pandas); it is imported lazily by the CLI so the harness runtime
has no plotting dependency. Kept out of `concurrency_scaling.py` so the core experiment stays
deployment-free and fake-testable.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib  # noqa: E402 - grouped with the other plotting deps below
from pydantic import BaseModel, ValidationError

from wmh.research.concurrency_scaling import ConcurrencyScalingReport

matplotlib.use("Agg")  # headless: write a file, never open a window
import matplotlib.pyplot as plt  # noqa: E402 - must follow the Agg backend selection
import matplotlib.ticker as mticker  # noqa: E402
import pandas as pd  # noqa: E402


class _PlotRow(BaseModel):
    """One tidy row for seaborn: a (level, side) timing point of a single report."""

    level: int
    side: str
    wall: float
    wall_std: float
    speedup: float
    efficiency: float
    differential: float


def _load_points(path: str) -> pd.DataFrame:
    """Flatten one report JSON into a tidy long DataFrame (one row per level×side).

    Parses the file through `ConcurrencyScalingReport`, so a malformed or truncated report raises a
    clean `ValueError` (the CLI maps it to a friendly error) rather than a raw KeyError, and missing
    optional fields fall back to their model defaults.
    """
    report = _load_report(path)
    rows: list[_PlotRow] = []
    for point in report.points:
        if point.world_wall_mean:
            rows.append(
                _PlotRow(
                    level=point.level,
                    side="world model",
                    wall=point.world_wall_mean,
                    wall_std=point.world_wall_std,
                    speedup=point.speedup,
                    efficiency=point.efficiency,
                    differential=point.differential,
                )
            )
        if point.real_wall_mean:
            rows.append(
                _PlotRow(
                    level=point.level,
                    side="real sandbox",
                    wall=point.real_wall_mean,
                    wall_std=point.real_wall_std,
                    speedup=0.0,
                    efficiency=0.0,
                    differential=point.differential,
                )
            )
    if not rows:
        raise ValueError(f"no timed points in {path}")
    return pd.DataFrame([r.model_dump() for r in rows])


# Brand system (AGENTS.md rule 15): white bg, near-black ink, hairline grid, brand-palette accents.
# Ref: scripts/plot_trace_scaling.py.
_INK = "#0a0a0a"
_MUTED = "#8a8a8a"
_GRID = "#ececec"
_WORLD_COLOR = "#0070f3"  # world model — primary blue
_REAL_COLOR = "#7928ca"  # real sandbox — purple
_DIFF_COLOR = "#e00"  # differential — red
_IDEAL_COLOR = "#8a8a8a"  # ideal / parity reference — muted grey
# Region tints for the differential panel — light washes of the brand teal (positive) and red.
_FASTER_FILL = "#e8faf6"  # region where the world model is faster (light teal, tint of #50e3c2)
_SLOWER_FILL = "#fdeaea"  # region where the real sandbox is faster (light red, tint of #ee0000)
# Distinct per-benchmark accents for the cross-benchmark combined figure — brand palette, in order.
_BENCH_COLORS = ("#0070f3", "#f5a623", "#ee0000", "#7928ca", "#50e3c2")


def _bench_color(index: int) -> str:
    """Stable brand accent for benchmark `index` in the combined overlay."""
    return _BENCH_COLORS[index % len(_BENCH_COLORS)]


def _fmt_secs(seconds: float) -> str:
    """Compact seconds label for point annotations (e.g. 3.6s, 27.6s, 1224s)."""
    return f"{seconds:.0f}s" if seconds >= 100 else f"{seconds:.1f}s"


def _load_report(path: str) -> ConcurrencyScalingReport:
    """Parse a report JSON, mapping a schema failure to a clean ValueError the CLI can surface."""
    text = Path(path).read_text(encoding="utf-8")
    try:
        return ConcurrencyScalingReport.model_validate_json(text)
    except ValidationError as exc:
        raise ValueError(f"{path} is not a valid concurrency-scaling report: {exc}") from exc


def _styled_line(ax: plt.Axes, xs: list[int], ys: list[float], *, color: str, label: str) -> None:
    """Plot one series in the shared brand marker/line style (used by every panel and figure)."""
    ax.plot(
        xs,
        ys,
        "-o",
        color=color,
        label=label,
        linewidth=2.2,
        markersize=6,
        markerfacecolor="white",
        markeredgecolor=color,
        markeredgewidth=1.6,
        zorder=3,
    )


def _style_panel(ax: plt.Axes, levels: list[int], *, title: str, xlabel: str, ylabel: str) -> None:
    """Apply the shared brand chrome to a panel: hairline grid, no top/right spine, muted ticks."""
    ax.set_title(title, fontsize=13, color=_INK, fontweight="bold", loc="left", pad=12)
    ax.set_xlabel(xlabel, fontsize=11, color=_INK)
    ax.set_ylabel(ylabel, fontsize=11, color=_INK)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(_GRID)
    ax.grid(axis="y", color=_GRID, linewidth=1)
    ax.set_axisbelow(True)
    ax.tick_params(colors=_MUTED, labelsize=10, length=0)
    _style_level_axis(ax, levels)
    leg = ax.get_legend()
    if leg is not None:
        leg.set_frame_on(False)
        for text in leg.get_texts():
            text.set_color(_INK)


def _style_level_axis(ax: plt.Axes, levels: list[int]) -> None:
    """Log2 x-axis with plain integer concurrency ticks (1, 2, 4, ...), shared by every panel."""
    ax.set_xscale("log", base=2)
    ax.set_xticks(levels)
    ax.get_xaxis().set_major_formatter(mticker.ScalarFormatter())
    ax.set_xlim(levels[0] * 0.85, levels[-1] * 1.18)
    ax.margins(y=0.12)


def render_report(path: str, out: str, *, title: str = "Concurrency scaling law") -> str:
    """Render the report JSON at `path` to an image at `out`; return `out`.

    Three panels, all comparing the SAME two sides so the world-model-vs-real-sandbox story threads
    through every one:
      1. batch wall-clock per side (absolute time, log y, endpoint seconds labelled),
      2. the headline time differential T_real/T_world (how many times faster, with the parity
         crossing shaded green=world-faster / red=real-faster),
      3. how each side parallelizes (speed-up vs W=1 against ideal-linear, log-log).
    Fixed benchmark-agnostic styling so all benchmarks line up. When only the world side was timed
    (`--side world`), the real-sandbox series are simply absent and panel 2 says so.
    """
    df = _load_points(path)
    has_real = bool((df["side"] == "real sandbox").any())
    has_diff = bool((df["differential"] > 0).any())
    levels = sorted(df["level"].unique())
    fig, axes = plt.subplots(1, 3, figsize=(19, 5.5), dpi=200)
    fig.patch.set_facecolor("white")

    def line(ax: plt.Axes, xs: list[int], ys: list[float], color: str, label: str) -> None:
        _styled_line(ax, xs, ys, color=color, label=label)

    # 1) Batch wall-clock vs. concurrency (log y), one line per side, mean±std band, endpoint labels
    # so the absolute seconds (and the gap between the two sides) read at a glance.
    ax = axes[0]
    ax.set_facecolor("white")
    for side, color in (("world model", _WORLD_COLOR), ("real sandbox", _REAL_COLOR)):
        grp = df[df["side"] == side].sort_values("level")
        if grp.empty:
            continue
        xs, ys = list(grp["level"]), list(grp["wall"])
        line(ax, xs, ys, color, side)
        lo = [w - s for w, s in zip(grp["wall"], grp["wall_std"], strict=True)]
        hi = [w + s for w, s in zip(grp["wall"], grp["wall_std"], strict=True)]
        ax.fill_between(grp["level"], lo, hi, color=color, alpha=0.12, linewidth=0, zorder=2)
        for x, y in ((xs[0], ys[0]), (xs[-1], ys[-1])):  # label the W=1 and W=max endpoints
            ax.annotate(
                _fmt_secs(y),
                (x, y),
                textcoords="offset points",
                xytext=(0, 10),
                ha="center",
                fontsize=8.5,
                color=color,
                fontweight="bold",
            )
    ax.set_yscale("log")
    ax.legend(loc="best", fontsize=10)
    _style_panel(
        ax,
        levels,
        title="Wall-clock to obtain the batch (log scale, lower = faster)",
        xlabel="concurrency (scenarios at once)",
        ylabel="batch wall-clock (s)",
    )

    # 2) The headline: time differential T_real / T_world — how many times faster the world model.
    # Shade the world-faster (>1) and real-faster (<1) regions so the parity crossing reads at a
    # glance (tau dips below 1 at high W; terminal/swe stay above). Log y when the range is wide.
    ax = axes[1]
    ax.set_facecolor("white")
    diff = df[(df["side"] == "real sandbox") & (df["differential"] > 0)].sort_values("level")
    if has_diff and not diff.empty:
        dl, dvals = list(diff["level"]), list(diff["differential"])
        line(ax, dl, dvals, _DIFF_COLOR, "T_real / T_world")
        for lvl, d in zip(dl, dvals, strict=True):
            ax.annotate(
                f"{d:.2f}×",
                (lvl, d),
                textcoords="offset points",
                xytext=(6, 6),
                fontsize=9,
                color=_INK,
                fontweight="bold",
            )
        # Log y only when the curve's OWN dynamic range is wide (max/min >= 8) — keying on absolute
        # magnitude would log a high-but-flat curve (e.g. swe ~70-138×, ~2× spread) and cram it into
        # the top of the panel with empty decades below. These curves read best on a linear axis.
        if max(dvals) / min(dvals) >= 8:
            ax.set_yscale("log")
            ax.yaxis.set_major_formatter(mticker.ScalarFormatter())
        lo = min(0.9, min(dvals) * 0.85)
        hi = max(dvals) * 1.25
        ax.set_ylim(lo, hi)
        ax.axhspan(1.0, hi, color=_FASTER_FILL, zorder=0)  # world model faster
        if lo < 1.0:
            ax.axhspan(lo, 1.0, color=_SLOWER_FILL, zorder=0)  # real sandbox faster
        ax.axhline(1.0, linestyle=(0, (4, 3)), color=_IDEAL_COLOR, linewidth=1.6, label="parity")
        ax.legend(loc="best", fontsize=10)
    else:
        msg = (
            "world-model side only\n(run `--side both` for\nthe sandbox differential)"
            if not has_real
            else "no real-sandbox timings"
        )
        ax.text(0.5, 0.5, msg, ha="center", va="center", transform=ax.transAxes, color=_MUTED)
    _style_panel(
        ax,
        levels,
        title="World model speed-up over the real sandbox",
        xlabel="concurrency",
        ylabel="T_real / T_world  (>1 = world model faster)",
    )

    # 3) Mechanism: how each side parallelizes — speed-up vs W=1 against the ideal-linear diagonal,
    # log-log so ideal is a straight 45° reference and the sub-linear curves stay legible.
    ax = axes[2]
    ax.set_facecolor("white")
    ax.plot(
        levels,
        levels,
        linestyle=(0, (4, 3)),
        color=_IDEAL_COLOR,
        linewidth=1.6,
        label="ideal (linear)",
        zorder=1,
    )
    baseline_level = levels[0]
    for side, color in (("world model", _WORLD_COLOR), ("real sandbox", _REAL_COLOR)):
        grp = df[df["side"] == side].sort_values("level")
        base_rows = grp[grp["level"] == baseline_level]
        if grp.empty or base_rows.empty:
            # No timing at the baseline level -> can't normalize this side to the same W as the
            # other, so skip it rather than plot a curve on a different (higher) baseline.
            continue
        base = float(base_rows.iloc[0]["wall"])
        speedup = [base / w if w else 0.0 for w in grp["wall"]]
        line(ax, list(grp["level"]), speedup, color, side)
    ax.set_yscale("log", base=2)
    ax.yaxis.set_major_formatter(mticker.ScalarFormatter())
    ax.legend(loc="upper left", fontsize=10)
    _style_panel(
        ax,
        levels,
        title="How each side parallelizes (speed-up vs W=1)",
        xlabel="concurrency",
        ylabel="speed-up vs. W=1 (log)",
    )

    fig.suptitle(title, fontsize=15, color=_INK, fontweight="bold", x=0.02, ha="left")
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out


def _load_reports(paths: list[str]) -> tuple[list[tuple[str, ConcurrencyScalingReport]], list[int]]:
    """Load each report JSON (labelled by its `benchmark` field, else file stem) + shared levels.

    Raises a clean `ValueError` (surfaced by the CLI as a friendly error) when the list is empty or
    no report has a single timed point.
    """
    reports: list[tuple[str, ConcurrencyScalingReport]] = []
    for path in paths:
        report = _load_report(path)
        reports.append((report.benchmark or Path(path).stem, report))
    if not reports:
        raise ValueError("no reports to plot")
    levels = sorted({p.level for _, r in reports for p in r.points})
    if not levels:
        raise ValueError("reports have no timed points to plot")
    return reports, levels


def _header(fig: plt.Figure, title: str) -> None:
    """One clean left-aligned title at the top of a single-panel figure (Vercel/Notion minimal).

    No subtitle or caption: the title carries the message, the chart carries the rest.
    `subplots_adjust` reserves a generous top band so the title breathes above the axes.
    """
    fig.text(0.02, 0.955, title, fontsize=14.5, fontweight="bold", color=_INK, ha="left", va="top")
    fig.subplots_adjust(top=0.83, left=0.1, right=0.93, bottom=0.13)


def render_speedup(
    paths: list[str],
    out: str,
    *,
    title: str = "How much faster is the world model than the real environment?",
) -> str:
    """Render the cross-benchmark speed-up figure (T_real/T_world per benchmark) to `out`.

    One line per benchmark on a log y-axis: how many times faster the world model is than standing
    up the real environment, at each concurrency level. The region above the parity line is tinted
    (world model faster); a line dipping below it (e.g. tau-bench's cheap in-process env) is where
    the real sandbox wins. Lines are directly labelled at their right end, so the figure needs no
    legend lookup. This is the "what"; `render_cost` is the "why".
    """
    reports, levels = _load_reports(paths)
    fig, ax = plt.subplots(figsize=(8.6, 5.6), dpi=200)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    all_diffs: list[float] = []
    for idx, (label, report) in enumerate(reports):
        pts = [
            (p.level, p.differential)
            for p in report.points
            if p.differential and p.differential > 0
        ]
        if not pts:
            continue
        xs = [lvl for lvl, _ in pts]
        ys = [v for _, v in pts]
        all_diffs.extend(ys)
        color = _bench_color(idx)
        _styled_line(ax, xs, ys, color=color, label=label)
        for x, y in ((xs[0], ys[0]), (xs[-1], ys[-1])):  # value-label both endpoints
            ax.annotate(
                f"{y:.1f}×",
                (x, y),
                textcoords="offset points",
                xytext=(0, 9),
                ha="center",
                fontsize=8.5,
                color=color,
                fontweight="bold",
            )
        # Direct-label the line at its right end (replaces a legend the reader has to cross-check).
        ax.annotate(
            label,
            (xs[-1], ys[-1]),
            textcoords="offset points",
            xytext=(12, -1),
            ha="left",
            va="center",
            fontsize=10,
            color=color,
            fontweight="bold",
            annotation_clip=False,
        )

    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(mticker.ScalarFormatter())
    if all_diffs:
        hi = max(all_diffs) * 1.7
        lo = min(0.9, min(all_diffs) * 0.82)
        ax.set_ylim(lo, hi)
        # Colour alone carries the meaning (no labels): teal above parity = world model faster,
        # red below = real environment faster.
        ax.axhspan(1.0, hi, color=_FASTER_FILL, zorder=0)
        if lo < 1.0:
            ax.axhspan(lo, 1.0, color=_SLOWER_FILL, zorder=0)
    ax.axhline(1.0, linestyle=(0, (4, 3)), color=_IDEAL_COLOR, linewidth=1.4)
    ax.annotate(
        "parity (1×)",
        (levels[-1], 1.0),
        textcoords="offset points",
        xytext=(0, 6),
        ha="right",
        fontsize=8.5,
        color=_MUTED,
    )
    _style_panel(
        ax,
        levels,
        title="",
        xlabel="concurrency (scenarios reconstructed at once)",
        ylabel="times faster   (T_real / T_world)",
    )
    ax.set_xlim(levels[0] * 0.85, levels[-1] * 1.75)  # room for the right-end benchmark labels
    _header(fig, title)
    fig.savefig(out, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out


def render_cost(
    paths: list[str],
    out: str,
    *,
    title: str = "Why: reconstructing vs. standing up the real environment",
) -> str:
    """Render the cross-benchmark cost figure (world vs. real wall-clock at W=1) to `out`.

    Grouped log-y bars per benchmark: the world-model reconstruction cost (blue) beside the real
    environment's standup cost (purple), at the lowest concurrency level. Where the purple bar
    towers over the blue one the world model wins; where it is shorter it loses. The per-benchmark
    speed-up (×) is annotated above each pair to tie this mechanism back to `render_speedup`.
    """
    reports, levels = _load_reports(paths)
    base_level = levels[0]
    names: list[str] = []
    w_costs: list[float] = []
    r_costs: list[float] = []
    for label, report in reports:
        pt = next((p for p in report.points if p.level == base_level), None)
        if pt is None:
            continue
        names.append(label)
        w_costs.append(pt.world_wall_mean or 0.0)
        r_costs.append(pt.real_wall_mean or 0.0)
    if not names:
        raise ValueError(f"no report has a point at the baseline concurrency W={base_level}")

    fig, ax = plt.subplots(figsize=(8.6, 5.6), dpi=200)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    xs = list(range(len(names)))
    width = 0.36
    ax.bar(
        [i - width / 2 for i in xs],
        w_costs,
        width,
        color=_WORLD_COLOR,
        label="world model (reconstruct)",
        zorder=3,
    )
    ax.bar(
        [i + width / 2 for i in xs],
        r_costs,
        width,
        color=_REAL_COLOR,
        label="real environment (stand up)",
        zorder=3,
    )
    ax.set_yscale("log")

    def _bar_label(x: float, value: float, color: str) -> None:
        if value <= 0:
            return
        ax.annotate(
            _fmt_secs(value),
            (x, value),
            textcoords="offset points",
            xytext=(0, 3),
            ha="center",
            fontsize=8.5,
            color=color,
            fontweight="bold",
        )

    top = max([*w_costs, *r_costs, 1.0])
    for i, (w, r) in enumerate(zip(w_costs, r_costs, strict=True)):
        _bar_label(i - width / 2, w, _WORLD_COLOR)
        _bar_label(i + width / 2, r, _REAL_COLOR)
        if w > 0 and r > 0:  # speed-up (×) just above the pair's taller bar, tying back to fig 1
            ax.annotate(
                f"{r / w:.1f}× faster" if r >= w else f"{r / w:.2f}× (slower)",
                (i, max(w, r) * 1.6),
                ha="center",
                fontsize=9.5,
                color=_INK,
                fontweight="bold",
            )
    ax.set_ylim(top=top * 3.2)  # headroom so the tallest pair's ×-label clears the top
    _style_panel(ax, levels, title="", xlabel="", ylabel="batch wall-clock at W=1  (s, log)")
    ax.set_xscale("linear")  # categorical x (benchmarks), overriding the shared log-level axis
    ax.set_xticks(xs)
    ax.set_xticklabels(names, fontsize=11, color=_INK)
    ax.set_xlim(-0.6, len(names) - 0.4)
    # Upper-left: the empty band above the shortest (leftmost) pair, clear of every ×-label.
    ax.legend(loc="upper left", fontsize=10, frameon=False, labelcolor=_INK)
    _header(fig, title)
    fig.savefig(out, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out


__all__ = ["render_cost", "render_report", "render_speedup"]
