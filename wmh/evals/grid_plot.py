"""Render a `GridResult` as the World-Model Harness fidelity bar chart.

One vertical bar per (model x condition) cell, sorted ASCENDING by fidelity left->right, each
labeled with its fidelity and (when priced) its target-side cost. Matplotlib/seaborn live behind
the `viz` extra, so this module imports them lazily inside the function - the only sanctioned lazy
import here (the engine must import without the plotting deps installed).
"""

from __future__ import annotations

from pathlib import Path

from wmh.evals.grid import CONDITIONS, GridResult

_TITLE = "World-Model Harness Fidelity"

# Brand palette (AGENTS.md rule 15) - no ad-hoc colors. Ink for text/lines, a light gridline, and
# one brand hue per condition so a bar's color reads its condition regardless of sorted position.
_INK = "#0a0a0a"
_GRIDLINE = "#ececec"
_BRAND_BY_CONDITION = {
    "base": _INK,  # baseline is neutral; the wmh conditions carry the brand hues
    "base_rag": "#0070f3",  # blue
    "gepa": "#7928ca",  # purple
    "gepa_rag": "#f5a623",  # amber
}


def plot_grid(
    result: GridResult,
    out_path: str | Path,
    *,
    dataset_label: str,
    n_test_traces: int,
) -> Path:
    """Write the fidelity barplot for `result` to `out_path` (PNG). Returns the path.

    `dataset_label`/`n_test_traces` populate the subtitle, e.g.
    "armand0e/qwen3.7-max-pi-traces | 8 held-out test traces | 225 judged steps".
    """
    import matplotlib

    matplotlib.use("Agg")  # headless: write a file, never open a window
    import matplotlib.pyplot as plt
    import seaborn as sns

    cells = sorted(result.cells, key=lambda c: c.fidelity)  # ascending performance, left -> right
    if not cells:
        raise ValueError("grid result has no cells to plot")
    labels = [c.bar_label for c in cells]
    heights = [c.fidelity for c in cells]
    colors = [_BRAND_BY_CONDITION.get(c.condition, _INK) for c in cells]

    sns.set_theme(style="whitegrid", context="talk")
    fig, ax = plt.subplots(figsize=(max(8, 1.6 * len(cells)), 6.5))
    ax.grid(axis="y", color=_GRIDLINE)
    ax.set_axisbelow(True)
    bars = ax.bar(range(len(cells)), heights, color=colors, edgecolor="white", linewidth=0.8)

    # Per-bar text: fidelity on top, target cost above it (omit the $ line when cost is None).
    for bar, cell in zip(bars, cells, strict=True):
        x = bar.get_x() + bar.get_width() / 2
        top = bar.get_height()
        ax.text(x, top + 0.012, f"{cell.fidelity:.3f}", ha="center", va="bottom", fontsize=11)
        if cell.cost_usd is not None and cell.cost_usd > 0:
            ax.text(
                x,
                top + 0.052,
                f"${cell.cost_usd:.2f}",
                ha="center",
                va="bottom",
                fontsize=10,
                color=_INK,
            )

    # One legend entry per condition present, in canonical order, using the same brand hues.
    seen = {c.condition: c.condition_label for c in cells}
    handles = [
        plt.Rectangle((0, 0), 1, 1, color=_BRAND_BY_CONDITION.get(cond, _INK))
        for cond in CONDITIONS
        if cond in seen
    ]
    ax.legend(
        handles,
        [seen[cond] for cond in CONDITIONS if cond in seen],
        title="condition",
        frameon=False,
        fontsize=10,
        title_fontsize=10,
        loc="upper left",
    )

    ax.set_xticks(range(len(cells)))
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel("Mean fidelity")
    # Headroom above the tallest bar so the fidelity + cost labels never collide with the subtitle
    # (near-ceiling bars at ~0.97 would otherwise push the $ label past y=1.0).
    ax.set_ylim(0, max(1.0, max(heights) + 0.14))
    ax.set_title(_TITLE, fontsize=17, fontweight="bold", pad=28)
    subtitle = (
        f"{dataset_label} | {n_test_traces} held-out test traces | "
        f"{result.total_test_steps} judged steps | judge {result.judge_version}"
    )
    ax.text(
        0.5,
        1.02,
        subtitle,
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        fontsize=12,
        color=_INK,
        alpha=0.7,
    )
    sns.despine(ax=ax)
    fig.tight_layout()
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_grid_heatmap(
    results: dict[str, GridResult],
    out_path: str | Path,
    *,
    benchmark_order: list[str] | None = None,
) -> Path:
    """Render the whole grid as one heatmap: rows = model x condition, columns = benchmark.

    `results` maps a benchmark label to its merged `GridResult` (all 5 models x 4 conditions). Each
    cell is the mean fidelity, annotated and colored on a shared 0..1 scale so every benchmark reads
    on the same footing. A model x condition with no cell for a benchmark is left blank (NaN).
    Columns follow `benchmark_order` when given, else the dict's insertion order.
    """
    import matplotlib

    matplotlib.use("Agg")  # headless
    import matplotlib.pyplot as plt
    import numpy as np
    import seaborn as sns
    from matplotlib.colors import LinearSegmentedColormap

    if not results:
        raise ValueError("no grid results to plot")
    # Sequential brand ramp (light gridline -> brand blue -> brand purple) so higher fidelity reads
    # darker/brand-er; seaborn auto-picks light/dark annotation text per cell luminance.
    brand_cmap = LinearSegmentedColormap.from_list("wmh-brand", [_GRIDLINE, "#0070f3", "#7928ca"])
    benchmarks = benchmark_order or list(results)

    # Row order: models in first-seen order, each followed by its four conditions.
    models: list[str] = []
    labels: dict[str, str] = {}
    for res in results.values():
        for cell in res.cells:
            if cell.model_label not in models:
                models.append(cell.model_label)
            labels.setdefault(cell.condition, cell.condition_label)
    rows = [(m, c) for m in models for c in CONDITIONS]

    # (model, condition, benchmark) -> fidelity lookup.
    fid: dict[tuple[str, str, str], float] = {}
    for bench, res in results.items():
        for cell in res.cells:
            fid[(cell.model_label, cell.condition, bench)] = cell.fidelity

    matrix = np.full((len(rows), len(benchmarks)), np.nan)
    for r, (model, cond) in enumerate(rows):
        for c, bench in enumerate(benchmarks):
            if (model, cond, bench) in fid:
                matrix[r, c] = fid[(model, cond, bench)]
    row_labels = [f"{m}  ·  {labels.get(c, c)}" for m, c in rows]

    sns.set_theme(style="white", context="talk")
    fig, ax = plt.subplots(figsize=(1.7 * len(benchmarks) + 4, 0.42 * len(rows) + 2.2))
    sns.heatmap(
        matrix,
        ax=ax,
        cmap=brand_cmap,
        vmin=0.0,
        vmax=1.0,
        annot=True,
        fmt=".2f",
        annot_kws={"fontsize": 9},
        linewidths=0.6,
        linecolor="white",
        cbar_kws={"label": "Mean fidelity", "shrink": 0.6},
        xticklabels=benchmarks,
        yticklabels=row_labels,
    )
    # Separator lines between models (every 4 conditions) so the model blocks read as groups.
    for i in range(len(CONDITIONS), len(rows), len(CONDITIONS)):
        ax.axhline(i, color=_INK, linewidth=1.4)
    judge_versions = sorted({r.judge_version for r in results.values()})
    judge_tag = judge_versions[0] if len(judge_versions) == 1 else "/".join(judge_versions)
    ax.set_title(f"{_TITLE}: full grid (judge {judge_tag})", fontsize=17, fontweight="bold", pad=16)
    ax.tick_params(axis="y", labelsize=9, rotation=0)
    ax.tick_params(axis="x", labelsize=11)
    fig.tight_layout()
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out
