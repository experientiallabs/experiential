"""Render a `GridResult` as the World-Model Harness fidelity bar chart.

One vertical bar per (model x condition) cell, sorted ASCENDING by fidelity left->right, each
labeled with its fidelity and (when priced) its target-side cost. Matplotlib/seaborn live behind
the `viz` extra, so this module imports them lazily inside the function — the only sanctioned lazy
import here (the engine must import without the plotting deps installed).
"""

from __future__ import annotations

from pathlib import Path

from wmh.engine.grid import GridResult

_TITLE = "World-Model Harness Fidelity"


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

    sns.set_theme(style="whitegrid", context="talk")
    palette = sns.color_palette("deep", len(cells))
    fig, ax = plt.subplots(figsize=(max(8, 1.6 * len(cells)), 6.5))
    bars = ax.bar(range(len(cells)), heights, color=palette, edgecolor="white", linewidth=0.8)

    # Per-bar text: fidelity on top, target cost below it (omit the $ line when cost is None).
    for bar, cell in zip(bars, cells, strict=True):
        x = bar.get_x() + bar.get_width() / 2
        top = bar.get_height()
        ax.text(x, top + 0.012, f"{cell.fidelity:.3f}", ha="center", va="bottom", fontsize=11)
        if cell.cost_usd is not None and cell.cost_usd > 0:
            ax.text(
                x,
                top + 0.055,
                f"${cell.cost_usd:.2f}",
                ha="center",
                va="bottom",
                fontsize=10,
                color="#444",
            )

    ax.set_xticks(range(len(cells)))
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel("Mean fidelity")
    ax.set_ylim(0, 1.0)
    ax.set_title(_TITLE, fontsize=17, fontweight="bold", pad=28)
    subtitle = (
        f"{dataset_label} | {n_test_traces} held-out test traces | "
        f"{result.total_test_steps} judged steps"
    )
    ax.text(
        0.5,
        1.02,
        subtitle,
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        fontsize=12,
        color="#555",
    )
    sns.despine(ax=ax)
    fig.tight_layout()
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out
