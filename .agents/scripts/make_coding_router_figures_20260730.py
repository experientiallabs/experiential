"""Create plain Matplotlib figures and tables for the coding-router experiment."""

# The plotting code intentionally uses Matplotlib's dynamic axes and long chart annotations.
# ruff: noqa: E501, ANN401, B905, I001

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


COLORS = {
    "baseline": "#6b7280",
    "candidate": "#2563eb",
    "target": "#b45309",
    "secondary": "#0f766e",
    "grid": "#d1d5db",
    "text": "#111827",
}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trials", type=Path, required=True)
    parser.add_argument("--frontier", type=Path, required=True)
    parser.add_argument("--proxy-report", type=Path, required=True)
    parser.add_argument("--swe-leaderboard", type=Path, required=True)
    parser.add_argument("--fresh-candidate", type=Path, required=True)
    parser.add_argument("--fresh-baseline", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def _load_analysis_helpers() -> Any:
    script = Path(__file__).with_name("analyze_fast_proxy_and_router_20260729.py")
    spec = importlib.util.spec_from_file_location("fast_proxy_analysis", script)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {script}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _percent(value: float) -> str:
    return f"{value * 100:.1f}%"


def _style_axes(ax: Any) -> None:
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color=COLORS["grid"], linewidth=0.7, alpha=0.7)
    ax.set_axisbelow(True)
    ax.tick_params(colors=COLORS["text"])
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_color(COLORS["text"])


def _aggregate_figure(frontier: dict[str, Any], path: Path) -> None:
    baseline = frontier["baseline"]
    candidate = frontier["candidate"]
    labels = ["GPT-5.5\nxhigh", "GPT-5.6 Sol\nxhigh"]
    quality = [baseline["quality"] * 100, candidate["quality"] * 100]
    cost = [baseline["cost_usd_per_task"], candidate["cost_usd_per_task"]]

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.8), constrained_layout=True)
    fig.suptitle("DeepSWE 1.1 router frontier", fontsize=15, fontweight="bold", color=COLORS["text"])

    bars = axes[0].bar(labels, quality, color=[COLORS["baseline"], COLORS["candidate"]], width=0.58)
    axes[0].set_title("Whole-task quality", loc="left", fontweight="bold")
    axes[0].set_ylabel("Pass rate (%)")
    axes[0].set_ylim(0, 82)
    axes[0].axhline(95 * baseline["quality"], color=COLORS["target"], linestyle="--", linewidth=1.2)
    axes[0].text(1.43, 95 * baseline["quality"] + 1, "95% of baseline", color=COLORS["target"], fontsize=9, ha="right")
    for bar, value in zip(bars, quality):
        axes[0].text(bar.get_x() + bar.get_width() / 2, value + 1.5, f"{value:.1f}%", ha="center", fontsize=10)
    axes[0].text(0.5, 76.5, f"Candidate: +{(candidate['quality'] - baseline['quality']) * 100:.2f} pp", ha="center", fontsize=9, color=COLORS["candidate"])
    _style_axes(axes[0])

    bars = axes[1].bar(labels, cost, color=[COLORS["baseline"], COLORS["candidate"]], width=0.58)
    axes[1].set_title("Cost per task", loc="left", fontweight="bold")
    axes[1].set_ylabel("USD")
    axes[1].set_ylim(0, 8.5)
    for bar, value in zip(bars, cost):
        axes[1].text(bar.get_x() + bar.get_width() / 2, value + 0.18, f"${value:.2f}", ha="center", fontsize=10)
    axes[1].text(0.5, 7.7, f"Savings: {1 - candidate['cost_usd_per_task'] / baseline['cost_usd_per_task']:.1%}", ha="center", fontsize=10, color=COLORS["candidate"])
    _style_axes(axes[1])
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _model_breakdown_figure(
    analysis: Any,
    trials: Path,
    proxy_report: dict[str, Any],
    swe_leaderboard: Path,
    frontier: dict[str, Any],
    path: Path,
) -> list[dict[str, Any]]:
    grouped = analysis._load_trials(trials.resolve())
    tasks, arms, quality, _ = analysis._task_scores(grouped)
    proxy_arms = list(analysis.MAPPED_ARMS)
    swe_scores = analysis._latest_swe_scores(swe_leaderboard.resolve())
    proxy_tasks = proxy_report["fast_proxy"]["12"]["task_ids"]
    proxy_quality = analysis._arm_scores(proxy_tasks, proxy_arms, quality)
    full_quality = analysis._arm_scores(tasks, proxy_arms, quality)
    costs = {
        arm: float(
            np.mean(
                [row["cost_usd"] for task in tasks for row in grouped[task][arm]]
            )
        )
        for arm in proxy_arms
    }
    rows = []
    for arm in proxy_arms:
        model, effort = analysis.MAPPED_ARMS[arm]
        rows.append(
            {
                "arm": arm,
                "model": model,
                "reasoning_effort": effort,
                "tasks": len(tasks),
                "offline_trial_rows": len(tasks) * len(grouped[tasks[0]][arm]),
                "offline_allocation_share": 1 / len(proxy_arms),
                "deep_swe_quality": full_quality[arm],
                "fast_proxy_quality": proxy_quality[arm],
                "cost_usd_per_task": costs[arm],
                "swe_bench_resolved_rate": swe_scores.get(arm),
            }
        )
    rows += [
        {
            "arm": "GPT-5.5 xhigh baseline",
            "model": frontier["baseline"]["model"],
            "reasoning_effort": frontier["baseline"]["reasoning_effort"],
            "tasks": frontier["provenance"]["task_count"],
            "offline_trial_rows": frontier["provenance"]["task_count"],
            "offline_allocation_share": None,
            "deep_swe_quality": frontier["baseline"]["quality"],
            "fast_proxy_quality": None,
            "cost_usd_per_task": frontier["baseline"]["cost_usd_per_task"],
            "swe_bench_resolved_rate": None,
        },
        {
            "arm": "GPT-5.6 Sol xhigh candidate",
            "model": frontier["candidate"]["model"],
            "reasoning_effort": frontier["candidate"]["reasoning_effort"],
            "tasks": frontier["provenance"]["task_count"],
            "offline_trial_rows": frontier["provenance"]["task_count"],
            "offline_allocation_share": None,
            "deep_swe_quality": frontier["candidate"]["quality"],
            "fast_proxy_quality": None,
            "cost_usd_per_task": frontier["candidate"]["cost_usd_per_task"],
            "swe_bench_resolved_rate": None,
        },
    ]
    display_rows = sorted(rows, key=lambda row: row["deep_swe_quality"])
    y = np.arange(len(display_rows))
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 8.2), sharey=True, constrained_layout=True)
    fig.suptitle("Offline model and reasoning-effort breakdown", fontsize=15, fontweight="bold", color=COLORS["text"])
    labels = [row["arm"] for row in display_rows]
    quality_values = [row["deep_swe_quality"] * 100 for row in display_rows]
    cost_values = [row["cost_usd_per_task"] for row in display_rows]
    colors = []
    for row in display_rows:
        if row["arm"] == "GPT-5.6 Sol xhigh candidate":
            colors.append(COLORS["candidate"])
        elif row["arm"] == "GPT-5.5 xhigh baseline":
            colors.append(COLORS["baseline"])
        else:
            colors.append(COLORS["secondary"])
    axes[0].barh(y, quality_values, color=colors, height=0.66)
    axes[0].set_title("DeepSWE quality", loc="left", fontweight="bold")
    axes[0].set_xlabel("Whole-task pass rate (%)")
    axes[0].set_xlim(0, 82)
    axes[0].set_yticks(y, labels, fontsize=8)
    for index, value in enumerate(quality_values):
        axes[0].text(value + 0.8, index, f"{value:.1f}%", va="center", fontsize=8)
    axes[1].barh(y, cost_values, color=colors, height=0.66)
    axes[1].set_title("Cost", loc="left", fontweight="bold")
    axes[1].set_xlabel("USD per task")
    axes[1].set_xlim(0, max(cost_values) * 1.18)
    for index, value in enumerate(cost_values):
        axes[1].text(value + 0.12, index, f"${value:.2f}", va="center", fontsize=8)
    for ax in axes:
        _style_axes(ax)
        ax.grid(axis="x", color=COLORS["grid"], linewidth=0.7, alpha=0.7)
        ax.grid(axis="y", visible=False)
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return rows


def _validation_figure(
    frontier: dict[str, Any],
    proxy_report: dict[str, Any],
    trials: Path,
    swe_leaderboard: Path,
    path: Path,
) -> None:
    splits = frontier["heldout_splits"]
    seeds = [str(row["seed"]) for row in splits]
    retention = [row["quality_ratio"] * 100 for row in splits]
    savings = [row["cost_savings"] * 100 for row in splits]

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.8), constrained_layout=True)
    fig.suptitle("Validation stability and benchmark correlation", fontsize=15, fontweight="bold", color=COLORS["text"])
    x = np.arange(len(seeds))
    width = 0.36
    axes[0].bar(x - width / 2, retention, width, label="Quality retained", color=COLORS["candidate"])
    axes[0].bar(x + width / 2, savings, width, label="Cost savings", color=COLORS["secondary"])
    axes[0].axhline(95, color=COLORS["target"], linestyle="--", linewidth=1.1)
    axes[0].axhline(30, color=COLORS["target"], linestyle=":", linewidth=1.1)
    axes[0].set_xticks(x, [f"seed {seed}" for seed in seeds])
    axes[0].set_ylim(0, 112)
    axes[0].set_ylabel("Percent")
    axes[0].set_title("Five held-out splits", loc="left", fontweight="bold")
    axes[0].legend(frameon=False, fontsize=8, loc="lower right")
    for index, value in enumerate(retention):
        axes[0].text(index - width / 2, value + 1.4, f"{value:.1f}", ha="center", fontsize=8)
    for index, value in enumerate(savings):
        axes[0].text(index + width / 2, value + 1.4, f"{value:.1f}", ha="center", fontsize=8)
    _style_axes(axes[0])

    analysis = _load_analysis_helpers()
    proxy_tasks = proxy_report["fast_proxy"]["12"]["task_ids"]
    grouped = analysis._load_trials(trials.resolve())
    tasks, arms, quality, _ = analysis._task_scores(grouped)
    proxy_arms = list(analysis.MAPPED_ARMS)
    proxy_scores = analysis._arm_scores(proxy_tasks, proxy_arms, quality)
    swe = analysis._latest_swe_scores(swe_leaderboard.resolve())
    mapped = [arm for arm in proxy_arms if arm in swe]
    x_values = [proxy_scores[arm] * 100 for arm in mapped]
    y_values = [swe[arm] for arm in mapped]
    axes[1].scatter(x_values, y_values, color=COLORS["candidate"], s=34, alpha=0.9)
    for arm, xv, yv in zip(mapped, x_values, y_values):
        label = arm.replace("gpt-5.5-2026-04-23-", "GPT-5.5 ").replace("gpt-5.4-2026-03-05-", "GPT-5.4 ")
        axes[1].annotate(label, (xv, yv), xytext=(4, 4), textcoords="offset points", fontsize=6.5)
    axes[1].set_title("12-task proxy vs SWE-bench", loc="left", fontweight="bold")
    axes[1].set_xlabel("Fast-proxy pass rate (%)")
    axes[1].set_ylabel("SWE-bench resolved rate (%)")
    correlation = proxy_report["fast_proxy"]["12"]["swe_bench_correlation"]
    axes[1].text(
        0.03,
        0.96,
        f"Spearman rho = {correlation['spearman_rho']:.3f}\np = {correlation['spearman_p']:.2g}",
        transform=axes[1].transAxes,
        va="top",
        fontsize=9,
        bbox={"facecolor": "#f3f4f6", "edgecolor": "none", "pad": 4},
    )
    _style_axes(axes[1])
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _write_tables(
    output_dir: Path,
    frontier: dict[str, Any],
    model_rows: list[dict[str, Any]],
    candidate_fresh: dict[str, Any],
    baseline_fresh: dict[str, Any],
    proxy_report: dict[str, Any],
) -> None:
    summary_rows = [
        {"metric": "benchmark", "value": "DeepSWE 1.1", "baseline": "", "candidate": "", "notes": "113-task historical/shared ledger"},
        {"metric": "whole-task quality", "value": "", "baseline": _percent(frontier["baseline"]["quality"]), "candidate": _percent(frontier["candidate"]["quality"]), "notes": "candidate minus baseline: +3.76 percentage points"},
        {"metric": "cost per task", "value": "USD", "baseline": f"${frontier['baseline']['cost_usd_per_task']:.2f}", "candidate": f"${frontier['candidate']['cost_usd_per_task']:.2f}", "notes": f"candidate savings: {frontier['overall']['cost_savings']:.1%}"},
        {"metric": "mean held-out quality ratio", "value": "ratio", "baseline": "1.000", "candidate": f"{frontier['summary']['mean_split_quality_ratio']:.3f}", "notes": "five deterministic 70/30 splits"},
        {"metric": "mean held-out cost savings", "value": "percent", "baseline": "0.0%", "candidate": _percent(frontier["summary"]["mean_split_cost_savings"]), "notes": f"weakest split: {frontier['summary']['worst_split_cost_savings']:.1%}"},
        {"metric": "fast proxy to SWE-bench", "value": "Spearman rho", "baseline": "", "candidate": f"{proxy_report['fast_proxy']['12']['swe_bench_correlation']['spearman_rho']:.3f}", "notes": f"p={proxy_report['fast_proxy']['12']['swe_bench_correlation']['spearman_p']:.2g}, n=12 arms"},
        {"metric": "fresh verifier probe", "value": "reward", "baseline": str(baseline_fresh["score"]), "candidate": str(candidate_fresh["score"]), "notes": "one matched task; both 163/182 tests, 0/14 feature tests"},
    ]
    prefix = "coding-router-small-agent-20260730-"
    _write_csv(output_dir / f"{prefix}summary.csv", list(summary_rows[0]), summary_rows)
    _write_csv(
        output_dir / f"{prefix}heldout-splits.csv",
        ["seed", "heldout_task_count", "quality_ratio", "cost_savings", "baseline_quality", "candidate_quality"],
        [
            {
                "seed": row["seed"],
                "heldout_task_count": row["heldout_task_count"],
                "quality_ratio": f"{row['quality_ratio']:.6f}",
                "cost_savings": f"{row['cost_savings']:.6f}",
                "baseline_quality": f"{row['baseline_quality']:.6f}",
                "candidate_quality": f"{row['candidate_quality']:.6f}",
            }
            for row in frontier["heldout_splits"]
        ],
    )
    _write_csv(
        output_dir / f"{prefix}fresh-verifier.csv",
        ["arm", "model", "reasoning_effort", "task", "reward", "feature_tests", "passed_tests", "total_tests", "infra_failed"],
        [
            {
                "arm": "candidate",
                "model": candidate_fresh["logical_model"],
                "reasoning_effort": candidate_fresh["reasoning_effort"],
                "task": candidate_fresh["task_ids"][0],
                "reward": candidate_fresh["score"],
                "feature_tests": "0/14",
                "passed_tests": 163,
                "total_tests": 182,
                "infra_failed": candidate_fresh["cells"][0]["infra_failed"],
            },
            {
                "arm": "baseline",
                "model": baseline_fresh["logical_model"],
                "reasoning_effort": baseline_fresh["reasoning_effort"],
                "task": baseline_fresh["task_ids"][0],
                "reward": baseline_fresh["score"],
                "feature_tests": "0/14",
                "passed_tests": 163,
                "total_tests": 182,
                "infra_failed": baseline_fresh["cells"][0]["infra_failed"],
            },
        ],
    )
    lines = [
        "# Coding router figures and tables",
        "",
        "These artifacts summarize the isolated coding-model-router experiment on DeepSWE 1.1.",
        "The historical/shared ledger is the source for the 113-task frontier. Fresh verifier data is a one-task matched probe.",
        "Offline allocation shares describe experiment coverage, not production serving telemetry.",
        "",
        "## Figures",
        "",
        "![Aggregate quality and cost](coding-router-small-agent-20260730-01-aggregate-quality-cost.png)",
        "",
        "![Model and effort breakdown](coding-router-small-agent-20260730-02-model-effort-breakdown.png)",
        "",
        "![Validation and correlation](coding-router-small-agent-20260730-03-validation-and-correlation.png)",
        "",
        "## Tables",
        "",
        "See the `coding-router-small-agent-20260730-*.csv` files in this directory.",
        "",
        "## Interpretation",
        "",
        "The candidate is GPT-5.6 Sol with xhigh reasoning effort. It improves aggregate whole-task quality by 3.76 percentage points while reducing cost by 34.85% versus GPT-5.5 xhigh.",
        "The fast 12-task proxy correlates significantly with the SWE-bench snapshot, but the snapshot is not a fresh leaderboard result. The fresh one-task verifier pair is neutral and should not be read as broad live superiority.",
    ]
    (output_dir / f"{prefix}figures-and-tables.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def main() -> None:
    args = _parser().parse_args()
    output_dir = args.output_dir.resolve()
    figures_dir = output_dir
    tables_dir = output_dir
    frontier = json.loads(args.frontier.read_text(encoding="utf-8"))
    proxy_report = json.loads(args.proxy_report.read_text(encoding="utf-8"))
    candidate_fresh = json.loads(args.fresh_candidate.read_text(encoding="utf-8"))
    baseline_fresh = json.loads(args.fresh_baseline.read_text(encoding="utf-8"))
    prefix = "coding-router-small-agent-20260730-"
    _aggregate_figure(frontier, figures_dir / f"{prefix}01-aggregate-quality-cost.png")
    analysis = _load_analysis_helpers()
    model_rows = _model_breakdown_figure(
        analysis,
        args.trials,
        proxy_report,
        args.swe_leaderboard,
        frontier,
        figures_dir / f"{prefix}02-model-effort-breakdown.png",
    )
    _validation_figure(
        frontier,
        proxy_report,
        args.trials,
        args.swe_leaderboard,
        figures_dir / f"{prefix}03-validation-and-correlation.png",
    )
    _write_csv(
        tables_dir / f"{prefix}model-effort-breakdown.csv",
        list(model_rows[0]),
        model_rows,
    )
    _write_tables(tables_dir, frontier, model_rows, candidate_fresh, baseline_fresh, proxy_report)
    manifest = {
        "benchmark": "DeepSWE 1.1",
        "frontier_source": str(args.frontier.resolve()),
        "proxy_source": str(args.proxy_report.resolve()),
        "fresh_candidate_source": str(args.fresh_candidate.resolve()),
        "fresh_baseline_source": str(args.fresh_baseline.resolve()),
        "outputs": [str(path.relative_to(output_dir)) for path in sorted(output_dir.rglob("*")) if path.is_file()],
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
