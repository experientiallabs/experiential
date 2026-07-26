"""Render THE figure for the TerminalBench-2 lane: where the run stands against its subgoal.

ONE panel, no subplots. The subgoal is a single question — does distilling GLM-5.2
into Qwen3.5-9B measurably improve the student on TerminalBench-2, measured through
one harness? So the figure shows exactly three quantities on one axis: the baseline
the run must beat, the teacher ceiling it is aiming at, and the post-training
result, which is drawn as an explicit NOT YET MEASURED slot rather than omitted.

Everything else this lane has measured — the +23.4 pp headroom that justifies the
run, the 29-47% holdout attrition that bounds it, per-episode cost — is context and
belongs in prose. A reader who sees only the page title and this figure should know
whether the thing is working, and right now the honest answer is "no result yet,
here is the bar it has to clear".

Numbers come from the run's own eval reports, never hardcoded: a figure that
asserts a constant keeps asserting it after the page has been corrected.

Usage:
    .venv/bin/python .agents/distill/figures_tb2.py --out /tmp/tb2_figure.html
"""

from __future__ import annotations

import argparse
import io
import json
import tomllib
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams["svg.fonttype"] = "none"
matplotlib.rcParams["svg.hashsalt"] = "tb2"
import matplotlib.pyplot as plt
import seaborn as sns

INK = "#0a0a0a"
GRID = "#ececec"
BLUE = "#0070f3"
AMBER = "#f5a623"
MUTED = "#b8b8b8"

RUN = Path(
    "/Users/admin/Documents/experientiallabs/world-model-harness/.claude/worktrees"
    "/xtoken-t2/.wmh/runs/t2c"
)


def _report(name: str) -> dict:
    return json.loads((RUN / "evals" / f"{name}.json").read_text())


def _progress() -> tuple[int, int]:
    """(completed optimizer steps, total planned) — both read from the run dir.

    A completed step appends one row to `metrics.jsonl`, so its line count IS the
    step count; the file is absent until the first step lands. The total comes from
    the run's own config snapshot rather than this script, so re-shaping the run
    cannot leave the figure asserting a stale denominator.
    """
    metrics = RUN / "metrics.jsonl"
    done = (
        sum(1 for line in metrics.read_text().splitlines() if line.strip())
        if metrics.exists()
        else 0
    )
    config = tomllib.loads((RUN / "config.toml").read_text())
    return done, int(config["train"]["steps"])


def render() -> str:
    student = _report("baseline-student-before")
    teacher = _report("baseline-teacher")
    done, total = _progress()

    # BINARY on both bars, deliberately. The teacher report was imported from the
    # previous scaffold and carries no graded score, so plotting the student's
    # graded 0.425 against the teacher's binary 0.500 would put two DIFFERENT
    # metrics on one axis — the exact mismatched comparison this lane has had to
    # retract twice. The graded number goes in the caption instead.
    before = student["solve_rate"]
    ceiling = teacher["solve_rate"]
    executed = student["executed_trials"]
    trials = student["trials"]

    sns.set_theme(style="white", font_scale=1.05)
    figure, axis = plt.subplots(figsize=(10.5, 3.9))
    figure.patch.set_facecolor("white")

    labels = ["student\nBEFORE", "student\nAFTER", "teacher\n(ceiling)"]
    values = [before, 0.0, ceiling]
    colors = [AMBER, MUTED, BLUE]
    bars = axis.barh(labels, values, color=colors, height=0.55, edgecolor="none")
    bars[1].set_hatch("///")
    bars[1].set_facecolor("white")
    bars[1].set_edgecolor(MUTED)

    axis.text(before + 0.012, 0, f"{before:.3f}", va="center", fontsize=13,
              fontweight="bold", color=INK)
    axis.text(ceiling + 0.012, 2, f"{ceiling:.3f}", va="center", fontsize=13,
              fontweight="bold", color=INK)
    # "{done} of {total} COMPLETE", not "step {done}": `done` counts FINISHED optimizer
    # steps, so while step 1 is mid-flight it reads 0 — and "step 0 of 8" beside a run
    # log saying "step 1/8" reads as a contradiction that the page then has to explain
    # in prose. A headline figure must stand alone.
    axis.text(0.012, 1, f"NOT YET MEASURED  —  {done} of {total} training steps complete",
              va="center", fontsize=12, fontweight="bold", color="#8a8a8a")

    axis.axvline(before, color=AMBER, linewidth=1.1, linestyle=(0, (4, 3)), alpha=0.7, zorder=0)
    axis.set_xlim(0, max(ceiling, before) * 1.35)
    axis.set_xlabel("TerminalBench-2 tasks solved (binary rate)", fontsize=10)
    axis.set_title(
        "Does distilling GLM-5.2 into Qwen3.5-9B improve it on TerminalBench-2?\n"
        "No result yet. The dashed line is the bar the run has to clear.",
        loc="left", color=INK, fontsize=13.5, fontweight="bold", pad=14,
    )
    axis.grid(axis="x", color=GRID, linewidth=1)
    axis.set_axisbelow(True)
    axis.invert_yaxis()
    sns.despine(ax=axis, left=True)

    figure.tight_layout(rect=(0, 0.13, 1, 1))
    figure.text(
        0.008, 0.10,
        f"Binary rate on both bars: it is the only metric both arms have, because the "
        f"teacher report predates graded scoring. BEFORE is "
        f"{round(student['solve_rate'] * executed)} of {executed} tasks solved, over the "
        f"{executed} of {trials} holdout trials that produced verifier evidence — the "
        f"other {trials - executed} died on context overflow and are excluded, which "
        f"flatters whatever survives. With partial credit the student scores "
        f"{student['graded_solve_rate']:.3f}; {student['scaffold_loss_rate']:.1%} of trials "
        f"never reached an explicit submit. "
        f"CEILING is the teacher on the same 16 tasks, measured under the previous "
        f"scaffold, so it sets the promotion gate and is not a like-for-like comparison. "
        f"AFTER will be measured on the identical holdout through the identical harness.",
        fontsize=7.8, color="#666666", ha="left", va="top", wrap=True,
    )

    buffer = io.StringIO()
    figure.savefig(buffer, format="svg", bbox_inches="tight", facecolor="white")
    plt.close(figure)
    return buffer.getvalue()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="/tmp/tb2_figure.html")
    args = parser.parse_args()

    svg = render()
    Path(args.out).write_text(
        "<!doctype html><meta charset=utf-8><title>TB2: GLM-5.2 -> Qwen3.5-9B</title>"
        f"<body style='margin:0;background:white'>{svg}</body>"
    )
    print(f"wrote {args.out} ({len(svg) / 1024:.1f} KiB svg)")


if __name__ == "__main__":
    main()
