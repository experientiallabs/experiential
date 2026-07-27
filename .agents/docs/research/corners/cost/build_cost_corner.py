"""COST-MAX corner analysis: every savings number, from the scorecard, with provenance.

Reads the canonical tau grid's per-arm matrices through common/data.py (read-only, never
regenerates) and computes every COST figure through `wmo.optimize.scorecard`, the one
aggregation the savings claims are allowed to use (it implements the binding D-COMPRESS
accounting rule: cache-adjusted effective cost per COMPLETED task, compressor inference
folded in as RowOverhead, unscored spend excluded and reported). Every QUALITY delta claim
additionally carries common/stats.paired_delta evidence (paired per scenario, bootstrap CI,
sign test, noise-floor flag), per the binding conventions in common/README.md.

Anchors. Every headline delta is stated against the NAMED fable-5 anchor (its rows in the
identity arm). Savings against a weak anchor overstate, so the best single pool model by
mean reward is computed per run and every delta is ALSO stated against it; both columns
land in numbers.json and on the charts.

Degrades gracefully: arms whose matrix has not landed are named as pending (on the figures
too, per the no-silent-caps rule), and the charts computable today (the ours9 dial anchors,
the cycle-1 training-stage panel) render regardless. Zero LLM spend; pure offline
computation.

Run from the repo root:

    uv run --extra viz python .agents/docs/research/corners/cost/build_cost_corner.py

Routed rungs need fitted per-arm policies from the joint-tau master (bank-refit-per-arm
fits, per the grid design). Until those exist this script covers single-model configs and
compression arms; mounting a fit adds a `rows_for_policy` arm here.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

from matplotlib import pyplot as plt
from matplotlib.lines import Line2D

from wmo.optimize.knn import COST_QUALITY_ANCHORS
from wmo.optimize.outcomes import OutcomeMatrix
from wmo.optimize.scorecard import (
    Arm,
    ConditionLabel,
    RowOverhead,
    Scorecard,
    build_scorecard,
    rows_for_model,
)

CORNERS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CORNERS / "common"))

from data import (  # noqa: E402
    CYCLE1_JUDGE,
    GRID_ARMS,
    IDENTITY_ARM,
    cycle1_rewards_by_task,
    grid_dir,
    load_arm_matrix,
    load_cycle1_rows,
    rewards_by_scenario,
)
from palette import (  # noqa: E402
    BLUE,
    MUTED,
    NOISE_BAND_ALPHA,
    NOISE_BAND_COLOR,
    PURPLE,
    RED,
    apply_style,
    footnote,
    label_point,
    save_fig,
)
from stats import NOISE_FLOOR_REWARD, PairedDelta, paired_delta  # noqa: E402

DEFAULT_OUT = Path(__file__).resolve().parent
ANCHOR_MODEL = "fable-5"

# The WM episodes' scorer, per the canonical grid design (rubric-v2 judge pinned on
# opus-4-8). The matrices do not carry a judge field; this label is the grid cohort's pin
# and travels on every ConditionLabel so a differently-judged matrix can never be silently
# compared in.
GRID_JUDGE = "rubric-v2 (opus-4-8)"

# Fixed arm-identity colors for the grid's three compression arms, consistent across every
# cost figure. llmlingua2-endpoint wears red because it IS the compaction lever
# (SERIES_COLORS assigns red to "+compaction"); amber/teal are banned for marks this small
# (contrast floor, see palette.py).
ARM_COLORS = {IDENTITY_ARM: BLUE, "truncate": PURPLE, "llmlingua2-endpoint": RED}

# The noise floor in quality POINTS (scorecard deltas are reward x 100).
NOISE_FLOOR_POINTS = NOISE_FLOOR_REWARD * 100


@dataclass
class CostCornerReport:
    """Everything this run computed, with the labels the honesty rules require."""

    grid_dir: str
    arms_present: list[str] = field(default_factory=list)
    arms_pending: list[str] = field(default_factory=list)
    anchor: str = ANCHOR_MODEL
    best_single: str | None = None
    # scorecard + paired-stats summaries keyed "<arm>/<model>", per anchor column
    vs_fable5: dict[str, dict[str, object]] = field(default_factory=dict)
    vs_best_single: dict[str, dict[str, object]] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)


def _condition(model: str, optimizer: str) -> ConditionLabel:
    return ConditionLabel(
        base_model=model,
        optimizer=optimizer,
        dataset="tau-bench",
        split="wm-test-band",
        judge=GRID_JUDGE,
        provenance="wm_simulated",
    )


def _compressor_overheads(rows: list) -> list[RowOverhead]:
    """The compressor's own bill, folded in as the accounting rule requires."""
    return [
        RowOverhead(
            scenario_id=row.scenario_id,
            model=row.model,
            episode=row.episode,
            component="compressor",
            cost_usd=row.compressor_cost_usd,
            latency_s=row.compressor_latency_s,
        )
        for row in rows
        if row.compressor_cost_usd > 0.0 or row.compressor_latency_s > 0.0
    ]


def _arm_for(matrix: OutcomeMatrix, model: str, arm_name: str) -> Arm:
    rows = rows_for_model(matrix, model)
    optimizer = "none" if arm_name == IDENTITY_ARM else f"compaction({arm_name})"
    return Arm(
        name=f"{model} [{arm_name}]",
        condition=_condition(model, optimizer),
        rows=rows,
        overheads=_compressor_overheads(rows),
    )


def _summary(card: Scorecard, evidence: PairedDelta) -> dict[str, object]:
    """The numbers a chart or a reader needs, plus what they must never be read without."""
    return {
        "label": "measured",
        "provenance": card.provenance,
        "judge": card.judge,
        "quality_delta_points": round(card.quality_delta_points, 2),
        "quality_delta_paired": {
            "mean": round(evidence.mean_delta, 4),
            "ci95": [round(evidence.ci_low, 4), round(evidence.ci_high, 4)],
            "sign_test_p": evidence.sign_test_p,
            "n_pairs": evidence.n_pairs,
            "within_noise_floor": evidence.within_noise_floor,
        },
        "cost_delta_percent": None
        if card.cost_delta_percent is None
        else round(card.cost_delta_percent, 1),
        "latency_p50_delta_percent": None
        if card.latency_p50_delta_percent is None
        else round(card.latency_p50_delta_percent, 1),
        "cost_per_completed_task_usd": card.cost.cost_per_completed_task_usd,
        "anchor_cost_per_completed_task_usd": card.anchor_cost.cost_per_completed_task_usd,
        "task_success_rate": round(card.quality.task_success_rate, 4),
        "mean_reward": round(card.quality.mean_reward, 4),
        "latency_p50_model_s": round(card.latency.p50_model_s, 2),
        "latency_p95_model_s": round(card.latency.p95_model_s, 2),
        "scenarios_compared": card.scenarios_compared,
        "scenarios_excluded": card.scenarios_excluded,
        "n_scored": card.cost.n_scored,
        "n_excluded_episodes": card.cost.n_excluded,
        "excluded_cost_usd": round(card.cost.excluded_cost_usd, 4),
        "overhead_components": card.cost.overhead_components,
        "cost_assumptions": card.cost_assumptions,
    }


def _load_arms(root: Path, report: CostCornerReport) -> dict[str, OutcomeMatrix]:
    matrices: dict[str, OutcomeMatrix] = {}
    for arm in GRID_ARMS:
        matrix = load_arm_matrix(arm, root=root)
        if matrix is None:
            report.arms_pending.append(arm)
        else:
            matrices[arm] = matrix
            report.arms_present.append(arm)
    return matrices


def _best_single(matrix: OutcomeMatrix) -> str:
    """The strongest pool model by mean reward over its scored identity rows."""
    means: dict[str, float] = {}
    for model in matrix.model_names():
        rewards = [o.reward for o in rows_for_model(matrix, model) if o.reward is not None]
        if rewards:
            means[model] = sum(rewards) / len(rewards)
    return max(means, key=lambda m: means[m])


def _scorecards(
    matrices: dict[str, OutcomeMatrix], report: CostCornerReport, anchor_model: str
) -> dict[str, dict[str, tuple[Scorecard, PairedDelta]]]:
    """Every (config, anchor) scorecard plus its paired-delta evidence, per anchor column."""
    identity = matrices[IDENTITY_ARM]
    anchor = Arm(
        name=f"{anchor_model} [anchor]",
        condition=_condition(anchor_model, "none"),
        rows=rows_for_model(identity, anchor_model),
    )
    best_name = _best_single(identity)
    report.best_single = best_name
    best_anchor = Arm(
        name=f"{best_name} [best-single anchor]",
        condition=_condition(best_name, "none").replace(notes="best-single anchor"),
        rows=rows_for_model(identity, best_name),
    )

    cards: dict[str, dict[str, tuple[Scorecard, PairedDelta]]] = {"fable5": {}, "best_single": {}}
    for arm_name, matrix in matrices.items():
        for model in matrix.model_names():
            key = f"{arm_name}/{model}"
            config = _arm_for(matrix, model, arm_name)
            if not any(row.reward is not None for row in config.rows):
                report.notes.append(f"{key}: no scored rows yet, skipped")
                continue
            config_rewards = rewards_by_scenario(config.rows, model=model)
            for column, ref in (("fable5", anchor), ("best_single", best_anchor)):
                if config.condition.key() == ref.condition.key():
                    continue
                card = build_scorecard(arm=config, anchor=ref)
                evidence = paired_delta(
                    config_rewards, rewards_by_scenario(ref.rows, model=ref.condition.base_model)
                )
                cards[column][key] = (card, evidence)
    return cards


def chart_savings_frontier(
    cards: dict[str, tuple[Scorecard, PairedDelta]], out: Path, anchor: str, pending: list[str]
) -> None:
    """Savings vs the anchor per config and per rung: cost on x, quality on y (noise band
    drawn), latency labeled per point."""
    fig, ax = plt.subplots(figsize=(9.0, 6.0))
    ax.axhspan(
        -NOISE_FLOOR_POINTS,
        NOISE_FLOOR_POINTS,
        color=NOISE_BAND_COLOR,
        alpha=NOISE_BAND_ALPHA,
        zorder=0,
    )
    ax.text(0.99, NOISE_FLOOR_POINTS, "noise floor ", transform=ax.get_yaxis_transform(),
            ha="right", va="bottom", fontsize=7.5, color=MUTED)
    for key, (card, evidence) in cards.items():
        if card.cost_delta_percent is None:
            continue
        arm_name, model = key.split("/", 1)
        marker = "o" if not evidence.within_noise_floor else "^"
        ax.scatter(
            card.cost_delta_percent,
            card.quality_delta_points,
            color=ARM_COLORS[arm_name],
            s=42,
            marker=marker,
            zorder=3,
        )
        ax.annotate(
            f"{model} · p50 {card.latency.p50_model_s:.0f}s",
            xy=(card.cost_delta_percent, card.quality_delta_points),
            xytext=(6, 4),
            textcoords="offset points",
            fontsize=8,
        )
    ax.axhline(0.0, color=MUTED, linewidth=0.8, linestyle=":")
    ax.axvline(0.0, color=MUTED, linewidth=0.8, linestyle=":")
    handles = [
        Line2D([], [], marker="o", linestyle="", color=color, label=arm)
        for arm, color in ARM_COLORS.items()
    ]
    handles.append(
        Line2D([], [], marker="^", linestyle="", color=MUTED, label="Δ within noise floor")
    )
    ax.legend(handles=handles, loc="lower left")
    ax.set_title(f"Savings vs {anchor}, per config and per compression rung")
    ax.set_xlabel(f"effective cost per completed task, % vs {anchor} (negative = cheaper)")
    ax.set_ylabel(f"quality, points vs {anchor} (mean reward x 100)")
    pending_note = f" · PENDING arms: {', '.join(pending)}" if pending else ""
    footnote(
        fig,
        f"measured · wm_simulated · judge {GRID_JUDGE} · anchor {anchor} (identity arm) · "
        "effective cost = cache-adjusted provider spend + compressor inference, per COMPLETED "
        "task, unscored spend excluded and reported (wmo.optimize.scorecard) · compaction "
        "rungs are a measured tradeoff, not a recommendation (accuracy verdict pending)"
        + pending_note,
    )
    save_fig(fig, out)


def chart_cost_per_task(
    cards: dict[str, tuple[Scorecard, PairedDelta]], out: Path, anchor: str, pending: list[str]
) -> None:
    """Absolute effective cost per completed task, dot plot per model per arm."""
    by_model: dict[str, dict[str, float]] = {}
    anchor_cost: float | None = None
    for key, (card, _) in cards.items():
        arm_name, model = key.split("/", 1)
        cost = card.cost.cost_per_completed_task_usd
        if cost is not None:
            by_model.setdefault(model, {})[arm_name] = cost
        anchor_cost = card.anchor_cost.cost_per_completed_task_usd or anchor_cost
    if not by_model:
        return
    order = sorted(by_model, key=lambda m: min(by_model[m].values()))
    fig, ax = plt.subplots(figsize=(9.0, 0.45 * len(order) + 1.8))
    for y, model in enumerate(order):
        for arm_name, cost in by_model[model].items():
            ax.scatter(cost, y, color=ARM_COLORS[arm_name], s=40, zorder=3)
    if anchor_cost is not None:
        ax.axvline(anchor_cost, color=MUTED, linewidth=1.2, linestyle="--")
        ax.text(anchor_cost, len(order) - 0.3, f"  {anchor} anchor", color=MUTED, fontsize=9)
    ax.set_yticks(range(len(order)), order)
    ax.set_xscale("log")
    ax.set_title("Effective cost per completed task (log $), by model and compression rung")
    ax.set_xlabel("cache-adjusted effective $ per completed task")
    pending_note = f" · PENDING arms: {', '.join(pending)}" if pending else ""
    footnote(
        fig,
        f"measured · wm_simulated · judge {GRID_JUDGE} · dots colored by arm: identity blue, "
        "truncate purple, llmlingua2-endpoint red · scorecard accounting (unscored spend "
        "excluded and reported in numbers.json) · quality and latency per config are in "
        "numbers.json and the frontier chart" + pending_note,
    )
    save_fig(fig, out)


def chart_dial_curve(out: Path) -> None:
    """The dial's measured cost curve. Left: ours9 anchors AS MEASURED. Right: tau, when
    fitted.

    The two corpora are never blended: the D-DIAL anchors were measured on
    routerbench-ours9 and stay as measured until re-anchored (D-DIAL v2); the tau panel
    stays empty until the master's per-arm fits produce measured tau dial points.
    """
    fig, (left, right) = plt.subplots(1, 2, figsize=(11.0, 4.5), sharey=True)
    dials = [a.cost_quality for a in COST_QUALITY_ANCHORS]
    costs = [a.cost_delta_percent for a in COST_QUALITY_ANCHORS]
    left.plot(dials, costs, color=BLUE, marker="o")
    for a in COST_QUALITY_ANCHORS:
        left.annotate(
            f"{a.named_point}\n{a.quality_delta_points:+.2f} pt",
            xy=(a.cost_quality, a.cost_delta_percent),
            xytext=(0, -22),
            textcoords="offset points",
            fontsize=7.5,
            ha="center",
        )
    left.set_title("Dial cost curve: routerbench-ours9 (measured)")
    left.set_xlabel("cost_quality dial position")
    left.set_ylabel("cost % vs best single pool model")
    left.set_ylim(min(costs) - 12, 6)

    right.set_title("Dial cost curve: tau grid (pending per-arm fits)")
    right.set_xlabel("cost_quality dial position")
    right.text(
        0.5,
        0.5,
        "awaiting the master's bank-refit-per-arm fits",
        transform=right.transAxes,
        ha="center",
        color=MUTED,
        fontsize=9,
    )
    footnote(
        fig,
        "left: the five COST_QUALITY_ANCHORS (wmo/optimize/knn.py), measured on "
        "routerbench-ours9 (1199 scenarios, 9 models, 5 seeds) vs ITS best single model; "
        "quality delta annotated per point · corpora never blended: tau dial points plot "
        "separately when measured (D-DIAL v2 re-anchoring) · latency: not measured on the "
        "ours9 anchors (latency_delta is a D-DIAL v2 field, unknown renders nothing)",
    )
    save_fig(fig, out)


def chart_training_stage(out: Path) -> None:
    """The shared training-stage chart through the cost lens (v1: cycle-1 real data only).

    Quality on y per the charter; the cost lens annotates each stage point with its
    effective cost delta vs anchor. Cycle 1's rows carry no per-episode cost and its
    adapter was not promoted, so the honest annotation today is "cost n/a". Grid ablation
    lines (distill-only / +routing / +compaction) attach when the matrices land. Computed
    from episode-rows.jsonl, not transcribed from the result note.
    """
    rows = load_cycle1_rows()
    by_arm = {arm: cycle1_rewards_by_task(rows, arm=arm) for arm in
              ("teacher", "student-before", "student-after")}
    solve = {
        arm: 100 * sum(sum(r) / len(r) for r in tasks.values()) / len(tasks)
        for arm, tasks in by_arm.items()
    }
    evidence = paired_delta(by_arm["student-after"], by_arm["student-before"])

    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    stages = ["student-before", "student-after"]
    labels = ["student base\n(Qwen3.5-9B)", "cycle 1\n(warmup LoRA, NOT promoted)"]
    values = [solve[s] for s in stages]
    # Points with a dotted connector: the gate read this drop as noise (sign test over the
    # movers), so a solid trend line would draw a regression that was not measured.
    ax.plot(range(len(stages)), values, color=PURPLE, linestyle=":", linewidth=1.2)
    ax.scatter(range(len(stages)), values, color=PURPLE, s=48, zorder=3)
    for x, v in enumerate(values):
        ax.annotate(
            f"{v:.1f}%\ncost Δ n/a",
            xy=(x, v),
            xytext=(0, -30),
            textcoords="offset points",
            ha="center",
            fontsize=8,
        )
    p_label = "p n/a" if evidence.sign_test_p is None else f"p={evidence.sign_test_p:.2f}"
    label_point(
        ax, len(stages) - 1, values[-1], f"gate REJECTED\n(noise, sign test {p_label})"
    )
    ax.axhline(solve["teacher"], color=MUTED, linewidth=1.0, linestyle="--")
    ax.text(
        0.02,
        solve["teacher"] + 0.5,
        f"teacher reference (Qwen3.6-27B, {solve['teacher']:.1f}%)",
        fontsize=8,
        color=MUTED,
        transform=ax.get_yaxis_transform(),
    )
    ax.set_xticks(range(len(stages)), labels, fontsize=8)
    ax.set_xlim(-0.4, len(stages) - 0.3)
    ax.set_ylim(55, 85)
    ax.set_title("Training stage vs quality, cost lens (cycle 1: no measurable effect)")
    ax.set_ylabel("tau2 solve rate, %")
    footnote(
        fig,
        f"real_episode · judge: {CYCLE1_JUDGE} · k=3 x 20 pinned holdout tasks · gate "
        f"rejected: no teacher headroom; before-vs-after within noise "
        f"({evidence.n_up} up / {evidence.n_down} down / {evidence.n_tied} tied, sign test "
        f"{p_label}) · rows carry no per-episode $ so cost deltas are n/a until the grid's "
        f"student cells land · fable-5 anchor reference attaches when the grid lands "
        f"(different provenance: wm_simulated, separate panel rule)",
    )
    save_fig(fig, out)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grid-dir", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument(
        "--anchor",
        default=ANCHOR_MODEL,
        help="anchor pool model; the canonical grid uses fable-5 (override for smoke tests only)",
    )
    args = parser.parse_args()
    root = args.grid_dir or grid_dir()

    apply_style()
    figures = args.out_dir / "figures"
    report = CostCornerReport(grid_dir=str(root), anchor=args.anchor)

    matrices = _load_arms(root, report)
    identity = matrices.get(IDENTITY_ARM)
    if identity is not None and args.anchor in identity.model_names():
        cards = _scorecards(matrices, report, args.anchor)
        report.vs_fable5 = {k: _summary(c, e) for k, (c, e) in cards["fable5"].items()}
        report.vs_best_single = {
            k: _summary(c, e) for k, (c, e) in cards["best_single"].items()
        }
        chart_savings_frontier(
            cards["fable5"], figures / "savings_vs_fable5.png", args.anchor,
            report.arms_pending,
        )
        chart_cost_per_task(
            cards["fable5"], figures / "effective_cost_per_task.png", args.anchor,
            report.arms_pending,
        )
    else:
        report.notes.append(
            "identity matrix (or its anchor rows) not landed yet: grid charts pending"
        )

    chart_dial_curve(figures / "dial_cost_curve.png")
    chart_training_stage(figures / "training_stage_cost_lens.png")

    out = args.out_dir / "numbers.json"
    out.write_text(json.dumps(report.__dict__, indent=2, default=str) + "\n")
    status = [
        f"arms present: {report.arms_present or 'none'}; pending: {report.arms_pending}",
        f"wrote {out} and figures under {figures}",
        *(f"note: {note}" for note in report.notes),
    ]
    sys.stdout.write("\n".join(status) + "\n")


if __name__ == "__main__":
    main()
