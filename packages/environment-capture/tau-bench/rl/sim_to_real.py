"""Sim-to-real: does the world-model tau matrix rank the pool the way real tau2 does?

Two matrices, two very different measurements of the same candidates:

- REAL: `real_episodes.py`'s `rows.jsonl`, Sierra's tau2-bench, tau2's own reward (DB state,
  action, communicate, and env-assertion checks; NL-assertion tasks additionally use tau2's
  built-in judge, flagged per row). Binary per episode.
- WM: an `OutcomeMatrix` produced by `wmo.env.closed_loop`: our world model, LLM judge,
  continuous.

The headline is rank agreement over MODEL MEANS. When both legs run the same pinned eval split
(`scenarios_eval.jsonl`), scenario ids line up exactly and the paired comparison is the stronger
number; when they do not, tasks are matched on a normalized `reason_for_call` and the paired
overlap is reported as a secondary check.

Sampling unit is the SCENARIO, not the episode: a model is scored several times on each scenario,
so the SE over per-scenario means (rather than over all episodes) is the one that does not pretend
two trials of one task are independent draws.

Headline number caveat (read before quoting a correlation): the widely quoted Spearman +0.639 was
a 630-row snapshot taken mid-capture. On the finished 720-row corpus the same computation gives
+0.630 raw and +0.743 excluding glm-5.2. Quote the row count with the number, always.

`--glm-clean` reruns the correlations with glm-5.2's wm mean recomputed to net out its inline
tool-call format failures. Two estimators, because the obvious one is biased:

- `clean_only` simply drops the broken episodes. It overstates glm badly: the format failure is
  concentrated on the HARD scenarios (the other models average 0.23-0.39 on the scenarios where
  glm inlined both episodes, against 0.59 where it never did), so dropping them also drops the
  tasks glm would have found hardest.
- `clean_if_available` keeps every scenario and, for those with one good and one broken episode,
  scores the scenario on its clean episode only. This is the number to report.

Three `wmo/env/llm_agent.py` changes split world-model matrices into a before and an after, so do
not pool them:

1. It now parses inline `tool_name({...})` calls and EXECUTES them. Note what this does and does
   not do to `inline%` below: the column counts replies whose TEXT contains call syntax, and the
   change alters how such a reply is dispatched, not whether the model writes one, so `inline%`
   stays roughly flat across the change. What changes is that those episodes stop scoring 0 for
   a formatting reason. Consequently `--glm-clean` is only meaningful on a matrix captured
   BEFORE the change; run against a newer one it would correct away episodes that ran fine.
2. Its observation/history cap went from 500 to 2000 characters, which gives the agent strictly
   more context per turn on tool-heavy domains. A matrix captured before that change measured a
   different (more starved) agent; recapture rather than mix.
3. It retries a blank completion up to twice. That shifts both cost (the blank attempts are
   billed) and per-call latency for models that blank often.

Run from the repo root:

    uv run python packages/environment-capture/tau-bench/rl/sim_to_real.py \\
        --real .wmo/evals/tau-bench-real/rows.jsonl \\
        --wm .wmo/evals/tau-bench/matrix.json --glm-clean
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import re
import statistics as st
from collections.abc import Iterable, Sequence
from pathlib import Path

from real_episodes import RealEpisodeRow, load_rows
from scipy import stats  # present in every wmo install: scikit-learn requires it

from wmo.optimize.outcomes import OutcomeMatrix, ScenarioOutcome

logger = logging.getLogger("sim-to-real")

# A tool call the model wrote into its prose instead of emitting as a structured call, e.g.
# `get_user_details({"user_id": ...})`. Harnesses without the bare-call parser never execute
# these, so the episode dies.
INLINE_CALL = re.compile(
    r"\b(get|list|search|update|cancel|book|modify|return|exchange|transfer|calculate|find|send"
    r"|think)_?\w*\(\s*[{\"]"
)


def has_inline_call(outcome: ScenarioOutcome) -> bool:
    return any(INLINE_CALL.search(reply or "") for reply in outcome.replies)


def scenario_clustered_stats(
    values_by_scenario: dict[str, list[float]],
) -> tuple[float, float, int]:
    """Mean of per-scenario means, plus the SE of that mean across scenarios."""
    per_scenario = [st.mean(values) for values in values_by_scenario.values()]
    mean = st.mean(per_scenario)
    se = (
        st.stdev(per_scenario) / len(per_scenario) ** 0.5 if len(per_scenario) > 1 else float("nan")
    )
    return mean, se, len(per_scenario)


def group_real(rows: Sequence[RealEpisodeRow], model: str) -> dict[str, list[float]]:
    grouped: dict[str, list[float]] = {}
    for row in rows:
        if row.model == model and row.reward is not None:
            grouped.setdefault(row.scenario_id, []).append(float(row.reward))
    return grouped


def group_wm(outcomes: Sequence[ScenarioOutcome], model: str) -> dict[str, list[float]]:
    grouped: dict[str, list[float]] = {}
    for outcome in outcomes:
        if outcome.model == model and outcome.reward is not None:
            grouped.setdefault(outcome.scenario_id, []).append(float(outcome.reward))
    return grouped


class RankAgreement:
    """Rank agreement between two per-model score maps over their shared models."""

    def __init__(self, real: dict[str, float], wm: dict[str, float]) -> None:
        self.models = sorted(set(real) & set(wm))
        left = [real[m] for m in self.models]
        right = [wm[m] for m in self.models]
        self.spearman = stats.spearmanr(left, right)
        self.kendall = stats.kendalltau(left, right)
        self.pearson = stats.pearsonr(left, right)
        self.top3_real = sorted(sorted(self.models, key=lambda m: real[m], reverse=True)[:3])
        self.top3_wm = sorted(sorted(self.models, key=lambda m: wm[m], reverse=True)[:3])
        self.top3_overlap = len(set(self.top3_real) & set(self.top3_wm))
        self.best_real = max(self.models, key=lambda m: real[m])
        self.best_wm = max(self.models, key=lambda m: wm[m])

    def lines(self, label: str, rows: int) -> list[str]:
        out = [f"== rank agreement ({label}; {len(self.models)} models, {rows} real rows)"]
        for name, result in (
            ("spearman", self.spearman),
            ("kendall", self.kendall),
            ("pearson", self.pearson),
        ):
            # A correlation is undefined when either side is constant (every candidate scored
            # the same). Printing "+nan (p=nan)" in a column of real numbers reads as a
            # measurement that came out near zero, so say what actually happened.
            if math.isnan(result.statistic):
                out.append(f"  {name:9} n/a (undefined: one side is constant)")
            else:
                out.append(f"  {name:9} {result.statistic:+.3f}  (p={result.pvalue:.3f})")
        out.append(f"  top-3 real: {self.top3_real}")
        out.append(f"  top-3 wm:   {self.top3_wm}")
        out.append(
            f"  overlap:    {self.top3_overlap}/3   "
            f"best real={self.best_real} best wm={self.best_wm}"
        )
        return out


def _repeated(keys: Iterable[str]) -> set[str]:
    """The keys that appear more than once, i.e. cannot identify a single scenario."""
    counts: dict[str, int] = {}
    for key in keys:
        counts[key] = counts.get(key, 0) + 1
    return {key for key, count in counts.items() if count > 1}


def _reason(blob: str) -> str:
    try:
        parsed = json.loads(blob)
    except json.JSONDecodeError:
        return ""
    reason = parsed.get("reason_for_call") if isinstance(parsed, dict) else None
    return re.sub(r"\W+", "", reason or "").lower()


def paired_scores(
    real: Sequence[RealEpisodeRow], wm: Sequence[ScenarioOutcome]
) -> tuple[dict[str, float], dict[str, float], int]:
    """Per-model means restricted to scenarios both legs measured.

    Scenario ids match directly when both legs ran the pinned eval split. Older world-model
    matrices hash the task instead, so those fall back to matching on a normalized
    `reason_for_call`; a real task whose text is shared by more than one scenario is dropped as
    ambiguous rather than double-counted.

    Returns:
        (real means, wm means, number of shared scenarios). Empty maps when nothing is shared.
    """
    real_ids = {row.scenario_id for row in real}
    wm_ids = {outcome.scenario_id for outcome in wm}
    shared_ids = real_ids & wm_ids
    if shared_ids:
        real_key = {row.scenario_id: row.scenario_id for row in real}
        wm_key = {outcome.scenario_id: outcome.scenario_id for outcome in wm}
        shared = shared_ids
    else:
        real_key = {row.scenario_id: _reason(row.task) for row in real}
        wm_key = {outcome.scenario_id: _reason(outcome.task) for outcome in wm}
        shared = (set(real_key.values()) & set(wm_key.values())) - {""}
        # A key must identify ONE scenario on BOTH sides. Checking only the real side let two
        # distinct world-model scenarios that share a `reason_for_call` average into a single
        # paired cell. On the committed split this is not hypothetical: two telecom pairs
        # normalize to the same text, so 4 of 20 scenarios are ambiguous.
        ambiguous = _repeated(real_key.values()) | _repeated(wm_key.values())
        dropped = sorted(shared & ambiguous)
        shared -= ambiguous
        if dropped:
            logger.warning(
                "dropping %d scenario(s) whose reason_for_call is shared by more than one "
                "scenario on one side: %s",
                len(dropped),
                [key[:40] for key in dropped],
            )
    if not shared:
        return {}, {}, 0

    real_means: dict[str, float] = {}
    wm_means: dict[str, float] = {}
    for model in sorted({row.model for row in real} & {o.model for o in wm}):
        left, right = [], []
        for key in sorted(shared):
            rv = [
                float(row.reward)
                for row in real
                if row.model == model
                and row.reward is not None
                and real_key[row.scenario_id] == key
            ]
            wv = [
                float(o.reward)
                for o in wm
                if o.model == model and o.reward is not None and wm_key[o.scenario_id] == key
            ]
            if rv and wv:
                left.append(st.mean(rv))
                right.append(st.mean(wv))
        if left:
            real_means[model] = st.mean(left)
            wm_means[model] = st.mean(right)
    return real_means, wm_means, len(shared)


def report(real: Sequence[RealEpisodeRow], matrix: OutcomeMatrix, glm_clean: bool) -> list[str]:
    """Build the whole report as lines, so callers (and tests) can assert on it."""
    scored = [row for row in real if row.reward is not None]
    # Both sides must be SCORED, not merely present. `wmo.env.closed_loop` leaves a cell
    # unscored on a provider throttle or an agent crash, so a candidate that was rate-limited
    # across the whole sweep appears in the matrix with nothing to average, and taking its mean
    # used to abort the entire report after a capture that had already been paid for.
    wm_scored = {o.model for o in matrix.outcomes if o.reward is not None}
    models = sorted({row.model for row in scored} & wm_scored)
    if not models:
        return ["no model is scored in both the real rows and the world-model matrix"]
    silent = sorted({o.model for o in matrix.outcomes} - wm_scored)
    if silent:
        logger.warning("world-model models with no scored episode, excluded: %s", silent)

    lines = [
        f"== REAL (tau2 reward, {len(scored)} rows) vs WM (LLM judge), per model",
        f"{'model':16} {'real':>16} {'eps':>5} {'wm':>16} {'inline%':>8}",
    ]
    real_mean: dict[str, float] = {}
    wm_mean: dict[str, float] = {}
    wm_clean: dict[str, float] = {}
    for model in models:
        rmean, rse, _ = scenario_clustered_stats(group_real(scored, model))
        wm_rows = [o for o in matrix.outcomes if o.model == model and o.reward is not None]
        wmean, wse, _ = scenario_clustered_stats(group_wm(wm_rows, model))
        episodes = sum(1 for row in scored if row.model == model)
        inline = sum(1 for o in wm_rows if has_inline_call(o))
        by_scenario: dict[str, list[ScenarioOutcome]] = {}
        for outcome in wm_rows:
            by_scenario.setdefault(outcome.scenario_id, []).append(outcome)
        # clean_if_available: keep every scenario, score it on its unbroken episodes when it has
        # any. Dropping broken episodes outright would also drop the hard scenarios, which is
        # where the format failure concentrates.
        wm_clean[model] = st.mean(
            st.mean(
                float(e.reward or 0.0)
                for e in ([c for c in episodes_of if not has_inline_call(c)] or episodes_of)
            )
            for episodes_of in by_scenario.values()
        )
        real_mean[model], wm_mean[model] = rmean, wmean
        lines.append(
            f"{model:16} {rmean:.3f} +/- {rse:.3f} {episodes:5d} "
            f"{wmean:.3f} +/- {wse:.3f} {100 * inline / len(wm_rows):7.0f}%"
        )

    if len(models) < 3:
        lines.append("")
        lines.append(f"only {len(models)} shared models: too few for a rank correlation")
        return lines

    arms = [("headline", wm_mean)]
    if glm_clean and "glm-5.2" in wm_mean:
        arms.append(("glm-format-corrected", {**wm_mean, "glm-5.2": wm_clean["glm-5.2"]}))
    for label, scores in arms:
        lines.append("")
        lines.extend(RankAgreement(real_mean, scores).lines(label, len(scored)))

    paired_real, paired_wm, shared = paired_scores(scored, matrix.outcomes)
    lines.append("")
    if shared and len(paired_real) >= 3:
        lines.append(f"== paired overlap ({shared} scenarios measured by both legs)")
        for model in sorted(paired_real):
            lines.append(f"  {model:16} real {paired_real[model]:.3f}   wm {paired_wm[model]:.3f}")
        lines.extend(RankAgreement(paired_real, paired_wm).lines("paired", len(scored)))
    else:
        lines.append("== paired overlap: no scenario measured by both legs")

    lines.append("")
    lines.append("== per-episode real means (episode noise)")
    for model in models:
        parts = []
        for episode in sorted({row.episode for row in scored}):
            values = [
                float(row.reward)
                for row in scored
                if row.model == model and row.episode == episode and row.reward is not None
            ]
            if values:
                parts.append(f"e{episode}={st.mean(values):.3f} (n={len(values)})")
        lines.append(f"  {model:16} {'  '.join(parts)}")
    return lines


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--real", type=Path, default=Path(".wmo/evals/tau-bench-real/rows.jsonl"))
    parser.add_argument("--wm", type=Path, required=True, help="world-model OutcomeMatrix json")
    parser.add_argument("--glm-clean", action="store_true", help="add the format-corrected arm")
    args = parser.parse_args(argv)

    rows = load_rows(args.real)
    if not rows:
        parser.error(f"no real episodes at {args.real}; run real_episodes.py first")
    for line in report(rows, OutcomeMatrix.load(args.wm), args.glm_clean):
        print(line)  # noqa: T201 - operator-run CLI, this report IS the product output
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
