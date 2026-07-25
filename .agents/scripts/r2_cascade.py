"""The cascade: cheap model answers, the distilled verifier decides, escalate when in doubt.

Pre-registered in findings/r2.md (2026-07-25) BEFORE this ran; bars, selection rules, and
permutation-averaging are copied from there, not invented here. Economics: cost = 1x cheap +
escalation_rate x strong, against best-of-2's fatal flat 2x.

Arms per (matrix, seed):
- r2-oracle-cascade: the ceiling. Reads the cheap episode's TRUE reward at decision time
  (labeled oracle-* per the evaluate_call_sequences information boundary; never deployable).
- r2-cascade: deployable. The absolute-head reply verifier (master's validated recipe:
  full-dim ridge alpha=1 over 3-large reply embeddings, trained on the FIT split only)
  scores the cheap reply; escalate below a threshold chosen fit-side out-of-fold.
- r2-cascade-shuffled (seed 0): control; the verifier trained on permuted rewards must not
  produce a real gain.

All cascade rows are the mean over the 2 episode-order permutations (the order-dependence
trap); the best-single baseline is an evaluate_choices row (master a51f917f). Fit outputs
(chosen pair, threshold, escalation rate) go in NOTES, never in params (cohort ruling #1).

Usage: uv run python .agents/scripts/r2_cascade.py [wm-all ...] [--seeds=0,1,2,3,4]
       [--oracle-only]
"""

from __future__ import annotations

import hashlib
import importlib.util
import logging
import sys
import uuid
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from wmh.optimize.outcomes import OutcomeMatrix, ScenarioOutcome
from wmh.research.reply_verifier import (
    EpisodeKey,
    ReplyVerifier,
    episode_key,
    fit_absolute,
    scenario_folds,
    shuffled_rewards,
)
from wmh.research.routing_runs import (
    Finish,
    RunRecord,
    append_run,
    evaluate_call_sequences,
    evaluate_choices,
)

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("r2cascade")

DATA = Path("~/Desktop/Projects/wmh-routing-data").expanduser()
RUNS = DATA / "runs" / "r2.jsonl"
CACHE = DATA / "cache" / "wm-oai3l-replies.npz"
ALPHA = 1.0  # master's validated verifier recipe: full-dim ridge, alpha=1
# Round 8b selection constraints (pre-registered in findings/r2.md before running):
CAP_FACTOR = 0.85  # fit cost cap safety margin against fit->test cost drift (+8-15%)
CHEAP_RATIO = 0.6  # a real cascade: cheap must cost <= this fraction of strong
FOLDS = 5
SEEDS = [0, 1, 2, 3, 4]
MAX_CHARS = 28_000  # MUST match fit_reply_verifier.reply_text or cache hashes miss


def _driver():  # noqa: ANN202
    spec = importlib.util.spec_from_file_location(
        "r2drv", Path(__file__).parent / "run_routing_r2.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def reply_text(outcome: ScenarioOutcome) -> str:
    return "\n\n".join(outcome.replies)[:MAX_CHARS]


def load_embeddings(matrix: OutcomeMatrix) -> dict[EpisodeKey, np.ndarray]:
    """Episode -> cached 3-large reply embedding (sha256-of-text keyed, master's cache)."""
    blob = np.load(CACHE, allow_pickle=False)
    cache = dict(zip(blob["hashes"].tolist(), blob["vectors"], strict=True))
    out: dict[EpisodeKey, np.ndarray] = {}
    missing = 0
    for outcome in matrix.outcomes:
        if outcome.reward is None or not outcome.replies:
            continue
        digest = hashlib.sha256(reply_text(outcome).encode()).hexdigest()
        vector = cache.get(digest)
        if vector is None:
            missing += 1
            continue
        out[episode_key(outcome)] = np.asarray(vector, dtype=np.float64)
    logger.info(
        "embeddings: %d episodes covered, %d scored-with-reply missing from cache",
        len(out),
        missing,
    )
    return out


def _cells(matrix: OutcomeMatrix, ids: set[str]) -> dict[tuple[str, str], list[ScenarioOutcome]]:
    cells: dict[tuple[str, str], list[ScenarioOutcome]] = {}
    for outcome in matrix.outcomes:
        if outcome.scenario_id in ids and outcome.reward is not None:
            cells.setdefault((outcome.scenario_id, outcome.model), []).append(outcome)
    return cells


def _mean(values: list[float]) -> float:
    return sum(values) / len(values)


def _fit_stats(cells: dict) -> tuple[dict, dict]:
    rewards = {key: _mean([o.reward for o in episodes]) for key, episodes in cells.items()}
    costs = {key: _mean([o.cost_usd for o in episodes]) for key, episodes in cells.items()}
    return rewards, costs


def _oof_scores(
    matrix: OutcomeMatrix,
    fit_ids: list[str],
    embeddings: dict[EpisodeKey, np.ndarray],
    seed: int,
    *,
    shuffle: bool = False,
) -> dict[EpisodeKey, float]:
    """Out-of-fold verifier scores for every embedded FIT episode (no self-scoring)."""
    episodes = [
        o
        for o in matrix.outcomes
        if o.scenario_id in set(fit_ids) and o.reward is not None and episode_key(o) in embeddings
    ]
    scores: dict[EpisodeKey, float] = {}
    folds = scenario_folds(fit_ids, FOLDS, seed)
    for fold in folds:
        fold_set = set(fold)
        train = [o for o in episodes if o.scenario_id not in fold_set]
        held = [o for o in episodes if o.scenario_id in fold_set]
        if not train or not held:
            continue
        features = np.asarray([embeddings[episode_key(o)] for o in train])
        rewards = np.asarray([o.reward for o in train], dtype=float)
        if shuffle:
            rewards = shuffled_rewards(rewards, seed)
        verifier = fit_absolute(features, rewards, alpha=ALPHA)
        held_features = np.asarray([embeddings[episode_key(o)] for o in held])
        for outcome, score in zip(held, verifier.score(held_features), strict=True):
            scores[episode_key(outcome)] = float(score)
    return scores


def _select(
    matrix: OutcomeMatrix,
    fit_ids: list[str],
    best_name: str,
    *,
    scorer: dict[EpisodeKey, float] | None,
    fixed_pair: tuple[str, str] | None = None,
) -> tuple[str, str, float, float, float] | None:
    """(cheap, strong, threshold, fit_acc, fit_cost) or None when the cascade declines.

    `scorer=None` is the ORACLE arm: the per-scenario decision statistic is the cheap cell's
    true mean reward, thresholds over a reward grid. Otherwise the statistic is the mean
    out-of-fold verifier score of the cheap cell's replies, thresholds over its deciles.
    All quantities are fit-side cell MEANS (selection never sees episode order or test data).
    """
    cells = _cells(matrix, set(fit_ids))
    rewards, costs = _fit_stats(cells)
    models = [entry.name for entry in matrix.pool]
    model_cost = {
        m: _mean([costs[(sid, m)] for sid in fit_ids if (sid, m) in costs]) for m in models
    }
    cap = CAP_FACTOR * _mean(
        [costs[(sid, best_name)] for sid in fit_ids if (sid, best_name) in costs]
    )

    def statistic(sid: str, cheap: str) -> float | None:
        if scorer is None:
            return rewards.get((sid, cheap))
        keys = [episode_key(o) for o in cells.get((sid, cheap), [])]
        values = [scorer[k] for k in keys if k in scorer]
        return _mean(values) if values else None

    rng = np.random.default_rng(0)
    resamples = [
        [fit_ids[i] for i in rng.integers(0, len(fit_ids), size=len(fit_ids))]
        for _ in range(200)
    ]

    best = None
    for cheap in models:
        for strong in models:
            if fixed_pair is not None and (cheap, strong) != fixed_pair:
                continue
            if cheap == strong:
                continue
            if model_cost.get(cheap, 0) > CHEAP_RATIO * model_cost.get(strong, 0):
                continue  # lateral hop, not a cascade (round 8a lesson)
            stats = {sid: statistic(sid, cheap) for sid in fit_ids}
            known = [v for v in stats.values() if v is not None]
            if len(known) < len(fit_ids) * 0.5:
                continue
            if scorer is None:
                grid = [round(t, 2) for t in np.arange(0.05, 1.0, 0.05)]
            else:
                grid = sorted({float(q) for q in np.quantile(known, np.arange(0.1, 1.0, 0.1))})
            for threshold in grid:
                accs, costs_out = [], []
                for sid in fit_ids:
                    value = stats[sid]
                    escalate = value is None or value < threshold
                    reward_cell = (sid, strong) if escalate else (sid, cheap)
                    if reward_cell not in rewards:
                        continue
                    accs.append(rewards[reward_cell])
                    cost = costs.get((sid, cheap), model_cost.get(cheap, 0.0))
                    if escalate:
                        cost += costs.get((sid, strong), model_cost.get(strong, 0.0))
                    costs_out.append(cost)
                if not accs:
                    continue
                acc, cost = _mean(accs), _mean(costs_out)
                if scorer is not None:
                    # 8c robust cost check: bootstrap the fit scenarios; the 80th-percentile
                    # cascade cost must clear the cap (mean caps do not survive the pooled
                    # corpora's heavy-tailed per-scenario costs - the 8a/8b lesson).
                    per_sid_cost = {}
                    for sid in fit_ids:
                        value = stats[sid]
                        escalate = value is None or value < threshold
                        cost_sid = costs.get((sid, cheap), model_cost.get(cheap, 0.0))
                        if escalate:
                            cost_sid += costs.get((sid, strong), model_cost.get(strong, 0.0))
                        per_sid_cost[sid] = cost_sid
                    boot = sorted(
                        _mean([per_sid_cost[sid] for sid in sample]) for sample in resamples
                    )
                    cost_check = boot[int(0.8 * len(boot))]
                else:
                    cost_check = cost
                if cost_check <= cap and (best is None or (acc, -cost) > (best[3], -best[4])):
                    best = (cheap, strong, threshold, acc, cost)
    return best


def _swap_episodes(matrix: OutcomeMatrix) -> OutcomeMatrix:
    """Permute per-cell episode order (0<->1) for the order-dependence average."""
    outcomes = []
    for outcome in matrix.outcomes:
        clone = outcome.model_copy()
        clone.episode = -outcome.episode  # 2-episode cells reverse; sort is stable for 1
        outcomes.append(clone)
    return OutcomeMatrix(pool=matrix.pool, outcomes=outcomes)


def run_matrix(name: str, matrix: OutcomeMatrix, seeds: list[int], *, oracle_only: bool) -> None:
    drv = _driver()
    embeddings = load_embeddings(matrix)
    ts = datetime.now(tz=UTC).isoformat()

    for seed in seeds:
        fit_ids, test_ids = drv.split_scenario_ids(matrix, train_fraction=0.7, seed=seed)
        best_name, _a, _c = drv.best_single_model(matrix, fit_ids=fit_ids, eval_ids=test_ids)
        best_eval = evaluate_choices(matrix, test_ids, lambda _sid, b=best_name: b)

        def record(
            variant: str,
            params: dict,
            result,  # noqa: ANN001
            notes: str,
            seed: int = seed,
            best_eval=best_eval,  # noqa: ANN001
            fit_ids: list[str] = fit_ids,
            test_ids: list[str] = test_ids,
        ) -> None:
            append_run(
                RunRecord(
                    run_id=f"r2-{name}-iid-s{seed}-{variant}-{uuid.uuid4().hex[:8]}",
                    ts=ts,
                    matrix=name,
                    variant=variant,
                    params={**params, "split": "iid"},
                    split_seed=seed,
                    fit_scenarios=len(fit_ids),
                    test_scenarios=len(test_ids),
                    result=result,
                    baselines={"best_single": best_eval},
                    notes=notes,
                ),
                RUNS,
            )
            logger.info(
                "%s/s%d %s: acc=%.4f cost=$%.5f (best-single %.4f/$%.5f) | %s",
                name,
                seed,
                variant,
                result.accuracy,
                result.cost_per_call,
                best_eval.accuracy,
                best_eval.cost_per_call,
                notes,
            )

        record(
            "r2-cascade-best-single",
            {"model": best_name},
            best_eval,
            "1x arm scored via evaluate_choices (order-independence)",
        )

        arms: list[tuple[str, dict[EpisodeKey, float] | None, bool]] = [
            ("r2-oracle-cascade", None, False)
        ]
        if not oracle_only:
            arms.append(("r2-cascade", _oof_scores(matrix, fit_ids, embeddings, seed), False))
            if seed == 0:
                arms.append(
                    (
                        "r2-cascade-shuffled",
                        _oof_scores(matrix, fit_ids, embeddings, seed, shuffle=True),
                        True,
                    )
                )

        oracle_pair: tuple[str, str] | None = None
        for variant, scorer, shuffled in arms:
            chosen = _select(
                matrix, fit_ids, best_name, scorer=scorer,
                fixed_pair=oracle_pair if scorer is not None else None,
            )
            if chosen is None:
                record(
                    variant,
                    {"family": "cascade", "declined": True},
                    best_eval,
                    "declined: no (pair, threshold) met the fit cost cap; = best-single",
                )
                continue
            cheap, strong, threshold, fit_acc, fit_cost = chosen
            if scorer is None:
                oracle_pair = (cheap, strong)  # fit-label pair choice, reused by real arms

            if scorer is None:

                def decide(
                    sid: str,
                    transcript: list,
                    cheap: str = cheap,
                    strong: str = strong,
                    threshold: float = threshold,
                ) -> str | Finish:
                    if not transcript:
                        return cheap
                    if len(transcript) == 1:
                        reward = transcript[0].reward or 0.0
                        return Finish(pick=0) if reward >= threshold else strong
                    return Finish(pick=1)
            else:
                # Deployable: retrain on the FULL fit split, score the consumed episode.
                fit_set = set(fit_ids)
                train = [
                    o
                    for o in matrix.outcomes
                    if o.scenario_id in fit_set
                    and o.reward is not None
                    and episode_key(o) in embeddings
                ]
                features = np.asarray([embeddings[episode_key(o)] for o in train])
                labels = np.asarray([o.reward for o in train], dtype=float)
                if shuffled:
                    labels = shuffled_rewards(labels, seed)
                final: ReplyVerifier = fit_absolute(features, labels, alpha=ALPHA)

                def decide(
                    sid: str,
                    transcript: list,
                    cheap: str = cheap,
                    strong: str = strong,
                    threshold: float = threshold,
                    final: ReplyVerifier = final,
                ) -> str | Finish:
                    if not transcript:
                        return cheap
                    if len(transcript) == 1:
                        vector = embeddings.get(episode_key(transcript[0]))
                        if vector is None:
                            return strong  # unverifiable reply: escalate
                        score = float(final.score(vector)[0])
                        return Finish(pick=0) if score >= threshold else strong
                    return Finish(pick=1)

            results = []
            for permuted in (matrix, _swap_episodes(matrix)):
                results.append(evaluate_call_sequences(permuted, test_ids, decide, max_calls=2))
            mean_result = results[0].model_copy(
                update={
                    "accuracy": _mean([r.accuracy for r in results]),
                    "cost_per_call": _mean([r.cost_per_call for r in results]),
                }
            )
            escalations = mean_result.model_mix.get(strong, 0.0)
            record(
                variant,
                {
                    "family": "cascade",
                    "verifier": "oracle"
                    if scorer is None
                    else ("shuffled" if shuffled else "absolute-a1-fulldim"),
                    "folds": FOLDS,
                },
                mean_result,
                f"pair={cheap}->{strong} thr={threshold:.4f} esc_rate={escalations:.2f} "
                f"fit_acc={fit_acc:.4f} fit_cost={fit_cost:.5f} "
                f"perm_accs={[round(r.accuracy, 4) for r in results]}",
            )


def main() -> None:
    args = sys.argv[1:]
    wanted = [a for a in args if not a.startswith("--")] or ["wm-all"]
    seeds = SEEDS
    for arg in args:
        if arg.startswith("--seeds="):
            seeds = [int(s) for s in arg.split("=", 1)[1].split(",")]
    oracle_only = "--oracle-only" in args
    drv = _driver()
    matrices = drv._matrices()
    for name in wanted:
        run_matrix(name, matrices[name], seeds, oracle_only=oracle_only)


if __name__ == "__main__":
    main()
