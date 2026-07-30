"""Reproduce the DeepSWE coding-router result through the PRODUCTION path.

The standalone experiment (experientiallabs/coding-router) measured a guarded-kNN router on
DeepSWE v1.1 at 2.15x cheaper than always using the strongest arm, graded parity, nested
repo-grouped CV. That was a parallel implementation. This re-runs the same measurement through
`wmo.optimize.knn.fit_knn_policy` + `wmo.optimize.routing.evaluate_policy` so the number belongs
to the shipped code rather than to a one-off script. If it moves, the implementations disagree
and that is the finding.

Two things make this comparable to the standalone run:
  * ARMS are model x reasoning-effort, not model. Effort is the dominant axis on this benchmark
    (gpt-5.6-luna spans 1.5% -> 67.2% binary from low to max), so collapsing it would delete the
    signal the router exploits.
  * REWARD is DeepSWE's graded f2p_passed/f2p_total, not the binary resolve flag. Binary
    overstates the arm gap ~3.5x on identical episodes, which inflates any headroom estimate.

Embeddings are served from the standalone run's cache, so nothing is re-embedded and the two
runs see identical vectors -- the point is to isolate the fitter, not to re-measure the encoder.
"""
from __future__ import annotations

import collections
import json
import pathlib
import random
import sys

import numpy as np

from wmo.optimize.knn import fit_knn_policy
from wmo.optimize.outcomes import OutcomeMatrix, ScenarioOutcome
from wmo.optimize.policy import KNN_BANK_FILENAME, EmbedderSpec
from wmo.optimize.routing import evaluate_policy
from wmo.providers.base import ProviderKind
from wmo.providers.pool import PoolEntry

STANDALONE = pathlib.Path("/Users/admin/Documents/experientiallabs/coding-router")
NINE = ["gpt_5_6_terra_high", "gpt_5_6_luna_xhigh", "gpt_5_6_luna_max",
        "gpt_5_6_sol_medium", "gpt_5_6_sol_high", "claude_opus_5_low",
        "claude_opus_5_medium", "claude_opus_5_high", "claude_fable_5_xhigh"]
EMBED_DIM = 3072  # text-embedding-3-large

# arm-handle prefix -> (provider runtime id, $/1M input, $/1M output). Fetched live 2026-07-28
# from developers.openai.com/api/docs/pricing and platform.claude.com/docs/en/about-claude/pricing.
# claude-opus-5 is priced at the Opus tier ($5/$25) per the live Anthropic page.
PRICES = {
    "gpt_5_6_terra": ("gpt-5.6-terra", 2.50, 15.00),
    "gpt_5_6_luna": ("gpt-5.6-luna", 1.00, 6.00),
    "gpt_5_6_sol": ("gpt-5.6-sol", 5.00, 30.00),
    "claude_opus_5": ("claude-opus-5", 5.00, 25.00),
    "claude_fable_5": ("claude-fable-5", 10.00, 50.00),
}


def load_deepswe() -> dict:
    sys.path.insert(0, str(STANDALONE))
    from loaders import deepswe
    return deepswe.load()


def build_matrix(d: dict) -> tuple[OutcomeMatrix, dict[str, str], dict[str, str]]:
    """DeepSWE -> OutcomeMatrix over the 9-arm pool. Returns (matrix, task_text, repo_of)."""
    arms = [a for a in d["arms"] if a.replace("mini_swe_agent_", "") in NINE]
    assert len(arms) == 9, f"resolved {len(arms)} arms, expected 9"
    ai = {a: d["arms"].index(a) for a in arms}
    score = np.array(d["score"], dtype=float)
    cost = np.array(d["cost"], dtype=float)
    # Drop tasks with any missing cell across the 9 arms: NaN propagation previously made
    # argmin and argmax return the same arm and every ratio NaN.
    rows = [ai[a] for a in arms]
    bad = np.isnan(score[rows]).any(axis=0) | np.isnan(cost[rows]).any(axis=0)
    tasks = [q for q, b in zip(d["tasks"], bad) if not b]
    if bad.any():
        print(f"  dropped {int(bad.sum())} tasks with missing cells")

    # PoolEntry refuses an unpriced model, which is the right discipline. `name` is the arm
    # handle (model x effort) that policy artifacts key on; `model` must be the real provider
    # runtime id. Prices are the live table fetched 2026-07-28. They are not used to compute
    # anything here -- DeepSWE ships measured cost_usd per trial -- but a wrong price in a pool
    # snapshot would silently mislead anyone who later re-prices from it.
    pool = []
    for a in arms:
        handle = a.replace("mini_swe_agent_", "")
        model_id, pin, pout = PRICES[next(m for m in PRICES if handle.startswith(m))]
        pool.append(PoolEntry(
            name=handle,
            kind=ProviderKind.ANTHROPIC if "claude" in a else ProviderKind.OPENAI,
            model=model_id, endpoint=None,
            input_per_mtok=pin, output_per_mtok=pout))
    outcomes = []
    for a in arms:
        i = ai[a]
        for q in tasks:
            j = d["tasks"].index(q)
            outcomes.append(ScenarioOutcome(
                scenario_id=q, task=d["text"].get(q, q), model=a.replace("mini_swe_agent_", ""),
                benchmark="deepswe-v1.1", reward=float(score[i, j]),
                cost_usd=float(cost[i, j]), success=bool(score[i, j] >= 1.0)))
    return (OutcomeMatrix(pool=pool, outcomes=outcomes),
            {q: d["text"].get(q, q) for q in tasks},
            {q: d["group"].get(q, q) for q in tasks})


class CachedEmbedder:
    """Serves the standalone run's vectors, so both runs see byte-identical embeddings.

    The fitter takes an object with `.embed(texts)`, not a bare callable. Serving from cache is
    the point: it isolates the fitter as the only thing that differs between the two runs.
    """

    def __init__(self, text_by_task: dict[str, str]):
        cache = json.loads((STANDALONE / "results" / "deepswe_embeddings.json").read_text())
        missing = [q for q in text_by_task if q not in cache]
        assert not missing, f"{len(missing)} tasks have no cached embedding: {missing[:3]}"
        self._by_text = {text_by_task[q]: cache[q] for q in text_by_task}

    def embed(self, texts: list[str]) -> list[list[float]]:
        out = []
        for t in texts:
            v = self._by_text.get(t)
            if v is None:
                raise KeyError(f"no cached embedding for task text {t[:60]!r}")
            out.append(v)
        return out


def repo_folds(tasks: list[str], repo_of: dict[str, str], n_folds: int, seed: int):
    repos = sorted(set(repo_of[q] for q in tasks))
    random.Random(seed).shuffle(repos)
    buckets: list[list[str]] = [[] for _ in range(n_folds)]
    for i, r in enumerate(repos):
        buckets[i % n_folds].append(r)
    return [[q for q in tasks if repo_of[q] in set(b)] for b in buckets]


def main() -> None:
    import tempfile

    d = load_deepswe()
    matrix, text_by_task, repo_of = build_matrix(d)
    tasks = sorted({o.scenario_id for o in matrix.outcomes})
    embed = CachedEmbedder(text_by_task)
    print(f"matrix: {len(matrix.pool)} arms x {len(tasks)} tasks x "
          f"{len(set(repo_of.values()))} repos, {len(matrix.outcomes)} outcomes")

    folds = repo_folds(tasks, repo_of, 5, 0)
    tot_r_router = tot_c_router = tot_r_base = tot_c_base = 0.0
    n = 0
    mix: collections.Counter = collections.Counter()
    with tempfile.TemporaryDirectory() as td:
        for f, te in enumerate(folds):
            tr = [q for q in tasks if q not in set(te)]
            policy = fit_knn_policy(matrix, bank_path=pathlib.Path(td) / f"{f}_{KNN_BANK_FILENAME}",
                                    fit_ids=tr, embedder=EmbedderSpec(dim=EMBED_DIM),
                                    embed_with=embed, floor_q=0.10)
            ev = evaluate_policy(policy, matrix, te, embedder=embed)
            base = policy.guard_model or policy.default_model
            br = [o.reward for o in matrix.outcomes
                  if o.model == base and o.scenario_id in set(te) and o.reward is not None]
            bc = [o.cost_usd for o in matrix.outcomes
                  if o.model == base and o.scenario_id in set(te)]
            r_cost = ev.cost_per_scenario * ev.scenarios
            tot_r_router += ev.accuracy * len(te); tot_c_router += r_cost
            tot_r_base += float(np.mean(br)) * len(te); tot_c_base += float(np.sum(bc))
            n += len(te)
            mix.update({k: v * ev.scenarios for k, v in ev.model_mix.items()})
            # unscored means the routed arm had no measured row for that task -- it would make
            # the reward silently optimistic, so it must be zero on a complete matrix.
            assert ev.unscored_scenarios == 0, f"fold {f}: {ev.unscored_scenarios} unscored"
            print(f"  fold {f}: n={len(te):3d} guard={base:22s} "
                  f"router r={ev.accuracy:.3f} ${r_cost:7.2f} | "
                  f"base r={np.mean(br):.3f} ${np.sum(bc):7.2f}")

    print(f"\nPRODUCTION PATH (wmo.optimize.knn), repo-grouped 5-fold over {n} tasks:")
    print(f"  baseline : reward {tot_r_base/n:.3f}  ${tot_c_base:.1f}")
    print(f"  router   : reward {tot_r_router/n:.3f}  ${tot_c_router:.1f}  "
          f"{tot_c_base/tot_c_router:.2f}x cheaper")
    print(f"  delta    : {tot_r_router/n - tot_r_base/n:+.3f}")
    print("\n  route distribution (share of all routed tasks):")
    for a, c in mix.most_common():
        print(f"    {c/n*100:5.1f}%  {a}")
    print("\n  standalone implementation measured 2.15x at delta -0.021 "
          "(nested CV, tau/k chosen in-fold).")
    print("  A materially different ratio here means the two implementations disagree.")


if __name__ == "__main__":
    main()
