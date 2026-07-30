"""Does the GUARD ARM choice explain the cost gap? Pin it to each arm and measure.

`guard_model=None` auto-discovers the best single arm on the fit split, which is the most
expensive-ish one -- so every revert lands on it. Pinning a cheaper guard makes reverts cheap,
at the price of a worse fallback. This measures that trade directly.
"""
from __future__ import annotations
import pathlib, statistics, sys, tempfile
import numpy as np
sys.path.insert(0, str(pathlib.Path(".agents/scripts").resolve()))
from deepswe_knn_repro import CachedEmbedder, build_matrix, load_deepswe, repo_folds
from wmo.optimize.knn import fit_knn_policy
from wmo.optimize.policy import KNN_BANK_FILENAME, EmbedderSpec
from wmo.optimize.routing import evaluate_policy

d = load_deepswe(); matrix, text, repo_of = build_matrix(d)
tasks = sorted({o.scenario_id for o in matrix.outcomes}); embed = CachedEmbedder(text)
arms = [p.name for p in matrix.pool]
C = {(o.model, o.scenario_id): o.cost_usd for o in matrix.outcomes}
G = {(o.model, o.scenario_id): o.reward for o in matrix.outcomes}
med = {a: float(np.median([C[(a, q)] for q in tasks])) for a in arms}
solo_q = {a: float(np.mean([G[(a, q)] for q in tasks])) * 100 for a in arms}
BASE_C = sum(C[("claude_opus_5_high", q)] for q in tasks)
BASE_Q = solo_q["claude_opus_5_high"]

print(f"always-best baseline: quality {BASE_Q:.1f}  ${BASE_C:.1f}\n")
print(f"{'pinned guard arm':24s} {'arm solo':>9s} {'$/task':>7s} | {'router q':>9s} "
      f"{'cost$':>8s} {'x cheap':>8s} {'spread':>7s}")
with tempfile.TemporaryDirectory() as td:
    for a in sorted(arms, key=lambda x: med[x]):
        qs, rs = [], []
        for seed in (0, 1, 2):
            folds = repo_folds(tasks, repo_of, 5, seed)
            R = Ct = 0.0; n = 0
            for f, te in enumerate(folds):
                tr = [q for q in tasks if q not in set(te)]
                pol = fit_knn_policy(matrix, bank_path=pathlib.Path(td)/f"{a}_{seed}_{f}_{KNN_BANK_FILENAME}",
                                     fit_ids=tr, embedder=EmbedderSpec(dim=3072), embed_with=embed,
                                     guard_model=a, floor_q=0.05, pick_lam=0.03)
                p = pol.model_copy(update={"guard_mode": "asymmetric"})
                ev = evaluate_policy(p, matrix, te, embedder=embed)
                R += ev.accuracy*len(te); Ct += ev.cost_per_scenario*ev.scenarios; n += len(te)
            qs.append(R/n*100); rs.append(BASE_C/Ct)
        print(f"{a:24s} {solo_q[a]:8.1f} {med[a]:7.2f} | {statistics.mean(qs):9.1f} "
              f"{BASE_C/statistics.mean(rs):8.1f} {statistics.mean(rs):8.2f} {max(rs)-min(rs):7.2f}")
print("\n  'arm solo' = that arm used alone on all tasks. 'router q' = quality with it as guard.")
print("  my threshold rule for reference: quality 93.9, 1.97x")
