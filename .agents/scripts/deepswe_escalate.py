"""Cheap guard + escalate up. Can the router beat its own cheap default?

With an expensive guard the router routes DOWN and every deviation risks quality. With a cheap
guard it escalates UP, so a deviation can only be an attempt to buy quality. Symmetric mode is
the natural pairing: a PRICIER pick must clear +z, i.e. go expensive only on evidence.

The bar to beat is not always-opus. It is `luna_max` used alone: graded 94.7 at USD 351.
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
G = {(o.model, o.scenario_id): o.reward for o in matrix.outcomes}
C = {(o.model, o.scenario_id): o.cost_usd for o in matrix.outcomes}
GUARD = "gpt_5_6_luna_max"
solo_q = float(np.mean([G[(GUARD, q)] for q in tasks]))*100
solo_c = sum(C[(GUARD, q)] for q in tasks)
opus_c = sum(C[("claude_opus_5_high", q)] for q in tasks)
print(f"bar to beat -- {GUARD} ALONE: quality {solo_q:.1f}  ${solo_c:.1f}  "
      f"({opus_c/solo_c:.2f}x vs always-opus)\n")
print(f"{'mode':11s} {'z':>4s} {'lam':>5s} | {'quality':>8s} {'cost$':>7s} {'vs solo q':>10s} "
      f"{'vs solo $':>10s} {'beats solo?':>12s}")
with tempfile.TemporaryDirectory() as td:
    for mode in ("symmetric", "asymmetric"):
        for z in (0.0, 0.5, 1.0):
            for lam in (0.0, 0.03):
                qs, cs = [], []
                for seed in (0, 1, 2):
                    folds = repo_folds(tasks, repo_of, 5, seed)
                    R = Ct = 0.0; n = 0
                    for f, te in enumerate(folds):
                        tr = [q for q in tasks if q not in set(te)]
                        pol = fit_knn_policy(matrix, bank_path=pathlib.Path(td)/f"{mode}{z}{lam}{seed}{f}_{KNN_BANK_FILENAME}",
                                             fit_ids=tr, embedder=EmbedderSpec(dim=3072),
                                             embed_with=embed, guard_model=GUARD, z=z,
                                             floor_q=0.05, pick_lam=lam)
                        p = pol.model_copy(update={"guard_mode": mode})
                        ev = evaluate_policy(p, matrix, te, embedder=embed)
                        R += ev.accuracy*len(te); Ct += ev.cost_per_scenario*ev.scenarios; n += len(te)
                    qs.append(R/n*100); cs.append(Ct)
                q, c = statistics.mean(qs), statistics.mean(cs)
                win = "YES" if (q >= solo_q - 0.05 and c < solo_c) or (q > solo_q and c <= solo_c*1.02) else "no"
                print(f"{mode:11s} {z:4.1f} {lam:5.2f} | {q:8.1f} {c:7.1f} {q-solo_q:+10.1f} "
                      f"{c-solo_c:+10.1f} {win:>12s}")
