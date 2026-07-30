"""Sweep the shipped cost/quality dial on DeepSWE, instead of weakening the guard.

`apply_cost_quality` holds knn_z fixed at every position: the confidence bar never relaxes.
What opens is coverage (floor_q 0.5 -> 0.05 over [0, 0.25]) and then a cost preference
(pick_lam over [0.25, 1.0]). So this asks "what does the disciplined fitter give at an explicit
cost preference", which is the sanctioned way to buy savings.
"""
from __future__ import annotations
import collections, pathlib, sys, tempfile
import numpy as np
sys.path.insert(0, str(pathlib.Path(".agents/scripts").resolve()))
from deepswe_knn_repro import CachedEmbedder, build_matrix, load_deepswe, repo_folds
from wmo.optimize.knn import (COST_QUALITY_ANCHORS, COST_QUALITY_BALANCED,
                              apply_cost_quality, cost_quality_knobs, fit_knn_policy)
from wmo.optimize.policy import KNN_BANK_FILENAME, EmbedderSpec
from wmo.optimize.routing import evaluate_policy

d = load_deepswe(); matrix, text_by_task, repo_of = build_matrix(d)
tasks = sorted({o.scenario_id for o in matrix.outcomes}); embed = CachedEmbedder(text_by_task)
folds = repo_folds(tasks, repo_of, 5, 0)
print(f"matrix {len(matrix.pool)} arms x {len(tasks)} tasks x {len(set(repo_of.values()))} repos")
print("\nshipped MEASURED anchors (their benchmark, not ours):")
for a in COST_QUALITY_ANCHORS:
    print(f"   dial {a.cost_quality:4.2f} {a.named_point:14s} quality {a.quality_delta_points:+5.2f}pt "
          f"cost {a.cost_delta_percent:+6.1f}%")

DIALS = [0.0, 0.25, 0.5, 0.75, 0.9, 1.0]
print(f"\n{'dial':>5s} {'floor_q':>8s} {'pick_lam':>9s} {'guard':>11s} {'reward':>7s} "
      f"{'cost$':>8s} {'x cheap':>8s} {'->guard':>8s}")
with tempfile.TemporaryDirectory() as td:
    fitted = []
    for f, te in enumerate(folds):
        tr = [q for q in tasks if q not in set(te)]
        fitted.append((fit_knn_policy(matrix, bank_path=pathlib.Path(td)/f"{f}_{KNN_BANK_FILENAME}",
                       fit_ids=tr, embedder=EmbedderSpec(dim=3072), embed_with=embed), te))
    for dial in DIALS:
        kn = cost_quality_knobs(dial)
        R = C = BR = BC = 0.0; n = 0; mix = collections.Counter()
        for pol, te in fitted:
            p = apply_cost_quality(pol, dial)
            ev = evaluate_policy(p, matrix, te, embedder=embed)
            base = p.guard_model or p.default_model
            br = [o.reward for o in matrix.outcomes if o.model == base
                  and o.scenario_id in set(te) and o.reward is not None]
            bc = [o.cost_usd for o in matrix.outcomes if o.model == base and o.scenario_id in set(te)]
            R += ev.accuracy*len(te); C += ev.cost_per_scenario*ev.scenarios
            BR += float(np.mean(br))*len(te); BC += float(np.sum(bc)); n += len(te)
            mix.update({k: v*ev.scenarios for k, v in ev.model_mix.items()})
            gm = base
        gshare = mix.get(gm, 0)/n
        print(f"{dial:5.2f} {kn.floor_q:8.2f} {kn.pick_lam:9.2f} {kn.guard_mode:>11s} "
              f"{R/n:7.3f} {C:8.1f} {BC/C:8.2f} {gshare*100:7.1f}%")
    print(f"\n  baseline reward {BR/n:.3f} ${BC:.1f}   (my standalone rule: 1.97x at reward 0.939)")
