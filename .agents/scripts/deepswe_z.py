"""What does lowering the confidence bar (z) actually buy, and cost?

z is the evidence margin a pick must clear. Under the asymmetric guard a CHEAPER pick has to
clear -z, i.e. it is accepted unless the evidence says it is significantly worse. Lowering z
therefore accepts weaker and weaker evidence that the cheap arm is fine.

Everything else is held at the measured optimum: pick_lam=0.03 (the peak of DeepSWE's savings
curve), floor_q=0.05, asymmetric guard, se_floor ON, min_pairs=8. So this isolates the safety
threshold. Two fold seeds, because the whole reason a bar exists is stability -- a setting that
looks good on one split and not the other is the failure mode, not a result.
"""
from __future__ import annotations
import pathlib, sys, tempfile
import numpy as np
sys.path.insert(0, str(pathlib.Path(".agents/scripts").resolve()))
from deepswe_knn_repro import CachedEmbedder, build_matrix, load_deepswe, repo_folds
from wmo.optimize.knn import DEFAULT_KNN_Z, fit_knn_policy
from wmo.optimize.policy import KNN_BANK_FILENAME, EmbedderSpec
from wmo.optimize.routing import evaluate_policy

d = load_deepswe(); matrix, text, repo_of = build_matrix(d)
tasks = sorted({o.scenario_id for o in matrix.outcomes}); embed = CachedEmbedder(text)
ZS = [2.0, 1.0, DEFAULT_KNN_Z, 0.25, 0.0]  # 0 is a hard floor: knn_z >= 0 is type-enforced
print(f"shipped bar DEFAULT_KNN_Z = {DEFAULT_KNN_Z}\n")
print(f"{'z':>6s} {'seed0 reward':>13s} {'x cheap':>8s} {'seed1 reward':>13s} {'x cheap':>8s} {'spread':>8s}")
with tempfile.TemporaryDirectory() as td:
    for z in ZS:
        line = []
        for seed in (0, 1):
            folds = repo_folds(tasks, repo_of, 5, seed)
            R = C = BR = BC = 0.0; n = 0
            for f, te in enumerate(folds):
                tr = [q for q in tasks if q not in set(te)]
                pol = fit_knn_policy(matrix, bank_path=pathlib.Path(td)/f"{seed}_{z}_{f}_{KNN_BANK_FILENAME}",
                                     fit_ids=tr, embedder=EmbedderSpec(dim=3072), embed_with=embed,
                                     z=z, floor_q=0.05, pick_lam=0.03)
                p = pol.model_copy(update={"guard_mode": "asymmetric"})
                ev = evaluate_policy(p, matrix, te, embedder=embed)
                base = p.guard_model or p.default_model
                br = [o.reward for o in matrix.outcomes if o.model == base
                      and o.scenario_id in set(te) and o.reward is not None]
                bc = [o.cost_usd for o in matrix.outcomes if o.model == base and o.scenario_id in set(te)]
                R += ev.accuracy*len(te); C += ev.cost_per_scenario*ev.scenarios
                BR += float(np.mean(br))*len(te); BC += float(np.sum(bc)); n += len(te)
            line.append((R/n, BC/C))
        spread = abs(line[0][1]-line[1][1])
        tag = "  <-- shipped" if z == DEFAULT_KNN_Z else ""
        print(f"{z:6.2f} {line[0][0]:13.3f} {line[0][1]:8.2f} {line[1][0]:13.3f} "
              f"{line[1][1]:8.2f} {spread:8.2f}{tag}")
print("\n  baseline reward 0.945.  my standalone rule (no significance test): 1.97x at 0.939")
