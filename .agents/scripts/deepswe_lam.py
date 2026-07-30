"""Where does DeepSWE's savings curve turn around? Sweep pick_lam past the shipped cap.

COST_QUALITY_MAX_LAM=0.03 was calibrated on RouterBench (lam=0.03 -> -46.2%, decaying to
-31.2% by lam=0.12). This asks whether DeepSWE's turnaround sits somewhere else.

Only the COST PREFERENCE (pick_lam) moves. knn_z stays at the fitted bar and se_floor/min_pairs
are untouched: the evidence test a cheaper pick must clear is exactly as strict as shipped.
"""
from __future__ import annotations
import pathlib, sys, tempfile
import numpy as np
sys.path.insert(0, str(pathlib.Path(".agents/scripts").resolve()))
from deepswe_knn_repro import CachedEmbedder, build_matrix, load_deepswe, repo_folds
from wmo.optimize.knn import COST_QUALITY_MAX_LAM, fit_knn_policy
from wmo.optimize.policy import KNN_BANK_FILENAME, EmbedderSpec
from wmo.optimize.routing import evaluate_policy

d = load_deepswe(); matrix, text, repo_of = build_matrix(d)
tasks = sorted({o.scenario_id for o in matrix.outcomes}); embed = CachedEmbedder(text)
folds = repo_folds(tasks, repo_of, 5, 0)
LAMS = [0.0, 0.01, 0.03, 0.05, 0.08, 0.12, 0.20, 0.35, 0.60]
print(f"shipped cap COST_QUALITY_MAX_LAM = {COST_QUALITY_MAX_LAM}\n")
print(f"{'pick_lam':>9s} {'reward':>7s} {'cost$':>8s} {'x cheap':>8s} {'vs base':>8s}")
with tempfile.TemporaryDirectory() as td:
    fitted = []
    for f, te in enumerate(folds):
        tr = [q for q in tasks if q not in set(te)]
        fitted.append((fit_knn_policy(matrix, bank_path=pathlib.Path(td)/f"{f}_{KNN_BANK_FILENAME}",
                       fit_ids=tr, embedder=EmbedderSpec(dim=3072), embed_with=embed,
                       floor_q=0.05), te))
    for lam in LAMS:
        R = C = BR = BC = 0.0; n = 0
        for pol, te in fitted:
            p = pol.model_copy(update={"pick_lam": lam, "guard_mode": "asymmetric"})
            ev = evaluate_policy(p, matrix, te, embedder=embed)
            base = p.guard_model or p.default_model
            br = [o.reward for o in matrix.outcomes if o.model == base
                  and o.scenario_id in set(te) and o.reward is not None]
            bc = [o.cost_usd for o in matrix.outcomes if o.model == base and o.scenario_id in set(te)]
            R += ev.accuracy*len(te); C += ev.cost_per_scenario*ev.scenarios
            BR += float(np.mean(br))*len(te); BC += float(np.sum(bc)); n += len(te)
        star = "  <-- shipped cap" if abs(lam-COST_QUALITY_MAX_LAM) < 1e-9 else ""
        print(f"{lam:9.2f} {R/n:7.3f} {C:8.1f} {BC/C:8.2f} {R/n-BR/n:+8.3f}{star}")
    print(f"\n  baseline reward {BR/n:.3f} ${BC:.1f}")
    print("  my standalone rule (NO significance test): 1.97x at reward 0.939")
