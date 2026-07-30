"""Ablate each safeguard one at a time. Is the significance machinery earning its keep?

The claim under test is NOT "safeguards improve the mean" -- it is "safeguards reduce variance
across data splits". So every row reports mean cost-ratio AND the spread across 4 fold seeds.
A config with a great mean and a huge spread is a config that got lucky on one split.

Ablations are single-knob edits off the full config, so each row attributes to one mechanism.
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
SEEDS = (0, 1, 2, 3)
FULL = dict(z=0.5, min_pairs=8, se_floor=True, floor_q=0.05, pick_lam=0.03)

# (label, fit-kwarg overrides, policy-field overrides)
ABL = [
    ("FULL (shipped knobs)",        {},                    {}),
    ("- significance test",         {},                    {"guard_margin": None}),
    ("- standard-error floor",      {"se_floor": False},   {}),
    ("- min support (8 -> 1)",      {"min_pairs": 1},      {}),
    ("- novelty floor",             {"floor_q": 0.0},      {}),
    ("- cost tilt",                 {"pick_lam": 0.0},     {}),
    ("- ALL of the above",          {"se_floor": False, "min_pairs": 1, "floor_q": 0.0},
                                                           {"guard_margin": None}),
]

def run(fit_kw, pol_kw, seed, td, tag):
    folds = repo_folds(tasks, repo_of, 5, seed)
    R = C = BR = BC = 0.0; n = 0
    kw = {**FULL, **fit_kw}
    for f, te in enumerate(folds):
        tr = [q for q in tasks if q not in set(te)]
        pol = fit_knn_policy(matrix, bank_path=pathlib.Path(td)/f"{tag}_{seed}_{f}_{KNN_BANK_FILENAME}",
                             fit_ids=tr, embedder=EmbedderSpec(dim=3072), embed_with=embed, **kw)
        p = pol.model_copy(update={"guard_mode": "asymmetric", **pol_kw})
        ev = evaluate_policy(p, matrix, te, embedder=embed)
        base = p.guard_model or p.default_model
        br = [o.reward for o in matrix.outcomes if o.model == base
              and o.scenario_id in set(te) and o.reward is not None]
        bc = [o.cost_usd for o in matrix.outcomes if o.model == base and o.scenario_id in set(te)]
        R += ev.accuracy*len(te); C += ev.cost_per_scenario*ev.scenarios
        BR += float(np.mean(br))*len(te); BC += float(np.sum(bc)); n += len(te)
    return R/n*100, BC/C, BR/n*100

print(f"{'config':26s} {'quality':>8s} {'x cheap':>8s} {'ratio range':>13s} {'spread':>7s}")
with tempfile.TemporaryDirectory() as td:
    for i, (label, fk, pk) in enumerate(ABL):
        q, r = [], []
        for s in SEEDS:
            qq, rr, base_q = run(fk, pk, s, td, f"a{i}")
            q.append(qq); r.append(rr)
        print(f"{label:26s} {statistics.mean(q):7.1f} {statistics.mean(r):8.2f} "
              f"{min(r):5.2f}-{max(r):5.2f} {max(r)-min(r):7.2f}")
    print(f"\n  baseline quality {base_q:.1f} (always best single arm)")
    print("  my standalone rule: quality 93.9, 1.97x, ratio range 0.20-4.54 (spread 4.34)")
