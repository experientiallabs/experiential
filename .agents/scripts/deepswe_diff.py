"""Where do the two rules actually disagree, and what does each disagreement cost?

No theory. For every task: what does the shipped policy pick, what does my threshold rule pick,
what did each cost, what did each actually score. Then attribute the whole cost gap to the
specific tasks where they differ.
"""
from __future__ import annotations
import collections, pathlib, sys, tempfile
import numpy as np
sys.path.insert(0, str(pathlib.Path(".agents/scripts").resolve()))
sys.path.insert(0, "/Users/admin/Documents/experientiallabs/coding-router")
from deepswe_knn_repro import CachedEmbedder, build_matrix, load_deepswe, repo_folds
from wmo.optimize.knn import fit_knn_policy
from wmo.optimize.policy import KNN_BANK_FILENAME, EmbedderSpec
from wmo.optimize.routing import route_scenarios

d = load_deepswe(); matrix, text, repo_of = build_matrix(d)
tasks = sorted({o.scenario_id for o in matrix.outcomes}); embed = CachedEmbedder(text)
arms = [p.name for p in matrix.pool]
G = {(o.model, o.scenario_id): o.reward for o in matrix.outcomes}
C = {(o.model, o.scenario_id): o.cost_usd for o in matrix.outcomes}

# my rule, verbatim: similarity-weighted P(fully solves) over 12 neighbours, cheapest above 0.5
emb_by_task = {q: np.array(embed.embed([text[q]])[0], float) for q in tasks}
for q in emb_by_task: emb_by_task[q] /= np.linalg.norm(emb_by_task[q])
def mine(te, tr):
    med = {a: float(np.median([C[(a, q)] for q in tr])) for a in arms}
    order = sorted(arms, key=lambda a: med[a])
    best = max(arms, key=lambda a: np.mean([G[(a, q)] >= 1.0 for q in tr]))
    out = {}
    for q in te:
        sims = np.array([emb_by_task[q] @ emb_by_task[t] for t in tr])
        nn = np.argsort(-sims)[:12]; w = np.clip(sims[nn], 0, None) + 1e-6
        p = {a: float(np.sum([(G[(a, tr[i])] >= 1.0) * w[k] for k, i in enumerate(nn)]) / w.sum())
             for a in arms}
        out[q] = next((a for a in order if p[a] >= 0.5), best)
    return out

folds = repo_folds(tasks, repo_of, 5, 0)
pick_t, pick_m = {}, {}
with tempfile.TemporaryDirectory() as td:
    for f, te in enumerate(folds):
        tr = [q for q in tasks if q not in set(te)]
        pol = fit_knn_policy(matrix, bank_path=pathlib.Path(td)/f"{f}_{KNN_BANK_FILENAME}",
                             fit_ids=tr, embedder=EmbedderSpec(dim=3072), embed_with=embed,
                             floor_q=0.05, pick_lam=0.03)
        p = pol.model_copy(update={"guard_mode": "asymmetric"})
        for q, dec in route_scenarios(p, matrix, te, embedder=embed).items():
            pick_t[q] = dec.model
        pick_m.update(mine(te, tr))

same = [q for q in tasks if pick_t[q] == pick_m[q]]
diff = [q for q in tasks if pick_t[q] != pick_m[q]]
ct = sum(C[(pick_t[q], q)] for q in tasks); cm = sum(C[(pick_m[q], q)] for q in tasks)
print(f"{len(tasks)} tasks: agree on {len(same)}, disagree on {len(diff)}\n")
print(f"  theirs total ${ct:.1f}   mine total ${cm:.1f}   ratio {ct/cm:.2f}x")
print(f"  on the {len(same)} AGREED tasks: theirs ${sum(C[(pick_t[q],q)] for q in same):.1f} "
      f"= mine ${sum(C[(pick_m[q],q)] for q in same):.1f}")
print(f"  on the {len(diff)} DISAGREED tasks: theirs ${sum(C[(pick_t[q],q)] for q in diff):.1f} "
      f"vs mine ${sum(C[(pick_m[q],q)] for q in diff):.1f}")
print(f"  quality on disagreed: theirs {np.mean([G[(pick_t[q],q)] for q in diff]):.3f} "
      f"vs mine {np.mean([G[(pick_m[q],q)] for q in diff]):.3f}")
print(f"\n  what THEIRS picks on the disagreed tasks:")
for a, k in collections.Counter(pick_t[q] for q in diff).most_common(5): print(f"    {k:3d}  {a}")
print(f"  what MINE picks on the disagreed tasks:")
for a, k in collections.Counter(pick_m[q] for q in diff).most_common(5): print(f"    {k:3d}  {a}")
