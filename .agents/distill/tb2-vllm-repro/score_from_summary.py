"""Final scoring for the TB2 off-Tinker reproduction.

Binary rates come from harbor's own job-level summary (reward_stats buckets list trial names,
and trial names carry the task), which avoids parsing 272 multi-megabyte per-trial result.json
files. Graded ctrf comes from the small verifier/*.json files. Paired delta is bootstrapped over
TASKS, not trials, because attempts within a task share a task.
"""

import glob
import json
import os
import random
import re
from collections import defaultdict

ROOT = "/scratch/repro-tb2"
ARMS = ("base", "distill")
TRIAL_RE = re.compile(r"^(?P<task>.+)__[A-Za-z0-9]+$")


def task_of(trial_name):
    m = TRIAL_RE.match(trial_name)
    return m.group("task") if m else trial_name


def load_binary(arm):
    """task -> list of 0/1, from harbor's reward_stats buckets."""
    p = "%s/jobs-%s/tb2-repro-%s/result.json" % (ROOT, arm, arm)
    d = json.load(open(p))
    stats = d["stats"]
    per_task = defaultdict(list)
    total = 0
    for ev in stats.get("evals", {}).values():
        for reward_str, trials in ev.get("reward_stats", {}).get("reward", {}).items():
            val = 1.0 if float(reward_str) == 1.0 else 0.0
            for tn in trials:
                per_task[task_of(tn)].append(val)
                total += 1
    return per_task, total, stats


def load_ctrf(arm):
    """task -> list of passed/total test fractions."""
    per_task = defaultdict(list)
    base = "%s/jobs-%s/tb2-repro-%s" % (ROOT, arm, arm)
    for trial_dir in sorted(glob.glob(base + "/*/")):
        name = os.path.basename(trial_dir.rstrip("/"))
        if "__" not in name:
            continue
        for jf in sorted(glob.glob(trial_dir + "verifier/*.json")):
            try:
                doc = json.load(open(jf))
            except Exception:
                continue
            s = doc.get("results", {}).get("summary")
            if isinstance(s, dict) and s.get("tests"):
                per_task[task_of(name)].append(s.get("passed", 0) / s["tests"])
                break
    return per_task


def rate(per_task):
    vals = [v for vs in per_task.values() for v in vs]
    return (sum(vals) / len(vals) if vals else 0.0), int(sum(vals)), len(vals)


def bootstrap(deltas, n=20000, seed=0):
    rng = random.Random(seed)
    means = []
    k = len(deltas)
    for _ in range(n):
        means.append(sum(deltas[rng.randrange(k)] for _ in range(k)) / k)
    means.sort()
    return means[int(0.025 * n)], means[int(0.975 * n)]


bin_ = {}
ctrf = {}
stats = {}
for a in ARMS:
    bin_[a], tot, stats[a] = load_binary(a)
    ctrf[a] = load_ctrf(a)
    print("%s: loaded %d trials across %d tasks" % (a, tot, len(bin_[a])))

print("\n%-10s %-20s %-18s %-14s" % ("arm", "binary", "graded ctrf", "errored"))
print("-" * 68)
for a in ARMS:
    br, bs, bn = rate(bin_[a])
    gr, _, gn = rate(ctrf[a])
    ne = stats[a].get("n_errored_trials")
    nt = stats[a].get("n_completed_trials")
    print("%-10s %5.1f%% (%3d/%3d)   %5.1f%% (n=%3d)   %3d/%3d = %4.1f%%"
          % (a, 100 * br, bs, bn, 100 * gr, gn, ne, nt, 100.0 * ne / nt))

shared = sorted(set(bin_["base"]) & set(bin_["distill"]))
rows = []
for t in shared:
    b = sum(bin_["base"][t]) / len(bin_["base"][t])
    d = sum(bin_["distill"][t]) / len(bin_["distill"][t])
    rows.append((t, b, d, len(bin_["base"][t]), len(bin_["distill"][t])))

deltas = [d - b for _, b, d, _, _ in rows]
mean = sum(deltas) / len(deltas)
lo, hi = bootstrap(deltas)

print("\n%-34s %6s %6s %8s   n(b/d)" % ("task", "base", "distill", "delta"))
print("-" * 74)
for t, b, d, nb, nd in rows:
    star = " *" if abs(d - b) > 1e-9 else ""
    print("%-34s %6.3f %6.3f %+8.3f   %d/%d%s" % (t, b, d, d - b, nb, nd, star))
print("-" * 74)
print("paired per-task delta: %+.4f   95%% bootstrap CI [%+.4f, %+.4f]  %s"
      % (mean, lo, hi, "INCLUDES ZERO" if lo <= 0 <= hi else "EXCLUDES ZERO"))

pinned = sum(1 for _, b, d, _, _ in rows if b == d and b in (0.0, 1.0))
moved = sum(1 for _, b, d, _, _ in rows if abs(d - b) > 1e-9)
print("tasks moved: %d/%d   pinned at floor/ceiling in both arms: %d" % (moved, len(rows), pinned))

# graded paired
gshared = sorted(set(ctrf["base"]) & set(ctrf["distill"]))
if gshared:
    gd = [sum(ctrf["distill"][t]) / len(ctrf["distill"][t])
          - sum(ctrf["base"][t]) / len(ctrf["base"][t]) for t in gshared]
    gm = sum(gd) / len(gd)
    glo, ghi = bootstrap(gd)
    print("\ngraded ctrf paired delta over %d tasks: %+.4f  95%% CI [%+.4f, %+.4f]"
          % (len(gshared), gm, glo, ghi))

teacher = 0.490
print("\ngate: 0.7 x teacher(%.3f) = %.3f -> distill %.3f : %s"
      % (teacher, 0.7 * teacher, rate(bin_["distill"])[0],
         "MET" if rate(bin_["distill"])[0] >= 0.7 * teacher else "NOT MET"))
