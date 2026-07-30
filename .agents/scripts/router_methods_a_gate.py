"""Zero-API-spend pre-flight gate for METHOD A.

Runs BEFORE any model spend is committed, because the one failure mode that could silently burn
the whole $4,000 is a broken grading path: every episode would come back ungradeable and no amount
of model spend would produce a number. This is a cheap assert, not a canary run -- it makes no
model calls at all.

Two checks per task, using the publisher's own artifacts:
  ORACLE  apply solution/solution.patch  -> f2p MUST be total/total, binary reward 1
  EMPTY   apply nothing                  -> f2p MUST be 0/total, binary reward 0

If ORACLE does not score 1.0 the verifier plumbing is wrong (tests not copied, patch path wrong,
reward.json not parsed). If EMPTY does not score 0.0 the f2p whitelist is not actually failing at
base, so the task cannot discriminate and the graded reward is meaningless.

    python3 router_methods_a_gate.py --n 2
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import router_methods_a_collect as M  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/scratch/router_a")
    ap.add_argument("--n", type=int, default=2)
    a = ap.parse_args()

    root = pathlib.Path(a.root)
    man = json.loads((root / "manifest.json").read_text())
    troot = root / "tree" / "deep-swe-main" / "tasks"
    M.semaphore(4)

    ok = True
    for e in man[: a.n]:
        tid = e["task_id"]
        tests = troot / tid / "tests"
        oracle = (troot / tid / "solution" / "solution.patch").read_text()
        print(f"\n=== {tid}  repo={e['repo']}  f2p_expected={e['n_f2p']}  "
              f"oracle={len(oracle)}B ===", flush=True)
        M.pull_image(e["image"], log=lambda m: print(f"  {m}", flush=True))
        got = {}
        for label, patch in (("ORACLE", oracle), ("EMPTY", "")):
            t0 = time.time()
            g = M.grade(e["image"], tests, patch, f"gate-{label.lower()}-{tid}"[:60], 4)
            f2p = (f"{g.get('f2p_passed')}/{g.get('f2p_total')}")
            print(f"  {label:6s} verifier_ok={g.get('verifier_ok')} f2p={f2p} "
                  f"p2p={g.get('p2p_passed')}/{g.get('p2p_total')} "
                  f"graded={g.get('f2p_passed')/g['f2p_total'] if g.get('f2p_total') else None} "
                  f"binary={g.get('reward')} apply_failed={g.get('apply_failed', 0)} "
                  f"{time.time() - t0:.0f}s", flush=True)
            if not g.get("verifier_ok"):
                print("  TAIL:", (g.get("tail") or "")[-1200:], flush=True)
            got[label] = g
        M._docker("rmi", "-f", e["image"], timeout=600.0, check=False)

        o, z = got["ORACLE"], got["EMPTY"]
        if not (o.get("verifier_ok") and o.get("f2p_total") and
                o["f2p_passed"] == o["f2p_total"]):
            print(f"  FAIL {tid}: oracle patch did not fully pass f2p", flush=True)
            ok = False
        if not (z.get("verifier_ok") and z.get("f2p_passed") == 0):
            print(f"  FAIL {tid}: base state did not score 0 f2p", flush=True)
            ok = False

    print(f"\nGATE {'PASS' if ok else 'FAIL'}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
