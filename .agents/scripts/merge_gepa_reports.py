#!/usr/bin/env python
"""Rebuild/merge a gepa-scaling `AblationReport` from runner logs and partial report JSONs.

Salvage tool: the runner writes its report JSON only at the END of a sweep, so a crash on the last
point loses the file even though every completed point is in the log (`  t64_b8  seed=0
fidelity=0.872` lines). This script re-parses those lines from any number of logs, merges in any
partial report JSONs (e.g. a single re-run point), recomputes per-condition mean/std across the
union of seeds, and writes a well-formed `AblationReport`.

    uv run python .agents/scripts/merge_gepa_reports.py --name gepa-scaling-law \
        --log /tmp/gepa_logs/tau_budget.log --report tau-bench_extra_point.json \
        --out tau-bench_budget.json
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from wmh.research.ablation import AblationReport, Condition, SeedScore, aggregate

_LINE_RE = re.compile(r"^\s*(?P<label>t\d+_b\d+)\s+seed=(?P<seed>\d+)\s+fidelity=(?P<score>[\d.]+)")
_LABEL_RE = re.compile(r"^t(?P<n>\d+)_b(?P<b>\d+)$")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log", action="append", default=[], help="Runner log to parse.")
    parser.add_argument("--report", action="append", default=[], help="Report JSON to merge.")
    parser.add_argument("--name", default="gepa-scaling-law", help="Report name.")
    parser.add_argument("--out", required=True, help="Merged AblationReport path.")
    args = parser.parse_args()

    # label -> {seed -> score}; later sources win on duplicates (log first, then reports).
    cells: dict[str, dict[int, float]] = {}
    for log in args.log:
        for line in Path(log).read_text(encoding="utf-8").splitlines():
            m = _LINE_RE.match(line)
            if m:
                cells.setdefault(m["label"], {})[int(m["seed"])] = float(m["score"])
    for report in args.report:
        data = json.loads(Path(report).read_text(encoding="utf-8"))
        for cell in data.get("conditions", []):
            label = cell["condition"]["label"]
            for entry in cell.get("per_seed", []):
                cells.setdefault(label, {})[int(entry["seed"])] = float(entry["score"])

    if not cells:
        raise SystemExit("nothing to merge — no fidelity lines or report conditions found")

    conditions = []
    seeds: set[int] = set()
    for label, by_seed in cells.items():
        m = _LABEL_RE.match(label)
        if not m:
            raise SystemExit(f"unrecognized condition label {label!r}")
        condition = Condition(label=label, params={"n_train": int(m["n"]), "budget": int(m["b"])})
        per_seed = [SeedScore(seed=s, score=by_seed[s]) for s in sorted(by_seed)]
        seeds.update(by_seed)
        conditions.append(aggregate(condition, per_seed))

    merged = AblationReport(name=args.name, seeds=sorted(seeds), conditions=conditions)
    Path(args.out).write_text(merged.model_dump_json(indent=2), encoding="utf-8")
    for cell in merged.conditions:
        print(f"  {cell.summary()}")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
