"""Pin the tau2 distill task-id splits from the BENCH-B pinned scenario files.

The distill CLI takes plain JSON arrays of task-id strings (`--task-ids` /
`--holdout-task-ids`). For the tau2 rollout source those ids are composite
`domain/tau2_task_id` strings. This script derives them from the committed,
leak-guarded scenario pins (`packages/environment-capture/tau-bench/rl/
scenarios_{train,eval}.jsonl`, seed 4405) by resolving each scenario's
provenance trace id against the corpus metadata, the same mapping the
sim-to-real study used. Consumers load the FILES; nothing re-derives.

Run from the repo root:

    uv run python .agents/distill/pin_tau2_task_ids.py

Idempotent: re-running on the same pins rewrites byte-identical files.
"""

from __future__ import annotations

import json
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
_TAU_DIR = _REPO / "packages" / "environment-capture" / "tau-bench"
_CORPUS = _TAU_DIR / "traces.otel.jsonl"
_SCENARIOS = {
    "train": _TAU_DIR / "rl" / "scenarios_train.jsonl",
    "holdout": _TAU_DIR / "rl" / "scenarios_eval.jsonl",
}
_OUT = {
    "train": Path(__file__).resolve().parent / "tau2-train-task-ids.json",
    "holdout": Path(__file__).resolve().parent / "tau2-holdout-task-ids.json",
}


def _trace_tasks() -> dict[str, str]:
    """trace_id -> 'domain/task_id' from the corpus span metadata."""
    resolved: dict[str, str] = {}
    for line in _CORPUS.read_text(encoding="utf-8").splitlines():
        if not line:
            continue
        span = json.loads(line)
        trace_id = span["traceId"]
        if trace_id in resolved:
            continue
        for attr in span.get("attributes", []):
            if attr["key"] in ("wmo.trace.metadata", "wmh.trace.metadata"):
                metadata = json.loads(attr["value"]["stringValue"])
                resolved[trace_id] = f"{metadata['domain']}/{metadata['task_id']}"
    return resolved


def main() -> int:
    by_trace = _trace_tasks()
    written: dict[str, list[str]] = {}
    for split, scenarios_path in _SCENARIOS.items():
        task_ids: list[str] = []
        for line in scenarios_path.read_text(encoding="utf-8").splitlines():
            if not line:
                continue
            scenario = json.loads(line)
            provenance = scenario["provenance"][0]
            if provenance not in by_trace:
                raise SystemExit(f"unresolved provenance {provenance} in {scenarios_path}")
            task_ids.append(by_trace[provenance])
        # The same real task can back several mined scenarios; the distill loop
        # keys rollouts by task id, so the pin is the DEDUPLICATED, order-stable list.
        deduped = sorted(set(task_ids))
        _OUT[split].write_text(json.dumps(deduped, indent=1) + "\n", encoding="utf-8")
        written[split] = deduped
    overlap = set(written["train"]) & set(written["holdout"])
    if overlap:
        raise SystemExit(f"train/holdout overlap at the real-task level: {sorted(overlap)}")
    for split, ids in written.items():
        domains: dict[str, int] = {}
        for task_id in ids:
            domain = task_id.partition("/")[0]
            domains[domain] = domains.get(domain, 0) + 1
        print(f"{split}: {len(ids)} task(s) {domains} -> {_OUT[split].name}")  # noqa: T201
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
