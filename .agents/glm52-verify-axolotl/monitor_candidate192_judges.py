#!/usr/bin/env python3
"""Append compact health snapshots for the durable candidate judge runs."""

from __future__ import annotations

import argparse
import datetime
import json
import shutil
import subprocess
from collections import Counter
from pathlib import Path


def output_summary(path: Path) -> dict[str, object]:
    """Summarize one append-only judgment output."""
    counts: Counter[str] = Counter()
    rows = 0
    if path.exists():
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                rows += 1
                value = json.loads(line)
                decision = value.get("decision")
                verdict = str(decision.get("verdict")) if isinstance(decision, dict) else "ERROR"
                counts[verdict] += 1
    return {"rows": rows, "verdict_counts": dict(sorted(counts.items()))}


def session_alive(name: str) -> bool:
    """Return whether a named tmux session exists."""
    result = subprocess.run(
        ["tmux", "has-session", "-t", name],
        check=False,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return result.returncode == 0


def gpu_snapshot() -> list[dict[str, int]]:
    """Read utilization and memory use for all local GPUs."""
    command = [
        "nvidia-smi",
        "--query-gpu=index,utilization.gpu,memory.used",
        "--format=csv,noheader,nounits",
    ]
    output = subprocess.check_output(command, text=True)
    rows = []
    for line in output.splitlines():
        index, utilization, memory = (int(value.strip()) for value in line.split(","))
        rows.append({"index": index, "utilization_percent": utilization, "memory_mib": memory})
    return rows


def main() -> int:
    """Append one machine-readable monitor snapshot."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--log", type=Path, required=True)
    args = parser.parse_args()
    total, used, free = shutil.disk_usage(args.root)
    snapshot = {
        "timestamp": datetime.datetime.now(datetime.UTC).isoformat(),
        "sessions": {
            "glm52-candidate192-sonnet46": session_alive("glm52-candidate192-sonnet46"),
            "glm52-candidate192-opus45": session_alive("glm52-candidate192-opus45"),
        },
        "judgments": {
            "sonnet46": output_summary(args.root / "judgments/sonnet46.jsonl"),
            "opus45": output_summary(args.root / "judgments/opus45.jsonl"),
        },
        "disk": {"total": total, "used": used, "free": free},
        "gpus": gpu_snapshot(),
    }
    args.log.parent.mkdir(parents=True, exist_ok=True)
    with args.log.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(snapshot, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
