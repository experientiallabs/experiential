#!/usr/bin/env python3
"""Append one health snapshot for both candidate-192 Axolotl training seeds."""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


METRIC_RE = re.compile(
    r"\{'loss':\s*'(?P<loss>[^']+)',\s*'grad_norm':\s*'(?P<grad>[^']+)',"
    r"\s*'learning_rate':\s*'(?P<lr>[^']+)'"
)
WANDB_RE = re.compile(r"https://wandb\.ai/[^\s\x1b]+/runs/[a-zA-Z0-9]+")


def run(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(args, check=False, capture_output=True, text=True)


def session_alive(name: str) -> bool:
    return run("tmux", "has-session", "-t", name).returncode == 0


def gpu_health() -> list[dict[str, int]]:
    proc = run(
        "nvidia-smi",
        "--query-gpu=index,utilization.gpu,memory.used,memory.total",
        "--format=csv,noheader,nounits",
    )
    if proc.returncode != 0:
        return []
    return [
        {
            "index": int(parts[0]),
            "utilization_percent": int(parts[1]),
            "memory_mib": int(parts[2]),
            "memory_total_mib": int(parts[3]),
        }
        for line in proc.stdout.splitlines()
        if (parts := [int(value.strip()) for value in line.split(",")])
    ]


def seed_health(root: Path, seed: int) -> dict[str, Any]:
    name = f"qwen35-4b-glm52-candidate-realverified-sft-lr1e5-r64-seed{seed}"
    log = root / "logs" / f"{name}.log"
    output = root / "checkpoints" / name
    session = f"glm52-candidate192-sft-seed{seed}"
    text = log.read_text(encoding="utf-8", errors="replace") if log.exists() else ""
    metrics = list(METRIC_RE.finditer(text))
    latest = metrics[-1].groupdict() if metrics else None
    checkpoints = sorted(
        int(path.name.rsplit("-", 1)[-1])
        for path in output.glob("checkpoint-*")
        if path.name.rsplit("-", 1)[-1].isdigit()
    ) if output.exists() else []
    urls = WANDB_RE.findall(text)
    lowered = text.lower()
    return {
        "seed": seed,
        "session_alive": session_alive(session),
        "step": len(metrics),
        "latest": latest,
        "checkpoints": checkpoints,
        "completed": "training completed!" in lowered and "model successfully saved" in lowered,
        "wandb_url": urls[-1] if urls else None,
        "signals": {
            "traceback": "traceback" in lowered,
            "oom": "out of memory" in lowered or "cuda oom" in lowered,
            "nan": bool(re.search(r"(?<![a-z])nan(?![a-z])", lowered)),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--health-log", type=Path, required=True)
    args = parser.parse_args()
    disk = shutil.disk_usage(args.root)
    snapshot = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "runs": [seed_health(args.root, seed) for seed in (20260809, 20260810)],
        "disk": {"free": disk.free, "used": disk.used, "total": disk.total},
        "gpus": gpu_health(),
    }
    args.health_log.parent.mkdir(parents=True, exist_ok=True)
    with args.health_log.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(snapshot, sort_keys=True) + "\n")
    print(json.dumps(snapshot, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
