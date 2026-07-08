"""Materialize the offline GAIA2 slice into this benchmark's on-disk shape (run under the venv).

GAIA2 (``meta-agents-research-environments/gaia2``, CC-BY-4.0) ships scenarios as JSON universes on
HuggingFace. This script reads the ``execution`` and ``search`` capability validation splits — the
two whose tasks are completed from the initial universe state by agent tool calls, with no
time-driven environment events — and writes:

  - ``data/{train,test}.jsonl`` — one row per in-scope scenario:
    ``{task_id: "gaia2-{split}-{i}", prompt: <USER message>, data: {config, gaia2_id,
    oracle: [{app, function, args}]}}``. The oracle actions are the scenario's ground-truth AGENT
    events; ``prompt`` is the scenario's USER message. Both are SMALL and committed.
  - ``datafiles/<task_id>.json`` — the scenario's full universe JSON, needed only to BOOT the world
    (``world_backend.py serve``). LARGE, gitignored, re-fetched here.

Scenarios are seeded-split ~70/30 into train/test; only train is captured (test never enters the
world model). In-scope = has a USER message, has >=1 oracle AGENT action, and no ENV/CONDITION
events (static — faithful without advancing the simulation clock).

Usage (from this directory, under the venv):
    ./.venv/bin/python backend/fetch_data.py
"""

from __future__ import annotations

import json
import random
import shutil
import sys
from pathlib import Path

from datasets import load_dataset

_ROOT = Path(__file__).resolve().parents[1]
_CONFIGS = ("execution", "search")
_SEED = 7
_TRAIN_FRAC = 0.7


def _args_map(action: dict) -> dict[str, object]:
    return {arg["name"]: arg.get("value") for arg in action.get("args", [])}


def _user_message(events: list[dict]) -> str:
    for event in events:
        if event.get("event_type") == "USER":
            args = _args_map(event.get("action") or {})
            message = args.get("content") or args.get("message")
            if message:
                return str(message).strip()
    return ""


def _oracle_actions(events: list[dict]) -> list[dict[str, object]]:
    oracle: list[dict[str, object]] = []
    for event in events:
        if event.get("event_type") != "AGENT":
            continue
        action = event.get("action") or {}
        oracle.append(
            {
                "app": action.get("app", ""),
                "function": action.get("function", ""),
                "args": _args_map(action),
            }
        )
    return oracle


def _in_scope(events: list[dict]) -> bool:
    if not _user_message(events) or not _oracle_actions(events):
        return False
    return not any(event.get("event_type") in ("ENV", "CONDITION") for event in events)


def main() -> None:
    for directory in ("data", "datafiles"):
        target = _ROOT / directory
        if target.exists():
            shutil.rmtree(target)
        target.mkdir()

    scenarios: list[tuple[str, str, dict]] = []  # (config, gaia2_id, raw_data)
    for config in _CONFIGS:
        dataset = load_dataset(
            "meta-agents-research-environments/gaia2", name=config, split="validation"
        )
        for row in dataset:
            raw = json.loads(row["data"])
            if _in_scope(raw.get("events", [])):
                scenarios.append((config, str(row["id"]), raw))

    scenarios.sort(key=lambda item: item[1])
    random.Random(_SEED).shuffle(scenarios)
    n_train = round(len(scenarios) * _TRAIN_FRAC)
    splits = {"train": scenarios[:n_train], "test": scenarios[n_train:]}

    for split, items in splits.items():
        rows: list[str] = []
        for index, (config, gaia2_id, raw) in enumerate(items):
            task_id = f"gaia2-{split}-{index}"
            (_ROOT / "datafiles" / f"{task_id}.json").write_text(json.dumps(raw), encoding="utf-8")
            rows.append(
                json.dumps(
                    {
                        "task_id": task_id,
                        "prompt": _user_message(raw["events"]),
                        "data": {
                            "config": config,
                            "gaia2_id": gaia2_id,
                            "oracle": _oracle_actions(raw["events"]),
                        },
                    }
                )
            )
        (_ROOT / "data" / f"{split}.jsonl").write_text("\n".join(rows) + "\n", encoding="utf-8")

    print(
        f"materialized {len(scenarios)} in-scope scenarios "
        f"({len(splits['train'])} train / {len(splits['test'])} test) from {len(_CONFIGS)} configs",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
