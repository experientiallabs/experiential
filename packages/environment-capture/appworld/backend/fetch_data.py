"""Materialize the real AppWorld dataset into this benchmark's on-disk shape (run under the venv).

AppWorld ships its apps + task data as encrypted ``.bundle`` files inside the ``appworld`` pip
package. This script unpacks and downloads them into the benchmark-local directory, then writes the
agent-visible task index the gate-checked adapter reads:

  - ``appworld install`` unpacks the app/engine source into the venv.
  - ``appworld download data`` fetches ``data/`` (apps' base databases, per-task specs + gold).
  - ``data/train.jsonl`` — one row per train task:
    ``{task_id: "aw-train-<i>", prompt: <instruction>, data: {appworld_id: <upstream id>}}``. The
    instruction comes straight from the task's ``specs.json`` (no world boot needed); the upstream
    id is what the adapter passes to the backend.

Only the ``train`` split is materialized: AppWorld's hidden test splits (``test_normal`` /
``test_challenge``) are never captured, so the world model can't absorb their dynamics. NOTHING here
is committed — the downloaded ``data/`` and this ``data/train.jsonl`` are AppWorld's protected data
(see the README's license note) and stay gitignored.

Usage (from this directory, under the venv):
    ./.venv/bin/python backend/fetch_data.py
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from appworld import load_task_ids

_ROOT = Path(__file__).resolve().parents[1]


def _ensure_data() -> None:
    """Unpack the engine and download AppWorld's data into this directory if not already present."""
    if not (_ROOT / "data" / "tasks").is_dir():
        subprocess.run([sys.executable, "-m", "appworld.cli", "install"], check=True)
        subprocess.run(
            [sys.executable, "-m", "appworld.cli", "download", "data", "--root", str(_ROOT)],
            check=True,
        )


def _instruction(appworld_id: str) -> str:
    specs = json.loads(
        (_ROOT / "data" / "tasks" / appworld_id / "specs.json").read_text(encoding="utf-8")
    )
    return str(specs["instruction"]).strip()


def main() -> None:
    _ensure_data()
    rows: list[str] = []
    for index, appworld_id in enumerate(load_task_ids("train")):
        rows.append(
            json.dumps(
                {
                    "task_id": f"aw-train-{index}",
                    "prompt": _instruction(appworld_id),
                    "data": {"appworld_id": appworld_id},
                }
            )
        )
    out = _ROOT / "data" / "train.jsonl"
    out.write_text("\n".join(rows) + "\n", encoding="utf-8")
    print(f"materialized {len(rows)} train tasks -> {out}", file=sys.stderr)


if __name__ == "__main__":
    main()
