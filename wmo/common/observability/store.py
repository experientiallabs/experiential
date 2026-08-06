"""Persist + list run records under `.wmo/runs/`.

One JSON file per run (`<run_id>.json`). Kept tiny and dependency-free so `wmo build`/`serve` can
write a record without pulling in the rest of the harness.
"""

from __future__ import annotations

from pathlib import Path

from wmo.common.observability.tracker import RunRecord


def save_run(record: RunRecord, runs_dir: str | Path) -> Path:
    """Write `record` to `<runs_dir>/<run_id>.json`, creating the directory if needed."""
    path = Path(runs_dir)
    path.mkdir(parents=True, exist_ok=True)
    out = path / f"{record.run_id}.json"
    out.write_text(record.model_dump_json(indent=2), encoding="utf-8")
    return out


def load_runs(runs_dir: str | Path) -> list[RunRecord]:
    """Load all run records from `runs_dir` (empty list if the directory doesn't exist)."""
    path = Path(runs_dir)
    if not path.exists():
        return []
    return [
        RunRecord.model_validate_json(p.read_text(encoding="utf-8"))
        for p in sorted(path.glob("*.json"))
    ]
