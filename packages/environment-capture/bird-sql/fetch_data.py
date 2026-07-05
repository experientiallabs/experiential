"""Materialize the real BIRD mini-dev dataset into this benchmark's on-disk shape.

BIRD mini-dev ships as a single zip (databases + questions) on the project's Google Drive; there
is no direct HTTP endpoint for the SQLite databases, so this script does NOT download — it
converts an ALREADY-UNZIPPED MINIDEV directory. Fetch it once (see README) and point
``--minidev-root`` at the unzipped ``.../minidev/MINIDEV`` dir (which holds ``mini_dev_sqlite.json``
and ``dev_databases/<db_id>/<db_id>.sqlite``).

Two modes:

- **Base** (default): materialize the databases and (re)write ``data/{train,test}.jsonl`` from
  scratch. For each selected database the records are seeded-shuffled, capped at ``--per-db``, then
  split disjointly into test/train (``--test-frac`` to test, the rest to train) so both splits draw
  from every database. This reproduces the original 52-train / 20-test corpus with its defaults.

- **Expand** (``--expand``): GROW the train split from more of the real upstream pool without
  touching the committed test split. It materializes every ``--databases`` db, then reads the
  existing committed splits and APPENDS only questions whose upstream ``question_id`` is not already
  in train or test, with fresh sequential ids continuing past the last train index (gold sidecars
  written alongside). The test split is never rewritten and no question is ever duplicated
  (``environment_capture.plan_appended_tasks`` enforces both). The same seed keeps the first
  ``--per-db`` questions per database identical to the base split, so raising the cap only ever adds
  new tail questions.

Written under this directory:
  - ``data/{train,test}.jsonl`` — agent-visible tasks (question + folded-in evidence hint).
  - ``gold/<task_id>.json`` — ``{"gold_sql": ...}`` sidecars (NEVER staged into the workspace).
  - ``schemas/<db>.sql`` — DDL only, staged as ``schema.sql`` for the agent to read.
  - ``databases/<db>.sqlite`` — a copy of the real db (gitignored; re-materialize with this script).

Usage (from the repo root):
    # base: reproduce the original split
    uv run python packages/environment-capture/bird-sql/fetch_data.py \
        --minidev-root /path/to/minidev/MINIDEV
    # expand: append ~150 new train tasks across all 11 mini-dev databases
    uv run python packages/environment-capture/bird-sql/fetch_data.py \
        --minidev-root /path/to/minidev/MINIDEV --expand
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
import sqlite3
from pathlib import Path

from environment_capture import CandidateTask, plan_appended_tasks
from environment_capture.trajectory import JsonValue

_HERE = Path(__file__).parent
# The four databases the base split was built from (schema variety, manageable size).
_BASE_DATABASES = ("superhero", "toxicology", "student_club", "california_schools")
# Every BIRD mini-dev database, used when expanding for maximum schema diversity.
_ALL_DATABASES = (
    *_BASE_DATABASES,
    "formula_1",
    "card_games",
    "european_football_2",
    "thrombosis_prediction",
    "codebase_community",
    "financial",
    "debit_card_specializing",
)
_TRAIN_ID_PREFIX = "bird-train-"


def _ddl(sqlite_path: Path) -> str:
    """The database's schema (DDL only) — tables, indexes, views, triggers; no data."""
    con = sqlite3.connect(f"file:{sqlite_path}?mode=ro", uri=True)
    try:
        rows = con.execute(
            "SELECT sql FROM sqlite_master "
            "WHERE sql IS NOT NULL AND name NOT LIKE 'sqlite_%' ORDER BY rowid"
        ).fetchall()
    finally:
        con.close()
    return "\n".join(f"{sql};" for (sql,) in rows) + "\n"


def _prompt(record: dict[str, str]) -> str:
    """The question the agent sees, with BIRD's evidence folded in as a hint."""
    prompt = record["question"].strip()
    evidence = record.get("evidence", "").strip()
    if evidence:
        prompt += f"\n\nHint: {evidence}"
    return prompt


def _capped_by_db(
    records: list[dict[str, str]],
    databases: tuple[str, ...],
    *,
    per_db: int,
    seed: int,
) -> list[dict[str, str]]:
    """The seeded, per-database capped selection (same shuffle the base split uses)."""
    rng = random.Random(seed)
    selected: list[dict[str, str]] = []
    for db_id in databases:
        db_records = [r for r in records if r["db_id"] == db_id]
        rng.shuffle(db_records)
        selected.extend(db_records[:per_db])
    return selected


def _split_records(
    records: list[dict[str, str]],
    databases: tuple[str, ...],
    *,
    per_db: int,
    test_frac: float,
    seed: int,
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    """Seeded, per-database disjoint test/train partition (every db appears in both)."""
    rng = random.Random(seed)
    test: list[dict[str, str]] = []
    train: list[dict[str, str]] = []
    for db_id in databases:
        db_records = [r for r in records if r["db_id"] == db_id]
        rng.shuffle(db_records)
        db_records = db_records[:per_db]
        n_test = round(len(db_records) * test_frac)
        test.extend(db_records[:n_test])
        train.extend(db_records[n_test:])
    return train, test


def _record_task_data(record: dict[str, str]) -> dict[str, JsonValue]:
    return {"db_name": record["db_id"], "question_id": record["question_id"]}


def _write_split(
    split: str,
    records: list[dict[str, str]],
    *,
    data_dir: Path,
    gold_dir: Path,
) -> None:
    rows: list[str] = []
    for i, record in enumerate(records):
        task_id = f"bird-{split}-{i}"
        rows.append(
            json.dumps(
                {"task_id": task_id, "prompt": _prompt(record), "data": _record_task_data(record)}
            )
        )
        (gold_dir / f"{task_id}.json").write_text(
            json.dumps({"gold_sql": record["SQL"]}) + "\n", encoding="utf-8"
        )
    (data_dir / f"{split}.jsonl").write_text("\n".join(rows) + "\n", encoding="utf-8")


def _materialize_databases(root: Path, databases: tuple[str, ...]) -> None:
    """Copy each database's real .sqlite and dump its DDL schema into this directory."""
    for db_id in databases:
        source_db = root / "dev_databases" / db_id / f"{db_id}.sqlite"
        shutil.copy(source_db, _HERE / "databases" / f"{db_id}.sqlite")
        (_HERE / "schemas" / f"{db_id}.sql").write_text(_ddl(source_db), encoding="utf-8")


def _used_question_ids(data_dir: Path) -> set[str]:
    """Every upstream question_id already present in the committed train or test split."""
    used: set[str] = set()
    for split in ("train", "test"):
        path = data_dir / f"{split}.jsonl"
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                used.add(str(json.loads(line)["data"]["question_id"]))
    return used


def _next_train_index(data_dir: Path) -> int:
    """One past the highest ``bird-train-<n>`` index in the committed train split."""
    indices = [
        int(json.loads(line)["task_id"].removeprefix(_TRAIN_ID_PREFIX))
        for line in (data_dir / "train.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    return max(indices) + 1 if indices else 0


def _expand(
    records: list[dict[str, str]],
    databases: tuple[str, ...],
    *,
    per_db: int,
    seed: int,
    data_dir: Path,
    gold_dir: Path,
) -> int:
    """Append new train tasks drawn from the upstream pool; leave the test split untouched."""
    candidates = [
        CandidateTask(
            upstream_id=str(record["question_id"]),
            prompt=_prompt(record),
            data=_record_task_data(record),
            gold={"gold_sql": record["SQL"]},
        )
        for record in _capped_by_db(records, databases, per_db=per_db, seed=seed)
    ]
    planned = plan_appended_tasks(
        candidates=candidates,
        used_upstream_ids=_used_question_ids(data_dir),
        id_prefix=_TRAIN_ID_PREFIX,
        next_index=_next_train_index(data_dir),
    )
    new_rows = [
        json.dumps({"task_id": task.task_id, "prompt": task.prompt, "data": task.data})
        for task in planned
    ]
    for task in planned:
        (gold_dir / f"{task.task_id}.json").write_text(
            json.dumps(task.gold) + "\n", encoding="utf-8"
        )
    train_path = data_dir / "train.jsonl"
    with train_path.open("a", encoding="utf-8") as handle:
        for row in new_rows:
            handle.write(row + "\n")
    return len(planned)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--minidev-root",
        required=True,
        help="unzipped MINIDEV dir (holds mini_dev_sqlite.json + dev_databases/)",
    )
    parser.add_argument(
        "--expand",
        action="store_true",
        help="append new train tasks from the upstream pool instead of rewriting the splits",
    )
    parser.add_argument(
        "--databases",
        default=None,
        help="comma-separated db_ids (default: 4 base dbs; all 11 when --expand)",
    )
    parser.add_argument(
        "--per-db",
        type=int,
        default=None,
        help="max questions per database (default: 18; 22 when --expand)",
    )
    parser.add_argument("--test-frac", type=float, default=0.3, help="fraction held out as test")
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    root = Path(args.minidev_root)
    default_dbs = _ALL_DATABASES if args.expand else _BASE_DATABASES
    databases = (
        tuple(d.strip() for d in args.databases.split(",") if d.strip())
        if args.databases
        else default_dbs
    )
    per_db = args.per_db if args.per_db is not None else (22 if args.expand else 18)
    records = json.loads((root / "mini_dev_sqlite.json").read_text(encoding="utf-8"))

    for sub in ("data", "gold", "schemas", "databases"):
        (_HERE / sub).mkdir(exist_ok=True)
    _materialize_databases(root, databases)

    if args.expand:
        added = _expand(
            records,
            databases,
            per_db=per_db,
            seed=args.seed,
            data_dir=_HERE / "data",
            gold_dir=_HERE / "gold",
        )
        print(
            f"expanded: materialized {len(databases)} databases, appended {added} new train tasks "
            f"(test split unchanged) -> {_HERE}"
        )
        return

    train, test = _split_records(
        records, databases, per_db=per_db, test_frac=args.test_frac, seed=args.seed
    )
    _write_split("train", train, data_dir=_HERE / "data", gold_dir=_HERE / "gold")
    _write_split("test", test, data_dir=_HERE / "data", gold_dir=_HERE / "gold")
    print(
        f"materialized {len(databases)} databases, {len(train)} train / {len(test)} test tasks "
        f"-> {_HERE}"
    )


if __name__ == "__main__":
    main()
