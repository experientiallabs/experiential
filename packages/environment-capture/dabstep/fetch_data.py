"""Fetch DABstep context files and grow the train split from the real upstream task pool.

Two jobs:

- **Context files** (default): the small context files (manual.md, fees.json, the reference CSVs)
  are committed under ``datafiles/``; the ~23 MB ``payments.csv`` is gitignored and downloaded here
  so a fresh clone is runnable. Stdlib-only for this path.

- **Expand** (``--expand``): DABstep publishes gold answers for only its 10-task ``dev`` split; the
  450-task pool (``data/tasks/all.jsonl``) ships with empty answers (the official server scores
  hidden submissions). This mode recovers gold from the dataset's own published leaderboard —
  ``data/task_scores/*.jsonl`` records, per submission, which ``agent_answer`` the official grader
  scored correct — then APPENDS new train tasks (question + guidelines + recovered gold sidecar)
  for pool questions not already in the committed splits. The test split is never rewritten and no
  question is duplicated (``environment_capture.plan_appended_tasks`` enforces both). See
  ``leaderboard_gold.py`` for the recovery and the confidence-based answerability filter. Needs the
  ``fetch`` extra (``huggingface_hub``).

Usage (from the repo root):
    uv run python packages/environment-capture/dabstep/fetch_data.py            # payments.csv only
    uv run python packages/environment-capture/dabstep/fetch_data.py --all      # every context file
    uv run python packages/environment-capture/dabstep/fetch_data.py --expand  # +~130 train tasks
"""

from __future__ import annotations

import argparse
import json
import random
import urllib.request
from pathlib import Path

from environment_capture import CandidateTask, plan_appended_tasks
from environment_capture.trajectory import JsonValue
from huggingface_hub import snapshot_download
from leaderboard_gold import canonical_gold, verified_answers

_HERE = Path(__file__).parent
_BASE_URL = "https://huggingface.co/datasets/adyen/DABstep/resolve/main/data/context"
_DATASET = "adyen/DABstep"
_TRAIN_ID_PREFIX = "dab-train-"

# The full context set upstream; only payments.csv is gitignored, the rest are committed.
_ALL_FILES = (
    "payments.csv",
    "acquirer_countries.csv",
    "fees.json",
    "manual.md",
    "merchant_category_codes.csv",
    "merchant_data.json",
    "payments-readme.md",
)
_LARGE_FILES = ("payments.csv",)
# Every DABstep task is answered from the same shared context, so new tasks stage the full set.
_CONTEXT_FILE_IDS: list[JsonValue] = [
    "acquirer_countries.csv",
    "fees.json",
    "manual.md",
    "merchant_category_codes.csv",
    "merchant_data.json",
    "payments-readme.md",
    "payments.csv",
]


def _download(file_id: str, dest: Path) -> None:
    url = f"{_BASE_URL}/{file_id}"
    print(f"fetching {url} -> {dest}")
    urllib.request.urlretrieve(url, dest)  # noqa: S310 - fixed https HuggingFace host
    print(f"  wrote {dest.stat().st_size / 1e6:.1f} MB")


def _normalize_question(question: str) -> str:
    """The dedup key for a task: its question text, whitespace-collapsed."""
    return " ".join(question.split())


def _used_questions(data_dir: Path) -> set[str]:
    """Normalized question text of every task already in the committed train or test split."""
    used: set[str] = set()
    for split in ("train", "test"):
        for line in (data_dir / f"{split}.jsonl").read_text(encoding="utf-8").splitlines():
            if line.strip():
                # The committed prompt is "<question>\n\n<guidelines>"; the question is the head.
                prompt = str(json.loads(line)["prompt"])
                used.add(_normalize_question(prompt.split("\n\n", 1)[0]))
    return used


def _next_train_index(data_dir: Path) -> int:
    indices = [
        int(json.loads(line)["task_id"].removeprefix(_TRAIN_ID_PREFIX))
        for line in (data_dir / "train.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    return max(indices) + 1 if indices else 0


def _candidates(scores_dir: Path, all_tasks_path: Path) -> list[CandidateTask]:
    """Pool tasks with a confidently-recovered gold, as append candidates (question is the key)."""
    answers = verified_answers(scores_dir)
    candidates: list[CandidateTask] = []
    for line in all_tasks_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        task = json.loads(line)
        gold = canonical_gold(answers.get(str(task["task_id"]), []))
        if gold is None:
            continue
        gold = {**gold, "upstream_task_id": str(task["task_id"])}
        prompt = f"{task['question'].strip()}\n\n{task['guidelines'].strip()}"
        candidates.append(
            CandidateTask(
                upstream_id=_normalize_question(task["question"]),
                prompt=prompt,
                data={"file_ids": list(_CONTEXT_FILE_IDS), "level": task["level"]},
                gold=gold,
            )
        )
    return candidates


def _expand(scores_dir: Path, all_tasks_path: Path, *, target: int, seed: int) -> int:
    """Append up to `target` new train tasks with recovered gold; leave the test split untouched."""
    data_dir = _HERE / "data"
    gold_dir = _HERE / "gold"
    candidates = _candidates(scores_dir, all_tasks_path)
    random.Random(seed).shuffle(candidates)
    planned = plan_appended_tasks(
        candidates=candidates[:target],
        used_upstream_ids=_used_questions(data_dir),
        id_prefix=_TRAIN_ID_PREFIX,
        next_index=_next_train_index(data_dir),
    )
    new_rows: list[str] = []
    for task in planned:
        data = dict(task.data)
        level = data.pop("level")
        new_rows.append(
            json.dumps(
                {"task_id": task.task_id, "prompt": task.prompt, "data": data, "level": level}
            )
        )
        (gold_dir / f"{task.task_id}.json").write_text(json.dumps(task.gold), encoding="utf-8")
    with (data_dir / "train.jsonl").open("a", encoding="utf-8") as handle:
        for row in new_rows:
            handle.write(row + "\n")
    return len(planned)


def _run_expand(args: argparse.Namespace) -> None:
    snapshot = Path(
        snapshot_download(
            _DATASET,
            repo_type="dataset",
            allow_patterns=["data/task_scores/*", "data/tasks/all.jsonl"],
        )
    )
    added = _expand(
        snapshot / "data" / "task_scores",
        snapshot / "data" / "tasks" / "all.jsonl",
        target=args.target,
        seed=args.seed,
    )
    print(f"expanded: appended {added} new train tasks (test split unchanged) -> {_HERE}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--all",
        action="store_true",
        help="Fetch every context file (default: only the gitignored large payments.csv)",
    )
    parser.add_argument(
        "--expand",
        action="store_true",
        help="Append new train tasks recovered from the leaderboard instead of fetching context",
    )
    parser.add_argument("--target", type=int, default=130, help="max new train tasks to append")
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    if args.expand:
        _run_expand(args)
        return

    datafiles = _HERE / "datafiles"
    datafiles.mkdir(exist_ok=True)
    wanted = _ALL_FILES if args.all else _LARGE_FILES
    for file_id in wanted:
        _download(file_id, datafiles / file_id)
    print(f"done: {len(wanted)} file(s) in {datafiles}")


if __name__ == "__main__":
    main()
