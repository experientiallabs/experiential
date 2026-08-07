"""Create a deterministic length-stratified trajectory calibration manifest."""

from __future__ import annotations

import argparse
import json
import logging
import random
from dataclasses import dataclass
from pathlib import Path


LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class Candidate:
    """Metadata used to stratify one corpus row."""

    row_index: int
    rollout_id: str
    task_id: str
    manifest_order: int
    replica: int
    transcript_characters: int
    message_count: int
    n_student_tokens: int
    n_supervised_tokens: int
    student_truncated_at_train_axis: bool


def load_candidates(path: Path) -> list[Candidate]:
    """Stream corpus metadata into compact calibration candidates."""
    candidates: list[Candidate] = []
    with path.open(encoding="utf-8") as corpus:
        for row_index, raw_line in enumerate(corpus):
            row = json.loads(raw_line)
            message_log_json = str(row["message_log_json"])
            candidates.append(
                Candidate(
                    row_index=row_index,
                    rollout_id=str(row["rollout_id"]),
                    task_id=str(row["task_id"]),
                    manifest_order=int(row["manifest_order"]),
                    replica=int(row["replica"]),
                    transcript_characters=len(message_log_json),
                    message_count=len(json.loads(message_log_json)),
                    n_student_tokens=int(row["n_student_tokens"]),
                    n_supervised_tokens=int(row["n_supervised_tokens"]),
                    student_truncated_at_train_axis=bool(
                        row["student_truncated_at_train_axis"]
                    ),
                )
            )
    return candidates


def select(candidates: list[Candidate], count: int, seed: int) -> list[Candidate]:
    """Select unique-task examples evenly across transcript-length ranks."""
    if count <= 0 or count > len(candidates):
        raise ValueError("count must be between 1 and the corpus size")
    ordered = sorted(candidates, key=lambda candidate: candidate.transcript_characters)
    randomizer = random.Random(seed)
    selected: list[Candidate] = []
    used_tasks: set[str] = set()
    bin_count = min(8, count)
    base_per_bin, extra = divmod(count, bin_count)
    for bin_index in range(bin_count):
        start = len(ordered) * bin_index // bin_count
        end = len(ordered) * (bin_index + 1) // bin_count
        choices = ordered[start:end]
        randomizer.shuffle(choices)
        target = base_per_bin + (1 if bin_index < extra else 0)
        selected_in_bin = 0
        for candidate in choices:
            if candidate.task_id in used_tasks:
                continue
            selected.append(candidate)
            used_tasks.add(candidate.task_id)
            selected_in_bin += 1
            if selected_in_bin >= target:
                break
    if len(selected) != count:
        raise RuntimeError(f"selected {len(selected)} rows, expected {count}")
    return sorted(selected, key=lambda candidate: candidate.row_index)


def main() -> None:
    """Parse arguments and write an append-independent JSONL manifest."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--count", type=int, default=32)
    parser.add_argument("--seed", type=int, default=20260807)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    candidates = load_candidates(args.corpus)
    selected = select(candidates, args.count, args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as output:
        for candidate in selected:
            output.write(json.dumps(candidate.__dict__, sort_keys=True) + "\n")
    LOGGER.info(
        "wrote %d rows from %d candidates to %s",
        len(selected),
        len(candidates),
        args.output,
    )


if __name__ == "__main__":
    main()
