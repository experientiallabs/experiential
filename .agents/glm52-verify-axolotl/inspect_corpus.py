"""Summarize the GLM trajectory corpus without emitting trajectory contents."""

from __future__ import annotations

import argparse
import collections
import json
import logging
import statistics
from pathlib import Path

from pydantic import JsonValue


LOGGER = logging.getLogger(__name__)


def percentile(values: list[int], quantile: float) -> int:
    """Return the nearest-rank percentile for a nonempty integer list."""
    ordered = sorted(values)
    index = min(len(ordered) - 1, round((len(ordered) - 1) * quantile))
    return ordered[index]


def content_length(value: JsonValue) -> int:
    """Return a stable serialized character length for message content."""
    if isinstance(value, str):
        return len(value)
    return len(json.dumps(value, sort_keys=True, ensure_ascii=False))


def inspect(path: Path) -> None:
    """Stream a JSONL corpus and log structural and size statistics."""
    row_keys: collections.Counter[str] = collections.Counter()
    message_keys: collections.Counter[str] = collections.Counter()
    roles: collections.Counter[str] = collections.Counter()
    id_counts: collections.Counter[str] = collections.Counter()
    row_chars: list[int] = []
    message_counts: list[int] = []

    with path.open(encoding="utf-8") as corpus:
        for raw_line in corpus:
            row = json.loads(raw_line)
            row_keys.update(row.keys())
            messages = json.loads(row["message_log_json"])
            message_counts.append(len(messages))
            row_char_count = 0
            for message in messages:
                message_keys.update(message.keys())
                roles[str(message.get("role"))] += 1
                row_char_count += content_length(message.get("content", ""))
            row_chars.append(row_char_count)
            row_id = row.get("task_id") or row.get("id")
            id_counts[str(row_id)] += 1

    summary = {
        "rows": len(row_chars),
        "row_keys": dict(row_keys),
        "message_keys": dict(message_keys),
        "roles": dict(roles),
        "unique_ids": len(id_counts),
        "none_ids": id_counts.get("None", 0),
        "row_characters": {
            "p50": percentile(row_chars, 0.50),
            "p90": percentile(row_chars, 0.90),
            "p99": percentile(row_chars, 0.99),
            "max": max(row_chars),
            "total": sum(row_chars),
        },
        "message_count": {
            "median": statistics.median(message_counts),
            "p90": percentile(message_counts, 0.90),
            "p99": percentile(message_counts, 0.99),
            "max": max(message_counts),
        },
    }
    LOGGER.info("%s", json.dumps(summary, indent=2, sort_keys=True))


def main() -> None:
    """Parse arguments and inspect the requested corpus."""
    parser = argparse.ArgumentParser()
    parser.add_argument("corpus", type=Path)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    inspect(args.corpus)


if __name__ == "__main__":
    main()
