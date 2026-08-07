"""Render compact human-review packets for trajectory judge calibration."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path


LOGGER = logging.getLogger(__name__)


def clip(value: str, limit: int) -> str:
    """Clip long text while preserving both the beginning and ending evidence."""
    if len(value) <= limit:
        return value
    half = (limit - 80) // 2
    return f"{value[:half]}\n...[{len(value) - 2 * half} characters omitted]...\n{value[-half:]}"


def message_text(message: dict[str, object]) -> str:
    """Return the best visible representation of one transcript message."""
    pieces: list[str] = []
    for key in ("reasoning_content", "visible_content", "content", "tool_calls"):
        value = message.get(key)
        if value in (None, "", []):
            continue
        rendered = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)
        pieces.append(f"{key}: {rendered}")
    return "\n".join(pieces)


def load_results(path: Path) -> dict[int, dict[str, object]]:
    """Load judge results keyed by source row index."""
    results: dict[int, dict[str, object]] = {}
    with path.open(encoding="utf-8") as source:
        for line in source:
            value = json.loads(line)
            results[int(value["row_index"])] = value
    return results


def render(corpus_path: Path, results_path: Path, output_path: Path, message_limit: int) -> None:
    """Join source trajectories to judge outputs and write compact Markdown."""
    results = load_results(results_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with corpus_path.open(encoding="utf-8") as corpus, output_path.open(
        "w", encoding="utf-8"
    ) as output:
        for row_index, line in enumerate(corpus):
            result = results.get(row_index)
            if result is None:
                continue
            row = json.loads(line)
            messages = json.loads(row["message_log_json"])
            user_messages = [message for message in messages if message.get("role") == "user"]
            tail_messages = messages[-6:]
            output.write(f"# Row {row_index}: {row['rollout_id']}\n\n")
            output.write("## Judge decision\n\n")
            output.write("```json\n")
            output.write(json.dumps(result.get("decision", result), indent=2, sort_keys=True))
            output.write("\n```\n\n")
            output.write("## Task\n\n")
            if user_messages:
                output.write(clip(message_text(user_messages[0]), message_limit))
            else:
                output.write("No user message found.")
            output.write("\n\n## Final six messages\n\n")
            for message in tail_messages:
                output.write(f"### {message.get('role', 'unknown')}\n\n")
                output.write(clip(message_text(message), message_limit))
                output.write("\n\n")
    LOGGER.info("rendered %d review packets to %s", len(results), output_path)


def main() -> None:
    """Parse command-line arguments and render calibration packets."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--message-limit", type=int, default=1800)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    render(args.corpus, args.results, args.output, args.message_limit)


if __name__ == "__main__":
    main()
