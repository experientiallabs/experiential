"""Build a provenance-rich Qwen chat SFT subset from independent judge passes."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def sha256_text(value: str) -> str:
    """Return the SHA-256 digest of UTF-8 text."""
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def load_judgments(path: Path) -> dict[int, dict[str, Any]]:
    """Load one judgment file keyed by source row index."""
    judgments: dict[int, dict[str, Any]] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            value = json.loads(line)
            row_index = value["row_index"]
            if row_index in judgments:
                raise ValueError(f"duplicate judgment for row {row_index} in {path}")
            judgments[row_index] = value
    return judgments


def admitted(value: dict[str, Any]) -> bool:
    """Return whether a normalized judgment strictly admits a row."""
    decision = value.get("decision", {})
    return bool(
        decision.get("verdict") == "PASS"
        and decision.get("confidence", 0) >= 90
        and decision.get("use_for_sft") is True
    )


def normalize_tool_calls(tool_calls: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Convert OpenAI JSON-string arguments into Qwen template mappings."""
    normalized: list[dict[str, Any]] = []
    for tool_call in tool_calls:
        function = tool_call.get("function")
        if not isinstance(function, dict) or not isinstance(function.get("name"), str):
            raise ValueError(f"invalid tool call: {tool_call!r}")
        arguments = function.get("arguments", {})
        if isinstance(arguments, str):
            arguments = json.loads(arguments)
        if not isinstance(arguments, dict):
            raise ValueError(f"tool arguments are not an object: {arguments!r}")
        normalized.append(
            {
                "type": "function",
                "function": {
                    "name": function["name"],
                    "arguments": arguments,
                },
            }
        )
    return normalized


def normalize_messages(raw_messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Keep only fields consumed by the pinned Qwen chat template."""
    messages: list[dict[str, Any]] = []
    for index, raw in enumerate(raw_messages):
        role = raw.get("role")
        if role not in {"system", "user", "assistant", "tool"}:
            raise ValueError(f"message {index} has unsupported role {role!r}")
        message: dict[str, Any] = {
            "role": role,
            "content": raw.get("content", ""),
        }
        if role == "assistant":
            reasoning = raw.get("reasoning_content")
            if isinstance(reasoning, str):
                message["reasoning_content"] = reasoning
            tool_calls = raw.get("tool_calls")
            if tool_calls:
                if not isinstance(tool_calls, list):
                    raise ValueError(f"message {index} tool_calls is not a list")
                message["tool_calls"] = normalize_tool_calls(tool_calls)
        messages.append(message)
    return messages


def build_subset(
    corpus_path: Path,
    primary_path: Path,
    adjudicator_path: Path,
) -> list[dict[str, Any]]:
    """Return rows admitted independently by both normalized judges."""
    primary = load_judgments(primary_path)
    adjudicator = load_judgments(adjudicator_path)
    selected = {
        row_index
        for row_index in primary.keys() & adjudicator.keys()
        if admitted(primary[row_index]) and admitted(adjudicator[row_index])
    }
    output: list[dict[str, Any]] = []
    with corpus_path.open(encoding="utf-8") as handle:
        for row_index, raw_line in enumerate(handle):
            if row_index not in selected:
                continue
            source = json.loads(raw_line)
            messages = normalize_messages(json.loads(source["message_log_json"]))
            tools = json.loads(source["tools_json"])
            if not isinstance(tools, list):
                raise ValueError(f"row {row_index} tools_json is not a list")
            output.append(
                {
                    "messages": messages,
                    "tools": tools,
                    "provenance": {
                        "source_row_index": row_index,
                        "source_row_sha256": sha256_text(raw_line.rstrip("\n")),
                        "rollout_id": source["rollout_id"],
                        "task_id": source["task_id"],
                        "manifest_order": source["manifest_order"],
                        "replica": source["replica"],
                        "source_student_tokens": source["n_student_tokens"],
                        "source_supervised_tokens": source["n_supervised_tokens"],
                        "primary_model": primary[row_index]["judge_model_id"],
                        "primary_prompt_sha256": primary[row_index]["prompt_sha256"],
                        "primary_confidence": primary[row_index]["decision"]["confidence"],
                        "adjudicator_model": adjudicator[row_index]["judge_model_id"],
                        "adjudicator_prompt_sha256": adjudicator[row_index]["prompt_sha256"],
                        "adjudicator_confidence": adjudicator[row_index]["decision"]["confidence"],
                    },
                }
            )
    missing = selected - {row["provenance"]["source_row_index"] for row in output}
    if missing:
        raise ValueError(f"selected source rows missing from corpus: {sorted(missing)}")
    return output


def main() -> None:
    """Write the independently admitted chat subset and a reproducibility summary."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", required=True, type=Path)
    parser.add_argument("--primary", required=True, type=Path)
    parser.add_argument("--adjudicator", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--summary", required=True, type=Path)
    args = parser.parse_args()

    rows = build_subset(args.corpus, args.primary, args.adjudicator)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    output_sha256 = hashlib.sha256(args.output.read_bytes()).hexdigest()
    summary = {
        "schema": "glm52-double-judge-qwen-chat-sft-summary-v1",
        "rows": len(rows),
        "unique_tasks": len({row["provenance"]["task_id"] for row in rows}),
        "source_row_indices": [row["provenance"]["source_row_index"] for row in rows],
        "output_sha256": output_sha256,
    }
    args.summary.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
