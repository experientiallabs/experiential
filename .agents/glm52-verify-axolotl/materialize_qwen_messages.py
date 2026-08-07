"""Materialize tool metadata and assistant traces for Axolotl's Qwen3.5 MM path."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from transformers import AutoTokenizer


SYSTEM_PREFIX = "<|im_start|>system\n"
SYSTEM_END = "<|im_end|>\n"
ASSISTANT_PREFIX = "<|im_start|>assistant\n"
TURN_END = "<|im_end|>\n"
SENTINEL = "__AXOLOTL_QWEN_MATERIALIZATION_SENTINEL__"


def extract_system_content(
    tokenizer: Any, system_content: str, tools: list[dict[str, Any]]
) -> str:
    """Render and extract the exact system body including tool definitions."""
    rendered = tokenizer.apply_chat_template(
        [
            {"role": "system", "content": system_content},
            {"role": "user", "content": SENTINEL},
        ],
        tools=tools,
        tokenize=False,
        add_generation_prompt=False,
    )
    user_block = f"<|im_start|>user\n{SENTINEL}<|im_end|>\n"
    if not rendered.startswith(SYSTEM_PREFIX) or not rendered.endswith(user_block):
        raise ValueError("unexpected pinned-template system rendering")
    system_block = rendered[: -len(user_block)]
    if not system_block.endswith(SYSTEM_END):
        raise ValueError("rendered system block has no ChatML terminator")
    return system_block[len(SYSTEM_PREFIX) : -len(SYSTEM_END)]


def extract_assistant_content(tokenizer: Any, message: dict[str, Any]) -> str:
    """Render and extract one exact assistant body with reasoning and tool calls."""
    rendered = tokenizer.apply_chat_template(
        [
            {"role": "user", "content": SENTINEL},
            message,
        ],
        tokenize=False,
        add_generation_prompt=False,
    )
    start = rendered.rfind(ASSISTANT_PREFIX)
    if start < 0 or not rendered.endswith(TURN_END):
        raise ValueError("unexpected pinned-template assistant rendering")
    return rendered[start + len(ASSISTANT_PREFIX) : -len(TURN_END)]


def materialize_sample(tokenizer: Any, sample: dict[str, Any]) -> dict[str, Any]:
    """Return a self-contained chat row that survives MM message normalization."""
    messages = sample["messages"]
    tools = sample.get("tools", [])
    output_messages: list[dict[str, str]] = []
    for index, message in enumerate(messages):
        role = message["role"]
        if role == "system":
            if index != 0:
                raise ValueError("system message is not first")
            content = extract_system_content(tokenizer, message["content"], tools)
        elif role == "assistant":
            content = extract_assistant_content(tokenizer, message)
        else:
            content = message.get("content", "")
        if not isinstance(content, str):
            raise ValueError(f"message {index} content is not text")
        output_messages.append({"role": role, "content": content})

    materialized = {
        "messages": output_messages,
        "provenance": dict(sample["provenance"]),
    }
    original_rendered = tokenizer.apply_chat_template(
        messages,
        tools=tools,
        tokenize=False,
        add_generation_prompt=False,
    )
    materialized_rendered = tokenizer.apply_chat_template(
        output_messages,
        tokenize=False,
        add_generation_prompt=False,
    )
    if original_rendered != materialized_rendered:
        raise ValueError("materialized messages changed the pinned Qwen rendering")
    materialized["provenance"]["qwen_template_sha256"] = hashlib.sha256(
        tokenizer.chat_template.encode("utf-8")
    ).hexdigest()
    materialized["provenance"]["rendered_chat_sha256"] = hashlib.sha256(
        materialized_rendered.encode("utf-8")
    ).hexdigest()
    materialized["provenance"]["materialization_schema"] = (
        "qwen35-axolotl-mm-self-contained-v1"
    )
    return materialized


def main() -> None:
    """Materialize every admitted row under the pinned tokenizer revision."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--summary", required=True, type=Path)
    parser.add_argument("--model", required=True)
    parser.add_argument("--revision", required=True)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        revision=args.revision,
        local_files_only=True,
    )
    rows: list[dict[str, Any]] = []
    with args.input.open(encoding="utf-8") as handle:
        for line in handle:
            rows.append(materialize_sample(tokenizer, json.loads(line)))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")

    summary = {
        "schema": "qwen35-axolotl-mm-materialization-summary-v1",
        "rows": len(rows),
        "input_sha256": hashlib.sha256(args.input.read_bytes()).hexdigest(),
        "output_sha256": hashlib.sha256(args.output.read_bytes()).hexdigest(),
        "model": args.model,
        "revision": args.revision,
        "chat_template_sha256": hashlib.sha256(
            tokenizer.chat_template.encode("utf-8")
        ).hexdigest(),
    }
    args.summary.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
