"""Audit Qwen chat rendering and Axolotl assistant-only loss masks."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from axolotl.processing_strategies import ProcessingStrategy
from transformers import AutoProcessor


ROLE_BOUNDARIES = [
    {
        "role": role,
        "start": f"<|im_start|>{role}\n",
        "end": "<|im_end|>",
        "include_start": False,
        "include_end": True,
    }
    for role in ("system", "user", "assistant")
]


def marker_roles(tokenizer: Any) -> dict[tuple[int, ...], str]:
    """Return token markers used to identify the active rendered role."""
    return {
        tuple(
            tokenizer.encode(
                f"<|im_start|>{role}\n", add_special_tokens=False
            )
        ): role
        for role in ("system", "user", "assistant")
    }


def role_at_each_token(
    input_ids: list[int], markers: dict[tuple[int, ...], str]
) -> list[str | None]:
    """Track the most recent role marker over the rendered token stream."""
    roles: list[str | None] = []
    current: str | None = None
    for index in range(len(input_ids)):
        for marker, role in markers.items():
            if tuple(input_ids[index : index + len(marker)]) == marker:
                current = role
                break
        roles.append(current)
    return roles


def main() -> None:
    """Render every sample, validate masks, and persist compact audit evidence."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--model", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--sequence-len", type=int, default=32768)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--dataset-index", type=int)
    args = parser.parse_args()

    processor = AutoProcessor.from_pretrained(
        args.model,
        revision=args.revision,
        local_files_only=True,
    )
    tokenizer = processor.tokenizer
    # Mirror the actual Axolotl config: tokenizer_default preserves the exact
    # pinned model template, while explicit boundaries enable assistant-only
    # loss in the generic multimodal processing strategy.
    strategy = ProcessingStrategy(
        processor,
        chat_template=tokenizer.chat_template,
        train_on_inputs=False,
        roles_to_train=["assistant"],
        train_on_eos="turn",
        role_boundaries_override=ROLE_BOUNDARIES,
    )
    markers = marker_roles(tokenizer)
    dataset_sha256 = hashlib.sha256(args.dataset.read_bytes()).hexdigest()
    rows: list[dict[str, Any]] = []
    with args.dataset.open(encoding="utf-8") as handle:
        for dataset_index, line in enumerate(handle):
            if (
                args.dataset_index is not None
                and dataset_index != args.dataset_index
            ):
                continue
            sample = json.loads(line)
            processed = strategy([sample])
            batch = processor.apply_chat_template(
                [processed[0]["messages"]],
                add_generation_prompt=False,
                tokenize=True,
                return_tensors="pt",
                return_dict=True,
                chat_template=strategy.chat_template,
                processor_kwargs={"padding": True},
            )
            input_ids = batch["input_ids"][0].tolist()
            labels = strategy.process_labels(batch["input_ids"])[0].tolist()
            if len(input_ids) != len(labels):
                raise ValueError(f"row {dataset_index}: input/label length mismatch")
            roles = role_at_each_token(input_ids, markers)
            supervised_indices = [
                index for index, label in enumerate(labels) if label != -100
            ]
            if not supervised_indices:
                raise ValueError(f"row {dataset_index}: no supervised tokens")
            wrong_role = [
                index
                for index in supervised_indices
                if roles[index] != "assistant"
            ]
            if wrong_role:
                preview = wrong_role[:10]
                context_start = max(0, preview[0] - 48)
                context_end = min(len(input_ids), preview[-1] + 49)
                context = tokenizer.decode(
                    input_ids[context_start:context_end], skip_special_tokens=False
                )
                raise ValueError(
                    f"row {dataset_index}: supervised tokens outside assistant spans "
                    f"{preview}; roles={[roles[index] for index in preview]}; "
                    f"decoded_context={context!r}"
                )
            if len(input_ids) > args.sequence_len:
                raise ValueError(
                    f"row {dataset_index}: {len(input_ids)} tokens exceeds {args.sequence_len}"
                )
            supervised_ids = [input_ids[index] for index in supervised_indices]
            supervised_text = tokenizer.decode(
                supervised_ids, skip_special_tokens=False
            )
            provenance = sample["provenance"]
            rows.append(
                {
                    "dataset_index": dataset_index,
                    "source_row_index": provenance["source_row_index"],
                    "task_id": provenance["task_id"],
                    "input_tokens": len(input_ids),
                    "supervised_tokens": len(supervised_indices),
                    "supervised_fraction": round(
                        len(supervised_indices) / len(input_ids), 6
                    ),
                    "supervised_text_head": supervised_text[:240],
                    "supervised_text_tail": supervised_text[-240:],
                }
            )

    summary = {
        "schema": "axolotl-qwen-assistant-mask-audit-v1",
        "dataset": str(args.dataset),
        "dataset_sha256": dataset_sha256,
        "model": args.model,
        "revision": args.revision,
        "chat_template_sha256": hashlib.sha256(
            tokenizer.chat_template.encode("utf-8")
        ).hexdigest(),
        "role_boundaries": ROLE_BOUNDARIES,
        "sequence_len": args.sequence_len,
        "rows": rows,
        "totals": {
            "rows": len(rows),
            "input_tokens": sum(row["input_tokens"] for row in rows),
            "supervised_tokens": sum(row["supervised_tokens"] for row in rows),
            "max_input_tokens": max(row["input_tokens"] for row in rows),
            "min_input_tokens": min(row["input_tokens"] for row in rows),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary["totals"], sort_keys=True))


if __name__ == "__main__":
    main()
