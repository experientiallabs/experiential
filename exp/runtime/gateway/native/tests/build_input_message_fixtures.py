"""Freeze the canonical message schema and Python replacement-message validation."""

from __future__ import annotations

import json
from pathlib import Path

from exp.runtime.gateway.contracts import GatewayMessage
from exp.runtime.gateway.guardrails.subjects import _subject_value


def schema() -> dict[str, object]:
    """Remove documentation while preserving validation and defaults exactly."""
    result = GatewayMessage.model_json_schema()

    def clean(value: object) -> None:
        if isinstance(value, dict):
            value.pop("description", None)
            value.pop("title", None)
            for item in value.values():
                clean(item)
        elif isinstance(value, list):
            for item in value:
                clean(item)

    clean(result)
    result["$defs"]["ToolCall"]["properties"]["arguments"]["default"] = {}
    return result


def cases() -> list[dict[str, object]]:
    """Exercise role coherence, defaulting, scalar conversion, and hidden field validation."""
    seeds = [
        {"role": "user", "content": "text"},
        {"role": "assistant", "tool_calls": [{"call_id": "c", "name": "fn"}]},
        {"role": "tool", "tool_call_id": "c", "content": "result"},
        {"role": "assistant", "provider_reasoning": [{"kind": "thinking", "text": "reason"}]},
        {"role": "assistant", "provider_native_item": {"type": "item"}},
    ]
    patches = [
        {},
        {"content": None},
        {"role": "tool"},
        {"role": "system"},
        {"role": "assistant"},
        {"role": "developer"},
        {"unknown": 1},
        {"tool_is_error": True},
        {"tool_is_error": "false"},
        {"tool_is_error": 1},
        {"tool_is_error": "yes"},
        {"tool_call_id": ""},
        {"tool_call_id": 1},
        {"content": 5},
        {"provider_item_id": "i", "provider_output_index": "3"},
        {"provider_item_id": "i", "provider_output_index": 3.0},
        {"provider_output_index": -1},
        {"provider_status": "completed"},
        {"provider_phase": "final_answer"},
        {"provider_tool_name": "fn"},
        {"provider_tool_namespace": "ns"},
        {"provider_tool_caller": {}},
        {"cache_control": {}},
        {"tool_references": ["fn"]},
        {"provider_text_blocks": [{"text": "text"}]},
        {"provider_anthropic_block": {}},
        {"provider_anthropic_blocks": []},
        {"capture_only_reasoning": [{"kind": "exposed_reasoning_content", "content": ""}]},
    ]
    raw = [{**seed, **patch} for seed in seeds for patch in patches]
    for number in [
        "1_2",
        "1__2",
        "_12",
        "12_",
        "1e2",
        "1.00",
        "1.0_0",
        " 1 ",
        "+1",
        "-0.0",
        "١٢",
        "１２",
        "1.",
        "1.0000",
        "1_000.00",
        "+1.0",
        "-0",
        "0_0",
        "00012",
        "+ 1",
        "1e0",
        "1.00e0",
        "0x1",
        "\x1c1\x1d",
    ]:
        raw.append(
            {
                "role": "assistant",
                "content": "",
                "provider_item_id": "item",
                "provider_output_index": number,
            }
        )
    for text in [
        None,
        True,
        False,
        {"v": "a'b", "nested": [None, 1.0]},
        ["\x00\n\t", '"', "é\u200b\u00a0"],
        1e-9,
    ]:
        for content in [str(text), "", json.dumps(text)]:
            raw.append(
                {"role": "user", "content": content, "provider_text_blocks": [{"text": text}]}
            )
    for arguments in ["{}", '{"v":1}', "[]", '{"v":1.0}', "bad"]:
        for parsed in [{}, {"v": 1}]:
            raw.append(
                {
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "call_id": "c",
                            "name": "f",
                            "arguments": parsed,
                            "raw_arguments": arguments,
                        }
                    ],
                }
            )
    for kind in ["image", "video", "document", "audio"]:
        for data in ["", "YQ==", "YR==", "AAAA====", "a", "====", "YQ==\n", "YQ==="]:
            raw.append(
                {
                    "role": "user",
                    "content": "",
                    "content_parts": [
                        {
                            "kind": kind,
                            "data": data,
                            "media_type": {
                                "image": "image/png",
                                "video": "video/mp4",
                                "document": "application/pdf",
                                "audio": "audio/wav",
                            }[kind],
                        }
                    ],
                }
            )
    result: list[dict[str, object]] = []
    for value in raw:
        try:
            expected = _subject_value(GatewayMessage.model_validate(value))
        except ValueError:
            expected = None
        result.append({"input": value, "expected": expected})
    return result


if __name__ == "__main__":
    directory = Path(__file__).parent
    (directory.parent / "src/generated/gateway_message_schema.json").write_text(
        json.dumps(schema(), separators=(",", ":")) + "\n"
    )
    (directory / "generated/input_messages.jsonl").write_text(
        "".join(
            json.dumps(case, ensure_ascii=False, separators=(",", ":")) + "\n" for case in cases()
        )
    )
