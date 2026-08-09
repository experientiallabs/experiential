#!/usr/bin/env python3
"""Apply and audit a matched safe token budget for Terminal-Bench configs."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import yaml

MATCHED_PATHS = (
    ("n_attempts",),
    ("n_concurrent_trials",),
    ("environment", "type"),
    ("agents", 0, "name"),
    ("agents", 0, "override_timeout_sec"),
    ("agents", 0, "kwargs", "temperature"),
    ("agents", 0, "kwargs", "max_turns"),
    ("agents", 0, "kwargs", "enable_summarize"),
    ("agents", 0, "kwargs", "model_info", "max_input_tokens"),
    ("agents", 0, "kwargs", "model_info", "max_output_tokens"),
    ("agents", 0, "kwargs", "llm_call_kwargs", "top_p"),
    ("agents", 0, "kwargs", "llm_call_kwargs", "max_tokens"),
    ("agents", 0, "kwargs", "llm_call_kwargs", "seed"),
    ("datasets", 0, "name"),
    ("datasets", 0, "ref"),
    ("datasets", 0, "task_names"),
)


def sha256_bytes(payload: bytes) -> str:
    """Return a lowercase SHA-256 digest."""
    return hashlib.sha256(payload).hexdigest()


def nested_get(value: object, path: tuple[str | int, ...]) -> object:
    """Read a nested mapping/list value."""
    for key in path:
        if isinstance(key, int):
            if not isinstance(value, list):
                raise TypeError(f"expected list before index {key}, found {type(value)}")
            value = value[key]
        else:
            if not isinstance(value, dict):
                raise TypeError(f"expected dict before key {key!r}, found {type(value)}")
            value = value[key]
    return value


def assert_matched(configs: list[dict[str, object]]) -> None:
    """Assert the evaluation knobs and task set are identical across arms."""
    if len(configs) < 2:
        raise ValueError("at least two matched configs are required")
    for path in MATCHED_PATHS:
        values = [nested_get(config, path) for config in configs]
        if any(value != values[0] for value in values[1:]):
            label = ".".join(map(str, path))
            raise ValueError(f"matched config mismatch at {label}: {values!r}")


def rewrite_configs(
    paths: list[Path],
    *,
    expected_input: int,
    safe_input: int,
    server_context: int,
    template_margin: int,
) -> dict[str, object]:
    """Validate, rewrite, and return a manifest for matched configs."""
    originals = [path.read_bytes() for path in paths]
    configs = [yaml.safe_load(payload) for payload in originals]
    assert_matched(configs)

    output_tokens = int(
        nested_get(
            configs[0],
            ("agents", 0, "kwargs", "model_info", "max_output_tokens"),
        )
    )
    if safe_input >= expected_input:
        raise ValueError("safe input budget must be below the rendered input budget")
    if safe_input + output_tokens + template_margin > server_context:
        raise ValueError(
            f"unsafe token budget: {safe_input}+{output_tokens}+{template_margin}>{server_context}"
        )

    records: list[dict[str, object]] = []
    for path, original, config in zip(paths, originals, configs, strict=True):
        kwargs = config["agents"][0]["kwargs"]
        model_info = kwargs["model_info"]
        rendered_input = int(model_info["max_input_tokens"])
        if rendered_input != expected_input:
            raise ValueError(
                f"{path}: expected max_input_tokens={expected_input}, found {rendered_input}"
            )
        if int(kwargs["llm_call_kwargs"]["max_tokens"]) != output_tokens:
            raise ValueError(f"{path}: llm max_tokens differs from max_output_tokens")

        model_info["max_input_tokens"] = safe_input
        rewritten = yaml.safe_dump(config, sort_keys=False).encode()
        path.write_bytes(rewritten)
        records.append(
            {
                "path": str(path),
                "model_name": config["agents"][0]["model_name"],
                "pre_sha256": sha256_bytes(original),
                "post_sha256": sha256_bytes(rewritten),
            }
        )

    assert_matched(configs)
    return {
        "server_context_tokens": server_context,
        "rendered_max_input_tokens": expected_input,
        "safe_max_input_tokens": safe_input,
        "max_output_tokens": output_tokens,
        "template_margin_tokens": template_margin,
        "total_reserved_tokens": safe_input + output_tokens + template_margin,
        "unused_context_tokens": server_context - safe_input - output_tokens - template_margin,
        "configs": records,
    }


def main() -> int:
    """Rewrite all configs and record the exact transformation."""
    parser = argparse.ArgumentParser()
    parser.add_argument("configs", nargs="+", type=Path)
    parser.add_argument("--expected-input", type=int, default=53240)
    parser.add_argument("--safe-input", type=int, default=53232)
    parser.add_argument("--server-context", type=int, default=65536)
    parser.add_argument("--template-margin", type=int, default=16)
    parser.add_argument("--manifest", required=True, type=Path)
    args = parser.parse_args()

    manifest = rewrite_configs(
        args.configs,
        expected_input=args.expected_input,
        safe_input=args.safe_input,
        server_context=args.server_context,
        template_margin=args.template_margin,
    )
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    sys.stdout.write(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
