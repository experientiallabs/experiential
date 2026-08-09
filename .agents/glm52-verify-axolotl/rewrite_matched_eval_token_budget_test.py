"""Tests for matched evaluation token-budget rewriting."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest
import yaml
from rewrite_matched_eval_token_budget import rewrite_configs


def sample_config(model_name: str) -> dict[str, object]:
    """Return a minimal official-style config."""
    return {
        "job_name": model_name,
        "n_attempts": 1,
        "n_concurrent_trials": 24,
        "environment": {"type": "e2b"},
        "agents": [
            {
                "name": "terminus-2",
                "override_timeout_sec": 2700,
                "model_name": f"hosted_vllm/{model_name}",
                "kwargs": {
                    "temperature": 1.0,
                    "max_turns": 100,
                    "enable_summarize": True,
                    "model_info": {
                        "max_input_tokens": 53240,
                        "max_output_tokens": 12288,
                    },
                    "llm_call_kwargs": {
                        "top_p": 1.0,
                        "max_tokens": 12288,
                        "seed": 1,
                    },
                },
            }
        ],
        "datasets": [
            {
                "name": "terminal-bench/terminal-bench-2-1",
                "ref": "sha256:test",
                "task_names": [f"task-{index}" for index in range(89)],
            }
        ],
    }


def write_configs(tmp_path: Path) -> list[Path]:
    """Write a base and adapter pair."""
    paths = [tmp_path / "base.yaml", tmp_path / "adapter.yaml"]
    for path, name in zip(paths, ("base", "adapter"), strict=True):
        path.write_text(yaml.safe_dump(sample_config(name), sort_keys=False))
    return paths


def test_rewrite_configs_reserves_template_margin(tmp_path: Path) -> None:
    paths = write_configs(tmp_path)
    manifest = rewrite_configs(
        paths,
        expected_input=53240,
        safe_input=53232,
        server_context=65536,
        template_margin=16,
    )

    assert manifest["total_reserved_tokens"] == 65536
    assert manifest["unused_context_tokens"] == 0
    assert len(manifest["configs"]) == 2
    for path in paths:
        config = yaml.safe_load(path.read_text())
        assert config["agents"][0]["kwargs"]["model_info"]["max_input_tokens"] == 53232


def test_rewrite_configs_rejects_unmatched_pair(tmp_path: Path) -> None:
    paths = write_configs(tmp_path)
    adapter = yaml.safe_load(paths[1].read_text())
    adapter["agents"][0]["kwargs"]["temperature"] = 0.5
    paths[1].write_text(yaml.safe_dump(adapter, sort_keys=False))

    with pytest.raises(ValueError, match="temperature"):
        rewrite_configs(
            paths,
            expected_input=53240,
            safe_input=53232,
            server_context=65536,
            template_margin=16,
        )


def test_rewrite_configs_rejects_unsafe_budget(tmp_path: Path) -> None:
    paths = write_configs(tmp_path)
    configs = [yaml.safe_load(path.read_text()) for path in paths]
    for path, config in zip(paths, configs, strict=True):
        unsafe = deepcopy(config)
        unsafe["agents"][0]["kwargs"]["model_info"]["max_output_tokens"] = 12300
        unsafe["agents"][0]["kwargs"]["llm_call_kwargs"]["max_tokens"] = 12300
        path.write_text(yaml.safe_dump(unsafe, sort_keys=False))

    with pytest.raises(ValueError, match="unsafe token budget"):
        rewrite_configs(
            paths,
            expected_input=53240,
            safe_input=53232,
            server_context=65536,
            template_margin=16,
        )
