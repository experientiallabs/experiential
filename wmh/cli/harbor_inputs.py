"""Shared local CLI loading and publication for exact Harbor scorer inputs."""

from __future__ import annotations

import json
from pathlib import Path

import typer
import yaml
from harbor.models.job.config import JobConfig
from pydantic import JsonValue, ValidationError

from wmh.harness.scoring import ScoreRequest


def load_harbor_config(path: Path) -> JobConfig:
    """Load one Harbor JobConfig from JSON or YAML with an actionable CLI error."""
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        return JobConfig.model_validate(raw)
    except (OSError, yaml.YAMLError, ValidationError, ValueError, TypeError) as error:
        raise typer.BadParameter(f"cannot load Harbor config from {path}: {error}") from error


def load_task_ids(path: Path) -> tuple[str, ...]:
    """Load one exact ordered task identity list using canonical score validation."""
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise typer.BadParameter(f"cannot load task IDs from {path}: {error}") from error
    if not isinstance(raw, list) or not all(isinstance(item, str) for item in raw):
        raise typer.BadParameter("task ID file must contain one JSON string list")
    try:
        validated = ScoreRequest.model_validate(
            {
                "context": {
                    "task_set_digest": "sha256:" + "0" * 64,
                    "evaluator_digest": "sha256:" + "0" * 64,
                    "execution_config_digest": "sha256:" + "0" * 64,
                },
                "task_ids": raw,
                "attempts": 1,
            }
        )
    except ValidationError as error:
        raise typer.BadParameter(f"invalid task ID file: {error}") from error
    return validated.task_ids


def write_json_atomic(path: Path, value: JsonValue) -> None:
    """Write deterministic JSON through a same-directory atomic rename."""
    temporary = path.with_name(f"{path.name}.tmp")
    temporary.parent.mkdir(parents=True, exist_ok=True)
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)
