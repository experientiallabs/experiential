"""Canonical JSON projection for Harbor models with set-valued fields."""

from __future__ import annotations

import json
from collections.abc import Mapping
from datetime import date, datetime, time
from enum import Enum
from pathlib import Path
from typing import cast
from uuid import UUID

from harbor.models.job.config import JobConfig
from pydantic import BaseModel

from wmh.core.types import JsonObject


def normalize_harbor_json(value: object) -> object:
    """Return JSON-shaped Harbor data while preserving sequence and set semantics.

    Pydantic's JSON mode turns sets into process-order lists. Normalize from Python mode instead:
    ordered lists and tuples retain their order, while sets become canonically sorted lists.
    """
    if isinstance(value, BaseModel):
        return normalize_harbor_json(value.model_dump(mode="python"))
    if isinstance(value, Mapping):
        return {
            str(key): normalize_harbor_json(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (set, frozenset)):
        normalized = [normalize_harbor_json(item) for item in value]
        return sorted(
            normalized,
            key=lambda item: json.dumps(
                item,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
            ),
        )
    if isinstance(value, (list, tuple)):
        return [normalize_harbor_json(item) for item in value]
    if isinstance(value, Enum):
        return normalize_harbor_json(value.value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, UUID):
        return str(value)
    if isinstance(value, (datetime, date, time)):
        return value.isoformat()
    return value


def canonical_harbor_job_config(value: object) -> JsonObject:
    """Validate and emit one stable, JSON-safe Harbor ``JobConfig`` payload.

    Re-validation restores Harbor's set types when reading a pre-canonical checkpoint, allowing
    the same projection to compare old and new process orderings without rewriting evidence.
    """
    # A checkpoint payload may predate this projection. Reject unknown fields instead of letting
    # Harbor's default ``extra=ignore`` erase input drift and make two identities compare equal.
    config = (
        value if isinstance(value, JobConfig) else JobConfig.model_validate(value, extra="forbid")
    )
    encoded = json.dumps(
        normalize_harbor_json(config.model_dump(mode="python")),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )
    payload = json.loads(encoded)
    if not isinstance(payload, dict):  # pragma: no cover - JobConfig always has an object root
        raise TypeError("canonical Harbor JobConfig must be a JSON object")
    return cast("JsonObject", payload)
