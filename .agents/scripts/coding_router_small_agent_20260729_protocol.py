"""Materialize the isolated coding-router experiment's no-spend protocol artifacts."""

from __future__ import annotations

import hashlib
import json
import logging
import math
from pathlib import Path
from typing import cast

from wmo.providers.base import ProviderKind
from wmo.providers.catalog import list_provider_models
from wmo.providers.models import resolve_provider_model
from wmo.tracking.pricing import price_for

LOGGER = logging.getLogger(__name__)
ROOT = Path(__file__).resolve().parents[2]
ARTIFACT_ROOT = ROOT / ".wmo/experiments/coding-router-small-agent-20260729"
SEEDS = (11, 23, 37, 41, 59)


def _read_json(path: Path) -> object:
    """Read one JSON input without consulting any external service."""
    return json.loads(path.read_text(encoding="utf-8"))


def _ids(path: Path) -> list[str]:
    """Read a JSON string-list task manifest."""
    value = _read_json(path)
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise ValueError(f"task manifest is not a string list: {path}")
    return list(cast(list[str], value))


def _digest(path: Path) -> str:
    """Return the SHA-256 digest of one pinned input file."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _split(groups: dict[str, list[str]], seed: int, benchmark: str) -> dict[str, object]:
    """Split task groups deterministically while keeping each group intact."""
    ordered = sorted(
        groups.items(),
        key=lambda item: hashlib.sha256(
            f"{seed}|{benchmark}|{item[0]}".encode()
        ).hexdigest(),
    )
    if len(ordered) < 2:
        raise ValueError("a deterministic split needs at least two groups")
    fit_count = min(max(1, math.ceil(len(ordered) * 0.7)), len(ordered) - 1)
    fit_groups = ordered[:fit_count]
    heldout_groups = ordered[fit_count:]
    return {
        "fit_groups": [group for group, _ in fit_groups],
        "heldout_groups": [group for group, _ in heldout_groups],
        "fit_task_ids": [task_id for _, ids in fit_groups for task_id in ids],
        "heldout_task_ids": [task_id for _, ids in heldout_groups for task_id in ids],
    }


def _wmo_roster() -> list[dict[str, object]]:
    """Snapshot the OpenAI and Anthropic candidates exposed by WMO's catalog."""
    roster: list[dict[str, object]] = []
    for kind in (ProviderKind.OPENAI, ProviderKind.ANTHROPIC):
        catalog = list_provider_models(kind)
        for entry in catalog.models:
            spec = resolve_provider_model(kind, entry.model_type or entry.id)
            price = price_for(entry.id) or price_for(spec.model_type)
            roster.append(
                {
                    "provider": kind.value,
                    "model": entry.id,
                    "model_version": spec.model_type,
                    "context_limit_tokens": None,
                    "input_usd_per_mtok": price.input_per_mtok if price else None,
                    "cached_input_usd_per_mtok": price.cache_read_per_mtok if price else None,
                    "output_usd_per_mtok": price.output_per_mtok if price else None,
                    "tool_use": "pending_provider_preflight",
                    "chat_max_tokens_field": spec.chat_max_tokens_field,
                    "forward_temperature": spec.forward_temperature,
                    "price_source": "WMO built-in pricing snapshot" if price else "unpriced",
                    "eligible_for_paid_matrix": price is not None,
                }
            )
    return roster


def _task_manifest() -> dict[str, object]:
    """Snapshot local benchmark task sources and their provenance hashes."""
    distill = ROOT / ".agents/distill"
    tb_fit = distill / "tb2-train-task-ids.json"
    tb_heldout = distill / "tb2-holdout-task-ids.json"
    tb_smoke = distill / "tb2-smoke-train-task-ids.json"
    swe_source = ROOT / "packages/environment-capture/swe-bench/instance_commits.json"
    swe = _read_json(swe_source)
    if not isinstance(swe, dict):
        raise ValueError("SWE-bench instance source is not an object")
    swe_groups: dict[str, list[str]] = {}
    swe_records: list[dict[str, object]] = []
    for task_id, raw in sorted(swe.items()):
        if not isinstance(raw, dict) or not isinstance(raw.get("repo"), str):
            raise ValueError(f"SWE-bench source record is malformed: {task_id}")
        repo = str(raw["repo"])
        swe_groups.setdefault(repo, []).append(task_id)
        swe_records.append({"task_id": task_id, **raw})
    return {
        "terminal_bench_2": {
            "development_task_ids": _ids(tb_smoke),
            "smoke_task_id": _ids(tb_smoke)[0],
            "fit_source": str(tb_fit.relative_to(ROOT)),
            "fit_source_sha256": _digest(tb_fit),
            "fit_task_ids": _ids(tb_fit),
            "heldout_source": str(tb_heldout.relative_to(ROOT)),
            "heldout_source_sha256": _digest(tb_heldout),
            "heldout_task_ids": _ids(tb_heldout),
        },
        "swe_bench_verified": {
            "source": str(swe_source.relative_to(ROOT)),
            "source_sha256": _digest(swe_source),
            "task_count": len(swe_records),
            "task_records": swe_records,
            "groups": swe_groups,
        },
    }


def main() -> None:
    """Write the roster, task, split, and provenance artifacts for this lane."""
    ARTIFACT_ROOT.mkdir(parents=True, exist_ok=True)
    roster = _wmo_roster()
    tasks = _task_manifest()
    terminal = cast(dict[str, object], tasks["terminal_bench_2"])
    terminal_ids = list(
        dict.fromkeys(
            cast(list[str], terminal["fit_task_ids"])
            + cast(list[str], terminal["heldout_task_ids"])
        )
    )
    swe = cast(dict[str, object], tasks["swe_bench_verified"])
    swe_groups = cast(dict[str, list[str]], swe["groups"])
    splits = {
        "seeds": list(SEEDS),
        "fit_fraction": 0.7,
        "assignment_rule": (
            "Sort groups by sha256(seed|benchmark|group_id), keep groups intact, "
            "assign ceil(70 percent) to fit."
        ),
        "terminal_bench_2": {
            str(seed): _split(
                {task_id: [task_id] for task_id in terminal_ids},
                seed,
                "terminal-bench-2",
            )
            for seed in SEEDS
        },
        "swe_bench_verified": {
            str(seed): _split(swe_groups, seed, "swe-bench-verified") for seed in SEEDS
        },
    }
    protocol = {
        "experiment_id": "coding-router-small-agent-20260729",
        "source_commit": "a734885b6a27224218ee73af1886ee44bb0ea697",
        "roster": "roster.json",
        "tasks": "task-manifest.json",
        "splits": "split-manifest.json",
        "paid_ceiling_usd": 20000.0,
        "paid_execution_authorized": True,
        "unique_runtime_prefix": "coding-router-small-agent-20260729",
    }
    for name, value in (
        (
            "roster.json",
            {
                "source": "wmo.providers.catalog",
                "models": roster,
                "external_crosscheck": {
                    "openai_official_ids_not_in_wmo_catalog": [
                        "gpt-5.6",
                        "gpt-5.6-terra",
                        "gpt-5.6-luna",
                        "gpt-5.4-pro",
                        "gpt-5.4-nano",
                        "gpt-5.3-codex",
                    ],
                    "anthropic_official_ids_not_in_wmo_catalog": [
                        "claude-fable-5",
                        "claude-sonnet-5",
                    ],
                    "status": "reconcile typed ids and WMO capability metadata before paid launch",
                },
            },
        ),
        ("task-manifest.json", tasks),
        ("split-manifest.json", splits),
        ("protocol-generated.json", protocol),
    ):
        (ARTIFACT_ROOT / name).write_text(
            json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    LOGGER.info("wrote no-spend protocol artifacts under %s", ARTIFACT_ROOT)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
