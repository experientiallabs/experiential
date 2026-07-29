"""Freeze the real-to-world-model router reproduction protocol."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, cast

import pandas as pd
import tomli_w

from wmo.core.files import write_text_atomic

EXPERIMENT_ID = "router-real-wm-20260728"
SOURCE_COMMIT = "c3267f1f9d5f35a14ad45b6a94b7b21d3b11c958"
ROUTERBENCH_SHA256 = "ba4f77f19517610a707c374e99322d7750c30fc4ae7ff5527888595a1e65d36d"
TAU2_COMMIT = "1d244f5dca42944b67a379b44bfeb9f5748f189d"
TB2_COMMIT = "69671fbaac6d67a7ef0dfec016cc38a64ef7a77c"
SEEDS = (0, 1, 2, 3, 4)
FIT_FRACTION = 0.7
MCQ_PREFIXES = ("mmlu-", "arc-challenge", "hellaswag", "winogrande")
PUBLIC_MODELS = (
    "WizardLM/WizardLM-13B-V1.2",
    "claude-instant-v1",
    "claude-v1",
    "claude-v2",
    "gpt-3.5-turbo-1106",
    "gpt-4-1106-preview",
    "meta/code-llama-instruct-34b-chat",
    "meta/llama-2-70b-chat",
    "mistralai/mistral-7b-chat",
    "mistralai/mixtral-8x7b-chat",
    "zero-one-ai/Yi-34B-Chat",
)
LETTER = re.compile(r"\b([A-E])\b[).:]?")

ROSTER: tuple[dict[str, Any], ...] = (
    {
        "name": "gpt-5.5",
        "kind": "azure",
        "model_type": "gpt-5.5",
        "deployment_env": "AZURE_FOUNDRY_GPT55_DEPLOYMENT",
        "endpoint_env": "AZURE_FOUNDRY_2_ENDPOINT",
        "key_env": "AZURE_FOUNDRY_2_API_KEY",
        "api_version": "2024-10-21",
        "effort": "high",
        "input_per_mtok": 5.0,
        "cached_input_per_mtok": 0.5,
        "output_per_mtok": 30.0,
    },
    {
        "name": "gpt-5.4-mini",
        "kind": "azure",
        "model_type": "gpt-5.4-mini",
        "deployment_env": "AZURE_FOUNDRY_GPT54_MINI_DEPLOYMENT",
        "endpoint_env": "AZURE_FOUNDRY_2_ENDPOINT",
        "key_env": "AZURE_FOUNDRY_2_API_KEY",
        "api_version": "2024-10-21",
        "effort": "high",
        "input_per_mtok": 0.75,
        "cached_input_per_mtok": 0.075,
        "output_per_mtok": 4.5,
    },
    {
        "name": "fable-5",
        "kind": "anthropic",
        "model": "claude-fable-5",
        "effort": "max",
        "input_per_mtok": 10.0,
        "cached_input_per_mtok": 1.0,
        "output_per_mtok": 50.0,
        "cache_write_per_mtok": 12.5,
    },
    {
        "name": "sonnet-5",
        "kind": "anthropic",
        "model": "claude-sonnet-5",
        "effort": "high",
        "input_per_mtok": 3.0,
        "cached_input_per_mtok": 0.3,
        "output_per_mtok": 15.0,
        "cache_write_per_mtok": 3.75,
    },
    {
        "name": "haiku-4-5",
        "kind": "anthropic",
        "model": "claude-haiku-4-5-20251001",
        "effort": None,
        "input_per_mtok": 1.0,
        "cached_input_per_mtok": 0.1,
        "output_per_mtok": 5.0,
        "cache_write_per_mtok": 1.25,
    },
    {
        "name": "opus-4-8",
        "kind": "bedrock",
        "model": "us.anthropic.claude-opus-4-8",
        "region": "us-east-1",
        "effort": None,
        "input_per_mtok": 5.0,
        "cached_input_per_mtok": 0.5,
        "output_per_mtok": 25.0,
        "cache_write_per_mtok": 6.25,
    },
    {
        "name": "deepseek-v4-pro",
        "kind": "azure",
        "model_type": "deepseek-v4-pro",
        "deployment_env": "AZURE_FOUNDRY_DEEPSEEK_DEPLOYMENT",
        "endpoint_env": "AZURE_FOUNDRY_ENDPOINT",
        "key_env": "AZURE_FOUNDRY_API_KEY",
        "api_version": "2024-10-21",
        "effort": None,
        "input_per_mtok": 1.74,
        "cached_input_per_mtok": 1.74,
        "output_per_mtok": 3.48,
    },
    {
        "name": "kimi-k2.6",
        "kind": "azure",
        "model_type": "kimi-k2.6",
        "deployment_env": "AZURE_FOUNDRY_KIMI_DEPLOYMENT",
        "endpoint_env": "AZURE_FOUNDRY_ENDPOINT",
        "key_env": "AZURE_FOUNDRY_API_KEY",
        "api_version": "2024-10-21",
        "effort": None,
        "input_per_mtok": 0.95,
        "cached_input_per_mtok": 0.95,
        "output_per_mtok": 4.0,
    },
    {
        "name": "glm-5.2",
        "kind": "azure",
        "model_type": "glm-5.2",
        "deployment_env": "AZURE_FOUNDRY_GLM52_DEPLOYMENT",
        "endpoint_env": "AZURE_FOUNDRY_3_ENDPOINT",
        "key_env": "AZURE_FOUNDRY_3_API_KEY",
        "api_version": "2024-10-21",
        "effort": None,
        "input_per_mtok": 1.54,
        "cached_input_per_mtok": 0.15,
        "output_per_mtok": 4.84,
        "price_source": (
            "Azure Retail Prices API, Azure Fireworks Models, FW GLM 5.2 Data Zone "
            "meters, queried 2026-07-29"
        ),
    },
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_text_atomic(path, json.dumps(value, indent=2, sort_keys=True) + "\n")


def _write_pool(path: Path) -> None:
    rows: list[dict[str, object]] = []
    for arm in ROSTER:
        row: dict[str, object] = {
            "name": arm["name"],
            "kind": arm["kind"],
            "model": arm.get("model", arm.get("model_type")),
            "reasoning_effort": arm.get("effort"),
            "input_per_mtok": arm["input_per_mtok"],
            "cached_input_per_mtok": arm["cached_input_per_mtok"],
            "output_per_mtok": arm["output_per_mtok"],
        }
        if arm["kind"] == "azure":
            row.update(
                {
                    "model_type": arm["model_type"],
                    "endpoint_env": arm["endpoint_env"],
                    "deployment_env": arm["deployment_env"],
                    "api_key_env": arm["key_env"],
                    "api_version": arm["api_version"],
                }
            )
        elif arm["kind"] == "anthropic":
            row["api_key_env"] = "ANTHROPIC_API_KEY"
        for optional in ("cache_write_per_mtok", "region"):
            value = arm.get(optional)
            if value is not None:
                row[optional] = value
        rows.append({key: value for key, value in row.items() if value is not None})
    path.parent.mkdir(parents=True, exist_ok=True)
    write_text_atomic(path, tomli_w.dumps({"model": rows}))


def _prompt(raw: object) -> str:
    if isinstance(raw, list):
        return str(raw[0])
    return str(raw)


def _letter(raw: object) -> str | None:
    match = LETTER.search(str(raw).strip().strip("'[]\"")[:80])
    return match.group(1) if match else None


def _routerbench_manifest(pickle_path: Path) -> dict[str, object]:
    if _sha256(pickle_path) != ROUTERBENCH_SHA256:
        raise ValueError(f"RouterBench SHA-256 mismatch: {pickle_path}")
    frame = pd.read_pickle(pickle_path)
    frame = frame[frame["eval_name"].str.startswith(MCQ_PREFIXES)]
    certified: list[dict[str, str]] = []
    drops: Counter[str] = Counter()
    for row in frame.to_dict("records"):
        winners: Counter[str] = Counter()
        losers: set[str] = set()
        for model in PUBLIC_MODELS:
            parsed = _letter(row[f"{model}|model_response"])
            if parsed is None:
                continue
            score = float(row[model])
            if score >= 1.0:
                winners[parsed] += 1
            elif score <= 0.0:
                losers.add(parsed)
        if len(winners) != 1:
            drops["no_or_conflicting_winner"] += 1
            continue
        answer, count = winners.most_common(1)[0]
        if count < 2 or answer in losers:
            drops["insufficient_or_contradicted"] += 1
            continue
        certified.append(
            {
                "task_id": f"{row['eval_name']}:{row['sample_id']}",
                "group": str(row["eval_name"]),
                "prompt": _prompt(row["prompt"]),
                "answer": answer,
            }
        )
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for item in certified:
        grouped[item["group"]].append(item)
    rng = random.Random(11)
    target = 1200
    chosen: list[dict[str, str]] = []
    for _group, items in sorted(grouped.items()):
        quota = max(1, round(target * len(items) / len(certified)))
        chosen.extend(rng.sample(items, min(quota, len(items))))
    chosen = sorted(chosen, key=lambda item: item["task_id"])[:target]
    return {
        "benchmark": "routerbench-ours9-refreshed",
        "source": str(pickle_path.resolve()),
        "source_sha256": ROUTERBENCH_SHA256,
        "certification": "official score-column consensus, historical Stage B procedure",
        "certified_count": len(certified),
        "drop_counts": dict(drops),
        "sample_seed": 11,
        "target_count": target,
        "tasks": chosen,
    }


def _stratified_split(
    tasks: list[dict[str, object]], *, benchmark: str, seed: int
) -> dict[str, list[str]]:
    by_group: dict[str, list[str]] = defaultdict(list)
    for item in tasks:
        by_group[str(item["group"])].append(str(item["task_id"]))
    fit: list[str] = []
    heldout: list[str] = []
    for group, ids in sorted(by_group.items()):
        ordered = sorted(
            ids,
            key=lambda task_id: (
                hashlib.sha256(f"{seed}:{benchmark}:{group}:{task_id}".encode()).digest(),
                task_id,
            ),
        )
        cut = min(max(round(len(ordered) * FIT_FRACTION), 1), len(ordered) - 1)
        fit.extend(ordered[:cut])
        heldout.extend(ordered[cut:])
    return {"fit": sorted(fit), "heldout": sorted(heldout)}


def _group_split(
    tasks: list[dict[str, object]], *, benchmark: str, seed: int
) -> dict[str, list[str]]:
    grouped: dict[str, list[str]] = defaultdict(list)
    for item in tasks:
        grouped[str(item["group"])].append(str(item["task_id"]))
    ordered = sorted(
        grouped,
        key=lambda group: (
            hashlib.sha256(f"{seed}:{benchmark}:{group}".encode()).digest(),
            group,
        ),
    )
    target = round(len(tasks) * FIT_FRACTION)
    reachable: dict[int, tuple[str, ...]] = {0: ()}
    for group in ordered:
        size = len(grouped[group])
        additions = {
            count + size: selected + (group,)
            for count, selected in reachable.items()
            if count + size < len(tasks)
        }
        for count, selected in additions.items():
            reachable.setdefault(count, selected)
    best_count = min(
        (count for count in reachable if count > 0),
        key=lambda count: (abs(count - target), count > target, count),
    )
    fit_groups = set(reachable[best_count])
    fit = sorted(
        str(item["task_id"]) for item in tasks if str(item["group"]) in fit_groups
    )
    heldout = sorted(
        str(item["task_id"])
        for item in tasks
        if str(item["group"]) not in fit_groups
    )
    return {"fit": fit, "heldout": heldout}


def _tau_manifest(path: Path) -> dict[str, object]:
    task_ids = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(task_ids, list) or not all(isinstance(item, str) for item in task_ids):
        raise ValueError(f"{path} must be a list of task IDs")
    tasks = [
        {"task_id": task_id, "group": task_id.split("/", 1)[0]}
        for task_id in sorted(task_ids)
    ]
    return {
        "benchmark": "tau2-real",
        "source_commit": TAU2_COMMIT,
        "source": str(path.resolve()),
        "tasks": tasks,
    }


def _terminal_manifest(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    raw_tasks = value.get("tasks") if isinstance(value, dict) else None
    if not isinstance(raw_tasks, list):
        raise ValueError(f"{path} does not contain tasks")
    tasks: list[dict[str, object]] = []
    for raw in raw_tasks:
        if isinstance(raw, str):
            task_id = raw
        elif isinstance(raw, dict) and isinstance(raw.get("task_id"), str):
            task_id = raw["task_id"]
        else:
            raise ValueError(f"{path} contains an invalid task row")
        family = (
            str(raw.get("group"))
            if isinstance(raw, dict) and isinstance(raw.get("group"), str)
            else "-".join(task_id.split("-")[:2])
        )
        tasks.append({"task_id": task_id, "group": family})
    return {
        "benchmark": "terminal-bench-2",
        "source_commit": TB2_COMMIT,
        "harbor_dataset": "terminal-bench@2.0",
        "tasks": sorted(tasks, key=lambda item: str(item["task_id"])),
    }


def _manifest_tasks(manifest: dict[str, object]) -> list[dict[str, object]]:
    raw = manifest.get("tasks")
    if not isinstance(raw, list) or not all(isinstance(item, dict) for item in raw):
        raise ValueError("manifest tasks must be a list of objects")
    return [
        {
            str(key): value
            for key, value in cast(dict[object, object], item).items()
        }
        for item in raw
    ]


def main() -> None:
    """Write immutable manifests, paired splits, roster, and freeze summary."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument(
        "--routerbench",
        type=Path,
        default=Path(
            "/Users/admin/Documents/experientiallabs/data/"
            "router-repro-20260728/routerbench_0shot.pkl"
        ),
    )
    parser.add_argument(
        "--tau-heldout",
        type=Path,
        default=Path(".agents/distill/tau2-holdout-task-ids.json"),
    )
    parser.add_argument(
        "--terminal-source",
        type=Path,
        default=Path(
            ".wmo/experiments/coding-router-20260728/tasks/terminal-bench-2.json"
        ),
    )
    args = parser.parse_args()
    root = args.artifact_root.resolve()
    manifests = {
        "routerbench": _routerbench_manifest(args.routerbench),
        "tau2": _tau_manifest(args.tau_heldout),
        "terminal_bench_2": _terminal_manifest(args.terminal_source),
    }
    manifest_hashes: dict[str, str] = {}
    for name, manifest in manifests.items():
        path = root / "freeze" / "tasks" / f"{name}.json"
        _write_json(path, manifest)
        manifest_hashes[name] = _sha256(path)
    _write_json(root / "freeze" / "roster.json", list(ROSTER))
    _write_pool(root / "freeze" / "pool.toml")
    split_hashes: dict[str, str] = {}
    for seed in SEEDS:
        split = {
            name: (
                _group_split(_manifest_tasks(manifest), benchmark=name, seed=seed)
                if name == "terminal_bench_2"
                else _stratified_split(
                    _manifest_tasks(manifest), benchmark=name, seed=seed
                )
            )
            for name, manifest in manifests.items()
        }
        path = root / "freeze" / "splits" / f"seed-{seed}.json"
        _write_json(path, split)
        split_hashes[str(seed)] = _sha256(path)
    summary = {
        "experiment_id": EXPERIMENT_ID,
        "source_commit": SOURCE_COMMIT,
        "budget_usd": 20_000.0,
        "split_seeds": list(SEEDS),
        "fit_fraction": FIT_FRACTION,
        "manifest_hashes": manifest_hashes,
        "roster_sha256": _sha256(root / "freeze" / "roster.json"),
        "pool_sha256": _sha256(root / "freeze" / "pool.toml"),
        "split_hashes": split_hashes,
        "protocol": str(
            Path(".agents/docs/research/router-reproduction-20260728.md").resolve()
        ),
        "protocol_sha256": _sha256(
            Path(".agents/docs/research/router-reproduction-20260728.md")
        ),
        "required_env_files": {
            "/Users/admin/Documents/experientiallabs/platform/.env.local": True,
            "/Users/admin/Documents/experientiallabs/coding-router/.env.local": True,
        },
    }
    _write_json(root / "freeze" / "summary.json", summary)


if __name__ == "__main__":
    main()
