"""Fresh matched DeepSWE quality evaluation for task and turn routing."""

from __future__ import annotations

import argparse
import asyncio
import collections
import json
import os
import shutil
import statistics
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from harbor.models.environment_type import EnvironmentType
from harbor.models.job.config import DatasetConfig, JobConfig
from harbor.models.task.config import TaskConfig
from harbor.models.trial.config import AgentConfig, EnvironmentConfig

from wmo.agents.default import default_agent
from wmo.config.dotenv import load_env_file
from wmo.evals.harbor.scorer import HarborScorer
from wmo.harness.doc import MAX_OUTPUT_TOKENS_ID, MAX_TURNS_ID, HarnessDoc
from wmo.optimize.policy import RoutingPolicy
from wmo.optimize.profile import fit_profile_policy
from wmo.providers.pool import PoolEntry

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import report_deepswe_knn_router_20260730 as matrix_report  # noqa: E402

BASELINE = "gpt-5.6-luna__max"
PREFIX = "coding-router-deepswe-turn-quality-20260730"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--trials", type=Path, required=True)
    parser.add_argument("--task-root", type=Path, required=True)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--policy", type=Path)
    parser.add_argument("--env-file", type=Path)
    parser.add_argument("--task-id", action="append", default=[])
    parser.add_argument("--n-tasks", type=int)
    parser.add_argument("--sample-seed", type=int, default=0)
    parser.add_argument("--cv-seed", type=int, default=11)
    parser.add_argument(
        "--strategy",
        action="append",
        choices=("baseline", "per-task", "per-turn"),
        default=[],
    )
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--max-turns", type=int, default=60)
    parser.add_argument("--max-output-tokens", type=int, default=16384)
    parser.add_argument("--episode-timeout-s", type=float, default=5400.0)
    return parser


def _load_credentials(path: Path | None) -> None:
    candidates = [path] if path is not None else []
    candidates.extend(
        [
            Path.cwd() / ".env.local",
            Path("/Users/admin/Documents/experientiallabs/coding-router/.env.local"),
        ]
    )
    for candidate in candidates:
        if candidate is not None and candidate.is_file():
            load_env_file(candidate)
            break
    missing = [
        name
        for name in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY", "E2B_API_KEY")
        if not os.environ.get(name)
    ]
    if missing:
        raise RuntimeError(f"credentials unavailable by environment name: {missing}")


def _task_ids(root: Path, requested: list[str], n_tasks: int | None, seed: int) -> list[str]:
    available = sorted(path.parent.name for path in root.glob("*/task.toml"))
    if not available:
        raise ValueError(f"DeepSWE dataset has no task.toml files: {root}")
    if requested:
        missing = sorted(set(requested) - set(available))
        if missing:
            raise ValueError(f"DeepSWE task ids are missing: {missing}")
        return requested
    if n_tasks is None:
        return available
    if n_tasks < 1 or n_tasks > len(available):
        raise ValueError(f"--n-tasks must be between 1 and {len(available)}")
    import hashlib

    return sorted(
        available,
        key=lambda task_id: hashlib.sha256(f"{seed}|deepswe-1.1|{task_id}".encode()).hexdigest(),
    )[:n_tasks]


def _job_template(root: Path, dataset_root: Path, strategy: str, concurrency: int) -> JobConfig:
    return JobConfig(
        job_name=f"{PREFIX}-{strategy}-job",
        jobs_dir=root / "harbor-jobs" / strategy,
        n_concurrent_trials=concurrency,
        environment=EnvironmentConfig(type=EnvironmentType.E2B),
        datasets=[DatasetConfig(path=dataset_root)],
        agents=[AgentConfig()],
    )


def _stage_harbor_dataset(source_root: Path, task_ids: list[str], artifact_root: Path) -> Path:
    """Stage DeepSWE's image-backed tasks for Harbor's local-task validator.

    DeepSWE supplies the environment in ``task.toml`` via a prebuilt image and intentionally
    omits Harbor's conventional ``environment/`` directory. Harbor's local dataset scanner
    still requires that directory to exist, so make a small, read-only source copy with empty
    environment directories. The task TOML, pinned image, and verifier configuration are left
    unchanged; no source benchmark files are modified.
    """
    staged_root = artifact_root / "harbor-tasks"
    staged_root.mkdir(parents=True, exist_ok=True)
    for task_id in task_ids:
        source = source_root / task_id
        target = staged_root / task_id
        if not source.is_dir():
            raise ValueError(f"DeepSWE task directory is missing: {source}")
        if not target.exists():
            shutil.copytree(source, target)
        (target / "environment").mkdir(exist_ok=True)
        task_config = TaskConfig.model_validate_toml(
            (target / "task.toml").read_text(encoding="utf-8")
        )
        verifier_environment = task_config.verifier.environment
        if verifier_environment is not None and verifier_environment.docker_image is None:
            verifier_environment.docker_image = task_config.environment.docker_image
            (target / "task.toml").write_text(
                task_config.model_dump_toml(), encoding="utf-8"
            )
    return staged_root


def _configure_doc(max_turns: int, max_output_tokens: int) -> HarnessDoc:
    doc = default_agent("pi")
    return doc.model_copy(
        update={
            "surfaces": [
                surface.model_copy(update={"content": str(max_turns)})
                if surface.id == MAX_TURNS_ID
                else surface.model_copy(update={"content": str(max_output_tokens)})
                if surface.id == MAX_OUTPUT_TOKENS_ID
                else surface
                for surface in doc.surfaces
            ]
        }
    )


def _trial_records(job_dir: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for result_path in sorted(job_dir.glob("*/result.json")):
        payload = json.loads(result_path.read_text(encoding="utf-8"))
        trial_dir = result_path.parent
        records.append(
            {
                "task_id": payload.get("task_name"),
                "trial_name": payload.get("trial_name"),
                "reward": ((payload.get("verifier_result") or {}).get("rewards") or {}).get(
                    "reward"
                ),
                "exception_type": (payload.get("exception_info") or {}).get("exception_type"),
                "graded": _graded_from_ctrf(trial_dir),
            }
        )
    return records


def _graded_from_ctrf(trial_dir: Path) -> float | None:
    path = trial_dir / "verifier" / "ctrf.json"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    summary = ((payload.get("results") or {}).get("summary") or {})
    passed = summary.get("passed")
    failed = summary.get("failed")
    if isinstance(passed, int) and isinstance(failed, int) and passed + failed > 0:
        return passed / (passed + failed)
    statuses = [item.get("status") for item in ((payload.get("results") or {}).get("tests") or [])]
    resolved = [status for status in statuses if status in ("passed", "failed")]
    return resolved.count("passed") / len(resolved) if resolved else None


def _route_records(job_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(job_dir.rglob("turn-router.jsonl")):
        trial_id = path.parent.parent.name
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                row["trial_name"] = trial_id
                rows.append(row)
    return rows


def _policy_for_holdout(
    args: argparse.Namespace,
) -> tuple[RoutingPolicy, list[str], dict[str, str]]:
    if args.policy is not None:
        return RoutingPolicy.load(args.policy.resolve()), [], {}
    scratch = args.artifact_root / "fit"
    scratch.mkdir(parents=True, exist_ok=True)
    matrix, tasks, repos = matrix_report._load_matrix(
        args.trials.resolve(), args.task_root.resolve(), scratch
    )
    fit_ids, report_ids = matrix_report._group_split(tasks, repos, args.cv_seed)
    policy = fit_profile_policy(
        matrix,
        fit_ids=fit_ids,
        guard_model=BASELINE,
        quality_tolerance=0.02,
        bins=3,
        fitted_from=f"DeepSWE 1.1 grouped-repo profile fit seed {args.cv_seed}",
    )
    pool = [
        entry.model_copy(
            update={
                "cached_input_per_mtok": entry.cached_input_per_mtok
                if entry.cached_input_per_mtok is not None
                else entry.price().cache_read_per_mtok,
                "cache_write_per_mtok": entry.cache_write_per_mtok
                if entry.cache_write_per_mtok is not None
                else entry.price().cache_write_per_mtok,
            }
        )
        for entry in policy.pool
    ]
    policy = policy.model_copy(
        update={
            "pool": pool,
            "cache_aware": True,
            "cache_switch_prefill_ratio": 4.0,
        }
    )
    policy_path = args.artifact_root / f"profile-policy-seed-{args.cv_seed}.json"
    policy.save(policy_path)
    return policy, [task_id.split(":", 1)[1] for task_id in report_ids], repos


def _entry(policy: RoutingPolicy, name: str) -> PoolEntry:
    return next(candidate for candidate in policy.pool if candidate.name == name)


def _run(
    args: argparse.Namespace,
    strategy: str,
    task_ids: list[str],
    policy: RoutingPolicy,
    doc: object,
    harbor_dataset_root: Path,
) -> dict[str, object]:
    baseline_entry = _entry(policy, BASELINE)
    provider_config = baseline_entry.provider_config()
    root = args.artifact_root.resolve()
    scorer_kwargs: dict[str, object] = {
        "job_template": _job_template(
            root, harbor_dataset_root, strategy, args.concurrency
        ),
        "task_ids": task_ids,
        "provider_config": provider_config,
        "task_environment": "e2b",
        "harness_backend": "e2b",
        "reward_key": "reward",
        "reward_mode": "raw",
        "attempts": 1,
        "harbor_retries": 0,
        "missing_reward": "zero",
        "agent_model_name": f"{provider_config.kind.value}/{provider_config.model}",
        "episode_timeout_s": args.episode_timeout_s,
        "agent_concurrency": args.concurrency,
    }
    if strategy != "baseline":
        scorer_kwargs.update(
            {
                "agent_import_path": "wmo.evals.harbor.turn_router:TurnRoutingHarborAgent",
                "extra_agent_kwargs": {
                    "router_policy_path": str(
                        (root / f"profile-policy-seed-{args.cv_seed}.json").resolve()
                    ),
                    "router_mode": "task" if strategy == "per-task" else "transcript",
                },
            }
        )
    started = time.monotonic()
    scorer = asyncio.run(HarborScorer.create(**scorer_kwargs))
    report = scorer.score(doc)
    job_dir = scorer.candidate_job_dir(doc).resolve()
    trial_records = _trial_records(job_dir)
    route_rows = _route_records(job_dir) if strategy != "baseline" else []
    graded = [float(row["graded"]) for row in trial_records if row["graded"] is not None]
    model_mix = collections.Counter(str(row["model"]) for row in route_rows)
    switches = sum(1 for row in route_rows if row.get("switched"))
    result = {
        "strategy": strategy,
        "task_count": len(task_ids),
        "quality_binary": report.score,
        "quality_graded": statistics.mean(graded) if graded else None,
        "graded_task_count": len(graded),
        "infra_failed": sum(1 for cell in report.cells if cell.infra_failed),
        "task_ids": task_ids,
        "model_mix": dict(sorted(model_mix.items())),
        "turn_count": len(route_rows),
        "switches": switches,
        "switches_per_task": switches / len(task_ids) if task_ids else 0.0,
        "uncached_cost_estimate_usd": sum(
            float(row.get("uncached_cost_usd") or 0.0) for row in route_rows
        ),
        "job_dir": str(job_dir),
        "elapsed_s": round(time.monotonic() - started, 3),
        "trial_records": trial_records,
    }
    (root / f"{strategy}-result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return result


def main() -> None:
    args = _parser().parse_args()
    if args.concurrency < 1 or args.max_turns < 1 or args.max_output_tokens < 1:
        raise ValueError("concurrency, max-turns, and max-output-tokens must be positive")
    _load_credentials(args.env_file)
    args.artifact_root.mkdir(parents=True, exist_ok=True)
    policy, holdout_ids, repos = _policy_for_holdout(args)
    requested = _task_ids(args.task_root.resolve(), args.task_id, args.n_tasks, args.sample_seed)
    if holdout_ids:
        holdout_set = set(holdout_ids)
        task_ids = [task for task in requested if task in holdout_set]
    else:
        task_ids = requested
    if not task_ids:
        raise ValueError("no evaluation tasks remain after the grouped holdout filter")
    strategies = args.strategy or ["baseline", "per-task", "per-turn"]
    doc = _configure_doc(args.max_turns, args.max_output_tokens)
    harbor_dataset_root = _stage_harbor_dataset(
        args.dataset_root.resolve(), task_ids, args.artifact_root.resolve()
    )
    manifest = {
        "benchmark": "DeepSWE 1.1",
        "baseline": BASELINE,
        "policy": policy.model_dump(mode="json"),
        "cv_seed": args.cv_seed,
        "fit_repo_count": (
            len(
                {
                    repos[task]
                    for task in repos
                    if task not in set(task_ids)
                }
            )
            if repos
            else None
        ),
        "eval_task_count": len(task_ids),
        "task_ids": task_ids,
        "harbor_dataset_root": str(harbor_dataset_root),
        "strategies": strategies,
        "doc_hash": doc.doc_hash,
        "timestamp": datetime.now(UTC).isoformat(),
    }
    (args.artifact_root / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    results = [
        _run(args, strategy, task_ids, policy, doc, harbor_dataset_root)
        for strategy in strategies
    ]
    summary = {"manifest": manifest, "results": results}
    (args.artifact_root / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    sys.stdout.write(json.dumps(summary, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
