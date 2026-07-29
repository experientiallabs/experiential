"""Complete the one frozen composite smoke with one Azure GPT-5.5 WMO step."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
from pathlib import Path

from wmo.core.files import write_text_atomic
from wmo.core.types import Step
from wmo.engine.world_model import WorldModel
from wmo.providers.base import TokenUsage
from wmo.providers.pool import load_pool, pool_provider

logger = logging.getLogger("router-real-smoke")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _first_step(model_dir: Path) -> Step:
    index = model_dir / "index" / "steps.jsonl"
    with index.open(encoding="utf-8") as handle:
        line = next((raw for raw in handle if raw.strip()), None)
    if line is None:
        raise ValueError(f"{index} contains no WMO steps")
    return Step.model_validate_json(line)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--pool",
        type=Path,
        default=Path(
            "/Users/admin/Documents/experientiallabs/data/router-repro-20260728/"
            "full/freeze/pool.toml"
        ),
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=Path("packages/environment-capture/tau-bench/models/tau-bench"),
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path(
            "/Users/admin/Documents/experientiallabs/data/router-repro-20260728/"
            "full/smoke/composite.json"
        ),
    )
    args = parser.parse_args()
    if args.out.exists():
        if args.resume:
            logger.info("composite smoke already complete: %s", args.out)
            return 0
        raise FileExistsError(f"{args.out} exists; pass --resume to preserve the paid cell")

    prerequisites = {
        "routerbench_objective_path": Path(
            "/Users/admin/Documents/experientiallabs/data/router-repro-20260728/"
            "routerbench-public-repro.json"
        ),
        "tau2_real_rows": Path(
            "/Users/admin/Documents/experientiallabs/data/router-repro-20260728/"
            "tau-real/rows.jsonl"
        ),
        "terminal_bench_real_summary": Path(
            "/Users/admin/Documents/experientiallabs/data/router-repro-20260728/"
            "tb2-smoke/summary.json"
        ),
    }
    missing = [str(path) for path in prerequisites.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"composite smoke prerequisites missing: {missing}")

    entry = load_pool(args.pool).entry("gpt-5.5")
    source = _first_step(args.model_dir)
    provider = pool_provider(entry)
    world_model = WorldModel.load(
        str(args.model_dir),
        provider,
        telemetry_root=args.out.parent / "telemetry",
    )
    session = world_model.new_session(task=source.task, enrich=False)
    observation = world_model.step(session.id, source.action)
    usage = world_model.end_session(session.id)
    priced = TokenUsage(
        input_tokens=usage.total.input_tokens,
        output_tokens=usage.total.output_tokens,
        cached_input_tokens=usage.total.cached_input_tokens,
        cache_write_input_tokens=usage.total.cache_write_input_tokens,
    )
    record = {
        "status": "passed",
        "experiment_id": "router-real-wm-20260728",
        "components": {
            name: {"path": str(path), "sha256": _sha256(path)}
            for name, path in prerequisites.items()
        },
        "azure_gpt55_world_model": {
            "provider": entry.model_dump(mode="json", exclude_none=True),
            "artifact_dir": str(args.model_dir.resolve()),
            "artifact_config_sha256": _sha256(args.model_dir / "config.toml"),
            "task": source.task,
            "action": source.action.model_dump(mode="json"),
            "observation": observation.model_dump(mode="json"),
            "usage": usage.model_dump(mode="json"),
            "realized_cost_usd": entry.cost_usd(priced),
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    write_text_atomic(args.out, json.dumps(record, indent=2, sort_keys=True) + "\n")
    logger.info(
        "composite smoke passed: one Azure GPT-5.5 WMO serve step, $%.6f; %s",
        entry.cost_usd(priced),
        args.out,
    )
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    raise SystemExit(main())
