"""Seed repair sidecars so a grid re-run buys ONLY a chunk's unscored cells.

The concurrency-12 run finished with an outage's fingerprints in two arms (compressor endpoint
rate-limited/unreachable; local connection exhaustion on truncate). Those cells are unscored rows
in finished chunk matrices, and `execute_sweep`'s resume deliberately reuses unscored rows rather
than re-buying them - re-running a failure is the CALLER's decision. This script is that caller:

For each chunk whose matrix holds unscored rows, it writes the sidecar `execute_sweep` resumes
from, seeded with the chunk's SCORED rows only, then deletes the chunk matrix and the arm's
merged matrix. The next `run_tau_grid.py` invocation re-runs the chunk, resumes the scored rows
for free, and buys exactly the cells that failed. Identity is not reconstructed by hand: the
plans are built through the runner's own `plan_sweep` + `slice_plan`, so the sidecar header is
byte-for-byte the identity the re-run will check.

Usage (from the repo the grid dir belongs to; same --env-file layering as the runner):

    uv run python .agents/scripts/seed_repair_sidecars.py \
        --grid-dir .wmo/jt/grid-c2 --model-dir .wmo/models/tau-bench \
        --pool .wmo/jt/grid/pool.toml --traces packages/.../traces.otel.jsonl \
        --env-file /Users/silen/Desktop/Projects/wmo-grid/.env

Read-only toward the ledger: seeding spends nothing and appends no ledger line.
"""

from __future__ import annotations

import importlib.util
import logging
import sys
from pathlib import Path

log = logging.getLogger("seed_repair")

REPO_ROOT = Path(__file__).resolve().parents[2]

spec = importlib.util.spec_from_file_location(
    "run_tau_grid", Path(__file__).parent / "run_tau_grid.py"
)
assert spec and spec.loader
grid = importlib.util.module_from_spec(spec)
sys.modules["run_tau_grid"] = grid
spec.loader.exec_module(grid)

from wmo.optimize.outcomes import OutcomeMatrix  # noqa: E402
from wmo.optimize.sweep import plan_sweep, resolve_config  # noqa: E402
from wmo.optimize.sweep_partial import PartialWriter, partial_path  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(message)s")
    args = grid.parse_args(argv)
    grid.load_env_file(REPO_ROOT / ".env")
    for extra in args.env_file or []:
        grid.load_env_file(Path(extra))
    config = grid.build_config(args)

    harness_config = resolve_config(config.model_dir)
    cohort = grid.Cohort(
        tip_sha=grid.tip_sha(),
        max_steps=config.max_steps,
        episodes=config.episodes,
        scenarios=config.scenarios,
        chunk_size=config.chunk_size,
        history_chars=config.history_chars,
        model_dir=str(config.model_dir),
        pool_file=str(config.grid_dir / "pool.toml"),
        traces_file=str(config.traces_file),
        created=grid.now_iso(),
    )
    state = grid.GridState(config.grid_dir, cohort)
    pool = grid.preflight_pool(state.pinned_pool_file(config.pool_file)).pool

    total_seeded = 0
    for arm in config.arms:
        compression = grid.arm_compression(
            arm,
            config,
            state,
            train_split=harness_config.train_split,
            adapter=harness_config.trace_adapter,
        )
        base_plan = plan_sweep(
            model_dir=config.model_dir,
            config=harness_config,
            pool=pool,
            out_path=config.grid_dir / arm / grid.MERGED_MATRIX,
            traces_file=config.traces_file,
            scenarios=config.scenarios,
            episodes=config.episodes,
            max_steps=config.max_steps,
            assume_input_tokens=2000,
            assume_output_tokens=250,
            history_chars=config.history_chars,
            compression=compression,
            max_concurrency=config.concurrency,
        )
        arm_dir = config.grid_dir / arm
        arm_dirty = False
        chunk_count = -(-len(base_plan.scenarios) // config.chunk_size)
        for index in range(chunk_count):
            chunk_file = arm_dir / f"chunk-{index}.json"
            if not chunk_file.exists():
                continue
            matrix = OutcomeMatrix.load(chunk_file)
            scored = [row for row in matrix.outcomes if row.reward is not None]
            unscored = len(matrix.outcomes) - len(scored)
            if unscored == 0:
                continue
            scenarios = base_plan.scenarios[
                index * config.chunk_size : (index + 1) * config.chunk_size
            ]
            plan = grid.slice_plan(base_plan, scenarios, chunk_file)
            sidecar = partial_path(chunk_file)
            with PartialWriter(sidecar, plan.identity) as writer:
                for row in scored:
                    writer.append(row)
            chunk_file.unlink()
            arm_dirty = True
            total_seeded += unscored
            log.info(
                "%s chunk %d: seeded %d scored row(s), %d cell(s) will be re-bought",
                arm,
                index,
                len(scored),
                unscored,
            )
        merged = arm_dir / grid.MERGED_MATRIX
        if arm_dirty and merged.exists():
            merged.unlink()
            log.info("%s: merged matrix removed; the re-run will merge a repaired one", arm)
    log.info("seeding done: %d unscored cell(s) staged for re-buy", total_seeded)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
