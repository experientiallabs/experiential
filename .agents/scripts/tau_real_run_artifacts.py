"""Turn a real-tau2 grid's rows into the artifacts `wmo runs backfill` replays.

The real-episode leg measures candidates outside `wmo optimize model`, so nothing emits its run
to the platform while it is running: `PipelineEmitter` hangs off the pipeline's stages and this
grid has none. The product's answer for a run measured elsewhere is `wmo runs backfill`, which
replays a run from artifacts on disk. It reads a grid directory shaped like the world-model
cohorts: `cohort.json`, a `ledger.jsonl` of per-chunk lines, and `<arm>/chunk-<k>.json` files.

This writes exactly that shape from `rows.jsonl`, so the real grid lands in the runs panel with
its cells, its spend, and its own clock, through the product's own importer rather than a second
ingest path.

Two things it will not do:

- INVENT A TIMESTAMP. Backfill's contract is that nothing is inferred that the artifacts do not
  say, so a chunk is stamped from the tau2 clocks its rows carry. Rows bought before the runner
  recorded those clocks cannot be stamped, and they are REPORTED and skipped rather than given
  the hour this script happened to run.
- RESHAPE AN OUTCOME. Chunk files carry `ScenarioOutcome` rows built by the runner's own
  `to_matrix`, the same function that writes `matrix.json`, so the cells the panel shows and the
  cells the analysis fits cannot drift apart.

A chunk here is one tau2 batch: one (candidate, domain, episode). That is the unit the runner
actually buys and the unit whose wall clock is meaningful, so a ledger line's `wall_s` is the
real span from the batch's first episode start to its last episode end.

    uv run python .agents/scripts/tau_real_run_artifacts.py --grid-dir <dir> --arm identity
    wmo runs backfill <dir> --arm identity --dry-run --out /tmp/events.jsonl
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path

_RL = Path(__file__).resolve().parents[1].parent / "packages/environment-capture/tau-bench/rl"
sys.path.insert(0, str(_RL))

from real_episodes import (  # noqa: E402  (path shim above; the runner is not an installed module)
    RealEpisodeRow,
    load_rows,
    to_matrix,
)

from wmo.providers.pool import load_pool  # noqa: E402

logger = logging.getLogger("tau-run-artifacts")

CHUNK_KEY_HELP = "one tau2 batch: (candidate, domain, episode)"


def _parse(stamp: str) -> datetime:
    """One tau2 stamp as an aware UTC datetime.

    tau2 writes NAIVE stamps in the machine's local zone (`datetime.now()`), and the platform
    reads a stamp without an offset as UTC. Handing its strings over untouched put the run's
    started_at seven hours before the created_at of the very events that reported it, which is
    what the first backfill of this grid actually showed. A naive stamp is therefore localized
    before conversion; an offset-carrying stamp is trusted as given.
    """
    parsed = datetime.fromisoformat(stamp)
    if parsed.tzinfo is None:
        parsed = parsed.astimezone()
    return parsed.astimezone(UTC)


def chunk_rows(rows: list[RealEpisodeRow]) -> dict[tuple[str, str, int], list[RealEpisodeRow]]:
    """Group rows into the batches they were bought in, cheapest-candidate order preserved."""
    grouped: dict[tuple[str, str, int], list[RealEpisodeRow]] = defaultdict(list)
    for row in rows:
        grouped[(row.model, row.domain, row.episode)].append(row)
    return dict(grouped)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grid-dir", type=Path, required=True, help="holds <arm>/rows.jsonl")
    parser.add_argument("--arm", default="identity")
    parser.add_argument("--pool", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True, help="the served world-model dir")
    parser.add_argument("--cohort-label", required=True, help="the protocol pin label on the rows")
    parser.add_argument("--tip-sha", required=True, help="the commit the grid ran at")
    parser.add_argument(
        "--max-turns", type=int, default=100, help="the runner's max_turns pin, as tau2 --max-steps"
    )
    parser.add_argument("--episodes", type=int, default=2, help="episodes per cell the grid buys")
    args = parser.parse_args(argv)

    arm_dir = args.grid_dir / args.arm
    rows = load_rows(arm_dir / "rows.jsonl")
    if not rows:
        logger.error("no rows at %s", arm_dir / "rows.jsonl")
        return 2
    pool = load_pool(args.pool).models

    labels = {row.cohort for row in rows}
    if labels != {args.cohort_label}:
        # Pooling two cohorts into one run would present episodes from different environments as
        # one measurement, which is the mistake the cohort label exists to catch.
        logger.error("rows span cohorts %s, expected only %r", sorted(labels), args.cohort_label)
        return 2

    chunks = chunk_rows(rows)
    unstamped = [key for key, group in chunks.items() if not any(r.ended_at for r in group)]
    stamped = {key: group for key, group in chunks.items() if any(r.ended_at for r in group)}
    if unstamped:
        logger.warning(
            "%d of %d batches carry no tau2 clock and are SKIPPED (bought before the runner "
            "recorded start/end times); backfill refuses to invent a timestamp: %s",
            len(unstamped),
            len(chunks),
            sorted(f"{m}/{d}/e{e}" for m, d, e in unstamped),
        )
    if not stamped:
        logger.error("no batch carries a tau2 clock; nothing can be backfilled honestly")
        return 2

    ordered = sorted(
        stamped.items(), key=lambda kv: min(_parse(r.ended_at) for r in kv[1] if r.ended_at)
    )
    ledger_lines: list[dict[str, object]] = []
    cumulative = 0.0
    for index, (key, group) in enumerate(ordered):
        matrix = to_matrix(group, pool)
        (arm_dir / f"chunk-{index}.json").write_text(
            matrix.model_dump_json(indent=2), encoding="utf-8"
        )
        starts = [_parse(r.started_at) for r in group if r.started_at]
        ends = [_parse(r.ended_at) for r in group if r.ended_at]
        candidate_usd = sum(r.cost_usd_pool for r in group)
        cumulative += candidate_usd
        model, domain, episode = key
        ledger_lines.append(
            {
                "event": "chunk",
                "arm": args.arm,
                "chunk": index,
                "cells": len(matrix.outcomes),
                "scored": sum(1 for o in matrix.outcomes if o.scored),
                "candidate_usd": candidate_usd,
                # No compressor in this leg, and the environment is the real benchmark rather than
                # a world model, so there is no world-model spend to report. Zero here is a
                # measured zero, not a missing number.
                "compressor_usd": 0.0,
                "wm_usd": 0.0,
                "wall_s": (max(ends) - min(starts)).total_seconds() if starts and ends else 0.0,
                "ts": max(ends).isoformat(),
                "cumulative_usd": cumulative,
                "tip_sha": args.tip_sha,
                "max_steps": args.max_turns,
                "episodes": args.episodes,
                # `LedgerLine` forbids extra keys on purpose: that is what makes "did the runner
                # count this line?" decidable, and so what lets a backfill derive the same seq
                # months later. Which batch this was therefore goes in the free-text note rather
                # than in fields of my own.
                "note": (
                    f"real tau2 episodes, {model} / {domain} / episode {episode}; "
                    f"reward is tau2's own, never a wmo judge; cohort {args.cohort_label}"
                ),
            }
        )

    (args.grid_dir / "ledger.jsonl").write_text(
        "".join(json.dumps(line) + "\n" for line in ledger_lines), encoding="utf-8"
    )
    first_start = min(_parse(r.started_at) for _, group in ordered for r in group if r.started_at)
    (args.grid_dir / "cohort.json").write_text(
        json.dumps(
            {
                "created": first_start.isoformat(),
                "model_dir": str(args.model_dir),
                "pool_file": str(args.pool),
                "cohort": args.cohort_label,
                "provenance": "real_episode",
                "judge": "tau2 reward",
                "scenarios": len({r.scenario_id for r in rows}),
                "episodes": len({r.episode for r in rows}),
                "candidates": len({r.model for r in rows}),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    logger.info(
        "wrote %d chunk file(s), ledger.jsonl and cohort.json under %s (%d cells, $%.4f)",
        len(ledger_lines),
        args.grid_dir,
        sum(int(line["cells"]) for line in ledger_lines),  # type: ignore[arg-type]
        cumulative,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
