"""Live wiring for the concurrency scaling law: build the world/real batch runners.

`concurrency_scaling.run_concurrency_scaling` is deployment-free — it takes `world_runner(level)`
and `real_runner(level)` callables and never imports a provider or a subprocess. This module builds
those callables against the *deployed* primitives, so the experiment measures what the harness
actually does:

- **World side** — for each held-out scenario, replay its recorded steps teacher-forced through the
  shared `predict_observation` (the exact serving/eval predict path: recorded `state_before` +
  history + leak-free demos), timing only the predictions. Each scenario runs in its own thread with
  its own provider client, so a batch at concurrency W overlaps W scenarios; timing the batch gives
  the wall-clock the world model needs to reconstruct N scenarios W-at-a-time.
- **Real side** — for each scenario, shell the example's `run.sh --trace-id <id>` (which stands up
  the real environment and replays the SAME scenario the world side ran, pinned by `trace_id` so the
  two sides never diverge on corpus ordering), one subprocess per scenario, W at a time. The
  subprocess is process-isolated, so concurrency is bounded by host resources — the asymmetry the
  differential measures.

The CLI (`wmh research concurrency`) calls these; unit tests exercise the core driver with fakes, so
nothing here is imported on the no-network test path.
"""

from __future__ import annotations

import concurrent.futures as cf
import logging
import os
import subprocess
import time
from collections.abc import Callable
from pathlib import Path

from pydantic import BaseModel

from wmh.core.types import Trace
from wmh.optimize.gepa import predict_observation
from wmh.providers.base import Provider
from wmh.research.concurrency_scaling import RealBatch, RealRunner, WorldBatch, WorldRunner
from wmh.retrieval.leakfree import DemoRetriever
from wmh.tracking.metered import MeteredProvider
from wmh.tracking.tracker import Phase, RunTracker

_log = logging.getLogger(__name__)

# A fresh provider per worker: boto3/httpx clients are not safe to share across threads.
ProviderFactory = Callable[[], Provider]


class _ScenarioTiming(BaseModel):
    """One scenario's reconstruction cost: wall-clock, steps, error-flag matches, tokens, USD."""

    seconds: float
    steps: int
    matches: int
    tokens: int
    cost_usd: float


def _replay_one(
    provider: Provider, prompt: str, trace: Trace, demos: DemoRetriever
) -> _ScenarioTiming:
    """Time reconstructing one trace teacher-forced, metering tokens/cost per scenario.

    Times the full per-step reconstruction — leak-free demo retrieval + the `predict_observation`
    call — but NOT any judge, since the scaling law is about reconstruction throughput, and
    retrieval is part of what serving actually pays. (With the offline HashingEmbedder retrieval is
    a near-no-op; a live embedding-backed retriever would add its query latency here, exactly as it
    would in production.) Each step predicts from the recorded `state_before` + the real prior steps
    as history + leak-free demos, exactly like `wmh eval`. Metering goes through a
    `MeteredProvider`/`RunTracker` local to this scenario, so concurrent scenarios never share
    tracker state (the tracker is not thread-safe).
    """
    tracker = RunTracker(run_id=trace.trace_id, kind="concurrency")
    metered = MeteredProvider(provider, tracker, base_phase=Phase.SERVE)
    start = time.monotonic()
    matches = 0
    for i, step in enumerate(trace.steps):
        predicted = predict_observation(
            metered,
            prompt,
            step.task,
            step.state_before,
            step.action,
            demos=demos.demos_for(trace.trace_id, step),
            history=trace.steps[:i],
        )
        if predicted.is_error == step.observation.is_error:
            matches += 1
    seconds = time.monotonic() - start
    totals = tracker.totals()
    return _ScenarioTiming(
        seconds=seconds,
        steps=len(trace.steps),
        matches=matches,
        tokens=totals.total_tokens,
        cost_usd=totals.cost_usd,
    )


def build_world_runner(
    provider_factory: ProviderFactory,
    prompt: str,
    demos: DemoRetriever,
    selected: list[tuple[int, Trace]],
) -> WorldRunner:
    """A `WorldRunner`: reconstruct the fixed batch of `selected` scenarios at concurrency `level`.

    Every level runs the same N=`len(selected)` scenarios through a pool of `level` workers, so only
    concurrency varies across levels — the like-for-like measurement the scaling law needs.

    Each scenario gets its own provider (via `provider_factory`) so concurrent threads never share a
    client, and its own `RunTracker` for metering; the leak-free `demos` index is built once and
    shared read-only (its query path is stateless), so index-build time never pollutes the timing.
    """

    def run_one(trace: Trace) -> _ScenarioTiming:
        return _replay_one(provider_factory(), prompt, trace, demos)

    def runner(level: int) -> WorldBatch:
        start = time.monotonic()
        per: list[_ScenarioTiming] = []
        with cf.ThreadPoolExecutor(max_workers=level) as executor:
            futures = [executor.submit(run_one, t) for _idx, t in selected]
            for future in cf.as_completed(futures):
                # A failed scenario counts as not-ok (ok < total) rather than aborting the batch —
                # otherwise one early failure would still block on the rest before surfacing, and
                # lose the partial timing. Log it so a systematically-failing run (bad creds, etc.)
                # is diagnosable instead of silently reporting a batch of zeros.
                try:
                    per.append(future.result())
                except Exception:  # noqa: BLE001 - provider/network errors -> a failed scenario
                    _log.warning("world scenario failed at concurrency %d", level, exc_info=True)
        wall = time.monotonic() - start
        steps = sum(p.steps for p in per)
        matches = sum(p.matches for p in per)
        return WorldBatch(
            wall_seconds=wall,
            work_seconds=sum(p.seconds for p in per),
            ok=len(per),
            total=len(selected),
            tokens=sum(p.tokens for p in per),
            cost_usd=sum(p.cost_usd for p in per),
            fidelity=(matches / steps) if steps else 0.0,
        )

    return runner


def _run_real_one(
    runner_sh: Path,
    cwd: Path,
    trace_id: str,
    train_split: float,
    extra_args: list[str],
    timeout: float | None,
) -> tuple[float, bool]:
    """Shell the example's run.sh for one scenario, pinned by trace_id; return (seconds, ok).

    Only the wall-clock and exit status are used, so the run.sh output (which for swe-bench/terminal
    streams multi-GB docker build logs) is discarded rather than buffered — otherwise W concurrent
    workers would each hold their whole build log in memory. Run `run.sh` directly to watch a build.
    """
    cmd = [
        str(runner_sh),
        "--trace-id",
        trace_id,
        "--train-split",
        str(train_split),
        *extra_args,
    ]
    start = time.monotonic()
    try:
        proc = subprocess.run(
            cmd,
            cwd=cwd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=timeout,
        )
    except (subprocess.TimeoutExpired, OSError):
        return time.monotonic() - start, False
    return time.monotonic() - start, proc.returncode == 0


def build_real_runner(
    example_dir: Path,
    selected: list[tuple[int, Trace]],
    *,
    train_split: float,
    extra_args: list[str] | None = None,
    timeout: float | None = None,
) -> RealRunner:
    """A `RealRunner`: stand up + replay real sandboxes at concurrency `level` -> `RealBatch`.

    Shells `example_dir/run.sh --trace-id <trace_id> --train-split <ts>` per scenario — pinning by
    the world side's exact `trace_id` (not a positional index, whose order differs between the two
    corpus loaders), so both sides provably replay the SAME scenarios. Raises if the example has no
    executable `run.sh`.

    Every level runs the same `selected` sandboxes through a pool of `level` workers, matching the
    world side's fixed-N batch so only concurrency varies across levels.
    """
    runner_sh = example_dir / "run.sh"
    if not runner_sh.exists():
        raise FileNotFoundError(f"missing real sandbox runner: {runner_sh}")
    if not os.access(runner_sh, os.X_OK):
        raise PermissionError(f"real sandbox runner is not executable: {runner_sh}")
    args = list(extra_args or [])
    all_ids = [trace.trace_id for _idx, trace in selected]

    def runner(level: int) -> RealBatch:
        start = time.monotonic()
        per: list[tuple[float, bool]] = []
        with cf.ThreadPoolExecutor(max_workers=level) as executor:
            futures = [
                executor.submit(
                    _run_real_one, runner_sh, example_dir, tid, train_split, args, timeout
                )
                for tid in all_ids
            ]
            for future in cf.as_completed(futures):
                # `_run_real_one` already returns (secs, False) on its own errors; guard here too so
                # an unexpected raise counts as a failed scenario, not a batch-wide abort. Log it so
                # a systematically-failing sandbox is diagnosable.
                try:
                    per.append(future.result())
                except Exception:  # noqa: BLE001 - a failed sandbox launch -> a failed scenario
                    _log.warning("real sandbox failed at concurrency %d", level, exc_info=True)
        wall = time.monotonic() - start
        return RealBatch(
            wall_seconds=wall,
            work_seconds=sum(secs for secs, _ok in per),
            ok=sum(1 for _s, ok in per if ok),
            total=len(all_ids),
        )

    return runner


__all__ = ["ProviderFactory", "build_world_runner", "build_real_runner"]
