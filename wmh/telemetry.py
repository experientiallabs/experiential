"""Best-effort anonymous usage telemetry."""

from __future__ import annotations

import os
import sys
from atexit import register
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from posthog import Posthog

from wmh.config import ARTIFACT_DIR
from wmh.config.settings import ensure_telemetry_anonymous_id, load_settings
from wmh.engine.reporting import BuildReporter
from wmh.tracking import RunRecord

POSTHOG_PROJECT_API_KEY = "phc_rPFfCufWpxyctR7duEZTTXovP4k5kbHqSqzd4Z4MQJdL"
POSTHOG_HOST = "https://us.i.posthog.com"

TelemetryValue = str | int | float | bool | None
TelemetryProperties = dict[str, TelemetryValue]

_FALSE_VALUES = {"0", "false", "off", "no"}
_TRUE_VALUES = {"1", "true", "on", "yes"}
_CLIENTS: dict[tuple[str, str], Posthog] = {}


@dataclass
class BuildTelemetryStats:
    input_trace_count: int = 0
    input_step_count: int = 0
    train_trace_count: int = 0
    val_trace_count: int = 0
    heldout_trace_count: int = 0
    indexed_step_count: int = 0


class TelemetryBuildReporter:
    def __init__(self, inner: BuildReporter, stats: BuildTelemetryStats) -> None:
        self._inner = inner
        self._stats = stats

    def ingest_done(self, traces: int, steps: int) -> None:
        self._stats.input_trace_count = traces
        self._stats.input_step_count = steps
        self._inner.ingest_done(traces, steps)

    def split_done(self, train: int, val: int, test: int) -> None:
        self._stats.train_trace_count = train
        self._stats.val_trace_count = val
        self._stats.heldout_trace_count = test
        self._inner.split_done(train, val, test)

    def index_done(self, steps: int) -> None:
        self._stats.indexed_step_count = steps
        self._inner.index_done(steps)

    def optimize_start(self, budget: int) -> None:
        self._inner.optimize_start(budget)

    def activity(self, line: str) -> None:
        self._inner.activity(line)

    def rollout(self, done: int, budget: int, score: float | None) -> None:
        self._inner.rollout(done, budget, score)

    def optimize_done(self, held_out_accuracy: float, frontier_size: int, rollouts: int) -> None:
        self._inner.optimize_done(held_out_accuracy, frontier_size, rollouts)


def capture(
    event: str,
    properties: TelemetryProperties | None = None,
    *,
    root: str | Path = ARTIFACT_DIR,
) -> bool:
    """Send one anonymous metadata-only event. Returns False when skipped or failed."""
    if not _enabled(root):
        return False
    api_key = os.getenv("WMH_POSTHOG_PROJECT_API_KEY", POSTHOG_PROJECT_API_KEY).strip()
    if not api_key:
        return False
    host = os.getenv("WMH_POSTHOG_HOST", POSTHOG_HOST).rstrip("/")
    try:
        distinct_id = ensure_telemetry_anonymous_id(root)
        event_properties: TelemetryProperties = {
            "$process_person_profile": False,
            "wmh_version": _wmh_version(),
            "python_version": f"{sys.version_info.major}.{sys.version_info.minor}",
            **(properties or {}),
        }
        # Never log prompts, traces, actions, observations, paths, models, credentials, or text.
        message_id = _posthog_client(api_key, host).capture(
            event,
            distinct_id=distinct_id,
            properties=event_properties,
        )
        return message_id is not None
    except (OSError, ValueError):
        return False


def capture_build_completed(
    *,
    stats: BuildTelemetryStats,
    gepa_budget: int,
    rollouts_used: int,
    frontier_size: int,
    record: RunRecord,
    root: str | Path,
) -> None:
    capture(
        "wmh build completed",
        {
            "success": True,
            "input_trace_count": stats.input_trace_count,
            "input_step_count": stats.input_step_count,
            "train_trace_count": stats.train_trace_count,
            "val_trace_count": stats.val_trace_count,
            "heldout_trace_count": stats.heldout_trace_count,
            "indexed_step_count": stats.indexed_step_count,
            "gepa_budget": gepa_budget,
            "rollouts_used": rollouts_used,
            "frontier_size": frontier_size,
            "duration_seconds": round(record.duration_seconds, 3),
            "llm_call_count": record.total.calls,
            "input_tokens": record.total.input_tokens,
            "output_tokens": record.total.output_tokens,
            "cost_usd": round(record.total.cost_usd, 6),
        },
        root=root,
    )


def capture_eval_completed(
    *,
    mode: str,
    file_count: int,
    scored_step_count: int,
    rag_enabled: bool,
    judge_mode: str,
    sample_turns: str,
    train_split: float,
    top_k: int,
    root: str | Path,
) -> None:
    capture(
        "wmh eval completed",
        {
            "success": True,
            "eval_mode": mode,
            "file_count": file_count,
            "scored_step_count": scored_step_count,
            "rag_enabled": rag_enabled,
            "judge_mode": judge_mode,
            "sample_turns": sample_turns,
            "train_split": train_split,
            "top_k": top_k,
        },
        root=root,
    )


def settings_root_from_results_root(results_root: str) -> Path:
    path = Path(results_root)
    return path.parent if path.name == "evals" else Path(ARTIFACT_DIR)


def _enabled(root: str | Path) -> bool:
    if _env_truthy("DO_NOT_TRACK"):
        return False
    env = os.getenv("WMH_TELEMETRY")
    if env is not None:
        return env.strip().lower() in _TRUE_VALUES
    if os.getenv("PYTEST_CURRENT_TEST"):
        return False
    try:
        return load_settings(root).telemetry.enabled
    except (OSError, ValueError):
        return False


def _env_truthy(name: str) -> bool:
    value = os.getenv(name)
    if value is None:
        return False
    normalized = value.strip().lower()
    if normalized in _FALSE_VALUES:
        return False
    return bool(normalized)


def _wmh_version() -> str:
    try:
        return version("world-model-harness")
    except PackageNotFoundError:
        return "unknown"


def _posthog_client(api_key: str, host: str) -> Posthog:
    key = (api_key, host)
    client = _CLIENTS.get(key)
    if client is None:
        client = Posthog(
            api_key,
            host=host,
            flush_interval=1.0,
            max_retries=1,
            timeout=0.5,
        )
        _CLIENTS[key] = client
        register(client.shutdown)
    return client
