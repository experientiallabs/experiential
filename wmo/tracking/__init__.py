"""Run tracking: time + cost + tokens across the harness lifecycle.

Instrument at the provider boundary (`MeteredProvider`) so the world model, GEPA, and the judge are
all metered without changes to those modules. `RunTracker` aggregates `UsageEvent`s into
`UsageTotals` (priced via `wmo.tracking.pricing`) plus a wall-clock duration from an injectable
`Clock`; `RunRecord`s persist under `.wmo/runs/`.
"""

from wmo.tracking.clock import Clock, SystemClock
from wmo.tracking.metered import MeteredProvider, classify_build_call
from wmo.tracking.pricing import ModelPrice, cost_usd, price_for
from wmo.tracking.store import load_runs, save_run
from wmo.tracking.tracker import (
    Phase,
    RunRecord,
    RunTracker,
    UsageEvent,
    UsageTotals,
    merge_run_records,
)

__all__ = [
    "Clock",
    "SystemClock",
    "MeteredProvider",
    "classify_build_call",
    "ModelPrice",
    "cost_usd",
    "price_for",
    "load_runs",
    "save_run",
    "Phase",
    "RunRecord",
    "RunTracker",
    "UsageEvent",
    "UsageTotals",
    "merge_run_records",
]
