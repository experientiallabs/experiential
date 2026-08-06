"""The grid ledger's on-disk schema, shared by the writer and every reader.

The tau grid's `ledger.jsonl` is written by the grid runner and read by two things that
must agree with it exactly: the runner itself, which counts a line's position among the
lines it accepted, and `wmo runs backfill`, which derives a telemetry seq from that same
position. One definition, because the alternative was measured and it is a silent
corruption: the runner validates with `extra="forbid"` and SKIPS what fails, so a reader
using a looser rule counts a line the runner ignored, numbers every later line
differently, and the platform ends up holding two copies of the run with a spend curve
that double-counts. Nothing downstream can see that happen.

This module holds no behavior on purpose. It is the contract both sides import, so
neither can drift from it without the other's tests failing.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict


class Calibration(BaseModel):
    """The ratio match between the learned compressor and its truncation control.

    Persisted per grid directory because three arm processes must agree on truncate's
    dial: two processes that each calibrated their own would produce two different
    controls, and there would be no single ratio-matched arm to compare the method
    against.
    """

    model_config = ConfigDict(extra="forbid")

    sample_size: int
    sample_tokens_raw: int
    endpoint_aggressiveness: float
    endpoint_achieved_ratio: float
    searched: list[tuple[float, float]]  # (truncate aggressiveness, achieved keep ratio)
    chosen_aggressiveness: float
    chosen_achieved_ratio: float
    tolerance: float
    measured_at: str
    tip_sha: str


class LedgerLine(BaseModel):
    """One appended fact about the grid's progress and its bill.

    Append-only JSONL rather than a rewritten summary: the ledger has to survive the
    SIGKILL that the thing it is tracking might not, and a partially written line is
    one bad line rather than a lost file.

    `extra="forbid"` is load-bearing rather than tidiness. It is what makes "did the
    runner count this line?" a decidable question, which is what lets a backfill months
    later derive the same seq for the same line. Loosening it would silently renumber
    every line after the first unrecognized one.
    """

    model_config = ConfigDict(extra="forbid")

    event: str  # chunk | chunk-skipped | retry | calibration | merge | stop
    arm: str
    chunk: int | None = None
    cells: int = 0
    scored: int = 0
    candidate_usd: float = 0.0
    compressor_usd: float = 0.0
    wm_usd: float = 0.0
    wall_s: float = 0.0
    ts: str
    cumulative_usd: float = 0.0
    tip_sha: str
    max_steps: int
    episodes: int
    note: str = ""
    calibration: Calibration | None = None
