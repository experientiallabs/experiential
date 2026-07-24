"""Token-span records tying harbor trials to the student tokens they sampled.

During distillation rollouts each harbor trial runs the pi agent against the
Tinker student, and the distill agent's `TokenRecorder` appends every sampled
`TokenSpan` to a per-trial JSONL sink named after the harbor trial
(`{sink_dir}/{trial_name}.jsonl`, one span JSON per line). This module reads
those sinks back and joins them with the scorer's reward cells into
`TrialRecord`s, the unit the datum builder consumes. The trial's WMH run
trace (`wmh-run.json`, written by the agent bridge into harbor's per-trial
agent logs dir) supplies the stop reason when it exists.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable, Sequence
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from wmh.harness.scoring import ScoreCell
from wmh.providers.tinker import TokenSpan

logger = logging.getLogger(__name__)

WMH_RUN_TRACE_FILENAME = "wmh-run.json"

StopReasonReader = Callable[[Path], str | None]
"""Reads one trial's stop reason from its artifact dir; None when unknown."""


class TrialRecord(BaseModel):
    """One harbor trial joined with the exact token spans it sampled.

    `spans` is the tokens-in-tokens-out training evidence (empty when the
    trial died before its first successful student completion); `reward` and
    `passed` come from the harbor verifier via the scorer's cells and are
    metrics/gating signal, never part of the distillation loss.
    """

    model_config = ConfigDict(extra="forbid")

    task_id: str = Field(min_length=1)
    attempt: int = Field(ge=1)
    trial_name: str = Field(min_length=1)
    reward: float = Field(ge=0.0, le=1.0, allow_inf_nan=False)
    passed: bool
    spans: list[TokenSpan] = Field(default_factory=list)
    stop_reason: str | None = None
    """The WMH run trace's stop reason; None when no trace was readable."""

    artifact_dir: str
    """The harbor trial directory holding this trial's raw evidence."""


def load_trial_spans(sink_dir: Path, trial_name: str) -> list[TokenSpan]:
    """Read the token spans one trial's recorder sink captured.

    The sink is the JSONL file the distill agent's `TokenRecorder` writes:
    `{trial_name}.jsonl` under `sink_dir`, one `TokenSpan` JSON per line,
    flushed after every successful completion.

    Args:
        sink_dir: The per-trial token sink directory for one rollout batch.
        trial_name: The harbor trial name the sink file is keyed by.

    Returns:
        The recorded spans in call order. A missing sink file yields an empty
        list: the trial made no successful student completion (it died before
        the first sample), which callers count in their stats rather than
        raise on.

    Raises:
        ValueError: If a line is not a valid `TokenSpan`, or the call_index
            sequence is not exactly 0..n-1 (the sink was appended by more than
            one recorder); delete the step's token sink directory and re-run
            the step to rebuild it.
    """
    sink_path = sink_dir / f"{trial_name}.jsonl"
    try:
        text = sink_path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return []
    spans: list[TokenSpan] = []
    for line_number, line in enumerate(text.splitlines(), 1):
        if not line.strip():
            continue
        try:
            spans.append(TokenSpan.model_validate_json(line))
        except ValidationError as exc:
            raise ValueError(
                f"invalid token span on line {line_number} of {sink_path}: {exc}; "
                "the sink is corrupt, so delete this step's token sink directory "
                "and re-run the step"
            ) from exc
    observed = [span.call_index for span in spans]
    if observed != list(range(len(spans))):
        raise ValueError(
            f"token sink {sink_path} has call_index sequence {observed}, expected "
            f"0..{len(spans) - 1}; more than one recorder appended to it (e.g. a "
            "re-run trial reused the sink), so delete this step's token sink "
            "directory and re-run the step"
        )
    return spans


def read_trial_stop_reason(artifact_dir: Path) -> str | None:
    """The WMH run trace's stop reason for one trial, when a trace exists.

    The agent bridge writes `wmh-run.json` into harbor's per-trial agent logs
    dir (`{trial_dir}/agent/` for single-step tasks); both full `RunResult`
    dumps and partial cancellation traces carry a string `stop_reason`. The
    trial-dir root is also checked for robustness against layout drift.

    Args:
        artifact_dir: The harbor trial directory (one cell's `artifact_dir`).

    Returns:
        The stop reason string, or None when no trace is present or readable.
        Assembly must not fail on the trials that died hardest; a missing
        stop reason is itself the signal.
    """
    candidates = (
        artifact_dir / "agent" / WMH_RUN_TRACE_FILENAME,
        artifact_dir / WMH_RUN_TRACE_FILENAME,
    )
    for candidate in candidates:
        if not candidate.is_file():
            continue
        try:
            payload = json.loads(candidate.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            logger.warning("unreadable WMH run trace at %s; recording no stop reason", candidate)
            return None
        stop_reason = payload.get("stop_reason") if isinstance(payload, dict) else None
        return stop_reason if isinstance(stop_reason, str) else None
    return None


def assemble_trial_records(
    cells: Sequence[ScoreCell],
    sink_dir: Path,
    *,
    read_stop_reason: StopReasonReader | None = None,
) -> list[TrialRecord]:
    """Join scorer cells with their token sinks and run traces.

    Args:
        cells: The scored trial cells (one per task x attempt); each cell's
            `artifact_dir` must be the harbor trial directory, whose basename
            is the trial name the token sinks are keyed by.
        sink_dir: The per-trial token sink directory for this rollout batch.
        read_stop_reason: Reads one trial's stop reason from its artifact dir;
            defaults to `read_trial_stop_reason`.

    Returns:
        One `TrialRecord` per cell, in the cells' order. A trial without a
        sink file gets empty spans, never dropped: its reward is still real
        batch signal and callers count span-less trials explicitly.

    Raises:
        ValueError: If a cell carries no artifact dir (the trial name cannot
            be derived), or a sink file is corrupt (see `load_trial_spans`).
    """
    reader = read_stop_reason if read_stop_reason is not None else read_trial_stop_reason
    records: list[TrialRecord] = []
    for cell in cells:
        artifact_dir = Path(cell.artifact_dir)
        trial_name = artifact_dir.name
        if not cell.artifact_dir or not trial_name:
            raise ValueError(
                f"score cell for task {cell.task_id!r} attempt {cell.attempt} carries no "
                "artifact dir, so its trial name (the token-sink key) cannot be derived; "
                "collect rollouts through a scorer that records per-trial directories "
                "(HarborScorer does)"
            )
        records.append(
            TrialRecord(
                task_id=cell.task_id,
                attempt=cell.attempt,
                trial_name=trial_name,
                reward=cell.reward,
                passed=cell.passed,
                spans=load_trial_spans(sink_dir, trial_name),
                stop_reason=reader(artifact_dir),
                artifact_dir=cell.artifact_dir,
            )
        )
    return records
