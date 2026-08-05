"""Token-span records tying harbor trials to the student tokens they sampled.

Two span sources feed the same `TrialRecord` shape, one per rollout agent:

- **harbor's own terminus-2** (`load_trial_rollout_spans`, the distillation
  path): the agent runs `harbor.llms.tinker.TinkerLLM` with
  `collect_rollout_details=True`, harbor persists the per-turn
  `prompt_token_ids` / `completion_token_ids` / `logprobs` into the trial's
  `result.json` under `agent_result.rollout_details`, and this module reads
  them straight back out. Stop reasons come from
  `read_terminus_stop_reason`, which reconstructs them from the trial's
  recorded exception, its episode count, and the ATIF trajectory's final
  `mark_task_complete`.
- **the WMO pi bridge** (`load_trial_spans`, the harness-optimization path):
  the distill agent's `TokenRecorder` appends every sampled `TokenSpan` to a
  per-trial JSONL sink named after the harbor trial
  (`{sink_dir}/{trial_name}.jsonl`, one span JSON per line), and the WMO run
  trace (`wmo-run.json`, written into harbor's per-trial agent logs dir)
  supplies the stop reason.

Either way `assemble_trial_records` joins the spans with the scorer's reward
cells into `TrialRecord`s, the unit the datum builder consumes.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable, Sequence
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from wmo.core.types import JsonObject
from wmo.optimize.harness.runtime import StopReason
from wmo.optimize.harness.scoring import GradedTests, ScoreCell
from wmo.providers.tinker import TokenSpan

logger = logging.getLogger(__name__)

WMO_RUN_TRACE_FILENAME = "wmo-run.json"

HARBOR_TRIAL_RESULT_FILENAME = "result.json"
"""Harbor's per-trial `TrialResult` dump (`harbor.models.trial.paths.TrialPaths.result_path`)."""

HARBOR_TRAJECTORY_FILENAME = "trajectory.json"
"""Terminus-2's ATIF trajectory, dumped into its logs dir (`{trial_dir}/agent/`)."""

TERMINUS_COMPLETE_TOOL = "mark_task_complete"
"""The ATIF tool call terminus-2 records when it declares the task finished."""

StopReasonReader = Callable[[Path], str | None]
"""Reads one trial's stop reason from its artifact dir; None when unknown."""

SpanLoader = Callable[[Path], list[TokenSpan]]
"""Reads one trial's sampled token spans from its artifact dir."""


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
    """The WMO run trace's stop reason; None when no trace was readable."""

    infra_failed: bool = False
    """True when the trial never produced verifier evidence.

    Two causes, one measurement status: the agent never ran (sandbox/transport death) or its work
    was never graded (the verifier timed out or wrote nothing parseable). Carried from the scorer's
    cell. Its `reward` is a stand-in, so metrics must exclude it from solve-rate denominators and
    count it separately (see `rollout_stats`)."""

    tests: GradedTests | None = None
    """The verifier's per-test counts, carried from the scorer's cell; None when it wrote no
    readable report.

    `graded_score` is the pass fraction over resolved tests: the same trials at test resolution,
    beside (never instead of) the binary `passed`. None is not a 0.0, so `rollout_stats` averages
    only the trials that have one. Absent on records written before this field existed."""

    artifact_dir: str
    """The harbor trial directory holding this trial's raw evidence."""

    @property
    def graded_score(self) -> float | None:
        """This trial's graded test-pass score in [0, 1]; None when no test report exists."""
        return None if self.tests is None else self.tests.score


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


def _read_json_object(path: Path) -> JsonObject | None:
    """One JSON object from disk, or None when it is missing or unreadable.

    Never raises: assembly must survive the trials that died hardest, where a
    truncated or absent artifact IS the evidence.
    """
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None
    except (OSError, json.JSONDecodeError):
        logger.warning("unreadable harbor artifact at %s", path)
        return None
    return payload if isinstance(payload, dict) else None


def _int_lists(value: object) -> list[list[int]] | None:
    """`value` as a list of int lists, or None when it is not shaped like one."""
    if not isinstance(value, list):
        return None
    rows: list[list[int]] = []
    for row in value:
        if not isinstance(row, list):
            return None
        ints: list[int] = []
        for item in row:
            if isinstance(item, bool) or not isinstance(item, int):
                return None
            ints.append(item)
        rows.append(ints)
    return rows


def _float_lists(value: object) -> list[list[float]] | None:
    """`value` as a list of float lists, or None when it is not shaped like one."""
    if not isinstance(value, list):
        return None
    rows: list[list[float]] = []
    for row in value:
        if not isinstance(row, list):
            return None
        floats: list[float] = []
        for item in row:
            if isinstance(item, bool) or not isinstance(item, (int, float)):
                return None
            floats.append(float(item))
        rows.append(floats)
    return rows


def load_trial_rollout_spans(artifact_dir: Path) -> list[TokenSpan]:
    """Read the token spans harbor recorded for one terminus-2 trial.

    The source is harbor's own persisted evidence, not a WMO sink: the trial's
    `result.json` carries `agent_result.rollout_details`, a list of
    `harbor.models.agent.rollout_detail.RolloutDetail`, and by harbor's
    convention its FIRST entry is the main agent's linear chat history. Each
    entry holds three per-turn lists (`prompt_token_ids`,
    `completion_token_ids`, `logprobs`) that terminus-2's `TinkerLLM`
    populates with the exact ids the sampler consumed and issued, so the
    tokens-in-tokens-out contract is preserved verbatim through the file.

    Later entries are SUBAGENT segments (terminus-2 appends one per
    summarization handoff); they are separate conversations whose prompts do
    not extend the main chat, so they are skipped and warned about. The
    distill collector disables summarization precisely so none exist.

    Args:
        artifact_dir: The harbor trial directory (one cell's `artifact_dir`).

    Returns:
        The spans in turn order, `call_index` 0..n-1. Empty when the trial
        never reached a completion, when harbor recorded no rollout details
        (the agent ran without `collect_rollout_details`, or died before its
        first sample), or when the recorded lists cannot be aligned. Every
        non-trivial empty case is logged at WARNING and surfaces in
        `RolloutStats.empty_span_trials`: a trial that produced no training
        evidence must be counted, never silently dropped and never allowed to
        abort the batch around it.
    """
    payload = _read_json_object(artifact_dir / HARBOR_TRIAL_RESULT_FILENAME)
    if payload is None:
        return []
    agent_result = payload.get("agent_result")
    if not isinstance(agent_result, dict):
        if payload.get("step_results"):
            logger.warning(
                "harbor trial %s is a MULTI-STEP trial: its per-step chats are separate "
                "conversations, so no single prefix-chained episode exists to train on",
                artifact_dir.name,
            )
        return []
    details = agent_result.get("rollout_details")
    if not isinstance(details, list) or not details:
        return []
    if len(details) > 1:
        logger.warning(
            "harbor trial %s recorded %d rollout detail segments; only the first (the main "
            "agent chat) is trainable, the rest are subagent conversations whose prompts do "
            "not extend it. Summarization must stay OFF for distillation rollouts",
            artifact_dir.name,
            len(details),
        )
    detail = details[0]
    if not isinstance(detail, dict):
        logger.warning("harbor trial %s recorded a malformed rollout detail", artifact_dir.name)
        return []
    prompts = _int_lists(detail.get("prompt_token_ids"))
    completions = _int_lists(detail.get("completion_token_ids"))
    logprobs = _float_lists(detail.get("logprobs"))
    if prompts is None or completions is None or logprobs is None:
        logger.warning(
            "harbor trial %s recorded rollout details without well-formed per-turn token id "
            "lists (prompt=%s completion=%s logprobs=%s); it contributes no training data",
            artifact_dir.name,
            type(detail.get("prompt_token_ids")).__name__,
            type(detail.get("completion_token_ids")).__name__,
            type(detail.get("logprobs")).__name__,
        )
        return []
    if not (len(prompts) == len(completions) == len(logprobs)):
        # harbor's Chat appends to the three lists independently, so a turn that returned no
        # completion (or no logprobs) desynchronizes them for every later turn. Nothing can
        # realign them after the fact, and pairing them anyway would train the wrong prompts.
        logger.warning(
            "harbor trial %s recorded %d prompt turn(s), %d completion turn(s) and %d logprob "
            "turn(s); the per-turn lists cannot be aligned, so the trial contributes no "
            "training data",
            artifact_dir.name,
            len(prompts),
            len(completions),
            len(logprobs),
        )
        return []
    spans: list[TokenSpan] = []
    for index, (prompt, completion, row) in enumerate(
        zip(prompts, completions, logprobs, strict=True)
    ):
        try:
            spans.append(
                TokenSpan(
                    call_index=index,
                    prompt_token_ids=prompt,
                    sampled_token_ids=completion,
                    sampled_logprobs=row,
                )
            )
        except ValidationError as exc:
            logger.warning(
                "harbor trial %s turn %d records %d sampled token(s) against %d logprob(s) "
                "(%s); the trial contributes no training data",
                artifact_dir.name,
                index,
                len(completion),
                len(row),
                exc,
            )
            return []
    return spans


_TERMINUS_EXCEPTION_STOP_REASONS = {
    # Harbor's own agent-phase wall clock (`AgentConfig.override_timeout_sec`, which is where
    # `rollout.episode_timeout_s` lands): harbor swallows it and still verifies the work.
    "AgentTimeoutError": StopReason.BUDGET,
    # Terminus-2 re-raises this when summarization is off, which is how distillation runs it:
    # the next prompt outgrew `rollout.context_budget_tokens` and nothing was sampled from it.
    "ContextLengthExceededError": StopReason.PROVIDER_ERROR,
}
"""Terminus-2 trial exceptions that name their own stop reason; anything else is `ERROR`."""


def _terminus_declared_complete(artifact_dir: Path) -> bool:
    """True when terminus-2's final agent step declared the task complete.

    Read from the ATIF trajectory terminus-2 dumps after every episode, so it
    survives a trial that died later. `mark_task_complete` is the tool call it
    records for its own completion claim, which is the closest analogue of
    pi's explicit `submit`.
    """
    payload = _read_json_object(artifact_dir / "agent" / HARBOR_TRAJECTORY_FILENAME)
    steps = payload.get("steps") if payload is not None else None
    if not isinstance(steps, list):
        return False
    for step in reversed(steps):
        if not isinstance(step, dict) or step.get("source") != "agent":
            continue
        calls = step.get("tool_calls")
        if not isinstance(calls, list):
            return False
        return any(
            isinstance(call, dict) and call.get("function_name") == TERMINUS_COMPLETE_TOOL
            for call in calls
        )
    return False


def read_terminus_stop_reason(artifact_dir: Path, *, max_turns: int) -> str | None:
    """Why one terminus-2 episode ended, in WMO's `StopReason` vocabulary.

    Terminus-2 writes no WMO run trace, so the reason is reconstructed from
    the three artifacts harbor and the agent do leave behind, in the order
    that makes each one decisive:

    1. the trial's recorded exception (`AgentTimeoutError` is the wall clock,
       `ContextLengthExceededError` is the context budget, anything else is a
       harness/runtime error);
    2. the episode count terminus-2 records in
       `agent_result.metadata.n_episodes`: reaching `max_turns` means the loop
       was cut off by the cap;
    3. the ATIF trajectory's final `mark_task_complete`, terminus-2's own
       completion claim.

    Deliberately conservative at the boundary: an episode that declared
    completion on exactly its last allowed turn reads as `max_turns`, i.e. as
    a scaffold loss, rather than as a submission the cap happened to permit.

    Args:
        artifact_dir: The harbor trial directory (one cell's `artifact_dir`).
        max_turns: The turn cap the agent ran under (`rollout.max_turns`).

    Returns:
        A `StopReason` value, or None when no trial result was readable at all
        (which `rollout_stats` counts as `"unknown"` and excludes from the
        scaffold-loss denominator rather than guessing).
    """
    payload = _read_json_object(artifact_dir / HARBOR_TRIAL_RESULT_FILENAME)
    if payload is None:
        return None
    exception = payload.get("exception_info")
    if isinstance(exception, dict):
        raised = exception.get("exception_type")
        if isinstance(raised, str):
            return _TERMINUS_EXCEPTION_STOP_REASONS.get(raised, StopReason.ERROR).value
    agent_result = payload.get("agent_result")
    metadata = agent_result.get("metadata") if isinstance(agent_result, dict) else None
    episodes = metadata.get("n_episodes") if isinstance(metadata, dict) else None
    if isinstance(episodes, bool) or not isinstance(episodes, int):
        return None
    if episodes >= max_turns:
        return StopReason.MAX_TURNS.value
    if _terminus_declared_complete(artifact_dir):
        return StopReason.SUBMITTED.value
    # The loop returned early without claiming completion: terminus-2 does that only when its
    # tmux session died under it, which is a harness failure, not a task verdict.
    return StopReason.ERROR.value


def read_trial_stop_reason(artifact_dir: Path) -> str | None:
    """The WMO run trace's stop reason for one trial, when a trace exists.

    The agent bridge writes `wmo-run.json` into harbor's per-trial agent logs
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
        artifact_dir / "agent" / WMO_RUN_TRACE_FILENAME,
        artifact_dir / WMO_RUN_TRACE_FILENAME,
    )
    for candidate in candidates:
        if not candidate.is_file():
            continue
        try:
            payload = json.loads(candidate.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            logger.warning("unreadable WMO run trace at %s; recording no stop reason", candidate)
            return None
        stop_reason = payload.get("stop_reason") if isinstance(payload, dict) else None
        return stop_reason if isinstance(stop_reason, str) else None
    return None


def assemble_harbor_trial_records(
    cells: Sequence[ScoreCell], *, max_turns: int
) -> list[TrialRecord]:
    """Join scorer cells with the spans HARBOR recorded for its terminus-2 agent.

    The distillation path's assembler: spans come from each trial's
    `result.json` (`load_trial_rollout_spans`) and stop reasons are
    reconstructed from harbor's own artifacts (`read_terminus_stop_reason`).
    No WMO-side sink directory is involved, so nothing has to survive the
    scorer's destructive entry prune.

    Args:
        cells: The scored trial cells (one per task x attempt).
        max_turns: The turn cap the agent ran under, needed to tell a
            cap-terminated episode from a completed one.

    Returns:
        One `TrialRecord` per cell, in the cells' order.

    Raises:
        ValueError: If a cell carries no artifact dir.
    """
    return _assemble_trial_records(
        cells,
        load_spans=load_trial_rollout_spans,
        read_stop_reason=lambda artifact_dir: read_terminus_stop_reason(
            artifact_dir, max_turns=max_turns
        ),
    )


def assemble_trial_records(
    cells: Sequence[ScoreCell],
    sink_dir: Path,
    *,
    read_stop_reason: StopReasonReader | None = None,
) -> list[TrialRecord]:
    """Join scorer cells with their token sinks and run traces.

    The WMO pi-bridge path's assembler (see `assemble_harbor_trial_records`
    for the terminus-2 one).

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
    return _assemble_trial_records(
        cells,
        load_spans=lambda artifact_dir: load_trial_spans(sink_dir, artifact_dir.name),
        read_stop_reason=read_stop_reason or read_trial_stop_reason,
    )


def _assemble_trial_records(
    cells: Sequence[ScoreCell],
    *,
    load_spans: SpanLoader,
    read_stop_reason: StopReasonReader,
) -> list[TrialRecord]:
    """Join cells with whichever span source and stop-reason reader the caller uses."""
    reader = read_stop_reason
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
                spans=load_spans(artifact_dir),
                stop_reason=reader(artifact_dir),
                infra_failed=cell.infra_failed,
                tests=cell.tests,
                artifact_dir=cell.artifact_dir,
            )
        )
    return records
