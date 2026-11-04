"""Turn saved continuation branches, contrast cells and Harbor trial ledgers into episode evidence.

Each converter returns one :class:`EpisodeEvidence` for the frozen episode judge. The actor-visible
prefix retained from a checkpoint comes first as ``context`` events: the new actor is not
responsible for it. Everything after the scored start is ``scored``: every actor message, including
intermediate user-facing text and not only the final answer, each tool call with its arguments,
the world's reply (``tool_error`` when it returned ``is_error``, never treated as success), the
harness receipt, the evaluator-only state after each change, exogenous events and the final answer.
An action an experiment imposed is scored but owned by ``experiment``: no actor sampled it.

Event ids are deterministic locators (``prefix.3``, ``call.2.tool.0.reply``, ``ledger.7``), and
each event's ``source`` names the exact saved record field it came from. ``complete`` and
``missing`` are honest: an unrecorded tool reply, actor response or interrupted branch is missing
evidence. Hidden reasoning is never recorded and is not evidence, so it is never listed as missing.
"""

from collections.abc import Sequence
from typing import Literal

from pydantic import Field

from exp.common.core.artifacts import ContractModel, JsonObject, JsonValue
from exp.common.models import ModelFinishReason, ModelMessage, ModelResponse, ToolCall
from exp.optimize.evaluation.continuations.action_contrast import ActionContrastSpec, ContrastCell
from exp.optimize.evaluation.continuations.intervention import ActionIntervention
from exp.optimize.evaluation.continuations.study import ContinuationBranch
from exp.optimize.evaluation.executable.trial import TrialLedgerEntry
from exp.optimize.evaluation.variance.episodes import (
    EpisodeEvent,
    EpisodeEvidence,
    EventKind,
    EventOwner,
    EventScope,
)
from exp.runtime.environments.interface import Observation
from exp.simulation.continuations import ContinuationCheckpoint


class ExogenousRecord(ContractModel):
    """An externally initiated world event observed during the scored episode.

    Attributes:
        after_step: Index of the tool step (``outcome.steps``) it followed, or -1 when it
            occurred before the first scored step.
        owner: Who initiated it: the world (scheduled effects), a simulator or the user.
        content: Exact recorded event.
        source: Locator of the record it was taken from.
    """

    after_step: int = Field(ge=-1)
    owner: Literal["world", "simulator", "user"] = "world"
    content: JsonObject
    source: str = Field(min_length=1)


class _Log:
    """Append-only event builder that numbers events and tracks the latest state."""

    def __init__(self) -> None:
        """Start an empty log."""
        self.events: list[EpisodeEvent] = []
        self.missing: list[str] = []
        self.state: JsonValue | None = None

    def add(
        self,
        event_id: str,
        kind: EventKind,
        owner: EventOwner,
        scope: EventScope,
        content: JsonValue,
        source: str,
    ) -> None:
        """Append one event at the next sequence position."""
        self.events.append(
            EpisodeEvent(
                event_id=event_id,
                sequence=len(self.events),
                kind=kind,
                owner=owner,
                scope=scope,
                content=content,
                source=source,
            )
        )

    def snapshot(
        self, event_id: str, state: JsonValue | None, source: str, *, scope: EventScope = "scored"
    ) -> None:
        """Record an evaluator-only state projection when it differs from the last one."""
        if state is not None and state != self.state:
            self.add(event_id, "state_snapshot", "world", scope, state, source)
            self.state = state

    def evidence(self, *, episode_id: str, task_sha256: str, world_sha256: str) -> EpisodeEvidence:
        """Freeze the log; it is complete exactly when nothing is known to be missing."""
        return EpisodeEvidence(
            episode_id=episode_id,
            task_sha256=task_sha256,
            world_sha256=world_sha256,
            events=tuple(self.events),
            complete=not self.missing,
            missing=tuple(self.missing),
        )


def _prefix(log: _Log, checkpoint: ContinuationCheckpoint, source: str) -> None:
    """Record the retained actor-visible prefix as context, with owners taken from roles."""
    for index, message in enumerate(checkpoint.actor.messages):
        _message(log, index, message, f"{source}#actor.messages[{index}]")


def _message(log: _Log, index: int, message: ModelMessage, source: str) -> None:
    """Map one prefix message: instructions, user text, actor text and calls, tool replies."""
    identity = f"prefix.{index}"
    if message.role in ("system", "developer"):
        content: JsonObject = {"role": message.role, "content": message.content}
        log.add(identity, "harness_note", "harness", "context", content, source)
    elif message.role == "user":
        log.add(identity, "user_message", "user", "context", message.content, source)
    elif message.role == "tool":
        # A chat checkpoint keeps the reply text but not its error flag; None says so.
        reply: JsonObject = {
            "tool_call_id": message.tool_call_id,
            "content": message.content,
            "is_error": None,
        }
        log.add(identity, "tool_result", "world", "context", reply, source)
    else:
        action = message.assistant_action
        text = message.content if action is None else action.content
        if text:
            log.add(identity, "assistant_message", "actor", "context", text, source)
        for position, call in enumerate(() if action is None else action.tool_calls):
            log.add(
                f"{identity}.call.{position}",
                "tool_call",
                "actor",
                "context",
                call.model_dump(mode="json"),
                f"{source}.assistant_action.tool_calls[{position}]",
            )


def _step(log: _Log, step: JsonValue, identity: str, call: ToolCall, source: str) -> None:
    """Record one dispatched step: receipt, observed reply or failure, and resulting state."""
    if not isinstance(step, dict):
        raise ValueError("continuation step is not a JSON object")
    action = step.get("action")
    if not isinstance(action, dict) or action.get("call_id") != call.call_id:
        raise ValueError("continuation step does not belong to the actor's tool call")
    receipt = step.get("execution")
    if receipt is not None:
        log.add(
            identity + ".receipt", "receipt", "harness", "scored", receipt, source + ".execution"
        )
    status = step.get("status")
    if status == "valid":
        observation = Observation.model_validate(step.get("reply"))
        log.add(
            identity + ".reply",
            "tool_error" if observation.is_error else "tool_result",
            "world",
            "scored",
            {"tool_call_id": call.call_id, **observation.model_dump(mode="json")},
            source + ".reply",
        )
        log.snapshot(identity + ".state", step.get("after"), source + ".after")
    elif status == "blocked":
        note: JsonObject = {
            "tool_call_id": call.call_id,
            "status": "blocked",
            "dispatch": step.get("dispatch"),
            "reason": step.get("reason"),
        }
        log.add(identity + ".blocked", "harness_note", "harness", "scored", note, source)
    else:
        effects = receipt.get("effects") if isinstance(receipt, dict) else None
        note = {
            "tool_call_id": call.call_id,
            "status": "failure",
            "phase": step.get("phase"),
            "effects": effects,
        }
        log.add(identity + ".failure", "harness_note", "harness", "scored", note, source)
        log.missing.append(
            f"tool call {call.call_id} returned no observation (effects: {effects or 'unknown'})"
        )


def _exogenous(
    log: _Log, records: Sequence[ExogenousRecord], after_step: int, emitted: set[int]
) -> None:
    """Record exogenous events that followed one step, in their supplied order."""
    for position, record in enumerate(records):
        if record.after_step == after_step and position not in emitted:
            emitted.add(position)
            log.add(
                f"exogenous.{position}",
                "exogenous_event",
                record.owner,
                "scored",
                record.content,
                record.source,
            )


def _branch(
    log: _Log,
    branch: ContinuationBranch,
    source: str,
    exogenous: Sequence[ExogenousRecord],
    *,
    initial: bool,
) -> None:
    """Record a branch's scored window from its saved actor calls and steps.

    Args:
        log: Log already holding the context prefix.
        branch: Saved branch.
        source: Locator of the branch record.
        exogenous: Externally initiated events placed after their step.
        initial: Whether to record the branch's initial state as context.
    """
    if branch.cascade_decisions:
        raise ValueError(
            "cascade branches hold decider-authored steps their saved actor calls omit; "
            "judge an actor-only branch"
        )
    outcome = branch.outcome
    if initial:
        log.snapshot(
            "state.initial",
            outcome.get("initial_state"),
            source + "#outcome.initial_state",
            scope="context",
        )
    calls = outcome.get("model_calls", [])
    steps = outcome.get("steps", [])
    if not isinstance(calls, list) or not isinstance(steps, list):
        raise ValueError("continuation outcome has malformed calls or steps")
    emitted: set[int] = set()
    _exogenous(log, exogenous, -1, emitted)
    taken = 0
    for index, row in enumerate(calls):
        locator = f"{source}#outcome.model_calls[{index}].response"
        raw = row.get("response") if isinstance(row, dict) else None
        if raw is None:
            log.add(
                f"call.{index}.unanswered",
                "harness_note",
                "harness",
                "scored",
                {"actor_call": index, "status": "no response recorded"},
                f"{source}#outcome.model_calls[{index}]",
            )
            log.missing.append(f"actor call {index} has no recorded response")
            continue
        response = ModelResponse.model_validate(raw)
        output = response.output
        if output.content:
            final = not output.tool_calls
            log.add(
                f"call.{index}.{'final' if final else 'text'}",
                "final_answer" if final else "assistant_message",
                "actor",
                "scored",
                output.content,
                locator + ".output.content",
            )
        stopped = response.finish_reason == ModelFinishReason.LENGTH
        if stopped:
            log.add(
                f"call.{index}.length",
                "harness_note",
                "harness",
                "scored",
                {"actor_call": index, "finish_reason": "length", "tool_calls_dispatched": False},
                locator + ".finish_reason",
            )
        for position, call in enumerate(output.tool_calls):
            identity = f"call.{index}.tool.{position}"
            log.add(
                identity,
                "tool_call",
                "actor",
                "scored",
                call.model_dump(mode="json"),
                f"{locator}.output.tool_calls[{position}]",
            )
            if stopped:
                continue
            if taken == len(steps):
                log.missing.append(f"tool call {call.call_id} has no recorded step or reply")
                continue
            _step(log, steps[taken], identity, call, f"{source}#outcome.steps[{taken}]")
            _exogenous(log, exogenous, taken, emitted)
            taken += 1
    if taken != len(steps):
        raise ValueError("continuation steps do not match the actor's saved tool calls")
    if len(emitted) != len(exogenous):
        raise ValueError("an exogenous event follows a step this branch does not hold")
    log.snapshot("state.final", outcome.get("final_state"), source + "#outcome.final_state")
    stop: JsonObject = {
        "status": branch.status,
        "stop_reason": outcome.get("stop_reason"),
        "phase": outcome.get("execution_phase"),
        "actor_stop": branch.actor_stop,
        "failure": branch.failure,
    }
    log.add("stop", "harness_note", "harness", "scored", stop, source + "#status")
    if branch.status != "completed":
        log.missing.append(
            f"branch ended {branch.status} at phase {outcome.get('execution_phase')}; later actor "
            "output and world effects were not recorded"
        )
    elif "final_state" not in outcome:
        log.missing.append("completed branch has no final state projection")


def evidence_from_branch(
    branch: ContinuationBranch,
    checkpoint: ContinuationCheckpoint,
    *,
    episode_id: str,
    task_sha256: str,
    world_sha256: str,
    checkpoint_source: str,
    branch_source: str,
    exogenous: Sequence[ExogenousRecord] = (),
) -> EpisodeEvidence:
    """Convert one saved continuation branch and its checkpoint prefix into episode evidence.

    Args:
        branch: Saved branch of a continuation study started at ``checkpoint``.
        checkpoint: The study's starting checkpoint; its actor messages are the retained prefix.
        episode_id: Stable episode identity, typically the branch record id.
        task_sha256: Digest of the exact task the judge definition binds.
        world_sha256: Digest of the exact frozen world the judge definition binds.
        checkpoint_source: Locator of the checkpoint artifact.
        branch_source: Locator of the branch record.
        exogenous: Externally initiated world events, each placed after its step.

    Returns:
        Context prefix and initial state, then every scored event in order.

    Raises:
        ValueError: The branch's steps and actor calls disagree, it is a cascade branch, or an
            exogenous event names a step the branch does not hold.
    """
    log = _Log()
    _prefix(log, checkpoint, checkpoint_source)
    _branch(log, branch, branch_source, exogenous, initial=True)
    return log.evidence(episode_id=episode_id, task_sha256=task_sha256, world_sha256=world_sha256)


def evidence_from_contrast_cell(
    cell: ContrastCell,
    spec: ActionContrastSpec,
    checkpoint: ContinuationCheckpoint,
    *,
    intervention: ActionIntervention | None,
    suffix: ContinuationBranch | None,
    episode_id: str,
    task_sha256: str,
    world_sha256: str,
    contrast_source: str,
    checkpoint_source: str,
    intervention_source: str | None = None,
    suffix_source: str | None = None,
    exogenous: Sequence[ExogenousRecord] = (),
) -> EpisodeEvidence:
    """Convert one action-contrast cell: imposed action plus suffix, judged together.

    The imposed action is ``scored`` and owned by ``experiment``, so an unsafe imposed action
    cannot disappear behind a clean suffix and is never mistaken for an actor-sampled decision.

    Args:
        cell: Saved contrast cell.
        spec: The contrast's registered spec, naming the imposed action for the cell's arm.
        checkpoint: The original pre-action checkpoint; its messages are the context prefix.
        intervention: Saved imposed-action effect, None exactly when the cell has none.
        suffix: The suffix study's single branch, None exactly when the cell has no study.
        episode_id: Stable episode identity, typically the cell record id.
        task_sha256: Digest of the exact task the judge definition binds.
        world_sha256: Digest of the exact frozen world the judge definition binds.
        contrast_source: Locator of the contrast registration.
        checkpoint_source: Locator of the original checkpoint artifact.
        intervention_source: Locator of the intervention record, required with one.
        suffix_source: Locator of the suffix branch record, required with one.
        exogenous: Externally initiated events during the suffix, each placed after its step.

    Returns:
        Context prefix and pre-action state, the imposed action and its reply, then the suffix.

    Raises:
        ValueError: The supplied records do not belong to this cell or disagree.
    """
    if (cell.intervention is None) != (intervention is None) or (cell.study is None) != (
        suffix is None
    ):
        raise ValueError("supplied intervention or suffix does not match the saved cell")
    action = spec.baseline if cell.arm == "baseline" else spec.alternative
    log = _Log()
    _prefix(log, checkpoint, checkpoint_source)
    if intervention is not None:
        if intervention.spec.action != action or intervention_source is None:
            raise ValueError("intervention imposed a different action or has no source")
        log.snapshot(
            "imposed.before",
            intervention.before,
            intervention_source + "#before",
            scope="context",
        )
    log.add(
        "imposed.call",
        "tool_call",
        "experiment",
        "scored",
        action.model_dump(mode="json"),
        f"{contrast_source}#spec.{cell.arm}",
    )
    if intervention is None or intervention_source is None:
        log.missing.append(f"imposed action outcome is unknown ({cell.failure or 'unrecorded'})")
    else:
        log.add(
            "imposed.reply",
            "tool_error" if intervention.observation.is_error else "tool_result",
            "world",
            "scored",
            {"tool_call_id": action.call_id, **intervention.observation.model_dump(mode="json")},
            intervention_source + "#observation",
        )
        log.snapshot("imposed.state", intervention.after, intervention_source + "#after")
    if suffix is None or suffix_source is None:
        log.missing.append(f"suffix after the imposed action is unavailable ({cell.failure})")
    else:
        initial = suffix.outcome.get("initial_state")
        if intervention is not None and initial is not None and initial != intervention.after:
            raise ValueError("suffix did not resume the imposed action's state")
        _branch(log, suffix, suffix_source, exogenous, initial=False)
    return log.evidence(episode_id=episode_id, task_sha256=task_sha256, world_sha256=world_sha256)


def ledger_evidence(
    entries: Sequence[TrialLedgerEntry],
    checkpoint: ContinuationCheckpoint,
    *,
    episode_id: str,
    task_sha256: str,
    world_sha256: str,
    ledger_source: str,
    checkpoint_source: str,
) -> EpisodeEvidence:
    """Convert a closed Harbor trial's ledger into episode evidence.

    Read the ledger after the trial closes. A ledger without a submission is a complete record
    that the actor never submitted, noted as a scored harness event; a tool request without a
    recorded reply, or a failed dispatch, is missing evidence.

    Args:
        entries: The trial's ledger (``trial_ledger``), starting with its ``start`` entry.
        checkpoint: The executable task's starting checkpoint; its messages are the prefix.
        episode_id: Stable episode identity, typically the trial identity.
        task_sha256: Digest of the exact task the judge definition binds.
        world_sha256: Digest of the exact frozen world the judge definition binds.
        ledger_source: Locator prefix of the ledger records (the trial identity).
        checkpoint_source: Locator of the checkpoint artifact.

    Returns:
        Context prefix and initial state, then every ledger entry in sequence order.

    Raises:
        ValueError: The ledger is out of order, lacks its start entry, or answers a call twice.
    """
    if [entry.sequence for entry in entries] != list(range(len(entries))) or not entries:
        raise ValueError("trial ledger must be the contiguous saved sequence")
    if entries[0].kind != "start":
        raise ValueError("trial ledger must begin with its start entry")
    log = _Log()
    _prefix(log, checkpoint, checkpoint_source)
    pending: dict[str, ToolCall] = {}
    submitted = False
    for entry in entries:
        identity, source = f"ledger.{entry.sequence}", f"{ledger_source}-ledger-{entry.sequence}"
        content = entry.content
        if entry.kind == "start":
            initial = content.get("initial_state")
            log.snapshot(identity, initial, source + "#initial_state", scope="context")
        elif entry.kind == "tool_call":
            call = ToolCall.model_validate(content.get("action"))
            pending[call.call_id] = call
            log.add(identity, "tool_call", "actor", "scored", call.model_dump(mode="json"), source)
        elif entry.kind in ("tool_reply", "tool_failure"):
            step = content.get("step")
            action = step.get("action") if isinstance(step, dict) else None
            key = action.get("call_id") if isinstance(action, dict) else None
            if not isinstance(key, str) or key not in pending:
                raise ValueError("trial ledger answers a tool call it did not record")
            _step(log, step, identity, pending.pop(key), source + "#step")
        elif entry.kind == "message":
            log.add(identity, "assistant_message", "actor", "scored", content.get("text"), source)
        else:
            submitted = True
            answer = content.get("answer")
            log.add(identity, "final_answer", "actor", "scored", answer, source + "#answer")
            log.snapshot(identity + ".state", content.get("final_state"), source + "#final_state")
    for call_id in pending:
        log.missing.append(f"tool call {call_id} has no recorded reply or failure")
    if not submitted:
        log.add(
            "ledger.end",
            "harness_note",
            "harness",
            "scored",
            {"submitted": False, "entries": len(entries)},
            ledger_source,
        )
    return log.evidence(episode_id=episode_id, task_sha256=task_sha256, world_sha256=world_sha256)
