"""One guarded commit path for prepared engine proposals and durable actor observations."""

import copy
import json
from collections.abc import Callable, Sequence
from datetime import UTC, datetime
from functools import partial
from operator import not_
from pathlib import Path

from pydantic import AwareDatetime, Field

from exp.common.core.artifacts import (
    ArtifactEnvelope,
    ArtifactInput,
    ContractModel,
    JsonObject,
    JsonValue,
    Sha256,
    sha256_json,
    stable_id,
)
from exp.common.core.locks import CoordinationLock
from exp.common.models import ToolCall
from exp.common.models.client import StructuredRequestPreflight
from exp.common.models.unknown_outcome import resume_unknown
from exp.common.project import ArtifactStore, artifact_input
from exp.runtime.environments import Observation
from exp.runtime.models.call_evidence import call_consumer
from exp.runtime.sessions.edits import AppendValue, CopyValue, StateEdit, prefixed
from exp.runtime.sessions.errors import (
    ProviderBudgetExhaustedError,
    SessionNotFoundError,
    StepAmbiguousError,
    StepInFlightError,
)
from exp.runtime.sessions.projection import StateRef
from exp.runtime.sessions.records import SessionBinding, SessionRecord, StepStatus
from exp.runtime.sessions.store import SessionStore
from exp.simulation.engines.reasks import reask_proposal
from exp.simulation.engines.reconstructed import InformationChannels, ReconstructedTextEngine
from exp.simulation.engines.transition import Completion, PreparedPrediction, TransitionEngine
from exp.simulation.reconstruction.models import (
    InitializationArtifact,
    ReconstructionBundle,
    validate_state,
)
from exp.simulation.retrieval.selection import EvidencePolicy
from exp.simulation.world_model.resumption import Resumption
from exp.simulation.world_model.session_research import SessionResearch


class SessionPolicy(ContractModel):
    """Immutable execution limits for one candidate-owned world.

    Attributes:
        deadline: Latest guarded commit decision for this session.
        maximum_operations: Maximum dispatched logical operations, including failures.
        inference_identity: Exact model/route/output/budget configuration digest.
        code_revision: Source revision owning immutable input and proposal artifacts.
        inspection_policy_sha256: Explicit public-reader policy when inspection is enabled.
        maximum_reasks: Times a refused proposal is asked again, with the refusal reason, in
            the same operation. Refused output is never repaired; each re-ask is a new,
            separately saved and metered request. Omitted when zero.
    """

    deadline: AwareDatetime
    maximum_operations: int = Field(gt=0, strict=True)
    maximum_reasks: int = Field(default=0, ge=0, le=3, strict=True, exclude_if=not_)
    inspection_policy_sha256: Sha256 | None = None
    inference_identity: Sha256
    code_revision: str = Field(min_length=1)


class SessionSnapshot(ContractModel):
    """Typed candidate envelope retained in the generic runtime store.

    Attributes:
        business_state: Current modeled world.
        history: Ordered public instance observations.
        transitions: Committed evaluator records, excluded from actor capabilities.
    """

    business_state: JsonObject
    history: tuple[JsonObject, ...]
    transitions: tuple[JsonObject, ...]


class DurableWorldModelSession:
    """Coordinate proposals while runtime persistence owns identity and atomic publication."""

    def __init__(
        self,
        *,
        engine: TransitionEngine,
        store: SessionStore,
        artifacts: ArtifactStore,
        session_id: str,
        binding: SessionBinding,
        initial_state: JsonObject,
        public_prefix: tuple[JsonObject, ...],
        policy: SessionPolicy,
        complete: Completion,
        lock_path: Path,
        validate: Callable[[JsonObject], None],
        clock: Callable[[], datetime] | None = None,
        evaluation_context: JsonObject | None = None,
        research: SessionResearch | None = None,
        resumption: Resumption | None = None,
        preparation_dispatches: bool = True,
    ) -> None:
        """Create or reopen the exact same world; no reference state is accepted.

        ``resumption`` is the explicit authority to continue a step a stopped process left
        open; without it such a step blocks the session. ``preparation_dispatches`` says
        whether preparing a step can send requests (query embeddings or memory-recall
        decisions), so re-preparing an interrupted step is recorded as a retrieval retry.
        """
        if binding.deadline != policy.deadline:
            raise ValueError("session binding and policy deadline differ")
        self._engine, self._store, self._artifacts = engine, store, artifacts
        self._research = research
        self._resumption = resumption
        self._preparation_dispatches = preparation_dispatches
        self._evaluation_context = copy.deepcopy(evaluation_context or {})
        self._engine_identity = engine.identity
        self._binding, self._policy, self._complete = binding, policy, complete
        self._validate, self._clock = validate, clock or _utc_now
        self._lock = CoordinationLock(lock_path, timeout=0)
        self.session_id = session_id
        self.initial_state = copy.deepcopy(initial_state)
        validate(initial_state)
        with self._lock:
            try:
                record = store.read_session(session_id)
            except SessionNotFoundError:
                store.create_session(
                    session_id,
                    binding,
                    {
                        "business_state": initial_state,
                        "history": list(public_prefix),
                        "transitions": [],
                    },
                )
            else:
                if record.binding != binding:
                    raise ValueError("existing session belongs to different knowledge or policy")

    def observe_reply_feedback(
        self, operation_id: str, observed_reply: JsonValue, *, source: str
    ) -> JsonObject:
        """Investigate disclosed past feedback without changing state or earlier predictions."""
        if self._research is None:
            raise ValueError("session was not opened with an online research policy")
        return self._research.observe(operation_id, observed_reply, source)

    def observe_all_reply_feedback(
        self, feedback: Sequence[tuple[str, JsonValue]], *, source: str
    ) -> list[JsonObject | Exception]:
        """Disclose every reply after the whole trajectory is predicted (pinned delivery)."""
        if self._research is None:
            raise ValueError("session was not opened with an online research policy")
        return self._research.observe_all(feedback, source)

    @property
    def evaluation_context(self) -> JsonObject:
        """Return source and contract provenance only to the evaluation coordinator."""
        return copy.deepcopy(self._evaluation_context)

    @property
    def state(self) -> JsonObject:
        """Return a detached evaluator-visible state, never an actor capability."""
        return _business_state(self._store, self._store.read_head(self.session_id).state)

    @property
    def transitions(self) -> tuple[JsonObject, ...]:
        """Read committed operations with their ordered observations and evidence receipts."""
        return _objects(
            self._store.read_state(self._store.read_head(self.session_id).state, ("transitions",))
        )

    def committed_record(self) -> SessionRecord:
        """Return a detached authoritative record for provider-free checkpoint projection."""
        return self._store.read_session(self.session_id).model_copy(deep=True)

    def retain_history(self, owner_id: str, end_revision: int) -> None:
        """Durably retain committed revisions ``[0, end_revision)`` for a named owner.

        Args:
            owner_id: Checkpoint or evaluation identity that owns the retained history.
            end_revision: First revision after the retained range.
        """
        self._store.pin_revisions(owner_id, self.session_id, 0, end_revision)

    def recorded_proposal(self, operation_id: str) -> JsonObject | None:
        """Return one saved simulator proposal without changing its operation status."""
        return self._store.proposal(self.session_id, operation_id)

    def has_recorded_proposal(self, operation_id: str) -> bool:
        """Whether one logical operation reached durably recorded simulator output.

        Args:
            operation_id: Caller-owned logical operation identity.

        Returns:
            True after output was recorded, including rejected proposals and failed commits.
            False does not identify the cause of a preparation or transport failure.
        """
        return self._store.proposal(self.session_id, operation_id) is not None

    def outcome_unknown(self, operation_id: str) -> bool:
        """Whether one logical operation was left open after dispatch with no known outcome.

        A failed step that never reached the provider is rejected or released, so a step
        still open after its failure was dispatched and its provider outcome is unknown.

        Args:
            operation_id: Caller-owned logical operation identity.

        Returns:
            True while the operation's step remains open or ambiguous.
        """
        return operation_id in self._store.open_steps(self.session_id)

    def execute(self, action: ToolCall) -> Observation:
        """Execute one tool call using its caller-owned logical operation identity."""
        return self.step((action,), operation_id=action.call_id)[0]

    def step(self, actions: tuple[ToolCall, ...], *, operation_id: str) -> tuple[Observation, ...]:
        """Prepare, persist, dispatch, validate and publish at most one revision.

        With a resumption, a step a stopped process left open continues: its saved input and
        every saved proposal are reused without a provider call, refused proposals keep the
        bounded re-ask rule, and a request without a saved response is re-dispatched only
        after it is recorded as an unknown-outcome retry.

        Args:
            actions: Ordered tool calls; each engine declares supported batch semantics.
            operation_id: Stable logical identity. Identical actions with new IDs are new work.

        Returns:
            Exact saved observations on replay, without new retrieval or provider work.
        """
        if not actions or len({action.call_id for action in actions}) != len(actions):
            raise ValueError("ordered actions require distinct nonempty tool call IDs")
        if self._engine.identity != self._engine_identity:
            raise ValueError("frozen engine or evidence policy changed")
        fingerprint = sha256_json(
            {
                "actions": [a.model_dump(mode="json") for a in actions],
                "binding": self._binding.model_dump(mode="json"),
            }
        )
        with self._lock:
            current = self._store.read_head(self.session_id)
            left: StepStatus | None = None
            try:
                decision = self._store.begin_step(
                    self.session_id, operation_id, fingerprint, current.version, self._binding
                )
            except (StepInFlightError, StepAmbiguousError):
                if self._resumption is None:
                    raise
                claim, left = self._store.reclaim_step(
                    self.session_id, operation_id, fingerprint, self._binding
                )
            else:
                if decision.completed is not None:
                    return _observations(decision.completed.response)
                assert decision.claim is not None
                claim = decision.claim
            state = _business_state(self._store, current.state)
            history = _objects(self._store.read_state(current.state, ("history",)))
            scope = stable_id(
                "world-operation",
                {
                    "session": self.session_id,
                    "operation": operation_id,
                    "binding": self._binding.model_dump(mode="json"),
                },
            )
            saved_input = None if left is None else self._saved(scope + "-input")
            if saved_input is not None:
                prepared = PreparedPrediction.model_validate(saved_input[0]["prepared"])
                intent = saved_input[1]
            else:
                if left not in (None, StepStatus.CLAIMED):
                    raise ValueError("an interrupted dispatched step has no saved input")
                if left is not None and self._resumption and self._preparation_dispatches:
                    self._resumption.retried("retrieval", operation_id, scope + "-preparation")
                try:
                    if self._clock() >= self._deadline():
                        raise TimeoutError("session deadline reached before preparation")
                    prepared = self._engine.prepare(copy.deepcopy(state), tuple(history), actions)
                    if isinstance(self._complete, StructuredRequestPreflight):
                        self._complete.preflight(prepared.body)
                    intent = self._save(
                        scope + "-input",
                        {
                            "prepared": prepared.model_dump(mode="json"),
                            "base_revision": current.version,
                            "logical_scope": scope,
                        },
                    )
                    if self._clock() >= self._deadline():
                        raise TimeoutError("session deadline reached during preparation")
                except Exception:  # noqa: BLE001 - preparation failure consumes no provider call.
                    self._store.reject_step(claim, "preparation_refused")
                    raise
            if left in (None, StepStatus.CLAIMED):
                try:
                    self._store.mark_provider_started(claim, self._policy.maximum_operations)
                except ProviderBudgetExhaustedError:
                    self._store.release_step(claim)
                    raise
            retried = (
                None
                if self._resumption is None
                else self._resumption.recorder("prediction", operation_id)
            )
            body, logical, refusals = prepared.body, scope, []
            while True:
                saved = None if left is None else self._saved(logical + "-proposal")
                if saved is not None:
                    raw = _proposal(saved[0])
                    proposal_input = saved[1]
                    self._store.record_proposal(claim, raw)
                else:
                    try:
                        with call_consumer("simulator"):
                            raw = resume_unknown(partial(self._complete, body), logical, retried)
                    except Exception as error:  # noqa: BLE001 - dispatched failures remain ambiguous.
                        self._store.mark_ambiguous(claim, type(error).__name__)
                        raise
                    self._store.record_proposal(claim, raw)
                    proposal_input = self._save(logical + "-proposal", {"proposal": raw}, (intent,))
                try:
                    proposal = self._engine.parse(raw, actions, current_state=copy.deepcopy(state))
                    next_state, state_edits = proposal.state_edits(state)
                    self._validate(next_state)
                    observations = []
                    for row in proposal.observations:
                        if row.encoding == "text":
                            if not isinstance(row.reply, str):
                                raise ValueError("text observations require exact string replies")
                            content = row.reply
                        else:
                            content = json.dumps(
                                row.reply,
                                ensure_ascii=False,
                                separators=(",", ":"),
                                allow_nan=False,
                            )
                        observations.append(
                            Observation(
                                content=content,
                                is_error=row.is_error,
                                metadata={"call_id": row.call_id},
                            )
                        )
                    if self._engine.identity != self._engine_identity:
                        raise ValueError("engine identity changed during prediction")
                    break
                except Exception as error:  # noqa: BLE001 - invalid proposals leave state unchanged.
                    if len(refusals) >= self._policy.maximum_reasks:
                        self._store.reject_step(claim, "proposal_refused")
                        raise
                    reason = f"{type(error).__name__}: {error}"[:2000]
                    refusals.append(
                        {"proposal": proposal_input.model_dump(mode="json"), "reason": reason}
                    )
                    body = reask_proposal(body, raw, reason)
                    logical = f"{scope}-reask-{len(refusals)}"
                    intent = self._save(
                        logical + "-input",
                        {"request": body, "refused": refusals[-1], "logical_scope": logical},
                        (proposal_input,),
                    )
            response: JsonObject = {
                "observations": [row.model_dump(mode="json") for row in observations]
            }
            transition: JsonObject = {
                "operation_id": operation_id,
                "actions": [a.model_dump(mode="json") for a in actions],
                "replies": [row.reply for row in proposal.observations],
                "before": None,
                "after": None,
                "observations": response["observations"],
                "evidence": {**prepared.evidence, "refused_proposals": refusals}
                if refusals
                else prepared.evidence,
                "proposal": proposal_input.model_dump(mode="json"),
            }
            self._store.complete_step(
                claim,
                _commit_edits(current.version, transition, state_edits),
                response,
                window=None if self._resumption is None else self._resumption.window,
            )
            return tuple(observations)

    def _deadline(self) -> datetime:
        """The registered deadline, extended by an explicit resumption window."""
        if self._resumption is None:
            return self._policy.deadline
        return self._resumption.deadline(self._policy.deadline)

    def _saved(self, identity: str) -> tuple[JsonObject, ArtifactInput] | None:
        """A saved operation record and its reference, when an earlier attempt wrote it."""
        if not self._artifacts.exists(identity):
            return None
        record = json.loads(self._artifacts.read_bytes(identity, "record.json"))
        if not isinstance(record, dict):
            raise ValueError(f"saved operation record {identity} is not an object")
        return record, artifact_input(self._artifacts.read(identity).manifest)

    def _save(
        self, identity: str, payload: JsonObject, inputs: tuple[ArtifactInput, ...] = ()
    ) -> ArtifactInput:
        """Publish immutable evidence before referring to it from a state transaction.

        A resumed step reuses the identical record an earlier attempt saved; a different one
        means the step's inputs changed, and the step stops instead of overwriting history.
        """
        if self._resumption is not None and (saved := self._saved(identity)) is not None:
            if sha256_json(saved[0]) != sha256_json(payload):
                raise ValueError(f"saved {identity} differs from the resumed step's input")
            return saved[1]
        self._artifacts.write_json(
            artifact_id=identity,
            artifact_type="world-operation",
            envelope=ArtifactEnvelope(
                schema_version=1,
                created_at=self._clock(),
                inputs=inputs,
                code_revision=self._policy.code_revision,
            ),
            files={"record.json": payload},
        )
        return artifact_input(self._artifacts.read(identity).manifest)


def _commit_edits(
    revision: int, transition: JsonObject, state_edits: Sequence[StateEdit]
) -> tuple[StateEdit, ...]:
    """Translate one validated transition into structural edits of the session envelope.

    The envelope keeps ``business_state``, ``history`` and ``transitions`` in that order, and
    every committed revision appends exactly one transition, so the new transition's index is
    the base revision. Its ``before`` and ``after`` states are structural copies of the business
    state around the proposal's own edits, so the store shares their unchanged subtrees.

    Args:
        revision: Base revision the step was claimed against.
        transition: Transition record with ``before`` and ``after`` placeholders.
        state_edits: The proposal's edits relative to the business state.

    Returns:
        Ordered edits that publish the business state, the history entry and the transition.
    """
    record = ("transitions", revision)
    return (
        AppendValue(path=("transitions",), value=transition),
        CopyValue(path=(*record, "before"), source=("business_state",)),
        *prefixed(("business_state",), state_edits),
        CopyValue(path=(*record, "after"), source=("business_state",)),
        AppendValue(
            path=("history",),
            value={"actions": transition["actions"], "observations": transition["observations"]},
        ),
    )


def _business_state(store: SessionStore, ref: StateRef) -> JsonObject:
    """Materialize only the business state of one committed envelope revision."""
    state = store.read_state(ref, ("business_state",))
    if not isinstance(state, dict):
        raise ValueError("committed session business state is not an object")
    return state


def _objects(value: JsonValue) -> tuple[JsonObject, ...]:
    """Narrow one stored envelope list of objects, refusing any other shape."""
    if not isinstance(value, list) or not all(isinstance(row, dict) for row in value):
        raise ValueError("committed session history or transitions are not a list of objects")
    return tuple(row for row in value if isinstance(row, dict))


def _proposal(record: JsonObject) -> JsonObject:
    """The raw simulator output an earlier attempt saved for one logical request."""
    raw = record.get("proposal")
    if not isinstance(raw, dict):
        raise ValueError("saved simulator proposal is not an object")
    return raw


def _observations(response: JsonObject) -> tuple[Observation, ...]:
    """Decode the exact observation inventory saved with an earlier commit."""
    values = response["observations"]
    if not isinstance(values, list):
        raise ValueError("stored observation inventory is invalid")
    return tuple(Observation.model_validate(row) for row in values)


def _utc_now() -> datetime:
    """Return the aware session dispatch clock."""
    return datetime.now(UTC)


def open_reconstructed_session(
    *,
    bundle: ReconstructionBundle,
    initializer: InitializationArtifact,
    store: SessionStore,
    artifacts: ArtifactStore,
    session_id: str,
    policy: SessionPolicy,
    complete: Completion,
    lock_path: Path,
    selection: EvidencePolicy | None = None,
    channels: InformationChannels | None = None,
    observation_schema: JsonObject | None = None,
    representation: str = "text",
    clock: Callable[[], datetime] | None = None,
) -> DurableWorldModelSession:
    """Open the supported reconstructed profile through the maintained world-model owner.

    Python and text/batch action profiles are not silently substituted. The generated
    Python source may be retained as an artifact; this factory never executes it.
    """
    if representation != "text":
        raise ValueError("generated Python requires a qualified executor; choose text explicitly")
    if getattr(complete, "inspection_policy_sha256", None) != policy.inspection_policy_sha256:
        raise ValueError("inspection wrapper and frozen session policy differ")
    if initializer.bundle_sha256 != bundle.digest:
        raise ValueError("initializer and behavioral artifact differ")
    engine = ReconstructedTextEngine(
        bundle,
        observation_schema=observation_schema,
        selection=selection,
        channels=channels,
        initial_coverage={
            "unknown_paths": list(initializer.unknown_paths),
            "complete_scopes": list(initializer.complete_scopes),
            "field_statuses": dict(initializer.field_statuses),
            "authority": "initial_public_observations_only",
        },
    )
    binding = SessionBinding(
        deadline=policy.deadline,
        target_id="reconstructed@text",
        initialization_receipt_sha256=initializer.digest,
        knowledge_sha256=bundle.digest,
        corpus_sha256=bundle.inputs_sha256,
        representation_sha256=engine.identity,
        execution_policy_sha256=sha256_json(policy.model_dump(mode="json")),
    )

    def validate(state: JsonObject) -> None:
        """Enforce declared state shape without promoting unknown facts to native truth."""
        validate_state(bundle.interface.state_schema, state, initializer.unknown_paths)

    return DurableWorldModelSession(
        engine=engine,
        store=store,
        artifacts=artifacts,
        session_id=session_id,
        binding=binding,
        initial_state=initializer.state,
        public_prefix=initializer.public_prefix,
        policy=policy,
        complete=complete,
        lock_path=lock_path,
        validate=validate,
        clock=clock,
        evaluation_context={
            "knowledge_sha256": bundle.digest,
            "task_context": {
                "initialization": {
                    "unknown_paths": list(initializer.unknown_paths),
                    "complete_scopes": list(initializer.complete_scopes),
                    "field_statuses": dict(initializer.field_statuses),
                }
            },
            "sources": bundle.sources.source_map(),
            "contract": bundle.interface.model_dump(mode="json"),
        },
    )
