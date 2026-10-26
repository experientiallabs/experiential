"""One guarded commit path for prepared engine proposals and durable actor observations."""

import copy
import json
from collections.abc import Callable, Sequence
from datetime import UTC, datetime
from operator import not_
from pathlib import Path

from filelock import FileLock
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
from exp.common.models import ToolCall
from exp.common.models.client import StructuredRequestPreflight
from exp.common.project import ArtifactStore, artifact_input
from exp.runtime.environments import Observation
from exp.runtime.sessions.store import (
    ProviderBudgetExhaustedError,
    SessionBinding,
    SessionNotFoundError,
    SessionRecord,
    SessionStore,
)
from exp.simulation.engines.reconstructed import InformationChannels, ReconstructedTextEngine
from exp.simulation.engines.transition import Completion, TransitionEngine
from exp.simulation.reconstruction.models import (
    InitializationArtifact,
    ReconstructionBundle,
    validate_state,
)
from exp.simulation.retrieval.selection import EvidencePolicy
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
    ) -> None:
        """Create or reopen the exact same world; no reference state is accepted."""
        if binding.deadline != policy.deadline:
            raise ValueError("session binding and policy deadline differ")
        self._engine, self._store, self._artifacts = engine, store, artifacts
        self._research = research
        self._evaluation_context = copy.deepcopy(evaluation_context or {})
        self._engine_identity = engine.identity
        self._binding, self._policy, self._complete = binding, policy, complete
        self._validate, self._clock = validate, clock or _utc_now
        self._lock = FileLock(lock_path, timeout=0)
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
        return copy.deepcopy(self._snapshot().business_state)

    @property
    def transitions(self) -> tuple[JsonObject, ...]:
        """Read committed operations with their ordered observations and evidence receipts."""
        return tuple(copy.deepcopy(self._snapshot().transitions))

    def committed_record(self) -> SessionRecord:
        """Return a detached authoritative record for provider-free checkpoint projection."""
        return self._store.read_session(self.session_id).model_copy(deep=True)

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

    def _snapshot(self) -> SessionSnapshot:
        """Read the current complete session envelope from the authoritative store."""
        return SessionSnapshot.model_validate(self._store.read_session(self.session_id).state)

    def execute(self, action: ToolCall) -> Observation:
        """Execute one tool call using its caller-owned logical operation identity."""
        return self.step((action,), operation_id=action.call_id)[0]

    def step(self, actions: tuple[ToolCall, ...], *, operation_id: str) -> tuple[Observation, ...]:
        """Prepare, persist, dispatch, validate and publish at most one revision.

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
            current = self._store.read_session(self.session_id)
            decision = self._store.begin_step(
                self.session_id, operation_id, fingerprint, current.version, self._binding
            )
            if decision.completed is not None:
                return _observations(decision.completed.response)
            claim = decision.claim
            assert claim is not None
            snapshot = SessionSnapshot.model_validate(current.state)
            state, history = snapshot.business_state, snapshot.history
            try:
                if self._clock() >= self._policy.deadline:
                    raise TimeoutError("session deadline reached before preparation")
                prepared = self._engine.prepare(copy.deepcopy(state), tuple(history), actions)
                if isinstance(self._complete, StructuredRequestPreflight):
                    self._complete.preflight(prepared.body)
                scope = stable_id(
                    "world-operation",
                    {
                        "session": self.session_id,
                        "operation": operation_id,
                        "binding": self._binding.model_dump(mode="json"),
                    },
                )
                intent = self._save(
                    scope + "-input",
                    {
                        "prepared": prepared.model_dump(mode="json"),
                        "base_revision": current.version,
                        "logical_scope": scope,
                    },
                )
                if self._clock() >= self._policy.deadline:
                    raise TimeoutError("session deadline reached during preparation")
            except Exception:  # noqa: BLE001 - preparation failure consumes no provider call.
                self._store.reject_step(claim, "preparation_refused")
                raise
            try:
                self._store.mark_provider_started(claim, self._policy.maximum_operations)
            except ProviderBudgetExhaustedError:
                self._store.release_step(claim)
                raise
            body, logical, refusals = prepared.body, scope, []
            while True:
                try:
                    raw = self._complete(body, logical)
                except Exception as error:  # noqa: BLE001 - dispatched failures remain ambiguous.
                    self._store.mark_ambiguous(claim, type(error).__name__)
                    raise
                self._store.record_proposal(claim, raw)
                proposal_input = self._save(logical + "-proposal", {"proposal": raw}, (intent,))
                try:
                    proposal = self._engine.parse(raw, actions, current_state=copy.deepcopy(state))
                    next_state = proposal.next_state(state)
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
                    body = _reask(body, raw, reason)
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
                "before": state,
                "after": next_state,
                "observations": response["observations"],
                "evidence": {**prepared.evidence, "refused_proposals": refusals}
                if refusals
                else prepared.evidence,
                "proposal": proposal_input.model_dump(mode="json"),
            }
            self._store.complete_step(
                claim,
                {
                    "business_state": next_state,
                    "history": [
                        *history,
                        {
                            "actions": transition["actions"],
                            "observations": response["observations"],
                        },
                    ],
                    "transitions": [*snapshot.transitions, transition],
                },
                response,
            )
            return tuple(observations)

    def _save(
        self, identity: str, payload: JsonObject, inputs: tuple[ArtifactInput, ...] = ()
    ) -> ArtifactInput:
        """Publish immutable evidence before referring to it from a state transaction."""
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


def _reask(body: JsonObject, raw: JsonObject, reason: str) -> JsonObject:
    """The same request plus the refused output and why it was refused; nothing is repaired."""
    messages = body.get("messages")
    if not isinstance(messages, list):
        raise ValueError("a re-ask needs the original request messages")
    output = raw.get("output")
    calls = output.get("tool_calls") if isinstance(output, dict) else None
    text = output.get("text") if isinstance(output, dict) else None
    refused = (
        json.dumps(calls, ensure_ascii=False)
        if calls
        else str(text or json.dumps(raw, ensure_ascii=False))
    )
    return {
        **body,
        "messages": [
            *messages,
            {"role": "assistant", "content": refused[:8000]},
            {
                "role": "user",
                "content": f"That proposal was refused: {reason} Return one complete, valid "
                "proposal for the same actions, following the required format exactly.",
            },
        ],
    }


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
