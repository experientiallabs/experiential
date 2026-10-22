"""Run a frozen grounded predictor as the simulated owner in the existing world state machine.

The world kernel owns time, scheduling, ownership and rollback. This adapter proposes one tool
reply and state edit, retains the raw response, and never executes a native tool or learns from
an evaluator. Restoring a checkpoint restores state and predictor history, not a future model draw.

Instance facts are bound by the engine, not stored as state: every prediction receives the
engine's conditioning facts for its state version in the separate ``instance_facts`` channel,
exactly as fidelity replay delivers them. They are never a writable domain, never part of a
checkpoint image, and they change the frozen definition only when present.
"""

import json
from collections.abc import Callable
from copy import deepcopy
from typing import Literal
from uuid import uuid4

from exp.common.core.artifacts import (
    ContractModel,
    JsonObject,
    Sha256,
    canonical_json_bytes,
    redact_secret_text,
    sha256_json,
)
from exp.common.models import AssistantAction, ModelMessage, ToolCall
from exp.common.tasks import TaskCase
from exp.common.tasks.facts import ConditioningFacts, FactProvenance, FactValidation, InstanceFact
from exp.runtime.environments.interface import Observation
from exp.runtime.environments.world_contracts import NativeImage, WorldAccess, WorldManifest
from exp.runtime.environments.world_responses import (
    validate_reply,
    validate_reply_state,
    validate_state_projection,
)
from exp.runtime.environments.worlds import WorldContext
from exp.simulation.engines.reasks import reask_proposal
from exp.simulation.engines.state_policy import StateEditPolicy
from exp.simulation.engines.text.prompt import INSTANCE_FACTS_GUIDANCE
from exp.simulation.engines.transition import Completion
from exp.simulation.world_model.frozen_memory import TaskView
from exp.simulation.world_model.grounded_session import GroundedTransitionEngine


class PredictorEvent(ContractModel):
    """Private provider evidence; a proposal is never presented as a committed world transition.

    Attributes:
        operation_id: Unique draw, with intent durably retained before provider dispatch.
        stage: Intended request, returned raw response or explicit contract refusal.
        payload: Original bounded request/evidence or response before parsing.
    """

    operation_id: str
    stage: Literal["intent", "response", "rejection"]
    payload: JsonObject


class GroundedWorldFactory:
    """Existing grounded model adapted to simulated domains of one frozen world manifest.

    Attributes:
        engine: Frozen predictor, artifact, task, retrieval policy and output protocol. Its
            optional instance facts condition every prediction; they are not state domains.
        manifest: Shared world ownership/effect policy. State keys name simulated domains.
        initial_state: Exact starting values for every simulated domain.
        complete: Existing admitted completion owner, responsible for physical attempt budgets.
        record: Durable private evidence sink; it must return only after recording the event.
        memory: Optional already eligible, pinned memory view; never changed by this runtime.
        maximum_reasks: Finite extra proposals after known contract refusal. All attempts
            consume the existing physical budget and remain in private evidence.
    """

    def __init__(
        self,
        engine: GroundedTransitionEngine,
        manifest: WorldManifest,
        initial_state: JsonObject,
        *,
        complete: Completion,
        record: Callable[[PredictorEvent], None],
        memory: TaskView | None = None,
        maximum_reasks: int = 0,
    ) -> None:
        """Bind one task without retrieving, loading credentials or calling a provider."""
        if engine.output_protocol != "json-tool":
            raise ValueError("executable grounded worlds require the json-tool edit protocol")
        self.engine, self.manifest = engine, manifest
        self.initial_state = deepcopy(initial_state)
        self.complete, self.record, self.memory = complete, record, memory
        if type(maximum_reasks) is not int or not 0 <= maximum_reasks <= 3:
            raise ValueError("maximum_reasks must be an integer between zero and three")
        self.maximum_reasks = maximum_reasks
        self._identity = engine.identity
        self._memory_identity = None if memory is None else memory.identity
        self._facts_bound = engine.facts is not None
        self._domains = {row.name for row in manifest.domains if row.owner == "simulated"}
        self._check_state(initial_state)
        self._definition_identity = sha256_json(self.definition)

    @property
    def definition(self) -> JsonObject:
        """Freeze all predictor semantics that an executable checkpoint may resume."""
        return {
            "implementation": "grounded-domain-world-v2",
            "engine": self._identity,
            "memory": self._memory_identity,
            "manifest": self.manifest.model_dump(mode="json"),
            "initial_state": deepcopy(self.initial_state),
            "maximum_reasks": self.maximum_reasks,
            # Bound only when the engine delivers facts, so fact-free definitions are unchanged.
            # The engine identity already binds the facts; this binds the guidance framing them.
            **(
                {"facts_prompt_sha256": sha256_json(INSTANCE_FACTS_GUIDANCE)}
                if self._facts_bound
                else {}
            ),
        }

    def _check_state(self, state: JsonObject) -> None:
        """Refuse invented, missing or foreign-owned state domains."""
        if set(state) != self._domains:
            raise ValueError("grounded world state must name exactly its simulated domains")
        validate_state_projection(
            state,
            {
                row.name: row.state_schema
                for row in self.manifest.domains
                if row.name in self._domains and row.state_schema is not None
            },
        )

    def _require_frozen(self) -> None:
        """Reject mutable predictor or recall context drift before any retrieval or completion."""
        if (
            sha256_json(self.definition) != self._definition_identity
            or self.engine.identity != self._identity
            or (self.memory is not None and self.memory.identity != self._memory_identity)
        ):
            raise ValueError("grounded executable predictor or memory changed after freezing")

    def create(self, task: TaskCase) -> "_GroundedWorldBackend":
        """Create an isolated state owner for exactly the registered public task."""
        self._require_frozen()
        if task != self.engine.task:
            raise ValueError("grounded executable world belongs to another public task")
        return _GroundedWorldBackend(self, deepcopy(self.initial_state), ())

    def restore(self, image: NativeImage) -> "_GroundedWorldBackend":
        """Verify and restore both candidate state and its full visible predictor history."""
        self._require_frozen()
        if set(image.parts) != {"predictor.json"}:
            raise ValueError("grounded checkpoint has an unexpected part inventory")
        wire = json.loads(image.parts["predictor.json"])
        state = wire["state"]
        history = wire["history"]
        if not isinstance(state, dict) or not isinstance(history, list):
            raise ValueError("grounded checkpoint has malformed state or history")
        self._check_state(state)
        backend = _GroundedWorldBackend(self, state, tuple(history))
        if backend.domain_digests() != image.domains:
            raise ValueError("grounded checkpoint state digest differs")
        return backend


class _GroundedWorldBackend:
    """One isolated simulated state; the outer kernel validates effects before commit."""

    def __init__(
        self, factory: GroundedWorldFactory, state: JsonObject, history: tuple[JsonObject, ...]
    ) -> None:
        """Restore only detached, checkpoint-owned values."""
        self.factory, self.state, self.history = factory, deepcopy(state), deepcopy(history)
        self._closed = False

    def access(self, action: ToolCall) -> WorldAccess:
        """Declare only the domains delivered for this tool, including its writable domains."""
        policy = self.factory.manifest.tool(action.name)
        return WorldAccess(reads=policy.domains(), writes=frozenset(policy.writes))

    def domain_digests(self) -> dict[str, Sha256]:
        """Separate business-state effects from retained conversation metadata."""
        return {key: sha256_json(value) for key, value in self.state.items()}

    def state_projection(self) -> JsonObject:
        """Expose detached candidate-owned domains for declared reply consistency checks."""
        return deepcopy(self.state)

    def capture(self) -> NativeImage:
        """Capture complete predictor context without serializing credentials or model clients."""
        return NativeImage(
            parts={
                "predictor.json": canonical_json_bytes(
                    {
                        "state": self.state,
                        "history": list(self.history),
                    }
                )
            },
            domains=self.domain_digests(),
        )

    def execute(self, action: ToolCall, context: WorldContext) -> Observation:
        """Commit one valid proposal after bounded corrections; all refusals stay recorded."""
        if self._closed:
            raise RuntimeError("grounded world is closed")
        self.factory._require_frozen()
        policy = self.factory.manifest.tool(action.name)
        clock: JsonObject = {"logical_time": context.now}
        visible = {name: deepcopy(self.state[name]) for name in sorted(policy.domains())}
        state: JsonObject = {"domains": visible, "environment": clock}
        memory = None
        if self.factory.memory is not None:
            messages = []
            for turn in self.history:
                actions = turn["actions"]
                observations = turn["observations"]
                assert isinstance(actions, list) and isinstance(observations, list)
                calls = tuple(ToolCall.model_validate(value) for value in actions)
                messages.append(
                    ModelMessage(
                        role="assistant", assistant_action=AssistantAction(tool_calls=calls)
                    )
                )
                for call, observed in zip(calls, observations, strict=True):
                    observation = Observation.model_validate(observed)
                    messages.append(
                        ModelMessage(
                            role="tool", content=observation.content, tool_call_id=call.call_id
                        )
                    )
            memory = self.factory.memory.context(
                messages, AssistantAction(tool_calls=(action,)), state
            )
        current_facts = None
        if context.native_reads:
            version = self.factory.engine.facts_version + len(self.history)
            current_facts = ConditioningFacts(
                state_version=version,
                facts=tuple(
                    InstanceFact(
                        key="native-read/" + receipt.bridge_sha256,
                        value=receipt.model_dump(mode="json"),
                        visibility="conditioning",
                        provenance=FactProvenance(
                            adapter="executable-native-read-v1",
                            source="native-checkpoint/" + receipt.checkpoint_sha256,
                            source_sha256=receipt.checkpoint_sha256,
                            derived_from_traces=self.factory.engine.task.source_trace_ids,
                            validation=FactValidation(
                                status="validated",
                                method="native fork: unchanged image, clock, events and RNG",
                            ),
                            state_version=version,
                        ),
                    )
                    for receipt in context.native_reads
                ),
            )
        state_policy = StateEditPolicy(
            roots=tuple(("domains", name) for name in policy.writes),
            read_roots=tuple(("domains", name) for name in sorted(policy.domains())),
        )
        operation = "grounded-world-" + uuid4().hex
        prepared = self.factory.engine.prepare(
            state,
            self.history,
            (action,),
            memory=memory,
            current_facts=current_facts,
            operation_id=operation,
            state_policy=state_policy,
            state_schemas={
                row.name: row.state_schema
                for row in self.factory.manifest.domains
                if row.name in policy.domains() and row.state_schema is not None
            },
            reply_bindings={action.name: policy.reply_bindings},
            reply_schemas={action.name: schema}
            if (schema := self.factory.manifest.tool(action.name).reply_schema) is not None
            else None,
        )
        body = prepared.body
        for attempt in range(self.factory.maximum_reasks + 1):
            logical = operation if attempt == 0 else operation + f"-reask-{attempt}"
            self.factory.record(
                PredictorEvent(
                    operation_id=logical,
                    stage="intent",
                    payload={
                        "request": body,
                        "evidence": prepared.evidence,
                        "operation": operation,
                        "refused_proposals": attempt,
                    },
                )
            )
            # Provider exceptions are deliberately outside the refusal handler. A lost response
            # is not a known invalid proposal and must never silently buy another draw.
            raw = self.factory.complete(body, logical)
            self.factory.record(PredictorEvent(operation_id=logical, stage="response", payload=raw))
            self.factory._require_frozen()
            try:
                domains, observation = self._validate_proposal(raw, action, state, state_policy)
            except ValueError as error:
                reason = type(error).__name__ + ": " + redact_secret_text(str(error))[0][:2000]
                self.factory.record(
                    PredictorEvent(
                        operation_id=logical,
                        stage="rejection",
                        payload={
                            "reason": reason,
                            "operation": operation,
                            "attempt": attempt,
                        },
                    )
                )
                if attempt == self.factory.maximum_reasks:
                    raise
                body = reask_proposal(body, raw, reason)
                continue
            observation = observation.model_copy(
                update={
                    "metadata": {
                        **observation.metadata,
                        "prediction_operation": operation,
                        "refused_proposals": attempt,
                    }
                }
            )
            self.state = deepcopy(domains)
            self.history += (
                {
                    "actions": [action.model_dump(mode="json")],
                    "observations": [observation.model_dump(mode="json")],
                },
            )
            return observation
        raise RuntimeError("bounded proposal loop ended without a result")

    def adopt(self, action: ToolCall, observation: Observation, state: JsonObject) -> None:
        """Commit one exact transition computed outside the predictor, such as induced code.

        The new state replaces this owner's domains and the call joins the visible history,
        so later predictions see every applied action and their instance-fact versions keep
        counting applied actions. The outer kernel still verifies effects before commit.

        Raises:
            ValueError: The state names other domains or violates a declared state schema.
        """
        if self._closed:
            raise RuntimeError("grounded world is closed")
        self.factory._check_state(state)
        self.state = deepcopy(state)
        self.history += (
            {
                "actions": [action.model_dump(mode="json")],
                "observations": [observation.model_dump(mode="json")],
            },
        )

    def _validate_proposal(
        self,
        raw: JsonObject,
        action: ToolCall,
        state: JsonObject,
        state_policy: StateEditPolicy,
    ) -> tuple[JsonObject, Observation]:
        """Check a saved proposal completely before changing state or conversation history."""
        policy = self.factory.manifest.tool(action.name)
        proposal = self.factory.engine.parse(raw, (action,), current_state=state)
        if proposal.terminal or len(proposal.observations) != 1:
            raise ValueError(
                "simulated tool calls must return one observation without terminating the actor"
            )
        state_policy.require(proposal)
        next_state = proposal.next_state(state)
        if (
            set(next_state) != {"domains", "environment"}
            or next_state["environment"] != state["environment"]
        ):
            raise ValueError("predictor cannot change the environment-owned logical clock")
        domains = next_state["domains"]
        if not isinstance(domains, dict):
            raise ValueError("simulated domains must remain an object")
        if set(domains) != policy.domains():
            raise ValueError("prediction must retain exactly its declared visible domains")
        domains = {**deepcopy(self.state), **domains}
        self.factory._check_state(domains)
        predicted = proposal.observations[0]
        if predicted.call_id != action.call_id:
            raise ValueError("simulated observation belongs to another action")
        if predicted.encoding == "text" and not isinstance(predicted.reply, str):
            raise ValueError("text observation must preserve a string reply")
        observation = Observation(
            content=predicted.reply
            if isinstance(predicted.reply, str) and predicted.encoding == "text"
            else canonical_json_bytes(predicted.reply).decode(),
            is_error=predicted.is_error,
            metadata={"call_id": action.call_id, "authority": "simulated"},
        )
        validate_reply(observation, policy.reply_schema)
        validate_reply_state(
            observation,
            policy.reply_bindings,
            self.state,
            domains,
            action=action,
            domains=policy.domains(),
        )
        return domains, observation

    def close(self) -> None:
        """Invalidate only this state owner, retaining durable evidence in the supplied sink."""
        self._closed = True
