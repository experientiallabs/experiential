"""Run the existing grounded predictor through the shared durable commit owner."""

import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from exp.common.core.artifacts import ArtifactInput, JsonObject, Sha256, sha256_json
from exp.common.core.json_schema import expand_local_references
from exp.common.models import (
    AssistantAction,
    ModelFinishReason,
    ModelMessage,
    ModelRequest,
    ModelResponse,
    ToolCall,
    ToolChoice,
)
from exp.common.project import ArtifactStore
from exp.common.tasks import TaskCase, ToolSchema
from exp.common.tasks.facts import ConditioningFacts, InstanceFacts
from exp.runtime.environments import Observation
from exp.runtime.environments.world_responses import ReplyStateBinding
from exp.runtime.sessions.store import SessionBinding, SessionStore
from exp.simulation.engines.state_policy import StateEditPolicy
from exp.simulation.engines.text.prompt import (
    NATIVE_ARTIFACT_PLAN_GUIDANCE,
    MemoryContext,
    parse_world_model_transition,
    validate_transition_action,
)
from exp.simulation.engines.transition import (
    ApplyEdits,
    Completion,
    PreparedPrediction,
    ProposedObservation,
    ProposedTransition,
    ReplaceState,
    TransitionEngine,
)
from exp.simulation.retrieval import load_rag_index
from exp.simulation.retrieval.contracts import RAGTransition
from exp.simulation.retrieval.evidence_reading import EvidenceReader
from exp.simulation.retrieval.selection import EvidencePolicy
from exp.simulation.world_model.checkpoints import WorldCheckpoint, load_world_checkpoint
from exp.simulation.world_model.durable import DurableWorldModelSession, SessionPolicy
from exp.simulation.world_model.experience_memory import retrieved_source
from exp.simulation.world_model.memory_projection import (
    RESEARCH_HISTORY_ABLATION,
    project_research_request,
    without_research,
)
from exp.simulation.world_model.prediction_reading import PredictionPreparation
from exp.simulation.world_model.resumption import Resumption
from exp.simulation.world_model.runtime import GroundedWorldModel
from exp.simulation.world_model.session_research import (
    OnlineResearchPolicy,
    ResearchTransitionEngine,
    SessionResearch,
)


class _GroundedJsonProposal(ProposedTransition):
    """Typed observations plus compact edits to the candidate state.

    Attributes:
        effect: Ordered set/delete/append/remove_value/copy edits to the current candidate state.
            Re-writing the complete state every turn dominated output tokens; an edit that does
            not apply is refused and asked again like any invalid proposal.
        terminal: Tool observations cannot terminate the actor before delivery.
    """

    effect: ApplyEdits
    terminal: Literal[False] = False


def frame_grounded_output(
    request: ModelRequest,
    *,
    output_protocol: Literal["text", "json-tool"] = "text",
    json_object_output: bool = False,
    state_policy: StateEditPolicy | None = None,
) -> ModelRequest:
    """Change proposal framing while retaining identical permitted context and evidence.

    Capability and budget admission remain with the session and completion owners.
    This pure adapter performs no retrieval, inference, serialization repair or state change.
    """
    framed = request.model_copy(update={"json_object_output": json_object_output})
    if output_protocol == "json-tool":
        schema = _GroundedJsonProposal.model_json_schema()
        if state_policy is not None:
            schema["$defs"]["ApplyEdits"]["properties"]["edits"] = state_policy.edit_schema()
        framed = framed.model_copy(
            update={
                "tools": (
                    ToolSchema(
                        name="submit_transition",
                        description="Submit observations and edits to the candidate state.",
                        input_schema=expand_local_references(schema),
                    ),
                ),
                "tool_choice": ToolChoice(name="submit_transition"),
            }
        )
    return framed


class GroundedTransitionEngine:
    """Adapt existing request framing and parsing without translating the provider wire.

    Attributes:
        runtime: Verified grounded artifact, retriever, parser and capacity binding.
        task: Public task and initialization; no evaluator references are accepted.
        maximum_output_tokens: Frozen output allowance shared with completion admission.
        json_object_output: Explicit structured-output treatment, false for plain text calls.
        output_protocol: Plain JSON text or one typed transition proposal.
        served_model_id: Catalog pin for the provider-reported ID of the frozen model.
        facts: Validated instance facts; each prediction receives only the conditioning facts
            whose state version is not later than the state it predicts from.
        research_history: False removes current research delivery, not prior learning effects.
        facts_version: State version of the session's first action: the recorded actions
            applied before it, such as a disclosed public prefix.
    """

    def __init__(
        self,
        runtime: GroundedWorldModel,
        task: TaskCase,
        *,
        maximum_output_tokens: int,
        json_object_output: bool = False,
        output_protocol: Literal["text", "json-tool"] = "text",
        served_model_id: str | None = None,
        facts: InstanceFacts | None = None,
        facts_version: int = 0,
        research_history: bool = True,
        prediction_reader: PredictionPreparation | None = None,
    ) -> None:
        """Bind an existing predictor to a single public fixed-action task."""
        if maximum_output_tokens <= 0 or task.reference_answer is not None:
            raise ValueError("use positive output capacity and a task without reference answers")
        if json_object_output and runtime.capabilities.supports_structured_output is not True:
            raise ValueError("JSON output requires declared structured-output support")
        if output_protocol == "json-tool" and (
            runtime.capabilities.supports_tools is not True or json_object_output
        ):
            raise ValueError("Typed proposal requires tool support without JSON text mode")
        self.runtime, self.task = runtime, TaskCase.model_validate_json(task.model_dump_json())
        self.maximum_output_tokens = maximum_output_tokens
        self.json_object_output = json_object_output
        self.output_protocol = output_protocol
        self._models = [runtime.artifact.model]
        if served_model_id is not None:
            self._models.append(
                runtime.artifact.model.model_copy(update={"model_id": served_model_id})
            )
        self.research_history = research_history
        self.prediction_reader = prediction_reader
        if facts_version < 0:
            raise ValueError("the facts state version of the first action cannot be negative")
        self.facts = None if facts is None else facts.require_validated("this session")
        self.facts_version = facts_version

    @property
    def identity(self) -> Sha256:
        """Bind knowledge, source, task, selection and actual generation limits."""
        return sha256_json(
            {
                "artifact": self.runtime.artifact_input.model_dump(mode="json"),
                "knowledge": self.runtime.artifact.model_dump(mode="json"),
                "selection": self.runtime.evidence_selector.policy.identity
                if self.runtime.evidence_selector
                else None,
                "task": self.task.model_dump(mode="json"),
                "output_tokens": self.maximum_output_tokens,
                "json_object_output": self.json_object_output,
                "output_protocol": self.output_protocol,
                "protocol": "grounded-typed-response-v3",
                **(
                    {}
                    if self.prediction_reader is None
                    else {
                        "prediction_reader": self.prediction_reader.identity,
                    }
                ),
                **(
                    {} if self.research_history else {"research_history": RESEARCH_HISTORY_ABLATION}
                ),
                # Recorded only when present, so fact-free identities stay unchanged.
                **(
                    {}
                    if self.facts is None
                    else {
                        "facts": self.facts.model_dump(mode="json"),
                        "facts_version": self.facts_version,
                    }
                ),
                # Recorded only for native-artifact tasks, so other identities stay unchanged.
                **(
                    {"native_artifact_guidance_sha256": sha256_json(NATIVE_ARTIFACT_PLAN_GUIDANCE)}
                    if self.task.initial_context.get("native_artifact_binding") is not None
                    else {}
                ),
            }
        )

    def prepare(
        self,
        state: JsonObject,
        history: tuple[JsonObject, ...],
        actions: tuple[ToolCall, ...],
        *,
        memory: MemoryContext | None = None,
        current_facts: ConditioningFacts | None = None,
        reply_schemas: Mapping[str, JsonObject] | None = None,
        state_policy: StateEditPolicy | None = None,
        reply_bindings: Mapping[str, tuple[ReplyStateBinding, ...]] | None = None,
        state_schemas: Mapping[str, JsonObject] | None = None,
        operation_id: str | None = None,
    ) -> PreparedPrediction:
        """Render the typed context once, pack it, and frame the output protocol.

        Memory enters as typed channels before rendering, so packing and capacity admission
        see the complete request and no caller edits serialized prompt text.
        """
        if state_policy is not None and self.output_protocol != "json-tool":
            raise ValueError("bounded executable prediction requires the json-tool protocol")
        names = {tool.name for tool in self.task.tools}
        if not actions or any(action.name not in names for action in actions):
            raise ValueError("declare every supplied action in the public task tools")
        visible: list[ModelMessage] = []
        applied = 0
        for turn in history:
            calls, observations = turn.get("actions"), turn.get("observations")
            if not isinstance(calls, list) or not isinstance(observations, list):
                raise ValueError("stored grounded history requires actions and observations")
            prior = tuple(ToolCall.model_validate(call) for call in calls)
            applied += len(prior)
            decoded = tuple(Observation.model_validate(value) for value in observations)
            if len(prior) != len(decoded):
                raise ValueError("stored grounded history has a different result inventory")
            visible.append(
                ModelMessage(role="assistant", assistant_action=AssistantAction(tool_calls=prior))
            )
            for call, observation in zip(prior, decoded, strict=True):
                if observation.metadata.get("call_id") != call.call_id:
                    raise ValueError("stored grounded observation identity differs")
                visible.append(
                    ModelMessage(
                        role="tool", content=observation.content, tool_call_id=call.call_id
                    )
                )
        conditioning = (
            None if self.facts is None else self.facts.conditioning(self.facts_version + applied)
        )
        if current_facts is not None:
            if current_facts.state_version != self.facts_version + applied:
                raise ValueError("current native facts belong to another prediction state")
            facts = (() if conditioning is None else conditioning.facts) + current_facts.facts
            if len({row.key for row in facts}) != len(facts):
                raise ValueError("native facts collide with existing instance facts")
            conditioning = ConditioningFacts(state_version=current_facts.state_version, facts=facts)
        task = self.task
        if state_schemas:
            task = task.model_copy(
                update={
                    "tools": tuple(
                        tool.model_copy(
                            update={
                                "description": tool.description
                                + "\nPreserve these domain state_schemas on every transition: "
                                + json.dumps(dict(state_schemas), sort_keys=True)
                            }
                        )
                        if tool.name in {action.name for action in actions}
                        else tool
                        for tool in task.tools
                    )
                }
            )
        if reply_schemas:
            if not set(reply_schemas) <= {action.name for action in actions}:
                raise ValueError("reply schemas must belong to the current action batch")
            task = task.model_copy(
                update={
                    "tools": tuple(
                        tool.model_copy(
                            update={
                                "description": tool.description
                                + "\nSuccessful replies must be JSON matching this schema: "
                                + json.dumps(reply_schemas[tool.name], sort_keys=True)
                                + ". Error replies remain is_error=true; do not invent success."
                            }
                        )
                        if tool.name in reply_schemas
                        else tool
                        for tool in task.tools
                    )
                }
            )
        if reply_bindings:
            task = task.model_copy(
                update={
                    "tools": tuple(
                        tool.model_copy(
                            update={
                                "description": tool.description
                                + "\nJSON reply fields must equal the specified before/after "
                                + "state fields. Reply paths start at the reply root; state "
                                + "paths start inside environment_state.domains: "
                                + json.dumps(
                                    [
                                        row.model_dump(mode="json")
                                        for row in reply_bindings[tool.name]
                                    ]
                                )
                                + ". Path operations: argument reads a key/index from the call; "
                                + "select matches exactly one item by its field and argument; "
                                + "slice_start/slice_count select a range, subtracting offset "
                                + "from start. each applies that slice to every row. Only "
                                + "clamp=true allows a shorter or empty page; other ranges "
                                + "must exist in full. length returns the array length; fields "
                                + "projects the named relative paths for each item. "
                                + "argument_value uses the call value or its declared default. "
                                + "empty_array_default exposes the declared values only for an "
                                + "observed empty array; it never fills missing or null state. "
                                + "page_cursor returns start plus the returned page length, "
                                + "or null at or beyond the end of the array, including when "
                                + "the last page is full. It never points past the last page."
                            }
                        )
                        if reply_bindings.get(tool.name)
                        else tool
                        for tool in task.tools
                    )
                }
            )
        if state_policy is not None:
            task = task.model_copy(
                update={
                    "tools": tuple(
                        tool.model_copy(
                            update={
                                "description": tool.description
                                + "\nRuntime state authority: edit only these path roots or their "
                                + "descendants: "
                                + json.dumps(state_policy.roots)
                                + ". All other state is read-only. Never replace ancestors. "
                                + "The runtime alone advances environment.logical_time after the "
                                + "call; never propose clock edits. Empty roots require no edits."
                                + " To preserve existing structures exactly, use an edit "
                                + '{"op":"copy","from":[...],"path":[...]} then edit the copy. '
                                + "Copy sources must be within these readable roots: "
                                + json.dumps((*state_policy.roots, *state_policy.read_roots))
                                + ". Predict the exact supplied action, not the customer's "
                                + "desired outcome. An action may execute successfully while "
                                + "making the task fail. Preserve that mistake; never repair "
                                + "arguments or substitute a historical successful solution. "
                                + "Apply ordered operations to the evolving state, including "
                                + "index shifts. Never sort or treat them as simultaneous."
                            }
                        )
                        if tool.name in {action.name for action in actions}
                        else tool
                        for tool in task.tools
                    )
                }
            )
        if not self.research_history and memory is not None:
            memory = without_research(memory)
        prepared = self.runtime.prepare_turn(
            task=task,
            visible_messages=visible,
            candidate_response=AssistantAction(tool_calls=actions),
            excluded_lineage_ids=(self.task.lineage_group_id,),
            maximum_output_tokens=self.maximum_output_tokens,
            state=state,
            memory=memory,
            output_protocol=self.output_protocol,
            facts=conditioning,
        )
        if not self.research_history:
            prepared = replace(prepared, request=project_research_request(prepared.request))
        if self.prediction_reader is not None:
            if operation_id is None and len(actions) == 1:
                # The durable session's operation for one fixed action is its call ID.
                operation_id = actions[0].call_id
            if operation_id is None:
                raise ValueError("prediction reading requires an owned operation_id before calls")
            prepared = self.prediction_reader.resolve(prepared, operation_id)
        framed = frame_grounded_output(
            prepared.request,
            output_protocol=self.output_protocol,
            json_object_output=self.json_object_output,
            state_policy=state_policy,
        )
        prepared = replace(prepared, request=framed).fit_context(
            self.runtime.capabilities, self.runtime.token_counter
        )
        request = prepared.request.model_dump(mode="json")
        return PreparedPrediction(
            body=request,
            evidence={
                "request_sha256": sha256_json(request),
                "memory_claims": []
                if memory is None
                else [
                    {"id": row["id"], "sha256": sha256_json(row)}
                    for row in (*memory.session, *memory.prior_trajectories)
                    if row.get("kind") == "wiki-claim" and isinstance(row.get("id"), str)
                ],
                "grounded_transition_ids": [
                    match.transition.transition_id for match in prepared.matches
                ],
                # A digest, never keys or values: this evidence reaches investigations, which
                # must not learn instance facts into memory other traces read.
                **(
                    {}
                    if conditioning is None
                    else {
                        "instance_facts_version": conditioning.state_version,
                        "instance_facts_sha256": sha256_json(conditioning.rendered()),
                    }
                ),
            },
        )

    def parse(
        self,
        proposal: JsonObject,
        actions: tuple[ToolCall, ...],
        *,
        current_state: JsonObject | None = None,
    ) -> ProposedTransition:
        """Preserve raw text until the existing protocol parser validates the result."""
        response = ModelResponse.model_validate(proposal)
        if response.model not in self._models:
            raise ValueError("grounded response model differs from the frozen artifact")
        if response.finish_reason != ModelFinishReason.COMPLETED:
            raise ValueError("grounded generation did not complete; raw output retained")
        if self.output_protocol == "json-tool":
            calls = response.output.tool_calls
            if len(calls) != 1 or calls[0].name != "submit_transition":
                raise ValueError("Expected one typed transition proposal")
            transition = _GroundedJsonProposal.model_validate(calls[0].arguments)
            if tuple(row.call_id for row in transition.observations) != tuple(
                action.call_id for action in actions
            ):
                raise ValueError("proposal observation identity/order differs from the action")
            return transition
        transition = parse_world_model_transition(response.output)
        validate_transition_action(transition, AssistantAction(tool_calls=actions))
        return ProposedTransition(
            observations=tuple(
                ProposedObservation(
                    call_id=result.call_id,
                    reply=result.content,
                    encoding="text",
                    is_error=result.is_error,
                )
                for result in transition.tool_results
            ),
            effect=ReplaceState(state=transition.state),
            terminal=transition.terminal,
        )


def open_grounded_session(
    *,
    runtime: GroundedWorldModel,
    task: TaskCase,
    store: SessionStore,
    artifacts: ArtifactStore,
    session_id: str,
    policy: SessionPolicy,
    complete: Completion,
    lock_path: Path,
    maximum_output_tokens: int,
    json_object_output: bool = False,
    output_protocol: Literal["text", "json-tool"] = "text",
    served_model_id: str | None = None,
    clock: Callable[[], datetime] | None = None,
    checkpoint: WorldCheckpoint | None = None,
    research_policy: OnlineResearchPolicy | None = None,
    research_complete: Completion | None = None,
    research_selection: EvidencePolicy | None = None,
    research_reader: EvidenceReader | None = None,
    research_prior: Mapping[str, JsonObject] | None = None,
    research_prior_identity: JsonObject | None = None,
    research_overview: Sequence[JsonObject] = (),
    research_prior_sources: Mapping[str, Sequence[ArtifactInput]] | None = None,
    research_overview_sources: Sequence[ArtifactInput] = (),
    research_overview_view: Callable[[frozenset[str]], list[JsonObject]] | None = None,
    research_overview_sources_of: Callable[[Sequence[JsonObject]], Sequence[ArtifactInput]]
    | None = None,
    resumption: Resumption | None = None,
    preparation_dispatches: bool = True,
    facts: InstanceFacts | None = None,
    facts_version: int = 0,
    prediction_reader: PredictionPreparation | None = None,
) -> DurableWorldModelSession:
    """Open one candidate world using public context and the existing guarded store.

    Args:
        runtime: Verified existing predictor and retrieval configuration.
        task: Public current-instance context, with no evaluator answer.
        store: Shared durable session owner.
        artifacts: Immutable proposal and input owner.
        session_id: Caller-owned stable operation namespace.
        policy: Frozen knowledge, deadline and execution allowance.
        complete: Metered typed-model completion adapter.
        lock_path: Single-writer lease for the session.
        maximum_output_tokens: Frozen output capacity.
        json_object_output: Require provider JSON mode when the selected route supports it.
        output_protocol: Freeze text framing or typed native-value proposals.
        served_model_id: Catalog pin for the provider-reported ID of the frozen model.
        clock: Optional aware clock for deterministic deadline qualification.
        checkpoint: Verified candidate checkpoint with exactly matching world and public context.
        research_policy: Explicit opt-in to past-reply feedback and provisional memory.
        research_complete: Separately metered investigator, required with research policy.
        research_selection: Exact registered query-dependent selector for session notes.
        research_reader: Optional bounded reader for post-feedback investigations.
        research_prior: Active research from a frozen memory version of earlier trajectories.
        research_prior_identity: Memory version identity, required with prior research.
        research_overview: Budgeted memory overview delivered with every prediction.
        research_prior_sources: Source trajectories behind each prior record.
        research_overview_sources: Source trajectories behind the overview rows.
        research_overview_view: Re-renders the overview minus this session's retired notes.
        research_overview_sources_of: Resolves sources behind each delivered overview.
        resumption: Explicit authority to continue a session a stopped process left open.
        preparation_dispatches: Whether preparing a step can send requests (query embeddings
            or memory-recall decisions).
        facts: Validated instance facts, delivered in their own channel by state version.
        facts_version: State version of the session's first action.
        prediction_reader: Optional durable step that rewrites each framed prediction request,
            such as session-scoped guidance.

    Returns:
        Durable session; no reference state or tool outcomes are used to initialize it.
    """
    if checkpoint is not None:
        reference = ArtifactInput(
            artifact_id=checkpoint.artifact_id,
            sha256=sha256_json(artifacts.read(checkpoint.artifact_id).manifest),
        )
        if load_world_checkpoint(artifacts, reference) != checkpoint:
            raise ValueError("checkpoint differs from its persisted source")
        if checkpoint.world_model != runtime.artifact_input:
            raise ValueError("checkpoint belongs to a different simulator artifact")
        if checkpoint.public_context != task.initial_context:
            raise ValueError("checkpoint public context differs from the selected task")
    if (research_policy is None) != (research_complete is None):
        raise ValueError("online research requires both policy and metered investigator")
    if (research_selection is None) != (
        research_policy is None and research_prior_identity is None
    ) or (research_prior is not None and research_prior_identity is None):
        raise ValueError("research delivery requires a selector and an identified memory source")
    engine: TransitionEngine = GroundedTransitionEngine(
        runtime,
        task,
        maximum_output_tokens=maximum_output_tokens,
        json_object_output=json_object_output,
        output_protocol=output_protocol,
        served_model_id=served_model_id,
        facts=facts,
        facts_version=facts_version,
        prediction_reader=prediction_reader,
    )
    research = None
    if research_selection is not None:
        index: dict[str, RAGTransition] = {}

        def retrieved_sources_of(ids: Sequence[str]) -> tuple[ArtifactInput, ...]:
            """Typed sources for the recorded traces behind retrieved transitions."""
            if not index:
                rag = load_rag_index(artifacts, runtime.artifact.serving_rag.artifact_id)
                index.update((row.transition_id, row) for row in rag.transitions)
            missing = [key for key in ids if key not in index]
            if missing:
                raise ValueError("retrieved transition is absent from the simulator's index")
            rows = sorted({(index[key].trace_id, index[key].lineage_id) for key in ids})
            return tuple(
                retrieved_source(
                    artifacts,
                    trace_id=trace,
                    task_lineage=lineage,
                    created_at=(clock or (lambda: datetime.now(UTC)))(),
                    code_revision=policy.code_revision,
                )
                for trace, lineage in rows
            )

        research = SessionResearch(
            selection=research_selection,
            reader=research_reader,
            session_id=session_id,
            policy=research_policy,
            store=store,
            artifacts=artifacts,
            complete=research_complete,
            lock_path=lock_path,
            clock=clock or (lambda: datetime.now(UTC)),
            code_revision=policy.code_revision,
            prior=research_prior,
            prior_identity=research_prior_identity,
            overview=research_overview,
            prior_sources=research_prior_sources,
            overview_sources=research_overview_sources,
            overview_view=research_overview_view,
            overview_sources_of=research_overview_sources_of,
            retrieved_sources_of=retrieved_sources_of,
            resumption=resumption,
        )
        engine = ResearchTransitionEngine(engine, research)
    binding = SessionBinding(
        deadline=policy.deadline,
        target_id="grounded@text",
        initialization_receipt_sha256=sha256_json(task.initial_context)
        if checkpoint is None
        else sha256_json(
            {
                "public_context": task.initial_context,
                "checkpoint": checkpoint.artifact_id,
                "state": checkpoint.current.model_dump(mode="json"),
            }
        ),
        knowledge_sha256=sha256_json(runtime.artifact.model_dump(mode="json")),
        corpus_sha256=runtime.artifact.serving_rag.sha256,
        representation_sha256=engine.identity,
        execution_policy_sha256=sha256_json(policy.model_dump(mode="json")),
    )

    def validate(state: JsonObject) -> None:
        """Require JSON object state without introducing business semantics."""
        json.dumps(state, allow_nan=False)

    return DurableWorldModelSession(
        engine=engine,
        research=research,
        store=store,
        artifacts=artifacts,
        session_id=session_id,
        binding=binding,
        initial_state={} if checkpoint is None else checkpoint.current.business_state,
        public_prefix=() if checkpoint is None else checkpoint.current.history,
        policy=policy,
        complete=complete,
        lock_path=lock_path,
        validate=validate,
        clock=clock,
        resumption=resumption,
        preparation_dispatches=preparation_dispatches,
        evaluation_context={
            "task_context": task.initial_context,
            "sources": {
                "frozen_profile_sha256": sha256_json(
                    {
                        "world_model": runtime.artifact_input.model_dump(mode="json"),
                        "task": task.model_dump(mode="json", exclude={"task_id"}),
                        "checkpoint": None if checkpoint is None else checkpoint.artifact_id,
                        "output_protocol": output_protocol,
                        "json_object_output": json_object_output,
                        "served_model_id": served_model_id,
                        "output_tokens": maximum_output_tokens,
                        "selection": None
                        if runtime.evidence_selector is None
                        else runtime.evidence_selector.policy.identity,
                        "prior_memory": research_prior_identity,
                        "prior_records": research_prior,
                        "overview": research_overview,
                        "memory_selection": None
                        if research_selection is None
                        else research_selection.identity,
                        "research_policy": None
                        if research_policy is None
                        else research_policy.model_dump(mode="json"),
                    }
                ),
                **(
                    {"candidate_checkpoint": checkpoint.artifact_id}
                    if checkpoint is not None
                    else {}
                ),
                **(
                    {"online_research_policy": research_policy.model_dump(mode="json")}
                    if research_policy is not None
                    else {}
                ),
                **(
                    {"prior_memory": research_prior_identity}
                    if research_prior_identity is not None
                    else {}
                ),
            },
            "contract": {"state_observation": "No complete native state supplied"},
            "knowledge_sha256": runtime.artifact.knowledge.identity
            if runtime.artifact.knowledge
            else None,
        },
    )
