"""Run the existing grounded predictor through the shared durable commit owner."""

import json
from collections.abc import Callable, Mapping, Sequence
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
from exp.runtime.environments import Observation
from exp.runtime.sessions.store import SessionBinding, SessionStore
from exp.simulation.engines.text.prompt import (
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
from exp.simulation.retrieval.selection import EvidencePolicy
from exp.simulation.world_model.durable import DurableWorldModelSession, SessionPolicy
from exp.simulation.world_model.experience_memory import retrieved_source
from exp.simulation.world_model.runtime import GroundedWorldModel
from exp.simulation.world_model.session_research import (
    OnlineResearchPolicy,
    ResearchTransitionEngine,
    SessionResearch,
)


class _GroundedJsonProposal(ProposedTransition):
    """Typed observations plus compact edits to the candidate state.

    Attributes:
        effect: Ordered set/delete/append/remove_value edits to the current candidate state.
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
) -> ModelRequest:
    """Change proposal framing while retaining identical permitted context and evidence.

    Capability and budget admission remain with the session and completion owners.
    This pure adapter performs no retrieval, inference, serialization repair or state change.
    """
    framed = request.model_copy(update={"json_object_output": json_object_output})
    if output_protocol == "json-tool":
        framed = framed.model_copy(
            update={
                "tools": (
                    ToolSchema(
                        name="submit_transition",
                        description="Submit observations and edits to the candidate state.",
                        input_schema=expand_local_references(
                            _GroundedJsonProposal.model_json_schema()
                        ),
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
            }
        )

    def prepare(
        self,
        state: JsonObject,
        history: tuple[JsonObject, ...],
        actions: tuple[ToolCall, ...],
        *,
        memory: MemoryContext | None = None,
    ) -> PreparedPrediction:
        """Render the typed context once, pack it, and frame the output protocol.

        Memory enters as typed channels before rendering, so packing and capacity admission
        see the complete request and no caller edits serialized prompt text.
        """
        names = {tool.name for tool in self.task.tools}
        if not actions or any(action.name not in names for action in actions):
            raise ValueError("declare every supplied action in the public task tools")
        visible: list[ModelMessage] = []
        for turn in history:
            calls, observations = turn.get("actions"), turn.get("observations")
            if not isinstance(calls, list) or not isinstance(observations, list):
                raise ValueError("stored grounded history requires actions and observations")
            prior = tuple(ToolCall.model_validate(call) for call in calls)
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
        prepared = self.runtime.prepare_turn(
            task=self.task,
            visible_messages=visible,
            candidate_response=AssistantAction(tool_calls=actions),
            excluded_lineage_ids=(self.task.lineage_group_id,),
            maximum_output_tokens=self.maximum_output_tokens,
            state=state,
            memory=memory,
            output_protocol=self.output_protocol,
        ).fit_context(self.runtime.capabilities, self.runtime.token_counter)
        framed = frame_grounded_output(
            prepared.request,
            output_protocol=self.output_protocol,
            json_object_output=self.json_object_output,
        )
        request = framed.model_dump(mode="json")
        return PreparedPrediction(
            body=request,
            evidence={
                "request_sha256": sha256_json(request),
                "grounded_transition_ids": [
                    match.transition.transition_id for match in prepared.matches
                ],
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
    research_policy: OnlineResearchPolicy | None = None,
    research_complete: Completion | None = None,
    research_selection: EvidencePolicy | None = None,
    research_prior: Mapping[str, JsonObject] | None = None,
    research_prior_identity: JsonObject | None = None,
    research_overview: Sequence[JsonObject] = (),
    research_prior_sources: Mapping[str, Sequence[ArtifactInput]] | None = None,
    research_overview_sources: Sequence[ArtifactInput] = (),
    research_overview_view: Callable[[frozenset[str]], list[JsonObject]] | None = None,
    research_overview_sources_of: Callable[[Sequence[JsonObject]], Sequence[ArtifactInput]]
    | None = None,
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
        research_policy: Explicit opt-in to past-reply feedback and provisional memory.
        research_complete: Separately metered investigator, required with research policy.
        research_selection: Exact registered query-dependent selector for session notes.
        research_prior: Active research from a frozen memory version of earlier trajectories.
        research_prior_identity: Memory version identity, required with prior research.
        research_overview: Budgeted memory overview delivered with every prediction.
        research_prior_sources: Source trajectories behind each prior record.
        research_overview_sources: Source trajectories behind the overview rows.
        research_overview_view: Re-renders the overview minus this session's retired notes.
        research_overview_sources_of: Resolves sources behind each delivered overview.

    Returns:
        Durable session; no reference state or tool outcomes are used to initialize it.
    """
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
        )
        engine = ResearchTransitionEngine(engine, research)
    binding = SessionBinding(
        deadline=policy.deadline,
        target_id="grounded@text",
        initialization_receipt_sha256=sha256_json(task.initial_context),
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
        initial_state={},
        public_prefix=(),
        policy=policy,
        complete=complete,
        lock_path=lock_path,
        validate=validate,
        clock=clock,
        evaluation_context={
            "task_context": task.initial_context,
            "sources": {
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
